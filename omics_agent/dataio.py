"""数据 IO 层：CSV 解析缓存 + 数据集指纹。

两个动机都来自实测：
  1. expression.csv 37MB（19938×227）。一轮分析里 load_qc / diff / enrichment 各解析一遍，
     而 discover_dataset 为了"数行数认出分组表"，还会把另外几个大 csv（基因卡片、GO 注释）整读一遍——
     光"解析 CSV"就占 3~4 秒。按 (路径, mtime, size) 缓存解析结果，一轮分析只付一次成本。
  2. 缓存 key 必须能区分"同一句话、不同数据集"，否则会把上一份报告返回给新数据（正确性 bug）。

注意：read_csv_cached 返回的是**共享对象**，调用方不要原地修改它
（现有节点都是只读：diff / enrichment / 质检都不 mutate）。
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from typing import Any

import pandas as pd

from omics_agent import metrics

_MAX_ENTRIES = int(os.getenv("CSV_CACHE_ENTRIES", "6"))
_DF_CACHE: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()
_LOCK = threading.Lock()


def _file_sig(path: str) -> tuple[int, int]:
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size)


def _freeze(kwargs: dict[str, Any]) -> tuple | None:
    """把 read_csv 的关键字参数变成可哈希的 key；有不可哈希参数就不进缓存。"""
    frozen: list[tuple[str, Any]] = []
    for name in sorted(kwargs):
        value = kwargs[name]
        if isinstance(value, (str, int, float, bool, type(None))):
            frozen.append((name, value))
        elif isinstance(value, (list, tuple)) and all(
            isinstance(x, (str, int, float, bool)) for x in value
        ):
            frozen.append((name, tuple(value)))
        else:
            return None
    return tuple(frozen)


def read_csv_cached(path: str, **kwargs: Any) -> pd.DataFrame:
    """按文件签名缓存解析结果；文件被改过（mtime/size 变了）自动失效。"""
    frozen = _freeze(kwargs)
    if frozen is None:
        return pd.read_csv(path, **kwargs)
    try:
        key = (os.path.abspath(path), _file_sig(path), frozen)
    except OSError:  # 文件不存在等：交给 pandas 抛原始错误
        return pd.read_csv(path, **kwargs)

    with _LOCK:
        cached = _DF_CACHE.get(key)
        if cached is not None:
            _DF_CACHE.move_to_end(key)
    if cached is not None:
        metrics.incr("csv_cache_hit_total")
        return cached

    metrics.incr("csv_cache_miss_total")
    with metrics.timer("csv.parse_ms"):
        df = pd.read_csv(path, **kwargs)
    with _LOCK:
        _DF_CACHE[key] = df
        _DF_CACHE.move_to_end(key)
        while len(_DF_CACHE) > _MAX_ENTRIES:
            _DF_CACHE.popitem(last=False)
    return df


def clear_csv_cache() -> None:
    with _LOCK:
        _DF_CACHE.clear()


def csv_cache_info() -> dict[str, int]:
    with _LOCK:
        return {"entries": len(_DF_CACHE), "max_entries": _MAX_ENTRIES}


def dataset_fingerprint(dataset_path: str | None) -> str:
    """数据集指纹：目录内数据文件的 名字+字节数+mtime 的 sha256 前 16 位。

    不读文件内容（37MB 读一遍不划算），但足以识别"换了一份上传 / 文件被覆盖"。
    """
    if not dataset_path or not os.path.isdir(dataset_path):
        return "missing"
    digest = hashlib.sha256()
    try:
        names = sorted(os.listdir(dataset_path))
    except OSError:
        return "missing"
    for name in names:
        if not name.lower().endswith((".csv", ".tsv", ".txt")):
            continue
        full = os.path.join(dataset_path, name)
        if not os.path.isfile(full):
            continue
        try:
            st = os.stat(full)
        except OSError:
            continue
        digest.update(f"{name}:{st.st_size}:{st.st_mtime_ns};".encode("utf-8"))
    return digest.hexdigest()[:16]
