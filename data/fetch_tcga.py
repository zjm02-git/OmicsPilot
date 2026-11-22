# -*- coding: utf-8 -*-
"""从 GDC 官方 API 下载 TCGA-BRCA 真实表达数据，转成项目 data/ 需要的格式。

数据源：GDC（Genomic Data Commons，TCGA 官方门户）
    - 表达文件：STAR - Counts 流程的 per-sample TSV（含 fpkm_uq 列）
    - 分组依据：样本类型 Primary Tumor=肿瘤 / Solid Tissue Normal=正常
    - 规模：取 BRCA 全部正常样本，肿瘤取同样数量保持两组平衡

两条"错得看不出来"的历史教训（所以这里的写法刻意留了痕迹）：

  1. 缺失值不能再补 0。旧版写的是 expr.get(g, {}).get(sid, 0.0)，于是
     "这个样本没测到这个基因"和"这个基因表达量确实是 0"变成同一个数 log2(0+1)=0。
     后果：QC 的缺失率体检（nodes.py 的 expr_df.isna().mean()）永远报 0.0%，
     而存量 data/expression.csv 里其实有 13%（595861 / 4505988）的格子正好是 0.0，
     已经没人分得清哪些是真 0、哪些是被补出来的 0。现在缺失一律写空单元格，
     pandas 读回来就是 NaN；每个样本缺了多少个基因记在 metadata 的
     n_missing_genes 列里，下游据此能算真实缺失率。

  2. 同一病人的重复等分（aliquot）不是独立的生物学重复。TCGA-A7-A0DC 同时有
     01A/01B 两份肿瘤组织，当成两个样本会让 n 虚高、方差被低估、p 值偏小。
     现在按 (病人, 分组) 去重，优先保留 01A/11A 这类 A 瓶；B 瓶只有在它是
     该病人该组织的**唯一**等分时才保留——否则是把真实样本白扔掉，而不是去重复。

生成文件（在 data/ 下）：
    - expression.csv : gene × 样本矩阵，值 = log2(fpkm_uq + 1)，首列 gene；
                       未测到的格子留空（NaN），不是 0
    - metadata.csv   : sample_id, condition, patient_id, aliquot, n_missing_genes

用法（在项目根目录执行）：
    python data/fetch_tcga.py --redownload
不加 --redownload 时，若 data/expression.csv 已存在则拒绝执行：存量文件是旧脚本
产物，缺失值已经被补成 0，没有"原地修复"的办法，只能重新拉一遍。
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
import os
import tempfile
import time
import urllib.request

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")


def _post(query: dict) -> dict:
    """GDC 查询接口（带重试）。

    GDC 会偶发 WinError 10054（远端强制关闭连接）。这一段没有重试的话，
    一次瞬时抖动就会让整个下载在第一步死掉 —— 实测踩到过一次：
    0 个文件、进程直接退出，看起来像"代码坏了"，其实只是连接被重置。
    """
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                GDC_FILES, data=json.dumps(query).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as exc:  # noqa: BLE001 - 统一重试
            last_err = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"GDC 查询失败（重试 3 次后）：{last_err}")


def _split_barcode(submitter_id: str) -> dict:
    """把 TCGA barcode 拆成「病人」和「等分」。

    TCGA-A7-A0DC-01A → patient_id=TCGA-A7-A0DC, aliquot=01A
    前三段标识病人，第四段才是组织+等分号（01=肿瘤、11=正常；A/B=同一次取材的不同等分瓶）。
    去重要按前三段做，按整串做等于没去重。
    """
    parts = submitter_id.split("-")
    patient_id = "-".join(parts[:3])
    aliquot = parts[3] if len(parts) > 3 else ""
    return {"sample_id": submitter_id, "patient_id": patient_id, "aliquot": aliquot}


def query_samples(sample_type: str, n: int) -> list:
    """分页拉取最多 n 个指定样本类型的文件：[{file_id, sample_id, patient_id, aliquot}, ...]"""
    filters = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
        {"op": "in", "content": {"field": "files.data_type", "value": ["Gene Expression Quantification"]}},
        {"op": "in", "content": {"field": "files.analysis.workflow_type", "value": ["STAR - Counts"]}},
        {"op": "in", "content": {"field": "cases.samples.sample_type", "value": [sample_type]}},
    ]}
    out = []
    for start in range(0, n, 100):
        q = {
            "filters": filters,
            "fields": "file_id,cases.samples.submitter_id",
            "expand": "cases.samples",
            "size": min(100, n - len(out)),
            "from": start,
        }
        hits = _post(q)["data"]["hits"]
        for h in hits:
            # 一个病例可能挂着多个样本（比如同时有 01A 肿瘤和 11A 正常组织）。
            # 旧代码取 samples[0]：赶上是正常组织那一份时，肿瘤测序文件会被贴上正常
            # 样本的 barcode——分组标签与文件内容错位，同样是查不出来的静默错误。
            # 所以这里按 sample_type 精确挑，挑不到就跳过。
            picked = next(
                (s for s in h["cases"][0]["samples"]
                 if s.get("sample_type") == sample_type and s.get("submitter_id")),
                None,
            )
            if picked is None:
                continue
            out.append({"file_id": h["file_id"], **_split_barcode(picked["submitter_id"])})
        if len(hits) < 100:
            break
    return out


def _is_primary_aliquot(aliquot: str) -> bool:
    """01A / 11A 这类 A 瓶算首选等分（同一次取材最先切下来的那瓶）。"""
    return aliquot.endswith("A")


def dedupe_aliquots(samples: list) -> tuple:
    """按 (病人, 分组) 只留一个等分，优先 A 瓶。

    samples 里的每个元素必须带 patient_id / condition / aliquot。
    返回 (保留的样本, 被丢掉的重复等分)，两者的顺序都沿用入参顺序，保证可复现。

    为什么键里要带 condition：同一个病人的肿瘤和正常组织是两份**不同**的组织，
    配对分析里两者都要留下；只有"病人 × 同一种组织"出现多份等分才叫伪重复。
    """
    kept_idx = {}
    dropped_idx = []
    for i, s in enumerate(samples):
        key = (s["patient_id"], s["condition"])
        j = kept_idx.get(key)
        if j is None:
            kept_idx[key] = i
        elif not _is_primary_aliquot(samples[j]["aliquot"]) and _is_primary_aliquot(s["aliquot"]):
            # 后来者才是 A 瓶 → 换掉先前留下的 B 瓶
            dropped_idx.append(j)
            kept_idx[key] = i
        else:
            dropped_idx.append(i)
    kept = [samples[i] for i in sorted(kept_idx.values())]
    dropped = [samples[i] for i in sorted(dropped_idx)]
    return kept, dropped


# 并发 8：GDC 会限流，开太高反而更容易被拖住或拒连；
# 串行时单文件常有数分钟空闲等待，226 个样本要几十小时，不可接受。
# 并发与超时的取值来自实测，不是拍脑袋：
#   - 串行时单个文件出现过"8 分钟零字节，然后 55 秒拉完 4.2MB"；
#   - 并发 8 时观察到 8 个连接同时卡在 0 字节超过 5 分钟。
# 说明 GDC 会先把连接挂住排队、再一次性给数据。所以：
#   - 单文件超时必须显著大于排队时间，120s 会把合法下载杀掉（实测踩到）→ 480s；
#   - 并发不能太高，8 路会把 GDC 的排队拖长 → 6 路。
DOWNLOAD_WORKERS = int(os.getenv("TCGA_DOWNLOAD_WORKERS", "6"))
DOWNLOAD_TIMEOUT_S = int(os.getenv("TCGA_DOWNLOAD_TIMEOUT_S", "480"))
DOWNLOAD_RETRIES = int(os.getenv("TCGA_DOWNLOAD_RETRIES", "3"))


def download(file_id: str, dest: str) -> None:
    """下载一个样本文件（**单次尝试**，重试交给 fetch_sample 统一管）。

    超时放这里而不是外层：卡住的是这一个连接，超时才能把它单独打断，
    不牵连同批次其它样本。
    """
    req = urllib.request.Request(GDC_DATA + "/" + file_id, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as r, open(dest, "wb") as f:
        f.write(r.read())


def fetch_sample(file_id: str, dest: str) -> dict:
    """下载 + 完整性校验 + 解析；任一步失败都重试整条链路（最多 DOWNLOAD_RETRIES 次）。

    为什么校验必须在这里：并行之后"写半截文件"的危害被放大了 ——
    一个 0 字节的 TSV 会被解析成空 dict，静默变成"这个样本什么都没测到"，
    最后以 100% 缺失的形式污染整张矩阵，而且没有任何报错。
    所以两条都查：字节数 > 0、能解析出非空的 {gene: fpkm}。
    """
    last_err = None
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            download(file_id, dest)
            if os.path.getsize(dest) <= 0:
                raise RuntimeError("下载结果为空文件")
            vals = parse_sample(dest)
            if not vals:
                raise RuntimeError("解析不出任何蛋白编码基因")
            return vals
        except Exception as exc:  # noqa: BLE001 - 统一重试，最后一次抛出
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"下载/解析失败 {file_id}: {last_err}")


def parse_sample(path: str) -> dict:
    """解析一个样本 TSV，返回 {gene_name: fpkm_uq}，只留 protein_coding 且有符号名的。

    注意返回的是"这个样本真的测到的基因"：TSV 里没有该基因、或值是 NA 时，
    这个 key 直接不存在（不是 0）。写 CSV 时缺失的 key 会留空，变成 NaN。
    """
    vals = {}
    with open(path, encoding="utf-8") as f:
        col = None
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if col is None:
                col = {name: i for i, name in enumerate(parts)}
                continue
            if parts[col["gene_type"]] != "protein_coding":
                continue
            name = parts[col["gene_name"]]
            if not name:
                continue
            raw = parts[col["fpkm_uq_unstranded"]]
            if raw in ("", "NA"):
                # 测到了基因但没给出定量值 → 这个基因在这个样本上"没有值"（NaN），
                # 不能写 0：写 0 就是宣称"表达量为 0"，会和真实低表达混在一起。
                continue
            vals[name] = float(raw)
    return vals


def _report_dedupe(label: str, raw: list, kept: list, dropped: list) -> None:
    """打印去重前后的样本数/病人数：改口径时最容易糊弄过去的就是这一步。"""
    n_raw_patients = len({s["patient_id"] for s in raw})
    n_kept_patients = len({s["patient_id"] for s in kept})
    tail = "无重复等分"
    if dropped:
        tail = f"丢弃重复等分 {len(dropped)} 份（{', '.join(s['sample_id'] for s in dropped)}）"
    print(
        f"  {label}: 去重前 {len(raw)} 份等分 / {n_raw_patients} 个病人"
        f" → 去重后 {len(kept)} 个样本 / {n_kept_patients} 个病人；{tail}"
    )


def main(redownload: bool = False, per_group: int | None = None) -> None:
    """per_group：每组最多保留多少个样本（None = 不设上限，按 500 候选去重后配平）。

    先用小规模跑通链路是必要的：GDC 单文件实测要 8 分钟左右，
    226 个样本要 5 小时；120 个（60 vs 60）约 2.7 小时，够验证口径了。
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    expr_path = os.path.join(OUT_DIR, "expression.csv")
    meta_path = os.path.join(OUT_DIR, "metadata.csv")

    # 存量文件不能就地修：旧口径把缺失补成了 0，事后无法区分"真 0"和"补的 0"。
    if not redownload and os.path.exists(expr_path):
        raise SystemExit(
            f"{expr_path} 已存在，不覆盖。\n"
            "  这份存量文件是旧脚本产物（缺失值被补成 0.0，缺失率永远算成 0%），"
            "无法原地修复，只能重新拉一遍：\n"
            "      python data/fetch_tcga.py --redownload"
        )

    raw_dir = tempfile.mkdtemp(prefix="tcga_brca_")

    # 1. 先拉候选样本，再做配平/去重。
    #    顺序不能反：先按"含重复的次数"配平、再去重，两组的样本数就对不上了。
    print("== 去重（同一病人的重复等分只留一份）==")
    normal_raw = query_samples("Solid Tissue Normal", 500)
    normal_all = [{**s, "condition": "normal"} for s in normal_raw]
    normal, normal_dup = dedupe_aliquots(normal_all)
    _report_dedupe("normal", normal_all, normal, normal_dup)
    if per_group:
        normal = normal[:per_group]
        print(f"  normal: 按 --per-group 截到 {len(normal)} 个")

    # 肿瘤多取一倍候选，抵消重复等分被去掉的损耗；去重后再截到与正常组同规模，
    # 保证两组样本数平衡（n 不平衡会让方差估计和检验力都变差）。
    tumor_raw = query_samples("Primary Tumor", max(100, len(normal) * 2))
    tumor_all = [{**s, "condition": "tumor"} for s in tumor_raw]
    tumor_pool, tumor_dup = dedupe_aliquots(tumor_all)
    _report_dedupe("tumor", tumor_all, tumor_pool, tumor_dup)
    tumor = tumor_pool[: len(normal)]
    print(f"  tumor: 取前 {len(tumor)} 个与正常组配平")

    samples = tumor + normal
    if not samples:
        raise SystemExit("没有拉到任何样本（GDC 返回为空？），中止。")
    n_tumor = sum(1 for s in samples if s["condition"] == "tumor")
    n_normal = sum(1 for s in samples if s["condition"] == "normal")
    print(
        f"共 {len(samples)} 个样本（{n_tumor} 肿瘤 + {n_normal} 正常），"
        f"来自 {len({s['patient_id'] for s in samples})} 个病人；"
        f"按病人去重共去掉 {len(tumor_dup) + len(normal_dup)} 份重复等分"
    )

    # 2. 并行下载 + 解析，累积 gene -> {sample: value}
    print(f"== 下载并解析样本（并发 {DOWNLOAD_WORKERS}，单文件超时 {DOWNLOAD_TIMEOUT_S}s）==")
    t_start = time.time()
    results = {}       # sample_id -> {gene: fpkm}；先收齐再归并，避免线程互踩
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(fetch_sample, s["file_id"], os.path.join(raw_dir, s["file_id"] + ".tsv")): s
            for s in samples
        }
        failures = []
        for done, fut in enumerate(as_completed(futures), 1):
            s = futures[fut]
            try:
                results[s["sample_id"]] = fut.result()
            except Exception as exc:  # noqa: BLE001 - 单个样本失败不能拖死整批
                failures.append((s["sample_id"], f"{type(exc).__name__}: {exc}"[:120]))
            if done % 10 == 0 or done == len(samples):
                print(f"  进度 {done}/{len(samples)}，失败 {len(failures)}"
                      f"（{time.time() - t_start:.0f}s）", flush=True)

    # 有样本彻底失败就不落盘：宁可这次白跑，也不能产出一张"某几个样本全空"的残缺矩阵，
    # 那种矩阵能被下游一路算下去，且没有任何报错。
    if failures:
        print(f"  × 失败样本 {len(failures)} 个：")
        for sid, err in failures[:15]:
            print(f"    - {sid}: {err}")
        raise SystemExit(
            f"{len(failures)} 个样本下载失败，未写出 expression.csv。"
            "重跑一次即可（已下载的样本会重新拉，GDC 有缓存会快很多）。"
        )

    # 基因顺序取样本列表里的**第一个**：并行完成顺序是乱的，
    # 用"先完成"的那个当基准会让每次产出的列序不稳定。
    gene_order = list(results[samples[0]["sample_id"]].keys())
    expr = {}          # gene -> {sample_id: fpkm_uq}；缺失的 key 就是"没测到"
    measured = {}      # sample_id -> 有值的基因集合（用来数每个样本缺了多少）
    for s in samples:
        sid = s["sample_id"]
        vals = results[sid]
        measured[sid] = set(vals.keys())
        for g, v in vals.items():
            expr.setdefault(g, {})[sid] = v
    print(f"  下载完成：{len(samples)} 个样本，用时 {time.time() - t_start:.0f}s")

    sample_ids = [s["sample_id"] for s in samples]
    gene_set = set(gene_order)
    # 每个样本在最终矩阵里留空了多少个基因（= 该样本没测到，或落在首个文件基因集之外）
    n_missing = {sid: len(gene_set - measured.get(sid, set())) for sid in sample_ids}

    # 3. 写 expression.csv（log2(fpkm_uq+1)），缺失留空
    with open(expr_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gene"] + sample_ids)
        for g in gene_order:
            row = [g]
            per_gene = expr.get(g, {})
            for sid in sample_ids:
                v = per_gene.get(sid)
                # 关键修复：没测到（v is None）就写空单元格，pandas 读回来是 NaN。
                # 写 0 会让"没测到"和"测得为 0"变成同一个 log2(0+1)=0，
                # 下游 QC 的缺失率体检于是永远报 0.0%，再也没人发现数据是缺的。
                row.append("" if v is None else round(math.log2(v + 1.0), 6))
            w.writerow(row)
    n_cells = len(gene_order) * len(sample_ids)
    n_missing_cells = sum(n_missing.values())
    print(
        f"已写 expression.csv：{len(gene_order)} 基因 × {len(sample_ids)} 样本；"
        f"留空 {n_missing_cells}/{n_cells} 个格子"
        f"（真实缺失率 {n_missing_cells / max(1, n_cells):.4%}）"
    )

    # 4. 写 metadata.csv：patient_id 记录去重口径，n_missing_genes 让下游能查真实缺失率
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "condition", "patient_id", "aliquot", "n_missing_genes"])
        for s in samples:
            w.writerow([s["sample_id"], s["condition"], s["patient_id"], s["aliquot"],
                        n_missing[s["sample_id"]]])
    print(f"已写 metadata.csv：{len(samples)} 样本"
          f"（每样本缺 {min(n_missing.values())}~{max(n_missing.values())} 个基因）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="从 GDC 下载 TCGA-BRCA 表达数据")
    ap.add_argument(
        "--redownload", action="store_true",
        help="重新从 GDC 拉取并覆盖 data/expression.csv 与 data/metadata.csv",
    )
    ap.add_argument(
        "--per-group", type=int, default=None,
        help="每组最多保留多少个样本（不传=全量）。先用小规模验证口径时用它",
    )
    _args = ap.parse_args()
    main(redownload=_args.redownload, per_group=_args.per_group)
