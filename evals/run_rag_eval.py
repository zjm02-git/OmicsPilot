"""跑评测：把「混合检索到底好在哪、好多少」量成一张表，并把分数落盘。

用法（项目根目录；需 .env 里的 SILICONFLOW_API_KEY，重排那列还需 DEEPSEEK_API_KEY）：
  python evals/run_rag_eval.py                  # 默认 6 列：向量 / BM25×2 / 混合 rrf_k×3
  python evals/run_rag_eval.py --rerank         # 再加「混合+LLM重排」一列（会花 LLM 调用）
  python evals/run_rag_eval.py --limit 5        # 只跑前 5 条，先验证链路通不通
  python evals/run_rag_eval.py --methods vector hybrid:60 bm25
  python evals/run_rag_eval.py --no-write       # 只看表，不落盘

指标（K=5）：
  hit@5       前 5 个里有没有金标基因（0/1），均值 = 命中率 —— **主指标**
  precision@5 前 5 个里命中金标基因的比例（分母固定 5）
  MRR         第一个金标基因排第几的倒数，衡量「排得够不够靠前」
  recall@5    命中数 / 金标总数 —— **保留但别单看**：
              金标 5~43 个，分母一变指标就变。金标 43 个时 recall@5 的天花板
              只有 5/43≈11.6%，它衡量的是「金标本身有多宽」而不是「这条路好不好」。
              表里会打印每组的 recall@5 天花板均值，方便把分数换算成「占天花板多少」。

为什么要三组数字而不是一个平均分：
  1) 基线：没有「纯 BM25」「纯向量」两列，混合检索的增益就是无源之水；
  2) 消融：rrf_k 取 10/60/100 三档，证明那个 60 是抄来的还是选出来的；
  3) 分语言：英文 30 条 + 中文 10 条。混在一起平均会把中文的短板稀释掉。

落盘：evals/rag_scores.json —— 时间戳 + 评测集指纹 + 每组每语言指标 + 同源泄漏诊断。
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from hashlib import sha1
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 直接 python evals/run_rag_eval.py 跑时 sys.path[0] 是 evals/，import 不到 omics_agent；
# 把项目根插到最前面，脚本就能从任意目录启动（比要求「必须 -m 跑」友好）。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omics_agent import rag  # noqa: E402

EVAL_PATH = PROJECT_ROOT / "evals" / "rag_eval.json"
OUT_PATH = PROJECT_ROOT / "evals" / "rag_scores.json"

K = 5              # 主指标取的 top_k
RERANK_POOL = 20   # 重排前先粗召回几个（必须 > K，重排才有翻盘空间）
DEFAULT_METHODS = ["vector", "bm25", "bm25_nogloss", "hybrid:10", "hybrid:60", "hybrid:100"]

VALID_METHODS = {"vector", "bm25", "bm25_nogloss", "hybrid", "hybrid_nogloss", "rerank"}


# ── 表格对齐：中文占两格，否则等宽终端里表格会歪 ──────────────────
# 0x2460 起算：①这类带圈数字在中文字体里也是两格宽，只会按 CJK 起算会漏掉它
_WIDE_START = 0x2460


def _w(s: str) -> int:
    return sum(2 if ord(c) >= _WIDE_START else 1 for c in s)


def _rel(p: Path) -> str:
    """报告里记相对项目根的路径；评测集不在项目内（--out 指到别处）时退回绝对路径。"""
    try:
        return str(Path(p).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _pad(s: str, width: int, right: bool = False) -> str:
    gap = max(width - _w(s), 0)
    return (" " * gap + s) if right else (s + " " * gap)


# ── 评测集读取 ───────────────────────────────────────────────────
def _load_items(limit: int = 0) -> list:
    """读评测集。lang 字段缺失时按「问题里有没有汉字」推断——
    兼容旧版 rag_eval.json（还没重建时也能跑，只是没有中文那一组）。
    """
    items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    for it in items:
        if "lang" not in it:
            it["lang"] = "zh" if any("\u4e00" <= c <= "\u9fff" for c in it["question"]) else "en"
    return items[:limit] if limit else items


# ── 指标 ─────────────────────────────────────────────────────────
def _names(triplets) -> list:
    return [str(t[0]).upper() for t in triplets]


def _metrics(names: list, gold: list, k: int = K) -> dict:
    """一条问题的四项指标。金标统一大写比较（卡片里有大小写不一致）。"""
    gold_set = {str(g).upper() for g in gold}
    top = names[:k]
    hits = [g for g in top if g in gold_set]
    return {
        "hit@5": 1.0 if hits else 0.0,
        "precision@5": len(hits) / k,
        "recall@5": len(hits) / len(gold_set) if gold_set else 0.0,
        "mrr": next((1.0 / i for i, g in enumerate(top, 1) if g in gold_set), 0.0),
        "n_hits": len(hits),
        "gold": len(gold_set),
    }


def _agg(rows: list) -> dict:
    """一组问题的均值 + recall@5 的天花板（=min(5,|gold|)/|gold| 的均值）。"""
    n = len(rows)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "hit@5": round(sum(r["hit@5"] for r in rows) / n, 4),
        "precision@5": round(sum(r["precision@5"] for r in rows) / n, 4),
        "recall@5": round(sum(r["recall@5"] for r in rows) / n, 4),
        "mrr": round(sum(r["mrr"] for r in rows) / n, 4),
        # 天花板：金标越多，5 个名额能覆盖的比例越低；用它把 recall@5 归一化才可比
        "recall@5_ceiling": round(sum(min(K, r["gold"]) / r["gold"] for r in rows) / n, 4),
        "avg_gold": round(sum(r["gold"] for r in rows) / n, 1),
    }


# ── 检索分发 ─────────────────────────────────────────────────────
def _retrieve(question: str, name: str, rrf_k: int):
    """按方法名跑一次检索，返回 (基因名列表, 额外信息或 None)。"""
    if name == "vector":
        return _names(rag.retrieve_genes(question, top_k=K)), None
    if name == "bm25":
        return _names(rag.retrieve_bm25(question, top_k=K, expand_zh=True)), None
    if name == "bm25_nogloss":
        # 消融：只切 CJK 二元组、不查中→英词表，用来量「词表到底贡献了多少」
        return _names(rag.retrieve_bm25(question, top_k=K, expand_zh=False)), None
    if name == "hybrid":
        return _names(rag.retrieve_hybrid(question, top_k=K, rrf_k=rrf_k)), None
    if name == "hybrid_nogloss":
        return _names(rag.retrieve_hybrid(question, top_k=K, rrf_k=rrf_k, expand_zh=False)), None
    if name == "rerank":
        hits, info = rag.retrieve_hybrid_reranked(
            question, top_k=K, rrf_k=rrf_k, pool=RERANK_POOL, return_info=True)
        return _names(hits), info
    raise ValueError(f"未知方法 {name}")


def _label(name: str, rrf_k: int) -> str:
    """表里那一列的名字。序号写在前面，方便报告里直接引用。"""
    return {
        "vector": "①纯向量",
        "bm25": "②BM25(含词表)",
        "bm25_nogloss": "③BM25(无词表)",
        "hybrid": f"④混合 k={rrf_k}",
        "hybrid_nogloss": f"⑤混合(无词表) k={rrf_k}",
        "rerank": f"⑥混合+LLM重排 k={rrf_k}",
    }[name]


def _run(items: list, name: str, rrf_k: int) -> dict:
    """跑一整列：返回 {"all"/"en"/"zh": 聚合指标, "per_item": 明细}。"""
    per_item = []
    rerank_status = {}
    for it in items:
        gold = it["gold_genes"]
        t0 = time.perf_counter()
        names, info = _retrieve(it["question"], name, rrf_k)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        m = _metrics(names, gold)
        per_item.append({
            "question": it["question"],
            "lang": it["lang"],
            "pathway": it.get("pathway"),
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
            "top5": names,
            "elapsed_ms": elapsed_ms,
        })
        if info:  # 重排那一列：把每次的降级原因统计出来，别把降级当成功
            rerank_status[info["status"]] = rerank_status.get(info["status"], 0) + 1

    out = {
        "all": _agg(per_item),
        "en": _agg([r for r in per_item if r["lang"] == "en"]),
        "zh": _agg([r for r in per_item if r["lang"] == "zh"]),
        "per_item": per_item,
        "avg_latency_ms": round(sum(r["elapsed_ms"] for r in per_item) / max(len(per_item), 1), 1),
    }
    if rerank_status:
        out["rerank_status"] = rerank_status
    return out


# ── 同源泄漏诊断（T4）────────────────────────────────────────────
def _leakage(items: list) -> dict:
    """量「同源泄漏」的两条链路——这两条连起来才是「这题在考字符串」的证据。

    ① card_rate：金标卡片里是否**逐字印着**该通路名。
       评测集和卡片都由 GO 注释生成，所以这一项必然接近 100%。它证明的是
       「答案就印在语料里」——泄漏成立的前提。
    ② question_rate：**问句**里是否逐字包含那个通路名。
       英文模板题 100% 包含（"Which genes are involved in platelet degranulation?"），
       于是 BM25 只要匹配这个字符串就赢，测的是字符串对齐而不是语义泛化。
       中文题换掉了表面形式，这一项是 0%，所以中文组的分数更接近「真泛化」，
       但金标来源没变，仍属弱化而非消除。
    只看①会得出「全都泄漏、评测没救」的悲观结论；①+②一起看才能说清
    「英文分数偏乐观多少、中文为什么反而是更硬的证据」。
    """
    cards = rag.load_cards()
    desc = dict(zip(cards["gene"].astype(str).str.upper(), cards["description"].astype(str)))

    detail = []
    for it in items:
        pathway = (it.get("pathway") or "").lower()
        gold = [str(g).upper() for g in it["gold_genes"]]
        in_card = [g for g in gold if pathway and pathway in desc.get(g, "").lower()]
        detail.append({
            "question": it["question"],
            "lang": it["lang"],
            "pathway": it.get("pathway"),
            "gold": len(gold),
            "gold_with_verbatim_pathway": len(in_card),
            "card_has_pathway": bool(in_card),
            "question_has_pathway": bool(pathway) and pathway in (it.get("question") or "").lower(),
        })

    def _rate(rows, flag):
        n = len(rows)
        return round(sum(1 for r in rows if r[flag]) / n, 4) if n else 0.0

    groups = {"all": detail,
              "en": [r for r in detail if r["lang"] == "en"],
              "zh": [r for r in detail if r["lang"] == "zh"]}
    return {
        "n": len(items),
        # ① 语料侧：金标卡片逐字印着通路名
        "card_rate": {g: _rate(rows, "card_has_pathway") for g, rows in groups.items()},
        # ② 问句侧：问句逐字含通路名（英文模板题=100%，中文题=0%）
        "question_rate": {g: _rate(rows, "question_has_pathway") for g, rows in groups.items()},
        "detail": detail,
    }


# ── 打印 ─────────────────────────────────────────────────────────
def _print_table(results: dict, group: str, title: str) -> None:
    rows = [(label, res[group]) for label, res in results.items() if res[group]["n"]]
    if not rows:
        return
    print(f"\n=== {title}（n={rows[0][1]['n']}，平均金标 {rows[0][1]['avg_gold']} 个）===")
    print(_pad("方法", 22) + _pad("hit@5", 9, True) + _pad("prec@5", 9, True)
          + _pad("recall@5", 10, True) + _pad("天花板", 8, True) + _pad("MRR", 8, True))
    print("-" * 66)
    for label, a in rows:
        print(_pad(label, 22)
              + _pad(f"{a['hit@5'] * 100:.1f}%", 9, True)
              + _pad(f"{a['precision@5'] * 100:.1f}%", 9, True)
              + _pad(f"{a['recall@5'] * 100:.1f}%", 10, True)
              + _pad(f"{a['recall@5_ceiling'] * 100:.1f}%", 8, True)
              + _pad(f"{a['mrr']:.3f}", 8, True))


def _deltas_vs_baseline(results: dict, baseline_label: str) -> dict:
    """相对基线（默认「纯向量」）的差值。结论靠这张表说话，不靠形容词。"""
    if baseline_label not in results:
        return {}
    base = results[baseline_label]
    out = {}
    for label, res in results.items():
        if label == baseline_label:
            continue
        out[label] = {
            group: {
                "hit@5_pp": round((res[group]["hit@5"] - base[group]["hit@5"]) * 100, 2),
                "precision@5_pp": round((res[group]["precision@5"] - base[group]["precision@5"]) * 100, 2),
                "mrr": round(res[group]["mrr"] - base[group]["mrr"], 4),
                "latency_ms": round(res["avg_latency_ms"] - base["avg_latency_ms"], 1),
            }
            for group in ("all", "en", "zh")
            if res[group]["n"] and base[group]["n"]
        }
    return out


def _print_deltas(deltas: dict, baseline_label: str) -> None:
    if not deltas:
        return
    print(f"\n=== 相对「{baseline_label}」的变化（正 = 更好）===")
    print(_pad("方法", 22) + _pad("全部 hit@5", 12, True) + _pad("英文 hit@5", 12, True)
          + _pad("中文 hit@5", 12, True) + _pad("延迟差(ms)", 11, True))
    print("-" * 70)

    def _pp(d, g):
        return f"{d[g]['hit@5_pp']:+.1f}pp" if g in d else "—"

    for label, d in deltas.items():
        lat = d.get("all", {}).get("latency_ms")
        print(_pad(label, 22) + _pad(_pp(d, "all"), 12, True) + _pad(_pp(d, "en"), 12, True)
              + _pad(_pp(d, "zh"), 12, True)
              + _pad(f"{lat:+.0f}" if lat is not None else "—", 11, True))


# ── 主流程 ───────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 检索评测：基线 / 消融 / 中文 / 落盘")
    ap.add_argument("--rerank", action="store_true", help="追加「混合+LLM重排」一列（要花 LLM 调用）")
    ap.add_argument("--methods", nargs="+", default=None,
                    help=f"指定列，如 vector bm25 hybrid:10（默认 {DEFAULT_METHODS}）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（链路自检用）")
    ap.add_argument("--out", default=str(OUT_PATH), help="分数落盘路径，默认 evals/rag_scores.json")
    ap.add_argument("--no-write", action="store_true", help="只打印不落盘")
    args = ap.parse_args()

    items = _load_items(args.limit)
    n_zh = sum(1 for it in items if it["lang"] == "zh")
    print(f"评测集：{len(items)} 条（英文 {len(items) - n_zh} / 中文 {n_zh}），"
          f"K={K}，重排粗召回 pool={RERANK_POOL}")

    methods = list(args.methods) if args.methods else list(DEFAULT_METHODS)
    if args.rerank and not any(m.split(":")[0] == "rerank" for m in methods):
        methods.append("rerank:60")

    parsed = []
    for m in methods:
        name, _, rk = m.partition(":")
        if name not in VALID_METHODS:
            raise SystemExit(f"未知方法 {name}；可选 {sorted(VALID_METHODS)}"
                             "（hybrid/hybrid_nogloss/rerank 可带 :rrf_k）")
        parsed.append((name, int(rk) if rk else 60))

    if n_zh == 0:
        print("⚠ 评测集里没有中文题：先跑 python evals/build_rag_eval.py 重建（会带 10 条中文提问）")

    # 重排那一列要保证真跑：① 显式打开开关（评测不受部署开关影响）② 清缓存（避免列间串味）
    # 不做①的话，部署里 RERANK_ENABLED 没开时这一列会全是 disabled，
    # 跑出来和「混合」列一模一样——那种"消融"是最容易骗过自己的。
    if any(name == "rerank" for name, _ in parsed):
        os.environ["RERANK_ENABLED"] = "1"
        try:
            from omics_agent.rerank import clear_cache
            clear_cache()
        except Exception as e:  # noqa: BLE001
            print(f"⚠ 清重排缓存失败（不影响继续跑）：{e}")

    results = {}
    for name, rrf_k in parsed:
        label = _label(name, rrf_k)
        print(f"\n> 跑 {label} ...", end="", flush=True)
        try:
            results[label] = _run(items, name, rrf_k)
        except Exception as e:  # noqa: BLE001 - 单列跑挂了要能保住其它列
            print(f" 失败：{type(e).__name__}: {e}")
            print("  常见原因：.env 没配 SILICONFLOW_API_KEY / DEEPSEEK_API_KEY，或网络不通、余额不足")
            continue
        print(f" 完成（平均 {results[label]['avg_latency_ms']} ms/条）")
        if "rerank_status" in results[label]:
            dist = results[label]["rerank_status"]
            print("   重排状态分布：" + "，".join(f"{k}={v}" for k, v in sorted(dist.items()))
                  + "（disabled/llm_error/parse_error 都说明这一列没真的重排）")

    if not results:
        raise SystemExit("所有列都跑失败了，见上面的报错")

    _print_table(results, "all", "全部")
    _print_table(results, "en", "英文题")
    if n_zh:
        _print_table(results, "zh", "中文题")

    baseline_label = next((lb for lb in results if lb.startswith("①")), None)
    deltas = _deltas_vs_baseline(results, baseline_label) if baseline_label else {}
    _print_deltas(deltas, baseline_label or "")

    # ── 落盘 ──
    eval_bytes = EVAL_PATH.read_bytes()
    leakage = _leakage(items)
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "meta": {
            "k": K,
            "rerank_pool": RERANK_POOL,
            "n_items": len(items),
            "n_en": len(items) - n_zh,
            "n_zh": n_zh,
            "eval_file": _rel(EVAL_PATH),
            "eval_sha1": sha1(eval_bytes).hexdigest()[:12],
            "embedding_model": "BAAI/bge-m3",
            "methods": [f"{n}:{k}" for n, k in parsed],
            "python": sys.version.split()[0],
        },
        "metric_notes": {
            "hit@5": "前 5 个里是否含任一金标基因（0/1），均值=命中率，主指标",
            "precision@5": "前 5 个里命中金标的比例，分母固定 5",
            "mrr": "第一个金标基因名次的倒数",
            "recall@5": "命中数/金标总数；金标 5~43 个，天花板=min(5,|gold|)/|gold|，勿单看",
            "recall@5_ceiling": "该组 recall@5 的理论天花板均值，用来换算成「占天花板的百分比」",
        },
        "leakage": leakage,
        "results": results,
        "deltas_vs_vector": deltas,
    }

    if args.no_write:
        print("\n（--no-write：不落盘）")
        return
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n分数已落盘 → {out_path}")
    print("同源泄漏：金标卡片里逐字印着通路名的题 "
          f"{leakage['card_rate']['all'] * 100:.0f}%（语料侧，说明答案就在语料里）；"
          "问句里逐字含通路名的题 "
          f"英文 {leakage['question_rate']['en'] * 100:.0f}% / 中文 {leakage['question_rate']['zh'] * 100:.0f}%"
          "（问句侧，越高越像在考字符串对齐）")


if __name__ == "__main__":
    main()
