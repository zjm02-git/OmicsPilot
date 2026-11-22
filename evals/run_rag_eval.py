"""跑评测：同一批问题分别用「单路向量」和「向量+BM25+RRF」检索，
算 recall@5 / MRR，打印对比表；可追加 LLM 重排看 MRR 提升。

用法（在项目根目录，需 .env 里的 SILICONFLOW_API_KEY / DEEPSEEK_API_KEY）：
  python evals/run_rag_eval.py             # 单路 vs 混合
  python evals/run_rag_eval.py --rerank    # 再加 LLM 重排一列

指标：
  recall@5 = 返回前 5 个基因里，命中金标基因的比例（金标里有多少被找回来）
  MRR      = 每条问题「第一个命中的金标基因排第几位」的倒数，取平均（排名越靠前越好）
"""

import argparse
import json
from pathlib import Path

from omics_agent import rag, rerank

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = PROJECT_ROOT / "evals" / "rag_eval.json"
K = 5            # top_k
RERANK_POOL = 20  # LLM 重排前先粗召回 20 个候选


def _names(triplets) -> list[str]:
    return [t[0].upper() for t in triplets]


def _recall_at_k(names: list[str], gold: list[str]) -> float:
    return len(set(names) & set(gold)) / len(gold)


def _mrr(names: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for i, g in enumerate(names, start=1):
        if g in gold_set:
            return 1.0 / i
    return 0.0


def _run(items, mode: str):
    """对整批评测跑一种检索方式，返回 (平均 recall@5, 平均 MRR)。"""
    recalls, mrrs = [], []
    for it in items:
        gold = [g.upper() for g in it["gold_genes"]]
        q = it["question"]
        if mode == "vector":
            hits = rag.retrieve_genes(q, top_k=K)
        elif mode == "hybrid":
            hits = rag.retrieve_hybrid(q, top_k=K)
        elif mode == "rerank":
            hits = rerank.rerank(q, rag.retrieve_hybrid(q, top_k=RERANK_POOL), top_k=K)
        else:
            raise ValueError(f"未知模式 {mode}")
        names = _names(hits)
        recalls.append(_recall_at_k(names, gold))
        mrrs.append(_mrr(names, gold))
    return sum(recalls) / len(recalls), sum(mrrs) / len(mrrs)


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 检索评测：单路 vs 混合（+LLM 重排）")
    ap.add_argument("--rerank", action="store_true", help="追加 LLM 重排一列对比")
    args = ap.parse_args()

    items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    print(f"评测集：{len(items)} 条问题，top_k={K}\n")

    rows = [("单路向量", *_run(items, "vector")),
            ("向量+BM25+RRF", *_run(items, "hybrid"))]
    if args.rerank:
        rows.append(("+LLM重排", *_run(items, "rerank")))

    print(f"{'方法':<16}{'recall@5':>10}{'MRR':>8}")
    print("-" * 36)
    for name, r, m in rows:
        print(f"{name:<16}{r * 100:>9.1f}%{m:>8.3f}")

    base_r, base_m = rows[0][1], rows[0][2]
    print()
    for name, r, m in rows[1:]:
        print(f"{name}: recall@5 较单路 {r * 100 - base_r * 100:+.1f} 个百分点，"
              f"MRR {m - base_m:+.3f}")


if __name__ == "__main__":
    main()
