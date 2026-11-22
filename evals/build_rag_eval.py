"""自动造评测集：从 GO 注释抽 30 条通路，生成「英文问题 → 金标基因」问答对。

为什么能自动造（不用人工标）：
  知识库卡片是「基因 → 参与的通路列表」（gene_docs.csv），而 gene_ontology.csv
  反过来给了「通路 → 基因列表」。把通路翻成人话问题，它的基因列表就是金标答案，
  一分钱人工标注都不用花。

为什么用英文问题：
  卡片和通路名都是英文，BM25 只切英文词；英文问题才能公平地同时考向量和 BM25。
  （中文问题 BM25 会漏——这是 BM25 的固有边界，评测里正好体现出来。）

产物：evals/rag_eval.json —— 列表，每项 {"question", "pathway", "gold_genes"}。
用法：python evals/build_rag_eval.py（在项目根目录跑）
"""

import json
import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ONT_PATH = PROJECT_ROOT / "data" / "gene_ontology.csv"
DOCS_PATH = PROJECT_ROOT / "data" / "gene_docs.csv"
OUT_PATH = PROJECT_ROOT / "evals" / "rag_eval.json"

N_EVAL = 30      # 评测集条数
MIN_GENES = 5    # 金标基因太少没法算召回，跳过
MAX_GENES = 60   # 金标过多说明通路太宽泛（recall@5 天花板过低），跳过
SEED = 42        # 固定随机种子，保证每次造出来都一样、可复现


def term_to_pathway(term: str) -> str | None:
    """GOBP_DNA_REPAIR -> 'dna repair'。只处理 GOBP（生物过程），其余返回 None。"""
    if not term.startswith("GOBP_"):
        return None
    return term[5:].replace("_", " ").lower()


def main() -> None:
    ont = pd.read_csv(ONT_PATH, usecols=["gene", "term"])
    docs = pd.read_csv(DOCS_PATH, usecols=["gene"])
    card_genes = set(docs["gene"].astype(str).str.upper())  # 卡片里的基因名（统一大写对齐）

    # 1. 按通路分组，只保留「金标基因在卡片里」的 GOBP 通路
    grouped = ont.groupby("term")["gene"].apply(
        lambda g: [x for x in g.astype(str).str.upper() if x in card_genes]
    )

    pool = []
    for term, genes in grouped.items():
        pathway = term_to_pathway(term)
        if pathway is None:
            continue
        genes = list(dict.fromkeys(genes))  # 去重、保序
        if MIN_GENES <= len(genes) <= MAX_GENES:
            pool.append({"pathway": pathway, "gold_genes": genes})

    # 2. 固定种子抽 N_EVAL 条
    rng = random.Random(SEED)
    rng.shuffle(pool)
    selected = pool[:N_EVAL]

    # 3. 生成英文问题（模板一致，保证可复现、可对比）
    items = [
        {
            "question": f"Which genes are involved in {p['pathway']}?",
            "pathway": p["pathway"],
            "gold_genes": p["gold_genes"],
        }
        for p in selected
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已生成 {len(items)} 条评测 → {OUT_PATH}")
    print(f"（从 {len(pool)} 条候选通路里抽，金标基因数 {MIN_GENES}~{MAX_GENES}）\n")
    for it in items[:5]:
        print(f"  · {it['question']}  （金标 {len(it['gold_genes'])} 个基因）")


if __name__ == "__main__":
    main()
