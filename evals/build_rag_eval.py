"""自动造评测集：从 GO 注释抽通路，生成「问题 -> 金标基因」问答对（英文 30 条 + 中文 10 条）。

为什么能自动造（不用人工标）：
  知识库卡片是「基因 -> 参与的通路列表」（gene_docs.csv），而 gene_ontology.csv
  反过来给了「通路 -> 基因列表」。把通路翻成人话问题，它的基因列表就是金标答案，
  一分钱人工标注都不用花。

为什么原来只有英文（重要，这是个坑）：
  旧版注释写着「英文问题才能公平地同时考向量和 BM25」——这句话在旧切词下成立，
  但等于让评测**永远不测真实用户语言**：中文提问下 BM25 空转这件事，
  30 条英文模板题一条都暴露不出来。现在补 10 条中文提问，两件事一起看清：
    1) 中文提问下混合检索到底退化了多少；
    2) 中→英词表（bm25._ZH_EN_SEED）补的那一跳值多少钱。

中文题的写法约定：
  **不把英文通路名原样塞进问题**。如果写「哪些基因参与 platelet degranulation？」，
  那就还是字符串对齐题，测不出中文场景。所以这 10 条是纯中文表述，
  且刻意选了「能被词表覆盖」和「不会被词表覆盖」两类，让覆盖率的短板暴露出来。
  代价是翻译由人（我）手写，可能带个人偏好——这一点写在报告的局限里。

⚠ 同源泄漏（评测集的根本局限，别忽略）：
  问题和卡片同出一份 GO 注释，所以英文题里的通路名逐字印在金标卡片上——
  这测的是「字符串对齐」而不是「语义泛化」。详见 evals/run_rag_eval.py 的 _leakage()
  与报告里的缓解建议。中文题因为换了语言，表面泄漏天然被打断，但仍然共享同一份
  标注来源（金标本身就是从卡片来源推出来的），属于弱化而非消除。

产物：evals/rag_eval.json —— 列表，每项 {"question", "lang", "pathway", "gold_genes"}。
用法：python evals/build_rag_eval.py（在项目根目录跑）；重建后跑 python evals/run_rag_eval.py。
"""

import json
import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ONT_PATH = PROJECT_ROOT / "data" / "gene_ontology.csv"
DOCS_PATH = PROJECT_ROOT / "data" / "gene_docs.csv"
OUT_PATH = PROJECT_ROOT / "evals" / "rag_eval.json"

N_EVAL = 30      # 英文题条数
MIN_GENES = 5    # 金标基因太少没法算召回，跳过
MAX_GENES = 60   # 金标过多说明通路太宽泛（recall@5 天花板过低），跳过
SEED = 42        # 固定随机种子，保证每次造出来都一样、可复现

# 10 条中文提问：{英文通路名: 中文问法}
# 键必须与 term_to_pathway() 的输出逐字一致（GOBP_X_Y -> "x y" 小写）。
# 手写的理由：GO 里没有中文名，机器翻译会把"通路名"翻成看不懂的术语，
# 反而测不出「中文用户会怎么问」。这 10 条覆盖不同金标规模（5~43 个基因）。
ZH_QUESTIONS = {
    "platelet degranulation": "血小板脱颗粒涉及哪些基因？",
    "purine nucleotide transport": "嘌呤核苷酸转运涉及哪些基因？",
    "embryonic hemopoiesis": "胚胎期造血涉及哪些基因？",
    "cardiac muscle tissue regeneration": "心肌组织再生涉及哪些基因？",
    "very long chain fatty acid biosynthetic process": "超长链脂肪酸的生物合成过程有哪些基因参与？",
    "intracellular sodium ion homeostasis": "细胞内钠离子稳态由哪些基因调控？",
    "regulation of positive chemotaxis": "正向趋化运动的调控涉及哪些基因？",
    "resolution of meiotic recombination intermediates": "减数分裂重组中间体的解离由哪些基因完成？",
    "post embryonic eye morphogenesis": "胚胎期后的眼睛形态发生由哪些基因控制？",
    "regulation of killing of cells of another organism": "杀伤异种细胞的调控涉及哪些基因？",
}


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

    # 2. 英文题：从「不含中文题那 10 条通路」的池子里固定种子抽 N_EVAL 条
    #    排除它们的理由：中文题要单独成组统计，同一条通路用中英两种问法各出现一次，
    #    会让"两条题"变成"一条题测两遍"，统计量被摊薄、结论也不好解释。
    by_pathway = {p["pathway"]: p for p in pool}
    en_pool = [p for p in pool if p["pathway"] not in ZH_QUESTIONS]
    rng = random.Random(SEED)
    rng.shuffle(en_pool)
    selected = en_pool[:N_EVAL]

    # 3. 英文 30 条 + 中文 10 条（中文是**追加**，不是替换：
    #    替换会让英文基线从 30 缩到 20，"补中文题之后整体分数怎么变"就没法归因了）
    items = [{
        "question": f"Which genes are involved in {p['pathway']}?",
        "lang": "en",
        "pathway": p["pathway"],
        "gold_genes": p["gold_genes"],
    } for p in selected]

    used_zh = []
    for pathway, zh in ZH_QUESTIONS.items():
        p = by_pathway.get(pathway)
        if p is None:
            continue           # 通路不存在，或金标规模没过 MIN/MAX_GENES 的筛——下面统一报警
        items.append({
            "question": zh,
            "lang": "zh",
            "pathway": pathway,
            "gold_genes": p["gold_genes"],
        })
        used_zh.append(pathway)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    n_zh = len(used_zh)
    print(f"已生成 {len(items)} 条评测（英文 {len(items) - n_zh} / 中文 {n_zh}）→ {OUT_PATH}")
    print(f"（从 {len(pool)} 条候选通路里抽，金标基因数 {MIN_GENES}~{MAX_GENES}）\n")

    # 4. 中文题对不上就明确报警：GO 数据变了、或 SEED 变了都会导致这条
    missing = [pw for pw in ZH_QUESTIONS if pw not in used_zh]
    if missing:
        print(f"⚠ 有 {len(missing)} 条中文题的目标通路没被抽中，这些中文题没进评测集：")
        for pw in missing:
            print(f"    · {pw}")

    for it in items[:5]:
        print(f"  · [{it['lang']}] {it['question']}  （金标 {len(it['gold_genes'])} 个基因）")
    print()
    for it in [x for x in items if x["lang"] == "zh"][:10]:
        print(f"  · [zh] {it['question']}  （金标 {len(it['gold_genes'])} 个基因，通路 {it['pathway']}）")


if __name__ == "__main__":
    main()
