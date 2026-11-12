"""第三步 · 第二步：把基因卡片「向量化 + 入库」（带容错版）。

做三件事：
  1) 读 gene_cards.csv（约 1.8 万张基因卡片）
  2) 每张卡片的 description 用 bge-m3 变成向量（指纹）
  3) 向量 + 原文 + 基因名 一起存进 Chroma（指纹库）

容错：某批/某条偶发报 400 时，自动降级成单条重试，实在不行跳过，保证能跑完。
"""
import os
import pandas as pd
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("D:/zj002/多组学数据分析智能体/.env")

# ── 1. 读基因卡片 ──────────────────────────────
cards = pd.read_csv("D:/zj002/多组学数据分析智能体/data/gene_cards.csv")
print("基因卡片总数：", len(cards))

LIMIT = None                   # None = 全量；想先测试就填 100
if LIMIT:
    cards = cards.head(LIMIT)

# ── 2. embedding 客户端（文字变向量）────────────
embedder = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

# ── 3. 建向量库 Chroma ─────────────────────────
client = chromadb.PersistentClient(path="D:/zj002/多组学数据分析智能体/chroma_db")
collection = client.get_or_create_collection(name="gene_cards")

# ── 4. 分批向量化 + 入库（带容错）───────────────
BATCH = 50                    # 一批 50 条（硅基流动上限 64，别超）
FLUSH = 5000                  # 攒够 5000 条才写一次库（段少，索引才建得起来）

ids, docs, vecs, metas = [], [], [], []   # 攒结果的大列表
skipped = []                   # 记录最终跳过的基因

def embed_one(text):
    """给一条文本算向量，返回 1024 维列表。"""
    r = embedder.embeddings.create(model="BAAI/bge-m3", input=text)
    return r.data[0].embedding

def flush():
    """把攒好的结果写进 Chroma 再清空（大段写入，全库只有 4 段）。"""
    if ids:
        collection.add(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)
        ids.clear(); docs.clear(); vecs.clear(); metas.clear()

for i in range(0, len(cards), BATCH):
    batch = cards.iloc[i:i+BATCH]
    genes = list(batch["gene"])
    texts = [str(t)[:2000] for t in batch["description"]]  # 截断防超长

    # 先试整批
    try:
        resp = embedder.embeddings.create(model="BAAI/bge-m3", input=texts)
        batch_vecs = [d.embedding for d in resp.data]
    except Exception:
        # 整批失败 → 降级成一条一条试
        batch_vecs = []
        for g, t in zip(genes, texts):
            try:
                batch_vecs.append(embed_one(t))
            except Exception:
                skipped.append(g)          # 单条也失败，记下并跳过
                batch_vecs.append(None)

    # 把这批成功的结果攒进大列表
    for g, t, v in zip(genes, texts, batch_vecs):
        if v is not None:
            ids.append(f"gene_{g}")
            docs.append(t)
            vecs.append(v)
            metas.append({"gene": g})

    # 攒够 FLUSH 条就写一次库（避免每批写、段太多）
    if len(ids) >= FLUSH:
        flush()

    print(f"已处理 {min(i + BATCH, len(cards))}/{len(cards)} 条")

flush()   # 最后不足 5000 的尾巴也写进去

print("入库完成，库里共", collection.count(), "条")
if skipped:
    print("跳过了", len(skipped), "条：", skipped[:10])
