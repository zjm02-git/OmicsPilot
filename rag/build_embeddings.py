"""第三步 · 向量化：把基因卡片算成向量，存成 numpy 文件（绕开 chromadb 索引 bug）。

做两件事：
  1) 读 gene_cards.csv，每张卡片用 bge-m3 算成 1024 维向量（指纹）
  2) 向量存成 vecs.npy、基因名+原文存成 gene_docs.csv，之后检索直接读这两个文件

为什么不用 chromadb：它的 HNSW 索引在这环境反复报"Error loading hnsw index"，
1.8 万条其实用 numpy 暴力算余弦相似度就够（毫秒级），没必要上向量库。
"""
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("D:/zj002/多组学数据分析智能体/.env")

cards = pd.read_csv("D:/zj002/多组学数据分析智能体/data/gene_cards.csv")
print("基因卡片总数：", len(cards))

LIMIT = None                   # None = 全量；想先测试就填 100
if LIMIT:
    cards = cards.head(LIMIT)

embedder = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

BATCH = 50
vecs, genes, docs = [], [], []   # 三个列表，顺序一一对应
skipped = []

def embed_one(text):
    r = embedder.embeddings.create(model="BAAI/bge-m3", input=text)
    return r.data[0].embedding

for i in range(0, len(cards), BATCH):
    batch = cards.iloc[i:i+BATCH]
    b_genes = list(batch["gene"])
    texts = [str(t)[:2000] for t in batch["description"]]  # 截断防超长

    try:
        resp = embedder.embeddings.create(model="BAAI/bge-m3", input=texts)
        batch_vecs = [d.embedding for d in resp.data]
    except Exception:
        batch_vecs = []
        for g, t in zip(b_genes, texts):
            try:
                batch_vecs.append(embed_one(t))
            except Exception:
                skipped.append(g)          # 单条也失败，记下并跳过
                batch_vecs.append(None)

    for g, t, v in zip(b_genes, texts, batch_vecs):
        if v is not None:
            genes.append(g)
            docs.append(t)
            vecs.append(v)

    print(f"已处理 {min(i + BATCH, len(cards))}/{len(cards)} 条")

# 存盘：向量用 float32 省一半空间；基因名+原文存 csv（两文件行序一致）
vecs = np.array(vecs, dtype=np.float32)
np.save("D:/zj002/多组学数据分析智能体/data/vecs.npy", vecs)
pd.DataFrame({"gene": genes, "description": docs}).to_csv(
    "D:/zj002/多组学数据分析智能体/data/gene_docs.csv", index=False
)
print("存盘完成：向量", vecs.shape, "，卡片", len(genes), "张")
if skipped:
    print("跳过了", len(skipped), "条：", skipped[:10])
