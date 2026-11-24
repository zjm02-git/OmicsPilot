"""第三步 · 检索 + 溯源：问一个问题，用 numpy 暴力算余弦相似度找最像的基因卡片。

绕开 chromadb（HNSW 索引在这环境建不起来），1.8 万条用 numpy 矩阵乘法毫秒级
就能算完——小规模数据根本用不上向量库，直接暴力算更快更稳。
"""
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("D:/zj002/多组学数据分析智能体/.env")

# ── 1. 读已存盘的向量 + 基因卡片 ─────────────────
vecs = np.load("D:/zj002/多组学数据分析智能体/data/vecs.npy")      # (N, 1024)
cards = pd.read_csv("D:/zj002/多组学数据分析智能体/data/gene_docs.csv")

# ── 2. 把问题算成向量 ──────────────────────────
embedder = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)
question = "哪些基因参与 DNA 损伤修复？"      # 改成你想问的
r = embedder.embeddings.create(model="BAAI/bge-m3", input=question)
q = np.array(r.data[0].embedding, dtype=np.float32)

# ── 3. 归一化：把每个向量长度变成 1，这样「点积」就等于「余弦相似度」──
q = q / np.linalg.norm(q)
vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

# ── 4. 一次矩阵乘法算「问题 vs 所有基因」的相似度 ──
scores = vecs @ q                        # (N,) 每个基因和问题的相似度
top = np.argsort(-scores)[:5]            # 取最高的 5 个下标

# ── 5. 打印（基因名 + 原文 + 相似度 = 溯源）──────
for idx in top:
    gene = cards.iloc[idx]["gene"]
    doc = cards.iloc[idx]["description"]
    print(f"【{gene}】 相似度 {scores[idx]:.3f}")
    print(f"  {doc[:120]}...")
    print()
