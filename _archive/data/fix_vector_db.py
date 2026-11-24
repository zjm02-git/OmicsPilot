"""修复向量库查询报错（Error loading hnsw index），不用重算 embedding。

原因：之前每批 32 条 add，一共 add 了 561 次 = 561 个「小段」；
     Chroma 查询前要把这些小段合并成索引（compaction），段太多合并崩了。

修法：把数据取出来（get 按 id 取、不碰相似度索引，能成功），删掉旧库，
     用「大段」（每 5000 条一段、共 4 段）重新存回去——段少了，索引就能建好。
"""
import chromadb

DB_PATH = "D:/zj002/多组学数据分析智能体/chroma_db"
NAME = "gene_cards"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=NAME)

# ── 1. 取出全部数据（get 不需要相似度索引，应该能成功）──────
data = collection.get(include=["embeddings", "documents", "metadatas"], limit=20000)
ids = data["ids"]
embeddings = [list(e) for e in data["embeddings"]]   # 转成标准 list，喂 add 更稳
documents = data["documents"]
metadatas = data["metadatas"]
print("取出了", len(ids), "条")

# ── 2. 安全检查：数量必须对（跳过了 6 条 → 17945）────────────
if len(ids) != 17945:
    print("数量不对，先别删，停下来看看（期望 17945，实际", len(ids), "）")
    exit()

# ── 3. 删旧库，原地重建同名 ───────────────────────────────
client.delete_collection(NAME)
new_col = client.get_or_create_collection(name=NAME)

# ── 4. 用大段重存（每 5000 条一段，共 4 段，远小于上限 5461）──
BATCH = 5000
for i in range(0, len(ids), BATCH):
    new_col.add(
        ids=ids[i:i+BATCH],
        embeddings=embeddings[i:i+BATCH],
        documents=documents[i:i+BATCH],
        metadatas=metadatas[i:i+BATCH],
    )
    print(f"已重存 {min(i+BATCH, len(ids))}/{len(ids)} 条")

print("重建完成，库里共", new_col.count(), "条")
print("现在再跑 retrieve.py 试试")
