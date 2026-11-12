

"""第三步 · 第一步：把 GO 注释表（基因→通路）聚合成「基因功能卡片」。

输入：data/gene_ontology.csv   两列 gene(基因名)、term(通路名)，63 万行
输出：data/gene_cards.csv      两列 gene(基因名)、description(功能说明一句话)

只用 pandas，不用装任何新库。
"""
import pandas as pd # 导入 pandas，以后写 pd 就代表它

# ── 第 1 步：读原始表 ──────────────────────────────
# pd.read_csv(文件路径) = 把一个 csv 文件读成「表」，存进变量 df
# df 是 DataFrame，你就理解成「带列名的二维表格」
df=pd.read_csv("data/gene_ontology.csv")

# ── 清洗通路名：GOBP_DNA_DAMAGE → dna damage（去前缀 + 下划线转空格 + 小写）──
# 原格式是机器命名（大写+下划线+前缀），embedding 难分词；
# 转成自然英文后，模型能看懂每个通路的真实含义，检索更准。
def clean_term(t):
    t = str(t)
    for p in ("GOBP_", "GOMF_", "GOCC_"):   # 三类 GO 前缀，去掉
        if t.startswith(p):
            t = t[len(p):]
            break
    return t.replace("_", " ").lower()        # 下划线转空格 + 小写

df["term"] = df["term"].apply(clean_term)

# ── 第 2 步：统计每个基因有几个通路 ─────────────────
# df.groupby("gene") = 按 gene 这一列分组（同名基因归到一组）
# ["term"]           = 分组之后，只看 term 这一列
# .count()           = 数每组有几行 → 得到「基因名 → 通路个数」
# 结果存进 term_count，它是一个 Series（一列带名字的数字）
term_count=df.groupby("gene")["term"].count()

# ── 第 3 步：把每个基因的一堆通路名，连成一句话 ─────
# .agg(函数) = 对每一组「套用」这个函数，把一组值变成一个值
# lambda x: "、".join(x) = 一次性小函数：把这一组的所有 term 用顿号连起来
#                          x 就代表「这一组的所有 term」
term_list=df.groupby("gene")["term"].agg(lambda x: " , ".join(x))

# ── 第 4 步：拼成「基因功能卡片」 ───────────────────
rows=[]                       # 准备一个空列表，最后装结果
for gene in term_list.index:  # .index = 所有基因名；for 挨个拿出来
     # f"..." = f 字符串，花括号 {} 里的内容会被替换成真实值
     desc = f"{gene}：参与通路 {term_list[gene]}（共 {term_count[gene]} 个）"
     rows.append({"gene": gene, "description": desc})  # 每行存成一个字典

# ── 第 5 步：存成新文件 ─────────────────────────────
cards = pd.DataFrame(rows)      # 一堆字典的列表 → 变成表
cards.to_csv("data/gene_cards.csv", index=False)   # 存盘；index=False = 不写行号

print(cards.head(3))   # 打印前 3 行，看看对不对