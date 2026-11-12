"""第三步 · 检索：把「问基因功能」抽成一个可复用的函数，供 LangGraph 节点调用。

原理回顾（菜谱卡片 + 闻味道）：
  1. 每张基因卡片 = 一张菜谱，已经提前用 bge-m3 算好了「气味指纹」（1024 维向量）
  2. 用户的问题也算成一个「气味指纹」
  3. 两个指纹做余弦相似度（= 归一化后点积），越像 = 越相关
  4. 返回最像的 top_k 张卡片（基因名 + 原文 + 相似度），供 LLM 溯源

为什么单独抽成一个函数：
  - 检索逻辑写一遍，测试脚本用、LangGraph 节点也用，不重复
  - 数据是「惰性加载」：真正调用时才开始读文件，import 这个模块时不吃内存
"""
"""
导入 4 个工具
↓
找路：__file__ → dirname 两层 → PROJECT_ROOT → 拼出 vecs.npy / gene_docs.csv 路径 → load_dotenv 读密钥
↓
三个空箱子 _vecs/_cards/_embedder = None
↓
_load()：if _vecs is None 才读文件、建客户端，global 声明后返回三个
↓
retrieve_genes()：_load 拿三件套 → 算问题向量 q → 两个都归一化 → scores = vecs @ q
                  → argsort 取 top 下标 → for 循环拼 (基因名, 原文, 相似度) → return"""


import os   #管文件路径、环境变量

import numpy as np  #矩阵、向量运算全靠它
import pandas as pd  #表格处理库；读 csv、操作「带列名的表」
from dotenv import load_dotenv
from openai import OpenAI

# ── 第 1 块：定位项目根目录，拼数据路径 ─────────────────
# __file__        = 当前这个文件的完整路径
# os.path.abspath = 把路径变成绝对路径（去掉可能有的 ..）
# os.path.dirname = 取它所在的文件夹
# 套两层 dirname  =  先到 omics_agent 文件夹，再往上一级 = 项目根目录
# 这样不写死 D 盘，换台电脑也能用。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# #os.path.join(文件夹, 文件名, ...) = 把几段拼成一条完整路径（自动加 \ 或 /）
# 向量矩阵文件（N 行 × 1024 列，每一行是一个基因卡片的指纹）
VECS_PATH = os.path.join(PROJECT_ROOT, "data", "vecs.npy")

# 基因卡片文件（两列：gene 基因名、description 功能原文）
CARDS_PATH = os.path.join(PROJECT_ROOT, "data", "gene_docs.csv")

# load_dotenv = 读 .env 文件，把里面的 KEY=值 变成环境变量
# 读完以后，下面 os.getenv("SILICONFLOW_API_KEY") 才能拿到密钥
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# ── 第 2 块：三个空箱子（惰性加载）────────
_vecs = None      # 向量矩阵；None = 还没读
_cards = None     # 基因卡片表；None = 还没读
_embedder = None  # embedding 客户端；None = 还没建


# ================= 第 3 块：_load 开箱 =================
def _load():
    """第一次调用时，把向量、卡片、embedding 客户端一次性准备好（之后直接复用）。

    好处：import rag 时什么都不做；只有真正调 retrieve_genes 才读文件、
    才建客户端，避免程序一启动就吃掉 1.8 万 × 1024 的内存。
    """
    # global = 「我下面要改的是模块级那三个变量，不是新建局部变量」
    # 不写这句，_vecs 会被当成函数内部的新变量，改不到外面的
    global _vecs, _cards, _embedder

    if _vecs is None:                      ## is None = 「是空的吗」；空的才读
        _vecs = np.load(VECS_PATH).astype(np.float32)   # 读向量文件成矩阵，转 float32 省内存
        # np.load(路径) = 读 .npy 文件成 numpy 数组（矩阵）
        # .astype(类型) = 把数据类型转成指定的；np.float32 = 32 位小数，省一半内存

        _cards = pd.read_csv(CARDS_PATH)                # 读基因卡片成表格
        # pd.read_csv(路径) = 读 csv 文件成「表」（DataFrame）

        _embedder = OpenAI(                             # 建 embedding 客户端
            api_key=os.getenv("SILICONFLOW_API_KEY"),   # 密钥从环境变量拿
            base_url="https://api.siliconflow.cn/v1",   # 硅基流动的接口地址
        )
    return _vecs, _cards, _embedder         # 三个一起返回，谁要谁取


# ================= 第 4 块：retrieve_genes 正主 =================
def retrieve_genes(question: str, top_k: int = 5):
    """问一个问题，返回最相关的 top_k 个基因（基因名、原文、相似度）。

    参数：
      question : 想问的问题，比如 "哪些基因参与 DNA 损伤修复？"
      top_k    : 返回几个最像的基因，默认 5 个

    返回：
      一个列表，每项是一个 (基因名, 原文描述, 相似度) 三元组。
      相似度范围 0~1，越接近 1 表示越相关（这就是「溯源」需要的证据）。
    """
    vecs, cards, embedder = _load()
    # 调 _load() 拿三个东西，按顺序装进 vecs / cards / embedder 三个变量

    # ① 把问题算成向量（气味指纹）
    # .embeddings.create = 调用接口，把文字变成向量
    r = embedder.embeddings.create(model="BAAI/bge-m3", input=question)

    # r.data = 返回的数据列表；[0] = 第一条；.embedding = q = 问题的向量（长度 1024））
    # np.array(列表, dtype=类型) = 把列表变成 numpy 数组，指定类型 float32
    q = np.array(r.data[0].embedding, dtype=np.float32)

    # #② 归一化：让向量长度都变成 1，点积才等于余弦相似度1
    q = q / np.linalg.norm(q)
    # np.linalg.norm(q) = 向量 q 的「长度」（模长）；q / 长度 = 缩成单位向量

    # axis=1 = 按「每一行」分别算长度；keepdims=True = 保留二维形状，方便逐行相除
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    # 结果：vecs 每一行都变成单位向量

    # ③ 一次矩阵乘法算相似度
    # @ = 矩阵乘法；vecs(N×1024) @ q(1024,) = scores(N,)
    # 意思是「每个基因向量」和「问题向量」各算一个点积 = 相似度
    scores = vecs @ q

    # ④ 取相似度最高的 top_k 个下标
    top = np.argsort(-scores)[:top_k]
    # np.argsort(数组) = 返回「从小到大排好」后，每个位置原本的下标
    # -scores = 把相似度取负 → 大的变小的 → 排最前面的就是原来最大的
    # [:top_k] = 只取前 top_k 个（切片：从头切到第 top_k 个）

    # ⑤ 拼成 (基因名, 原文, 相似度) 的列表返回
    results = []                                       # 空列表，装最后结果
    for idx in top:                                    # for = 挨个取；idx = 一个基因的下标
        gene = cards.iloc[idx]["gene"]
# cards.iloc[idx] = 按「位置」取表格第 idx 行；["gene"] = 再取 gene 那一列的值

        doc = cards.iloc[idx]["description"]  # 同上，取 description 列（原文）

        results.append((gene, doc, float(scores[idx])))
        # .append(东西) = 往列表末尾塞一个
        # (a, b, c) = 三元组（小括号包三个值）
        # float(数) = 把 numpy 的小数转成普通 Python 小数
        # scores[idx] = 这个基因的相似度分数

    return results # 返回整个列表


def get_gene_function(gene: str):
    """按基因名精确查一张卡片的功能原文（直接对表查，不是语义检索）。

    参数：
      gene : 基因名，比如 "BRCA1"

    返回：
      这个基因的功能原文（字符串）；查不到就返回 None（空）。
    """
    _, cards, _ = _load()
    # 调 _load() 拿三件套，但这里只要卡片表 cards；
    # 向量、客户端用不到，用下划线 _ 占个位置表示「我不用你」

    row = cards[cards["gene"] == gene]
    # cards["gene"] = 取表格里 gene 这一列（一整列基因名）
    # == gene        = 拿这一列去和要查的基因名「逐一比较」，等于就是 True
    # cards[...]     = 用「真/假」来挑行：只留下 True 的那些行
    # row            = 匹配到的行（一个基因通常只有 0 或 1 行）

    if row.empty:                      # .empty = 这张小表「是空的吗」（一行都没匹配上）
        return None                    # 空的 = 查不到，返回 None

    return row.iloc[0]["description"]
    # row.iloc[0] = 按位置取第一行（以防万一有重名，只取第一个）
    # ["description"] = 再取这一行的 description 列 = 功能原文
