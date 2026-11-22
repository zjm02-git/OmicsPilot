"""第三步 · 检索：把「问基因功能」抽成一个可复用的函数，供 LangGraph 节点调用。

原理回顾（菜谱卡片 + 闻味道）：
  1. 每张基因卡片 = 一张菜谱，已经提前用 bge-m3 算好了「气味指纹」（1024 维向量）
  2. 用户的问题也算成一个「气味指纹」
  3. 两个指纹做余弦相似度（= 归一化后点积），越像 = 越相关
  4. 返回最像的 top_k 张卡片（基因名 + 原文 + 相似度），供 LLM 溯源

为什么单独抽成一个函数：
  - 检索逻辑写一遍，测试脚本用、LangGraph 节点也用，不重复
  - 数据是「惰性加载」：真正调用时才开始读文件，import 这个模块时不吃内存

检索从一路变成三档（都是同一个函数家族的成员，形状一致可互换）：
  retrieve_genes           纯向量        基线，也是最贵的语义路径
  retrieve_bm25            纯 BM25       字面基线，评测里用来证明「向量到底有没有用」
  retrieve_hybrid          向量+BM25+RRF 产品默认（nodes.py 现在就在用）
  retrieve_hybrid_reranked 上面这档 + LLM 重排（给调度方一行调用，带开关/缓存/降级）

中文这一跳的两个坑（修在 bm25.py，但结论写在这里，因为这里是检索入口）：
  1) 旧切词只认 ASCII，中文提问切出空词表，BM25 一路 0 命中 → 混合检索退化成纯向量；
  2) 光修切词还不够：卡片是英文通路名，汉字只出现在模板串里，所以中文提问
     仍然命不中——还要靠 bm25.expand_query 的中→英种子词表补一跳。
  评测（evals/run_rag_eval.py）把这两层分别量出来：bm25 / bm25_nogloss 两列。
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

from omics_agent.bm25 import BM25  # 稀疏检索（字面匹配），和向量检索互补

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
_bm25 = None      # BM25 索引；None = 还没建（惰性加载）


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


# ================= 第 4.5 块：混合检索（向量 + BM25 + RRF）=================
def _vector_ranks(question: str):
    """把问题算成向量，返回「按相似度从高到低」的 (doc_id, 相似度) 全量列表。

    和上面的 retrieve_genes 是同一套 embedding 逻辑，抽出来供混合检索复用；
    两处保持一致（模型名、归一化、点积）。
    """
    vecs, _, embedder = _load()
    r = embedder.embeddings.create(model="BAAI/bge-m3", input=question)
    q = np.array(r.data[0].embedding, dtype=np.float32)
    q = q / np.linalg.norm(q)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    scores = vecs @ q
    order = np.argsort(-scores)  # 相似度从高到低的下标
    return [(int(i), float(scores[i])) for i in order]


def _get_bm25():
    """惰性建 BM25 索引（在卡片 description 上建倒排表），只建一次。"""
    global _bm25
    if _bm25 is None:
        _, cards, _ = _load()
        _bm25 = BM25(cards["description"].astype(str).tolist())
    return _bm25


def load_cards():
    """返回卡片表（gene / description 两列），供评测脚本算「数据侧指标」。

    为什么要暴露这个：评测要算「同源泄漏」——问题里的通路名是否逐字出现在
    金标卡片里——这必须拿到原始卡片文本。让评测脚本去摸 _load() 这种私有函数
    会把内部实现焊死在脚本里，所以给一个显式出口。
    """
    _, cards, _ = _load()
    return cards


def retrieve_bm25(question: str, top_k: int = 5, expand_zh: bool = True):
    """只用 BM25 的字面检索（评测里的「BM25-only 基线」也走这里）。

    为什么单独暴露：混合检索的收益必须和「只用 BM25」比才说得清，
    否则回答不了「那向量那一路到底有没有用」。也方便排查
    「某条问题为什么没被召回」——把两路分开看比看融合结果直观。
    """
    _, cards, _ = _load()
    hits = _get_bm25().search(question, top_k=top_k, expand_zh=expand_zh)
    return [(cards.iloc[d]["gene"], cards.iloc[d]["description"], float(s)) for d, s in hits]


def retrieve_hybrid(question: str, top_k: int = 5, rrf_k: int = 60, expand_zh: bool = True):
    """混合检索：向量 + BM25 两路各自排名，用 RRF 融合后取前 top_k。

    为什么 RRF（Reciprocal Rank Fusion）：
      向量分数是余弦相似度（约 0~1），BM25 分数是无上界的加权词频，两把尺子
      不能直接相加。RRF 只看「排名」不看绝对分——每个基因在每路里的名次
      换算成 1/(rrf_k + 名次) 再累加，跨尺度也能公平融合。

    参数 rrf_k：RRF 的平滑常数。越小越「奖励第一名」（排名差距被放大），
      越大越「平均看待两路」。经典取值 60，评测里做了 10/60/100 的消融——
      这个数不该靠抄论文，应该在自己的数据上选。

    参数 expand_zh：是否启用中→英词表扩展（默认开）。传 False 就是
      「只有 CJK 二元组、没有双语词表」那一档，评测拿它做消融。

    返回：[(基因名, 原文, RRF 分数), ...]，形状和 retrieve_genes 一致。
    """
    _, cards, _ = _load()

    # ① 向量这一路的排名（全量）
    vec_ranked = _vector_ranks(question)  # [(doc_id, 相似度), ...] 已按相似度降序

    # ② BM25 这一路的排名（只返回命中了查询词的文档）
    #    expand_zh=True：除 CJK 二元组外，再并入中→英词表扩展项（英文提问不受影响）
    bm25_ranked = _get_bm25().search(question, top_k=len(cards), expand_zh=expand_zh)  # [(doc_id, bm25分), ...]

    # ③ RRF 融合：名次从 1 起，只在某一路出现过的文档只累加该路的分
    rrf = {}
    for rank, (doc_id, _) in enumerate(vec_ranked):
        rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, (doc_id, _) in enumerate(bm25_ranked):
        rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)

    order = sorted(rrf, key=lambda d: rrf[d], reverse=True)[:top_k]
    return [(cards.iloc[d]["gene"], cards.iloc[d]["description"], rrf[d]) for d in order]


# ================= 第 4.6 块：混合召回 + LLM 重排（一行调用）=================
def retrieve_hybrid_reranked(question: str, top_k: int = 5, rrf_k: int = 60, pool: int = 20,
                             expand_zh: bool = True, return_info: bool = False):
    """混合召回 + LLM 重排：给调度方的「一行调用」入口。

    为什么分两段（粗召回 pool 个 → 重排只取 top_k）：
      重排只能重排「已经在候选里的东西」。如果 pool == top_k，重排最多在
      5 个里调个顺序，翻盘空间几乎没有；先粗召回 20 个再让 LLM 挑 5 个，
      才有机会把融合排序里被埋掉的相关项捞上来。代价是多一次 LLM 往返。

    为什么 rerank 要在这里延迟 import：
      omics_agent.rerank 依赖 langchain + DeepSeek 客户端；而「不重排」的链路
      （nodes.py 现在用的 retrieve_hybrid）不该被迫背上这份依赖——
      import rag 还是像原来一样轻。

    返回：[(基因名, 原文, 分数), ...]，与 retrieve_hybrid 同形状，可直接替换。
          return_info=True 时返回 (结果, info)，info["status"] 说明这次是
          真的重排了（ok/cache_hit）还是降级了（disabled/llm_error/...）。
    """
    candidates = retrieve_hybrid(question, top_k=max(pool, top_k), rrf_k=rrf_k, expand_zh=expand_zh)
    from omics_agent.rerank import maybe_rerank   # 延迟导入，理由见上面的 docstring
    return maybe_rerank(question, candidates, top_k=top_k, return_info=return_info)


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
