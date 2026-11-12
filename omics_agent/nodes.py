"""8 个节点函数：每个节点只干一件事，从 State 读、往 State 写。

分工原则：
  - LLM 干"懂"的活（①理解需求、⑥解读结果）
  - Python 干"算"的活（②质检、④差异统计、⑤通路富集、⑧拼报告）
  - 人工介入：③人工确认；记忆：②读偏好、⑦写偏好
"""

import os
from typing import Literal

import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from omics_agent.config import get_model
from omics_agent.diff import differential_expression
from omics_agent.enrichment import enrichment_analysis

from omics_agent.rag import get_gene_function,retrieve_genes
# 一次 import 两个：get_gene_function 精确查名、retrieve_genes 语义检索
# 把 rag.py 里的「精确查名」函数拿进来用

# 默认数据目录、分组列名
DATASET_PATH = "data"
GROUP_COL = "condition"

# 长期记忆的命名空间和 key（记忆按"研究者"区分，存各自的分析偏好）
MEMORY_NS = ("omics",)
PREFS_KEY = "preferences"


#=====1._is_numeric —— 判断"是不是全是数字"===
def _is_numeric(df:pd.DataFrame)->bool:
    try:#能全部转化数字返回True
        df.apply(pd.to_numeric)#.apply(某函数) =把这个函数对每一列都执行一遍
        return True
    except Exception:
        return False


#=====2.找出"分组列".先排除 ID 列===
def _find_group_col(df:pd.DataFrame)->str | None:
    for c in df.columns:#df.columns = 这张表的所有列名
        n_unique=df[c].nunique()#.nunique() = 这一列有多少个不同的值。
        if n_unique==len(df):#len(df) = 表有几行。
            continue#跳过，是id列
        if 2 <= n_unique <=10:
            return c# 返回这个列名
    return None#循环跑完都没找到，返回空


#=====3.输入一个目录，输出识别结果===
def discover_dataset(dataset_path:str)->dict:
    #files = [...] 这一段是"列表推导式"，一句话生成一个列表
    #整句意思：把目录里所有 csv 文件的完整路径，收集成一个列表放进 files。
    files=[
        os.path.join(dataset_path,f)
        #列出 dataset_path 这个目录里所有文件名，返回一个列表
        for f in os.listdir(dataset_path)
        #f.lower() = 把文件名转小写(防止写成CSV)  判断结尾是不是.csv
        if f.lower().endswith(".csv")
    ]
    if not files:
        raise ValueError("目录里没有 csv文件")

    #========第一步  认表达矩阵=======

    expr_file=None
    for path in files:#遍历每个 csv 文件路径
        head=pd.read_csv(path,nrows=50)#读 csv 成表格； 只读前 50行
        #head.shape = 一个 (行数, 列数) 的元组  取第 2 个元素 = 列数
        if head.shape[1] > 20:
            #.iloc = 按位置取；[:, 0] = 所有行 : + 第 1 列,即"取第一列"
            #iloc[:, 1:] = 所有行 + 从第2列到末尾；_is_numeric(...) = 这些列是不是全数字
            if head.iloc[:, 0].nunique()==len(head) and _is_numeric(head.iloc[:,1:]):
                #== len(head)等于行数→ 第一列每行都不重复
                expr_file=path#两个条件都满足 → 记录"这个文件是表达矩阵"
                break# 直接跳出整个 for 循环
    if expr_file is None:
        raise ValueError("没识别出表达矩阵")

    #=====第二步 拿样本数=======
    n_samples=pd.read_csv(expr_file,nrows=0).shape[1]-1#只读表头，不读数据行（只要列名，最快）
    # .shape表格的 (行数, 列数)  .shape[1] 取第 2 个 = 列数 - 1减掉第一列（基因列

    #=====第三步  用行数找分组表===
    meta_file=go_file=None
    for path in files:
        if path ==expr_file:
            continue
        # pd.read_csv(path) = 读整个文件 len(...) =这个表的行数
        n_rows = len(pd.read_csv(path))
        if n_rows==n_samples:#行数恰好等于样本数（226）→ 这是分组表
            meta_file=path
        else:
            go_file=path

    #=====第 5 块：检查 + 认分组列 + 返回=====

    if meta_file is None:
        raise ValueError(f"没识别出分组表（行数应等于样本数 {n_samples}）")

    meta=pd.read_csv(meta_file)#完整读分组表
    # 在确定的分组表里认"哪列是分组列"。
    group_col=_find_group_col(meta)
    if group_col is None:
        raise ValueError("分组表里没找到分组列")


    #=====# 第6块. 返回识别结果=====
    return  {
            "expr_file":expr_file,
            "meta_file":meta_file,
            "go_file":go_file,
            "group_col":group_col,#分组列名
         }


# ══════════ ① 需求解析 ══════════
class IntentOutput(BaseModel):
    """让 LLM 输出"确定的值"，而不是自由文本，路由才可靠。"""

    analysis_type: Literal["differential", "pca","clustering","qa","other"] = Field(
        description="要做的分析类型。differential=差异表达分析,pca=主成分分析,"
                    "clustering=聚类,qa=问基因功能/知识问答,other=其他/不支持"
    )


def parse_intent_node(state: dict) -> dict:
    """结构化输出：判断用户想做什么分析。"""
    model = get_model().with_structured_output(IntentOutput)
    res = model.invoke([HumanMessage(state["user_request"])])
    return {"analysis_type": res.analysis_type}


# ══════════ ② 加载 + 质检 ══════════
def load_qc_node(state: dict, runtime: Runtime) -> dict:
    """读数据 + 质检，质检通过后从长期记忆读这个用户上次的分析偏好。

    runtime 是 LangGraph 自动注入的：节点函数签名里声明了 Runtime 参数，
    就能拿到它来读写长期记忆（runtime.store），不需要自己手动传。
    """
    dataset_path = state.get("dataset_path", DATASET_PATH)

    # 靠内容识别数据文件（不再写死文件名），识别失败 = 质检失败
    try:
        found=discover_dataset(dataset_path)
    except ValueError as e:
        return{"qc_passed":False,"qc_report":f"数据识别失败：{e}"}
    expr_file=found["expr_file"]
    meta_file=found["meta_file"]
    go_file=found["go_file"]
    group_col=found["group_col"]


    #state = 这次任务的记事本（写完就丢）；runtime = 工厂发的工具箱，
    # 里面最有用的是 runtime.store 这个长期记忆档案柜（写完还在）。
    # 节点想记/想查"用户上次的偏好"，就伸手进工具箱拿档案柜。


    # 1. 文件在不在
    try:
        expr_df = pd.read_csv(expr_file)
        metadata_df = pd.read_csv(meta_file)
    except FileNotFoundError as e:
        return {"qc_passed": False, "qc_report": f"数据文件缺失：{e}"}

    # 2. 分组列在不在
    if group_col not in metadata_df.columns:
        return {"qc_passed": False, "qc_report": f"metadata.csv 缺少分组列 {group_col}"}

    # 3. 必须恰好两组（自动识别 A/B 两组名字）
    groups = metadata_df[group_col].unique().tolist()
    if len(groups) != 2:
        return {"qc_passed": False, "qc_report": f"需要恰好 2 组，实际 {len(groups)} 组"}
    group_a, group_b = groups[0], groups[1]

    # 4. 每组样本数不能太少（t 检验至少要有几个样本才有意义）
    n_a = int((metadata_df[group_col] == group_a).sum())
    n_b = int((metadata_df[group_col] == group_b).sum())
    if n_a < 20 or n_b < 20:
        return {"qc_passed": False, "qc_report": f"每组至少 20 个样本，实际 A={n_a} / B={n_b}"}

    # 5. 缺失值不能太多
    na_ratio = float(expr_df.isna().mean().mean())
    if na_ratio > 0.1:
        return {"qc_passed": False, "qc_report": f"缺失值过多：{na_ratio:.1%}"}

    qc_report = (
        f"质检通过：{len(expr_df)} 个基因，"
        f"A 组({group_a}) {n_a} 样本，B 组({group_b}) {n_b} 样本，缺失值 {na_ratio:.1%}"
    )

    # 6. 读长期记忆：这个研究者上次做过分析，就复用他上次的阈值偏好
    user_id = state.get("user_id", "default")
    fc_threshold, fdr_threshold = 1.0, 0.05
    prefs_from_memory = False
    try:
        item = runtime.store.get((*MEMORY_NS, user_id), PREFS_KEY)
        if item and item.value:
            fc_threshold = float(item.value.get("fc_threshold", 1.0))
            fdr_threshold = float(item.value.get("fdr_threshold", 0.05))
            prefs_from_memory = True
    except Exception:
        pass  # 记忆读不到就用默认阈值，不影响主流程

    return {
        "qc_passed": True,
        "qc_report": qc_report,
        "group_a": group_a,
        "group_b": group_b,
        "fc_threshold": fc_threshold,
        "fdr_threshold": fdr_threshold,
        "prefs_from_memory": prefs_from_memory,
        "expr_file": expr_file,
        "meta_file": meta_file,
        "go_file": go_file,
        "group_col": group_col,
    }


# ══════════ ③ 人工确认（HITL） ══════════
def human_review_node(state: dict) -> dict:
    """质检通过后暂停，把结果给人看，等人确认后才继续分析。"""
    # interrupt() 会让图在这里暂停；返回值 = 外部 resume 回来的值
    decision = interrupt({
        "message": "质检通过，是否继续差异分析？",
        "qc_report": state.get("qc_report", ""),
        "group_a": state.get("group_a"),
        "group_b": state.get("group_b"),
    })
    if isinstance(decision, dict) and decision.get("approved") is False:
        return {"qc_passed": False, "qc_report": "已由用户取消分析"}
    return {}


# ══════════ ④ 差异表达分析 ══════════
def diff_analysis_node(state: dict) -> dict:
    """调 diff.py 做真实统计，把结果整理成摘要文本。"""
#从 state 里取出"表达矩阵的文件路径"，再把那个文件读成表格，存进 expr_df
    expr_df=pd.read_csv(state["expr_file"])#存"表达矩阵"这个表格
    metadata_df=pd.read_csv(state["meta_file"])

    #调用差异分析函数differential_expression,存差异分析的结果
    res = differential_expression(
        expr_df,
        metadata_df,
        state["group_col"],
        group_a=state["group_a"],
        group_b=state["group_b"],
        fc_threshold=state.get("fc_threshold", 1.0),
        fdr_threshold=state.get("fdr_threshold", 0.05),
    )

    sig = res[res["significant"]]
    significant_count = int(len(sig))
    sig_genes_up = sig[sig["log2fc"] > 0]["gene"].tolist()
    sig_genes_down = sig[sig["log2fc"] < 0]["gene"].tolist()

    # 取 top 10 显著基因，做成 markdown 表格
    top = sig.head(10)
    if top.empty:
        diff_summary = "未找到显著差异表达基因。"
    else:
        lines = [
            "| 基因 | log2FC | FDR | 变化方向 |",
            "| --- | --- | --- | --- |",
        ]
        for _, row in top.iterrows():
            direction = "上调" if row["log2fc"] > 0 else "下调"
            lines.append(
                f"| {row['gene']} | {row['log2fc']:.2f} | {row['fdr']:.2e} | {direction} |"
            )
        diff_summary = "\n".join(lines)

    return {
        "diff_summary": diff_summary,
        "significant_count": significant_count,
        "sig_genes_up": sig_genes_up,
        "sig_genes_down": sig_genes_down,
    }


# ══════════ ⑤ 通路富集分析 ══════════
def enrichment_node(state: dict) -> dict:
    """对显著基因做 GO 通路富集，判断它们是不是扎堆在某些通路上。"""
    # 从 state 取 go_file
    ann_file=state.get("go_file")#not ann_file是空（None）就成立

    # os.path.exists(路径) = 这个文件存不存在（返回 True/False）
    if not ann_file or not os.path.exists(ann_file):
        return {"enrichment_summary": "（缺少 GO 注释文件，跳过富集分析）"}
    # ann_file 可能是 None，而 None 不能传给 os.path.exists（会报错），所以先 not ann_file拦一道；

    ann_df=pd.read_csv(ann_file)# 读 GO 注释文件成表格
    expr_df=pd.read_csv(state["expr_file"])#读表达矩阵
    #取表达矩阵第一列（基因列）转成列表，作为"背景集"（所有基因）
    background=expr_df.iloc[:,0].tolist()

    up = state.get("sig_genes_up", [])
    down = state.get("sig_genes_down", [])

    parts = []
    for label, genes in [("上调基因", up), ("下调基因", down)]:
        if not genes:
            continue
        res = enrichment_analysis(genes, ann_df, background)
        if res.empty:
            continue
        sig_terms = res[res["significant"]]
        if sig_terms.empty:
            parts.append(f"**{label}**：未发现显著富集的通路。")
            continue
        lines = [f"**{label}** 显著富集的通路：", ""]
        for _, row in sig_terms.head(5).iterrows():
            lines.append(
                f"- {row['term']}（显著基因 {row['sig_count']}/{row['bg_count']}，"
                f"FDR={row['fdr']:.2e}）"
            )
        parts.append("\n".join(lines))

    summary = "\n\n".join(parts) if parts else "未发现显著富集的通路。"
    return {"enrichment_summary": summary}


# ══════════ ⑤ PCA 主成分分析 ══════════
def pca_analysis_node(state:dict) ->dict:
    """无监督降维：把 2 万基因压成几个主成分，看样本怎么分布。"""
    import numpy as np
    # 和 load_qc 一样先靠内容识别表达矩阵，不依赖别人填好 expr_file
    found = discover_dataset(state.get("dataset_path", DATASET_PATH))
    # 读表达矩阵（行=基因，列=样本）
    expr_df=pd.read_csv(found["expr_file"])

    # iloc[:, 1:] = 跳过第一列基因名，只要数值；values = 转成 numpy 数组
    # astype(float) = 强制变成小数（防止读进来是文本，SVD 会报错）
    mat=expr_df.iloc[:, 1:].values.astype(float)#取数值矩阵，去掉基因名列

    # 转置：PCA 要分析"样本之间的分布"，所以每个样本要变成一行
    X=mat.T#转置成样本×基因

    # ① 中心化：每个基因列减自己的均值，让数据"搬到原点"
    X=X-X.mean(axis=0)

    # ② SVD 分解：拆成 左奇异向量 × 奇异值 × 右奇异向量
    #    full_matrices=False = 只算有用部分，省内存
    U,S,Vt=np.linalg.svd(X,full_matrices=False)
    #数学核心：把一堆样本点拆成"几个主轴 +每个样本在轴上的坐标"

    # ③ 解释方差比例 = 奇异值平方 / 所有奇异值平方之和
    #    它回答"这个主成分保留了原数据百分之多少的信息"
    explained=(S ** 2) / (S ** 2).sum()# 算每个主成分的解释比例

    # ④ 样本在新坐标轴上的坐标 = U × S
    scores = U * S#样本投影坐标

    # ⑤ 拼成摘要文字（前 3 个主成分解释了多少方差）
    #拼摘要
    summary = (
        f"PCA 完成：共 {X.shape[0]} 个样本，"
        f"PC1 解释 {explained[0]:.1%}，PC2 解释 {explained[1]:.1%}，"
        f"PC3 解释 {explained[2]:.1%}"
        f"（前三合计 {explained[:3].sum():.1%}）"
    )
    return {"pca_summary": summary}


# ══════════ 聚类分析（KMeans） ══════════
def cluster_analysis_node(state: dict) -> dict:
    """无监督聚类：把样本自动分成几群（KMeans）。"""
    import numpy as np
    # 和 load_qc 一样先靠内容识别表达矩阵，不依赖别人填好 expr_file
    found = discover_dataset(state.get("dataset_path", DATASET_PATH))
    # 读表达矩阵 → 取数值 → 转置成 样本×基因（和 PCA 一样，每个样本一行）
    expr_df=pd.read_csv(state["expr_file"])
    # iloc[:, 1:] = 跳过第一列基因名，只要数值；values = 转成 numpy 数组
    # astype(float) = 强制变成小数（防止读进来是文本，SVD 会报错）
    X = expr_df.iloc[:, 1:].values.astype(float).T

    # k = 要分几群（先写死 3，以后可做成参数）
    k = 3
    np.random.seed(42)  # 固定随机种子：让每次运行结果一样（可复现）

    # ① 随机挑 k 个样本当"初始中心"（replace=False = 不重复挑）
    centers = X[np.random.choice(len(X), k, replace=False)]

    # ② 反复迭代：分配 → 重算中心，最多 100 轮
    for _ in range(100):
        #算每个样本到 k 个中心的距离平方（开方不影响比较大小，省了）
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)  # 每个样本归到"最近"的中心

        # 重算中心：每一簇所有样本的均值
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # 中心和上一轮几乎一样 → 收敛，停
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    # ③ 统计每簇有几个样本（bincount = 数每个编号出现几次）
    counts = np.bincount(labels, minlength=k)

    # ④ 拼摘要
    summary = f"聚类完成（k={k}）：" + "、".join(
        f"簇 {i} = {counts[i]} 个样本" for i in range(k)
    )
    return {"cluster_summary": summary}








# ══════════ ⑥ LLM 解读 ══════════
def interpret_node(state: dict) -> dict:
    """LLM 把一堆基因数字翻译成人话。（这次带上每个基因的已知功能）"""
    model = get_model()

    # ── 新加：把 top 显著基因的功能从知识库查出来，拼成「基因功能注释」──
    up=state.get("sig_genes_up", [])[:5]       #上调基因列表，只取前 5 个
    down = state.get("sig_genes_down", [])[:5] # 下调基因列表，只取前 5 个
    # state.get(键, 默认值) = 从记事本里取；没有就返回默认值（空列表
    # [:5] = 切片，只留前 5 个，防止显著基因太多把 prompt 撑爆

    func_lines = []       # 空列表，装「基因功能」的每句话
    for label, genes in [("上调基因", up), ("下调基因", down)]:
    # for ... in [...] = 循环两轮：第一轮 label="上调基因"、genes=up；第二轮换下调
        for g in genes:  # 再对里面的每个基因名循环
            desc = get_gene_function(g)  # 调 rag.py 精确查名，拿功能原文
            if desc:  # desc 不是 None（查到了）才写
                func_lines.append(f"- {g}：{desc}")
            # f"- {g}：{desc}" = 拼成一句 "基因名：功能原文" 的 markdown 列表项

    gene_funcs = "\n".join(func_lines) if func_lines else "（未检索到功能信息）"
    # "\n".join(列表) = 用换行符把列表里每句话连成一个多行文本
    # if ... else ... = 列表里有东西就连；空列表就写一句占位的话

    prompt = (
        "你是一名生物信息学专家。下面是差异表达分析、通路富集分析的结果，"
        "以及每个显著基因的已知功能。请用通俗的中文解读："
        "哪些基因显著上/下调、它们各自负责什么生物学功能、"
        "显著富集在哪些通路，这些通路可能提示什么生物学意义。\n\n"
        "【差异表达】\n"
        f"{state['diff_summary']}\n\n"
        f"（共 {state['significant_count']} 个显著差异基因）\n\n"
        "【通路富集】\n"
        f"{state.get('enrichment_summary', '无')}\n\n"
        "【基因功能注释】\n"          # ← 新加的这一节，来自知识库检索
        f"{gene_funcs}"
    )
    res = model.invoke([HumanMessage(prompt)])
    return {"interpretation": res.content}


# ══════════ 7 知识库问答（RAG 语义检索） ══════════
def qa_node(state: dict) -> dict:
    """问答节点：用户问开放问题（如"哪些基因和免疫相关"），
    先拿问题去知识库做语义检索，再让 LLM 依据检索到的基因卡片回答。

    分工：retrieve_genes 负责"找证据"，LLM 负责"照着证据说话"，不瞎编。
    """
    question = state["user_request"]  # 用户问的那句话（从记事本取）
    hits = retrieve_genes(question, top_k=5)  # 语义检索，返回最像的 5 个基因

    if not hits:  # hits 是空列表（一个都没检索到）
        return {"qa_answer": "知识库里没检索到相关基因，换个问法试试。"}

    # 把检索到的 5 张卡片拼成「证据」文本，喂给 LLM
    # doc[:500] = 每张卡片原文只取前 500 字，防止太长把 prompt 撑爆
    # doc 开头本来就带基因名（"VEGFD：参与通路…"），所以不再重复拼 gene，用 _ 占位
    evidence = "\n".join(
        f"- {doc[:500]}（相似度 {score:.2f}）"
        for _, doc, score in hits
    )
    # "\n".join(...) = 用换行把每一行连成一个大文本；for 挨个取出三元组

    model = get_model()
    prompt = (
        "你是一名生物信息学助手。下面是根据用户问题从基因知识库检索到的基因卡片，"
        "请严格依据这些卡片回答用户的问题，不要编造卡片里没有的功能。"
        "如果卡片信息不足以回答，就如实说明。\n\n"
        f"【用户问题】\n{question}\n\n"
        f"【检索到的基因卡片】\n{evidence}"
    )
    res = model.invoke([HumanMessage(prompt)])  # 让 LLM 照着证据回答
    return {"qa_answer": res.content}  # 答案写回记事本




# ══════════ 8 写长期记忆 ══════════
def save_memory_node(state: dict, runtime: Runtime) -> dict:
    """把这次分析用的参数写进长期记忆，下次自动复用。"""
    user_id = state.get("user_id", "default")
    try:
        runtime.store.put(
            (*MEMORY_NS, user_id),
            PREFS_KEY,
            {
                "fc_threshold": state.get("fc_threshold", 1.0),
                "fdr_threshold": state.get("fdr_threshold", 0.05),
                "group_a": state.get("group_a", ""),
                "group_b": state.get("group_b", ""),
            },
        )
    except Exception:
        pass  # 记忆写失败不影响主流程
    return {}


# ══════════ 9 汇总报告 ══════════
def report_node(state: dict) -> dict:
    """把所有结果拼成一份最终报告。要能处理三种情况。"""
    # 情况1：不支持的分析类型（没进质检，直接到这）
    if state.get("analysis_type") == "other":
        return {
            "final_report": "# 分析报告\n\n当前支持：差异表达(differential)、PCA(pca)、聚类(clustering)。"
        }


    # 情况2：PCA
    if state.get("analysis_type") == "pca":
        return {
            "final_report": "# 分析报告\n\n## 主成分分析\n\n" + state.get("pca_summary", "")
        }

    # 情况3：聚类
    if state.get("analysis_type") == "clustering":
        return {
            "final_report": "# 分析报告\n\n## 聚类分析\n\n" + state.get("cluster_summary", "")
        }


    # 情况4：知识库问答
    if state.get("analysis_type") == "qa":
        return {
            "final_report": "# 知识库问答\n\n" + state.get("qa_answer", "")
        }

    # 情况5：质检失败（或人工确认时取消）
    if not state.get("qc_passed", False):
        return {
            "final_report": "# 分析报告\n\n## 质检未通过\n\n" + state.get("qc_report", "")
        }

    # 情况6：正常完整报告
    param_src = "来自长期记忆" if state.get("prefs_from_memory") else "默认"
    lines = [
        "# 多组学数据分析报告",
        "",
        "## 一、质检结果",
        state.get("qc_report", ""),
        "",
        "## 二、差异表达分析",
        f"显著差异基因数：{state.get('significant_count', 0)}",
        f"参数：|log2FC| > {state.get('fc_threshold', 1.0)}，"
        f"FDR < {state.get('fdr_threshold', 0.05)}（{param_src}）",
        "",
        state.get("diff_summary", ""),
        "",
        "## 三、通路富集分析",
        state.get("enrichment_summary", "无显著富集通路"),
        "",
        "## 四、结果解读",
        state.get("interpretation", ""),
    ]
    return {"final_report": "\n".join(lines)}
