"""8 个节点函数：每个节点只干一件事，从 State 读、往 State 写。

分工原则：
  - LLM 干"懂"的活（①理解需求、⑥解读结果）
  - Python 干"算"的活（②质检、④差异统计、⑤通路富集、⑧拼报告）
  - 人工介入：③人工确认；记忆：②读偏好、⑦写偏好

本轮加固（都在"调用边界"上，不动图拓扑）：
  - 意图解析：超时 + 重试 + 自纠错，全部失败退回关键词路由 —— 不让流水线死在第一个节点；
  - 结果解读 / 问答：证据按优先级塞进 token 预算（低优先级的块整块丢弃并显式说明），
    LLM 调用带超时/重试/熔断；模型彻底不可用时降级为"统计摘要 + 原始证据"，任务仍然出报告；
  - 数据读取：统一走 dataio.read_csv_cached，一轮分析里同一个大 CSV 只解析一次（实测 0.38s → 0s）。
"""

import logging
import os
import re
from typing import Literal

import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from omics_agent import metrics
from omics_agent.config import get_model
from omics_agent.context import drop_note, fit_blocks
from omics_agent.dataio import read_csv_cached
from omics_agent.diff import differential_expression
from omics_agent.enrichment import enrichment_analysis
from omics_agent.rag import get_gene_function, retrieve_hybrid
# 一次 import 两个：get_gene_function 精确查名、retrieve_genes 语义检索
from omics_agent.resilience import (
    CircuitBreaker,
    keyword_intent,
    llm_call,
    structured_output_call,
)

LOGGER = logging.getLogger(__name__)

# 默认数据目录、分组列名
DATASET_PATH = "data"
GROUP_COL = "condition"

# 长期记忆的命名空间和 key（记忆按"研究者"区分，存各自的分析偏好）
MEMORY_NS = ("omics",)
PREFS_KEY = "preferences"

# 送进 LLM 的证据预算（估算 token）。超了就按优先级丢低优先级的块，
# 而不是让模型的上下文窗口在结尾悄悄截断——截掉的恰好是最关键的解读证据。
INTERPRET_TOKEN_BUDGET = int(os.getenv("INTERPRET_TOKEN_BUDGET", "6000"))
QA_TOKEN_BUDGET = int(os.getenv("QA_TOKEN_BUDGET", "3000"))

# 所有 LLM 调用共用一把熔断器：网关持续不可用时快速失败，不再每个请求都等满超时
LLM_BREAKER = CircuitBreaker("llm", failure_threshold=5, reset_timeout=30.0)


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
    #files = [...] 这一段是"列表推导式"：把目录里所有 csv 文件的完整路径收集成列表
    files=[
        os.path.join(dataset_path,f)
        for f in os.listdir(dataset_path)
        if f.lower().endswith(".csv")  #防止写成 CSV
    ]
    if not files:
        raise ValueError("目录里没有 csv文件")

    #========第一步  认表达矩阵=======
    expr_file=None
    for path in files:
        head=read_csv_cached(path,nrows=50)#只读前 50行；按文件签名缓存，重复调用不重复解析
        #.shape = (行数, 列数)，取第 2 个元素 = 列数
        if head.shape[1] > 20:
            #iloc[:,0]=第一列；iloc[:,1:]=从第2列到末尾
            if head.iloc[:, 0].nunique()==len(head) and _is_numeric(head.iloc[:,1:]):
                expr_file=path
                break
    if expr_file is None:
        raise ValueError("没识别出表达矩阵")

    #=====第二步 拿样本数=======
    n_samples=read_csv_cached(expr_file,nrows=0).shape[1]-1#只读表头，不读数据行（最快）

    #=====第三步  用行数找分组表 + 认 GO 注释表=====
    meta_file=go_file=None
    for path in files:
        if path ==expr_file:
            continue
        # 只解析第一列来数行数：比整表解析省内存、省时间（37MB 的矩阵不再重复进内存）
        n_rows = len(read_csv_cached(path, usecols=[0]))
        if n_rows==n_samples:#行数恰好等于样本数（226）→ 这是分组表
            meta_file=path
            continue
        # GO 注释表的特征：有 gene + term 两列。基因卡片(gene,description)等其它表
        # 行数也≠样本数，会被误认成 go_file，导致 enrichment 里 groupby("term") 报
        # KeyError。这里按列名精确认，不靠行数猜。
        cols = read_csv_cached(path, nrows=0).columns.tolist()
        if "gene" in cols and "term" in cols:
            go_file=path

    #=====第 5 块：检查 + 认分组列 + 返回=====
    if meta_file is None:
        raise ValueError(f"没识别出分组表（行数应等于样本数 {n_samples}）")

    meta=read_csv_cached(meta_file)#完整读分组表
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
    """结构化输出：判断用户想做什么分析。

    失败梯度（三级）：结构化输出 → 追加"上次不合格"提示再试 → 关键词规则兜底。
    后两级是为了保证"模型抽风"不会让整条流水线在第一个节点就 500。
    """
    request = state.get("user_request", "")

    def _fallback(exc: Exception) -> IntentOutput:
        analysis_type = keyword_intent(request)
        metrics.incr(f"intent_fallback_{analysis_type}_total")
        LOGGER.warning("意图解析降级为关键词规则（%s）：%s", type(exc).__name__, exc)
        return IntentOutput(analysis_type=analysis_type)

    # get_model() 必须单独包在 try 里：它一旦抛异常（没配 key / 网关地址错 / SDK 版本不符），
    # 异常发生在 structured_output_call 的实参求值阶段，重试、自纠错、fallback 全都还没轮到，
    # 结果就是整条流水线在第一个节点直接 500。这里先把它隔离掉。
    try:
        model = get_model()
    except Exception as exc:  # noqa: BLE001 - 模型造不出来也必须能路由
        metrics.incr("intent_model_unavailable_total")
        res = _fallback(exc)
        metrics.incr(f"intent_{res.analysis_type}_total")
        return {"analysis_type": res.analysis_type}

    res = structured_output_call(
        model,
        IntentOutput,
        [HumanMessage(request)],
        timeout_s=float(os.getenv("INTENT_TIMEOUT_S", "45")),
        attempts=2,
        breaker=LLM_BREAKER,
        fallback=_fallback,
    )
    metrics.incr(f"intent_{res.analysis_type}_total")
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

    # 1. 文件在不在
    try:
        expr_df = read_csv_cached(expr_file)
        metadata_df = read_csv_cached(meta_file)
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

    # 4. 样本对齐：差异分析要靠 sample_id 把分组表接到表达矩阵的列上。
    #    这一列缺了、或者样本名对不上，必须在质检阶段就拦下 —— 否则会一路走到
    #    人工确认之后才在 diff_analysis 里炸，用户看到的是"会话可能已失效"这种
    #    与真实原因无关的报错，排查成本极高。
    if "sample_id" not in metadata_df.columns:
        return {
            "qc_passed": False,
            "qc_report": "分组表缺少 sample_id 列（它负责把样本与表达矩阵的列对齐）",
        }
    expr_samples = {str(c) for c in expr_df.columns[1:]}
    missing = [str(s) for s in metadata_df["sample_id"] if str(s) not in expr_samples]
    if missing:
        return {
            "qc_passed": False,
            "qc_report": (
                f"分组表有 {len(missing)} 个样本在表达矩阵里找不到（例如 {', '.join(missing[:3])}），"
                "请确认两张表用的是同一套样本名"
            ),
        }

    # 5. 每组样本数不能太少（t 检验至少要有几个样本才有意义）
    n_a = int((metadata_df[group_col] == group_a).sum())
    n_b = int((metadata_df[group_col] == group_b).sum())
    if n_a < 20 or n_b < 20:
        return {"qc_passed": False, "qc_report": f"每组至少 20 个样本，实际 A={n_a} / B={n_b}"}

    # 6. 缺失值不能太多
    na_ratio = float(expr_df.isna().mean().mean())
    if na_ratio > 0.1:
        return {"qc_passed": False, "qc_report": f"缺失值过多：{na_ratio:.1%}"}

    qc_report = (
        f"质检通过：{len(expr_df)} 个基因，"
        f"A 组({group_a}) {n_a} 样本，B 组({group_b}) {n_b} 样本，缺失值 {na_ratio:.1%}"
    )

    # 7. 读长期记忆：这个研究者上次做过分析，就复用他上次的阈值偏好
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
        metrics.incr("memory_read_error_total")
        # 记忆读不到就用默认阈值，不影响主流程

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
#从 state 里取出"表达矩阵的文件路径"，再把那个文件读成表格（走缓存，只解析一次）
    expr_df=read_csv_cached(state["expr_file"])#存"表达矩阵"这个表格
    metadata_df=read_csv_cached(state["meta_file"])

    # 阈值从 state 取：Fork 分叉时每个 Send 分支带不同阈值，各自算各自的结果
    fc_threshold = state.get("fc_threshold", 1.0)
    fdr_threshold = state.get("fdr_threshold", 0.05)

    #调用差异分析函数differential_expression,存差异分析的结果
    with metrics.timer("diff.compute_ms"):
        res = differential_expression(
            expr_df,
            metadata_df,
            state["group_col"],
            group_a=state["group_a"],
            group_b=state["group_b"],
            fc_threshold=fc_threshold,
            fdr_threshold=fdr_threshold,
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

    # 返回一条「阈值结果记录」，交给 state 的 reducer(operator.add) 合并进列表。
    # 不再直接写 diff_summary 等单值字段——并行分支会互相覆盖，改由 compare 节点挑主结果写回。
    return {
        "threshold_results": [{
            "fc_threshold": fc_threshold,
            "fdr_threshold": fdr_threshold,
            "significant_count": significant_count,
            "sig_genes_up": sig_genes_up,
            "sig_genes_down": sig_genes_down,
            "diff_summary": diff_summary,
        }],
    }


# ══════════ ④.5 Fork 汇合：阈值对比 ══════════
def compare_thresholds_node(state: dict) -> dict:
    """Fork 汇合点：对比各阈值分支的结果，挑出主结果写回单值字段。

    主结果 = fc_threshold 与「fork 前用户阈值」一致的那条（默认/记忆阈值那条）。
    严格阈值那条只用于对比展示，不改变后续富集/解读用的主结果。
    """
    results = state.get("threshold_results", [])
    if not results:
        return {"compare_summary": "（无阈值对比结果）"}

    # 对比表：一行一组阈值
    lines = ["| 阈值组合 | 显著基因数 | 上调 | 下调 |", "| --- | --- | --- | --- |"]
    for r in results:
        lines.append(
            f"| |log2FC|>{r['fc_threshold']}, FDR<{r['fdr_threshold']} "
            f"| {r['significant_count']} | {len(r['sig_genes_up'])} | {len(r['sig_genes_down'])} |"
        )
    compare_summary = "\n".join(lines)

    # 挑主结果：阈值等于 fork 前用户阈值的那条
    main_fc = state.get("fc_threshold", 1.0)
    main = next((r for r in results if r["fc_threshold"] == main_fc), results[0])

    return {
        "compare_summary": compare_summary,
        "diff_summary": main["diff_summary"],
        "significant_count": main["significant_count"],
        "sig_genes_up": main["sig_genes_up"],
        "sig_genes_down": main["sig_genes_down"],
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

    ann_df=read_csv_cached(ann_file)# 读 GO 注释文件成表格（缓存，和识别阶段共用一次解析）
    expr_df=read_csv_cached(state["expr_file"])#读表达矩阵
    #取表达矩阵第一列（基因列）转成列表，作为"背景集"（所有基因）
    background=expr_df.iloc[:,0].tolist()

    up = state.get("sig_genes_up", [])
    down = state.get("sig_genes_down", [])

    parts = []
    with metrics.timer("enrichment.compute_ms"):
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
    expr_df=read_csv_cached(found["expr_file"])

    # iloc[:, 1:] = 跳过第一列基因名，只要数值；values = 转成 numpy 数组
    # astype(float) = 强制变成小数（防止读进来是文本，SVD 会报错）
    mat=expr_df.iloc[:, 1:].values.astype(float)#取数值矩阵，去掉基因名列

    # 转置：PCA 要分析"样本之间的分布"，所以每个样本要变成一行
    X=mat.T#转置成样本×基因

    # ① 中心化：每个基因列减自己的均值，让数据"搬到原点"
    X=X-X.mean(axis=0)

    # ② SVD 分解：拆成 左奇异向量 × 奇异值 × 右奇异向量
    #    full_matrices=False = 只算有用部分，省内存
    with metrics.timer("pca.compute_ms"):
        U,S,Vt=np.linalg.svd(X,full_matrices=False)
    #数学核心：把一堆样本点拆成"几个主轴 +每个样本在轴上的坐标"

    # ③ 解释方差比例 = 奇异值平方 / 所有奇异值平方之和
    explained=(S ** 2) / (S ** 2).sum()# 算每个主成分的解释比例

    # ④ 样本在新坐标轴上的坐标 = U × S
    scores = U * S#样本投影坐标

    # ⑤ 拼成摘要文字（前 3 个主成分解释了多少方差）
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
    expr_df=read_csv_cached(found["expr_file"])
    X = expr_df.iloc[:, 1:].values.astype(float).T

    # k = 要分几群（先写死 3，以后可做成参数）
    k = 3
    np.random.seed(42)  # 固定随机种子：让每次运行结果一样（可复现）

    # ① 随机挑 k 个样本当"初始中心"（replace=False = 不重复挑）
    centers = X[np.random.choice(len(X), k, replace=False)]

    # ② 反复迭代：分配 → 重算中心，最多 100 轮
    with metrics.timer("cluster.compute_ms"):
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
    """LLM 把一堆基因数字翻译成人话。（这次带上每个基因的已知功能）

    两道防线：
      1. 证据按优先级进 token 预算，被裁掉的块在 prompt 里明说（模型不该假装看全了）；
      2. 模型不可用（超时/限流/熔断）时降级为"统计摘要"，报告照样出得去。
    """
    up=state.get("sig_genes_up", [])[:5]       #上调基因列表，只取前 5 个
    down = state.get("sig_genes_down", [])[:5] # 下调基因列表，只取前 5 个
    # [:5] = 只留前 5 个，防止显著基因太多把 prompt 撑爆

    func_lines = []       # 空列表，装「基因功能」的每句话
    for label, genes in [("上调基因", up), ("下调基因", down)]:
        for g in genes:  # 再对里面的每个基因名循环
            try:
                desc = get_gene_function(g)  # 调 rag.py 精确查名，拿功能原文
            except Exception:            # 知识库读失败不影响主流程：少了注释，统计还在
                metrics.incr("gene_lookup_error_total")
                desc = None
            if desc:  # 查到了才写
                func_lines.append(f"- {g}：{desc}")

    gene_funcs = "\n".join(func_lines) if func_lines else "（未检索到功能信息）"

    # 按优先级塞进预算：差异表最不能丢，基因功能最先被裁
    texts, info = fit_blocks(
        [
            {"name": "差异表达", "text": state.get("diff_summary", ""), "priority": 3},
            {"name": "通路富集", "text": state.get("enrichment_summary", "无"), "priority": 2},
            {"name": "基因功能注释", "text": gene_funcs, "priority": 1},
        ],
        INTERPRET_TOKEN_BUDGET,
    )
    metrics.incr("interpret_context_used_tokens", info["used_tokens"])
    if info["dropped"] or info["truncated"]:
        metrics.incr("interpret_context_trimmed_total")
    jobs_note = drop_note(info)

    prompt = (
        jobs_note
        + "你是一名生物信息学专家。下面是差异表达分析、通路富集分析的结果，"
        "以及每个显著基因的已知功能。请用通俗的中文解读："
        "哪些基因显著上/下调、它们各自负责什么生物学功能、"
        "显著富集在哪些通路，这些通路可能提示什么生物学意义。\n\n"
        "【差异表达】\n"
        f"{texts['差异表达']}\n\n"
        f"（共 {state.get('significant_count', 0)} 个显著差异基因）\n\n"
        "【通路富集】\n"
        f"{texts['通路富集']}\n\n"
        "【基因功能注释】\n"
        f"{texts['基因功能注释']}"
    )

    try:
        with metrics.timer("llm.interpret_ms"):
            res = llm_call(
                get_model(),
                [HumanMessage(prompt)],
                timeout_s=float(os.getenv("INTERPRET_TIMEOUT_S", "120")),
                attempts=3,
                breaker=LLM_BREAKER,
            )
        return {"interpretation": res.content}
    except Exception as exc:  # noqa: BLE001 - 模型彻底不可用也要有报告
        metrics.incr("interpret_degraded_total")
        degraded = (
            f"（模型解读暂不可用：{type(exc).__name__}。以下为统计结果原文）\n\n"
            f"显著差异基因 {state.get('significant_count', 0)} 个，"
            f"上调 {len(state.get('sig_genes_up', []))} 个、"
            f"下调 {len(state.get('sig_genes_down', []))} 个。\n\n"
            f"{texts['差异表达']}\n\n{texts['通路富集']}"
        )
        return {"interpretation": degraded, "degraded": True, "degraded_stages": ["interpret"]}


# ══════════ 7 知识库问答（RAG 语义检索） ══════════
_GENE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,14}")


def _exact_lookup(question: str) -> list[tuple[str, str, float]]:
    """向量检索不可用时的兜底：从问题里抽出疑似基因名，逐个精确查卡片。"""
    out: list[tuple[str, str, float]] = []
    for token in _GENE_TOKEN_RE.findall(question or "")[:12]:
        try:
            desc = get_gene_function(token.upper())
        except Exception:  # noqa: BLE001
            desc = None
        if desc:
            out.append((token.upper(), f"{token.upper()}：{desc}", 0.0))
    return out


def qa_node(state: dict) -> dict:
    """问答节点：先拿问题去知识库做语义检索，再让 LLM 依据检索到的卡片回答。

    分工：retrieve_genes 负责"找证据"，LLM 负责"照着证据说话"，不瞎编。
    两级降级：向量检索失败 → 抽基因名做精确查；LLM 失败 → 直接回原始证据。
    """
    question = state["user_request"]
    hits: list = []
    try:
        with metrics.timer("rag.retrieve_ms"):
            hits = retrieve_hybrid(question, top_k=5)  # 混合检索（向量+BM25+RRF），返回最相关的 5 个基因
        metrics.incr("qa_retrieval_total")
    except Exception as exc:  # noqa: BLE001 - 向量库/embedding 服务挂了不能让问答直接 500
        metrics.incr("qa_retrieval_error_total")
        hits = _exact_lookup(question)
        if not hits:
            return {"qa_answer": f"知识库检索暂不可用（{type(exc).__name__}），请稍后重试。",
                    "degraded": True, "degraded_stages": ["qa_retrieve"]}
        metrics.incr("qa_fallback_exact_total")

    if not hits:  # 一个都没检索到
        return {"qa_answer": "知识库里没检索到相关基因，换个问法试试。"}

    # 每张卡片按相似度给优先级：预算不够时先丢相关性最低的卡片
    blocks = []
    for idx, (_, doc, score) in enumerate(hits):
        blocks.append({
            "name": f"卡片{idx + 1}",
            "text": f"- {doc[:800]}（相关度 {score:.2f}）",
            "priority": 10.0 - float(idx),  # 保持检索顺序，前面的更相关
        })
    texts, info = fit_blocks(blocks, QA_TOKEN_BUDGET)
    evidence = "\n".join(t for t in texts.values() if t)
    if info["dropped"]:
        metrics.incr("qa_evidence_trimmed_total")

    prompt = (
        drop_note(info)
        + "你是一名生物信息学助手。下面是根据用户问题从基因知识库检索到的基因卡片，"
        "请严格依据这些卡片回答用户的问题，不要编造卡片里没有的功能。"
        "如果卡片信息不足以回答，就如实说明。\n\n"
        f"【用户问题】\n{question}\n\n"
        f"【检索到的基因卡片】\n{evidence}"
    )

    try:
        with metrics.timer("llm.qa_ms"):
            res = llm_call(
                get_model(),
                [HumanMessage(prompt)],
                timeout_s=float(os.getenv("QA_TIMEOUT_S", "90")),
                attempts=3,
                breaker=LLM_BREAKER,
            )
        return {"qa_answer": res.content}  # 答案写回记事本
    except Exception as exc:  # noqa: BLE001 - 模型挂了就把证据直接交给人看
        metrics.incr("qa_degraded_total")
        return {
            "qa_answer": (
                f"（模型暂不可用：{type(exc).__name__}。以下是从知识库检索到的原始证据）\n\n"
                f"{evidence}"
            ),
            "degraded": True,
            "degraded_stages": ["qa_llm"],
        }


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
        metrics.incr("memory_write_error_total")
        # 记忆写失败不影响主流程
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
        "### 阈值对比（Fork 并行试了两组阈值）",
        state.get("compare_summary", "（无对比）"),
        "",
        "## 三、通路富集分析",
        state.get("enrichment_summary", "无显著富集通路"),
        "",
        "## 四、结果解读",
        state.get("interpretation", ""),
    ]
    return {"final_report": "\n".join(lines)}
