"""State：整个流程共享的"记事本"。

每个节点从这里读自己要的信息，算完把结果写回来。
total=False 表示这些字段不是一开始就全有，而是流程走到哪一步填哪一步。
"""

import operator
from typing import Annotated, TypedDict


class AnalysisState(TypedDict, total=False):
    # ── 输入 ──
    user_request: str     # 用户说了什么（比如"帮我找差异表达基因"）
    dataset_path: str     # 数据文件所在的目录

    expr_file:str         #表达矩阵文件路径（discover_dataset 识别出来的）
    meta_file:str         #分组表文件路径
    go_file:str           #GO 注释文件路径（可能为 None）
    group_col:str         #分组列名（识别出来的，不再写死 "condition"）

    user_id: str          # 用户标识（长期记忆按它区分不同研究者）

    # ── ① 需求解析的结果 ──
    analysis_type: str    # 判断出来要做什么分析，如 "differential"

    # ── ② 质检的结果 ──
    qc_passed: bool       # 质检过没过
    qc_report: str        # 质检报告（文字）
    group_a: str          # A 组名字
    group_b: str          # B 组名字
    review_decision: str  # HITL 的结论："approved" / "rejected"。
                          # 单独一个字段是必须的：用户主动取消 ≠ 质检没通过，
                          # 只看 qc_passed 会把"已取消"渲染成"质检未通过"。

    # ── 分析参数（可被长期记忆里的偏好覆盖）──
    fc_threshold: float   # log2FC 阈值，默认 1.0
    fdr_threshold: float  # FDR 阈值，默认 0.05
    prefs_from_memory: bool  # 本次阈值是否来自长期记忆

    # ── ③ 差异分析的结果 ──
    diff_summary: str     # 差异分析摘要（top 基因的文本表格）
    significant_count: int  # 显著差异基因的数量
    sig_genes_up: list    # 上调显著基因名列表
    sig_genes_down: list  # 下调显著基因名列表

    # ── ③.5 Fork 分叉：多组阈值并行对比 ──
    threshold_results: Annotated[list, operator.add]  # 各阈值分支的结果（Send 并行后 reducer 合并）
    compare_summary: str  # 阈值对比摘要（表格）

    # ── ④ 无监督分析的结果 ──
    pca_summary: str    # PCA 分析摘要（主成分解释比例等)
    cluster_summary:str # 聚类分析摘要


    # ── ⑤ 富集分析的结果 ──
    enrichment_summary: str  # 富集分析摘要（哪些通路被显著富集）

    # ── ⑥ 解读与报告 ──
    interpretation: str   # LLM 的解读
    final_report: str     # 最终报告

    # ── ⑦ 知识库问答（RAG 语义检索）──
    qa_answer: str  # 问答节点的答案（检索证据 + LLM 生成）

    # ── ⑧ 降级标记（内容质量受损，不能当正品缓存）──
    degraded: bool        # 是否有节点降级产出（模型不可用 → 只出统计摘要）
    degraded_stages: list # 哪些阶段降级了，如 ["interpret"]