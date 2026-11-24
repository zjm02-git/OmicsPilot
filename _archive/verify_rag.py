# -*- coding: utf-8 -*-
"""验证 RAG：跑一遍完整分析流程，打印最终报告的「结果解读」部分。

用法（在项目根目录运行）：
    python verify_rag.py

看什么：报告「四、结果解读」里有没有出现「某基因：参与通路 …」这种
知识库原文（说明 RAG 检索成功接进去了），而不是空泛的"该基因参与重要通路"。
"""
import os
import sys

# 把项目根目录加进 sys.path，这样下面才能 import omics_agent 包
# os.path.abspath(__file__) = 本文件的绝对路径；dirname 一层 = 项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["USE_POSTGRES"] = "false"   # 用内存版，不用起数据库

from langgraph.types import Command      # Command = 用来「续跑」被中断的图
from omics_agent.graph import build_graph

g = build_graph()                        # 造图（会自动加载改好的 interpret_node）
config = {"configurable": {"thread_id": "verify-rag"}}

# 第一次 invoke：跑图，会在 human_review 节点暂停（interrupt）
result = g.invoke(
    {"user_request": "帮我找两组样本的差异表达基因，并解读",
     "dataset_path": "data",
     "user_id": "researcher-1"},
    config=config,
)

# 只要还挂着中断，就自动「批准」继续，直到整张图跑完
while result.get("__interrupt__"):
    result = g.invoke(Command(resume={"approved": True}), config=config)

# 从最终报告里截出「四、结果解读」这一段打印（重点看基因功能有没有进来）
report = result.get("final_report", "")
idx = report.find("四、结果解读")        # find = 找这段标题的位置，找不到返回 -1
print("=" * 60)
print("最终报告 · 结果解读（截取）")
print("=" * 60)
print(report[idx:] if idx != -1 else report)
