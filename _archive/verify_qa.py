# -*- coding: utf-8 -*-
"""验证知识库问答（RAG 语义检索）：问一个开放问题，打印答案。

用法（在项目根目录运行）：
    python verify_qa.py

看什么：打印出「# 知识库问答」+ 一段提到 FAN1/WRN/TDP1 等 DNA 修复基因的回答，
就说明 retrieve_genes 语义检索成功接进了问答节点。
"""
import os
import sys

# 把项目根目录加进 sys.path，才能 import omics_agent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["USE_POSTGRES"] = "false"   # 内存版，不用起数据库

from omics_agent.graph import build_graph

g = build_graph()

# 问答流程：parse_intent → qa_node → report，没有质检、没有 interrupt，一次跑完
result = g.invoke(
    {"user_request": "哪些基因参与 DNA 损伤修复？", "user_id": "u1"},
    config={"configurable": {"thread_id": "qa-1"}},
)

print("=" * 60)
print("知识库问答 · 最终报告")
print("=" * 60)
print(result.get("final_report"))
