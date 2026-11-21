"""全流程 Tracing：跑通图的所有路径，抓每条路径的节点序列 + 耗时。

为什么自研而不接 LangSmith：
  - LangSmith 要花钱、数据要传到外部 SaaS；
  - 自研只需几十行，面试还能说「我自己用 callback 做了轻量 trace」。

原理：
  图每执行一个节点，会触发 on_chain_start / on_chain_end 两个回调（带同一个 run_id），
  用 run_id 把 start/end 配对，一减就是节点耗时——这就是「监听」而非「侵入」，
  不改任何节点代码。

  （注：不用 astream_events，因为默认 InMemorySaver 没实现异步接口，会抛
   NotImplementedError；同步 invoke + callback 走同步接口，最稳。）

覆盖全部 5 条路径（含 1 条带人工确认 interrupt 的长路径）：
  other        parse_intent → report
  pca          parse_intent → pca_analysis → report
  clustering   parse_intent → cluster_analysis → report
  qa           parse_intent → qa_node → report
  differential parse_intent → load_qc → human_review(批准) → diff_analysis
               → enrichment → interpret → save_memory → report

关键点：differential 路径里 human_review 调用了 interrupt()，第一次 invoke 会停在
那里等人确认；用 Command(resume=...) 恢复。human_review 在 trace 里耗时≈0——它本身
只是「挂起」，真正的等待发生在两次 invoke 之间，不归入节点执行时间。

用法（项目根目录）：
    python -B omics_agent/trace.py              # 跑全部 5 条路径
    python -B omics_agent/trace.py --case pca   # 只跑某一条
    python -B omics_agent/trace.py --qa         # 等价 --case qa（向后兼容）
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langgraph.types import Command  # noqa: E402

from omics_agent.graph import build_graph  # noqa: E402


class _TraceHandler(BaseCallbackHandler):
    """监听节点开始/结束，按 run_id 配对算耗时。

    节点名在 metadata["langgraph_node"] 里（serialized 是 None）。
    一个节点会触发两层事件：顶层（tags 含 graph:step:N）和内部子链
    （tags 含 seq:step:N，如节点里调 LLM）。只记顶层，才算的是「节点」本身耗时。
    """

    def __init__(self):
        self.spans: dict[str, dict] = {}   # run_id -> {"name", "start"}
        self.events: list[dict] = []       # 按完成顺序排的节点记录
        self.seen_nodes: set[str] = set()  # 见过的所有 langgraph_node（调试用）

    def on_chain_start(self, serialized, inputs, *, run_id, tags=None, metadata=None, **kwargs):
        node = (metadata or {}).get("langgraph_node")
        if not node:
            return
        self.seen_nodes.add(node)
        is_top = tags and any(t.startswith("graph:step") for t in tags)
        if is_top:
            self.spans[run_id] = {"name": node, "start": time.perf_counter()}

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        if run_id in self.spans:
            span = self.spans.pop(run_id)
            self.events.append({
                "node": span["name"],
                "elapsed_s": round(time.perf_counter() - span["start"], 3),
            })


def run_path(graph, initial: dict, config: dict, *, resume_value=None):
    """跑一条路径（同步），必要时在 interrupt 处 resume，返回节点记录。

    interrupt（human_review）时第一次 invoke 会停在半路：graph.get_state(config).next
    非空（= 还有节点没跑完）。用 Command(resume=resume_value) 恢复，直到 next 为空。
    """
    handler = _TraceHandler()
    cb = {"callbacks": [handler]}

    graph.invoke(initial, config={**config, **cb})
    while graph.get_state(config).next:
        graph.invoke(Command(resume=resume_value), config={**config, **cb})
    return handler.events, handler.seen_nodes


def print_trace(events: list[dict], seen_nodes: set[str] | None = None) -> None:
    if not events:
        print("（没抓到节点事件）")
        if seen_nodes:
            print("callback 实际出现的 langgraph_node：", sorted(seen_nodes))
        return
    total = sum(e["elapsed_s"] for e in events)
    print(f"{'#':<3}{'节点':<18}{'耗时(s)':>10}")
    print("-" * 36)
    for i, e in enumerate(events, 1):
        # 柱状图：每 0.05s 一格，直观看出哪个节点占大头
        bar = "█" * max(1, int(e["elapsed_s"] * 20))
        print(f"{i:<3}{e['node']:<18}{e['elapsed_s']:>10.3f}  {bar}")
    print("-" * 36)
    print(f"{'合计':<18}{total:>10.3f}")


# 全流程 5 条路径：(名字, 输入, resume 值)
# differential 的 resume 传 {"approved": True} = 批准继续；传 False = 取消（另一条出口）。
PATHS = (
    ("other", "今天天气怎么样", None),
    ("pca", "做个 PCA 分析", None),
    ("clustering", "把这些样本做聚类", None),
    ("qa", "BRCA1 是什么基因，有什么功能", None),
    ("differential", "找两组样本的差异表达基因", {"approved": True}),
)


def main() -> int:
    graph = build_graph()

    # 向后兼容：--qa = --case qa
    if "--qa" in sys.argv and "--case" not in sys.argv:
        sys.argv.append("--case")
        sys.argv.append("qa")

    if "--case" in sys.argv:
        name = sys.argv[sys.argv.index("--case") + 1]
        chosen = [p for p in PATHS if p[0] == name]
        if not chosen:
            print(f"未知路径：{name}（可选 {', '.join(p[0] for p in PATHS)}）")
            return 2
        paths = chosen
    else:
        paths = PATHS

    for name, request, resume in paths:
        config = {"configurable": {"thread_id": f"trace-{name}"}}
        initial = {"user_request": request, "dataset_path": "data", "user_id": "trace-demo"}
        print(f"\n===== 路径 {name}：「{request}」 =====")
        events, seen = run_path(graph, initial, config, resume_value=resume)
        print_trace(events, seen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
