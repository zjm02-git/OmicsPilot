"""意图路由评测：跑一遍评测集，算路由准确率，列出答错的题。

为什么做这个：意图解析是流水线的第一个节点，路由错了后面全白算。
这个脚本给「路由准不准」一个可复现的数字，改完代码跑一遍就知道有没有改坏。

两种模式：
  - 关键词模式（默认）：只跑 keyword_intent 纯函数，秒出、不花 token、可离线，
    测的是「兜底逻辑」本身准不准；
  - LLM 模式（--llm）：真调结构化输出，测「真实路由」准不准（慢、花钱）。

用法（项目根目录）：
    python -B evals/run_intent_eval.py
    python -B evals/run_intent_eval.py --llm
    python -B evals/run_intent_eval.py --json last.json
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from omics_agent.resilience import keyword_intent  # noqa: E402

CASES_PATH = os.path.join(ROOT, "evals", "intent_cases.json")


def load_cases(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


def predict_keyword(question: str) -> str:
    """关键词模式：直接跑兜底逻辑，不碰 LLM。"""
    return keyword_intent(question)


def predict_llm(question: str) -> str:
    """LLM 模式：真调意图解析节点。"""
    from omics_agent.nodes import parse_intent_node

    return parse_intent_node({"user_request": question})["analysis_type"]


def run(cases: list[dict], predictor) -> list[dict]:
    rows = []
    for c in cases:
        actual = predictor(c["question"])
        rows.append({**c, "actual": actual, "ok": actual == c["expected"]})
    return rows


def report(rows: list[dict], mode: str) -> None:
    n = len(rows)
    ok = sum(1 for r in rows if r["ok"])
    print(f"模式：{mode}")
    print(f"准确率：{ok}/{n} = {ok / n:.1%}\n")

    # 按类型统计：哪个类型最容易路由错，一眼看清盲区在哪
    by_type = {}
    for r in rows:
        t = r["expected"]
        by_type.setdefault(t, {"total": 0, "ok": 0})
        by_type[t]["total"] += 1
        by_type[t]["ok"] += int(r["ok"])

    print("按类型：")
    for t in ("differential", "pca", "clustering", "qa", "other"):
        if t in by_type:
            info = by_type[t]
            print(f"  {t:<12} {info['ok']}/{info['total']}")

    wrong = [r for r in rows if not r["ok"]]
    if wrong:
        print("\n答错的题：")
        for r in wrong:
            print(f"  [{r['id']}] 「{r['question']}」")
            print(f"        期望={r['expected']}  实际={r['actual']}")
    else:
        print("\n全部答对 ✅")


def main() -> int:
    use_llm = "--llm" in sys.argv
    cases = load_cases(CASES_PATH)
    predictor = predict_llm if use_llm else predict_keyword
    mode = "LLM（结构化输出）" if use_llm else "关键词（keyword_intent）"
    rows = run(cases, predictor)
    report(rows, mode)

    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mode": mode,
                    "total": len(rows),
                    "correct": sum(1 for r in rows if r["ok"]),
                    "accuracy": round(sum(1 for r in rows if r["ok"]) / len(rows), 4),
                    "cases": rows,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n结果已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
