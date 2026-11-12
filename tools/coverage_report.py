"""行覆盖率测量（不依赖 coverage.py / pytest）。

原理：`compile()` 后遍历代码对象的 `co_lines()` 得到"有字节码的行"作分母，
用 `sys.settrace` + `threading.settrace` 记录真实执行到的行作分子——分母不含空行
与注释行，因此不会靠"稀释"得到虚高的数字。

用法（项目根目录）：
    python -B tools/coverage_report.py
    python -B tools/coverage_report.py --json cov.json
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TARGETS = [
    os.path.join(ROOT, "omics_agent", "cache.py"),
    os.path.join(ROOT, "omics_agent", "context.py"),
    os.path.join(ROOT, "omics_agent", "dataio.py"),
    os.path.join(ROOT, "omics_agent", "jobs.py"),
    os.path.join(ROOT, "omics_agent", "metrics.py"),
    os.path.join(ROOT, "omics_agent", "resilience.py"),
    os.path.join(ROOT, "web.py"),
]

_hits: "dict[str, set[int]]" = {}
_watch = {os.path.normcase(os.path.abspath(p)) for p in TARGETS}


def _tracer(frame, event, arg):
    if event == "line":
        fname = os.path.normcase(frame.f_code.co_filename)
        if fname in _watch:
            _hits.setdefault(fname, set()).add(frame.f_lineno)
    return _tracer


def _executable_lines(path: str) -> set[int]:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    code = compile(src, path, "exec")
    lines: set[int] = set()

    def walk(co: types.CodeType) -> None:
        for _start, _end, lineno in co.co_lines():
            if lineno:
                lines.add(lineno)
        for const in co.co_consts:
            if isinstance(const, types.CodeType):
                walk(const)

    walk(code)
    return lines


def main() -> int:
    sys.settrace(_tracer)
    threading.settrace(_tracer)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.join(ROOT, "tests"), top_level_dir=ROOT, pattern="test_*.py")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)

    sys.settrace(None)
    threading.settrace(None)

    rows = []
    total_exec = total_hit = 0
    for path in TARGETS:
        if not os.path.exists(path):
            continue
        exec_lines = _executable_lines(path)
        covered = exec_lines & _hits.get(os.path.normcase(os.path.abspath(path)), set())
        pct = 100.0 * len(covered) / len(exec_lines) if exec_lines else 0.0
        total_exec += len(exec_lines)
        total_hit += len(covered)
        rows.append((os.path.basename(path), len(exec_lines), len(covered), round(pct, 1)))

    print(f"{'file':<16}{'exec':>6}{'hit':>6}{'cov%':>8}")
    print("-" * 36)
    for name, e, h, pct in rows:
        print(f"{name:<16}{e:>6}{h:>6}{pct:>8}")
    print("-" * 36)
    overall = 100.0 * total_hit / total_exec if total_exec else 0.0
    print(f"{'TOTAL':<16}{total_exec:>6}{total_hit:>6}{round(overall, 1):>8}")
    print(f"\ntests: run={result.testsRun} failures={len(result.failures)} "
          f"errors={len(result.errors)} skipped={len(result.skipped)}")
    if not result.wasSuccessful():
        print(stream.getvalue()[-3000:])

    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "files": [{"file": n, "exec_lines": e, "covered": h, "coverage_pct": p}
                          for n, e, h, p in rows],
                "total_exec_lines": total_exec,
                "total_covered": total_hit,
                "coverage_pct": round(overall, 1),
                "tests_run": result.testsRun,
            }, f, ensure_ascii=False, indent=2)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
