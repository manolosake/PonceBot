#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dis
import pathlib
import sys
import trace
import types
import unittest


# Initial coverage gate scope (phase-1 baseline): transactional state layer.
# We intentionally keep scope narrow before expanding to full bot/orchestrator modules.
TARGETS = [
    pathlib.Path("state_store.py"),
]
DEFAULT_TEST_PATTERNS = ["test_state_store.py"]


def _code_line_numbers(code: types.CodeType) -> set[int]:
    out = {int(ln) for _, ln in dis.findlinestarts(code) if int(ln) > 0}
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            out |= _code_line_numbers(c)
    return out


def _candidate_lines(path: pathlib.Path) -> set[int]:
    src = path.read_text(encoding="utf-8", errors="replace")
    code = compile(src, str(path.resolve()), "exec")
    return _code_line_numbers(code)


def _discover_suite(*, patterns: list[str]) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    for pattern in patterns:
        suite.addTests(unittest.defaultTestLoader.discover(".", pattern=pattern))
    return suite


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal coverage gate for hardening-critical modules")
    ap.add_argument("--min", dest="min_cov", type=float, default=0.70)
    ap.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="test discovery pattern; repeat to include multiple patterns (default: test_state_store.py)",
    )
    args = ap.parse_args()

    patterns = [str(p or "").strip() for p in (args.pattern or []) if str(p or "").strip()]
    if not patterns:
        patterns = list(DEFAULT_TEST_PATTERNS)
    suite = _discover_suite(patterns=patterns)
    tracer = trace.Trace(count=True, trace=False, ignoredirs=[sys.prefix, sys.exec_prefix])
    runner = unittest.TextTestRunner(verbosity=0)
    result = tracer.runfunc(runner.run, suite)
    if not result.wasSuccessful():
        return 1

    counts = tracer.results().counts
    total_exec = 0
    total_all = 0

    for rel in TARGETS:
        if not rel.exists():
            continue
        full = str(rel.resolve())
        cand = _candidate_lines(rel)
        hit = {ln for ln in cand if counts.get((full, ln), 0) > 0}
        total_exec += len(hit)
        total_all += len(cand)

    cov = (float(total_exec) / float(total_all)) if total_all else 1.0
    print(f"[coverage-gate] coverage={cov:.3f} min={args.min_cov:.3f} lines={total_exec}/{total_all}")
    if cov < float(args.min_cov):
        print("[coverage-gate] FAIL")
        return 2
    print("[coverage-gate] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
