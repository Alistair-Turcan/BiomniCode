#!/usr/bin/env python3
"""Mechanical implementation of the biomni-code acceptance rule.

Given the current-best and candidate benchmark statistics, decide accept/reject and
the acceptance branch, so that Phase 5 decisions are reproducible rather than vibes.

Runtime branch (primary) -- accept when ALL hold:
  1. (best_runtime_mean - new_runtime_mean) > 2 * best_runtime_std
  2. relative runtime improvement > --min-rel-improvement (default 3%)
  3. peak RSS does not regress by more than --mem-regression-cap (default 10%) of the
     current best beyond 2 * baseline_memory_std:
         new_mem_mean - best_mem_mean <= cap * best_mem_mean + 2 * baseline_mem_std
     (skipped with a warning if memory stats are unavailable)

Memory branch (secondary) -- accept as MEMORY-DRIVEN when ALL hold:
  1. (best_mem_mean - new_mem_mean) > 2 * best_mem_std
  2. relative peak-RSS reduction >= --mem-branch-reduction (default 25%)
  3. runtime regression <= --runtime-regression-cap (default 5%) of best beyond
     2 * best_runtime_std:
         new_runtime_mean - best_runtime_mean <= cap * best_runtime_mean + 2 * best_runtime_std

Exit codes: 0 = accept, 3 = reject, 2 = usage error. Always prints a JSON verdict.
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--best-runtime-mean", type=float, required=True)
    p.add_argument("--best-runtime-std", type=float, required=True)
    p.add_argument("--new-runtime-mean", type=float, required=True)
    p.add_argument("--new-runtime-std", type=float, default=None)
    p.add_argument("--best-mem-mean", type=float, default=None)
    p.add_argument("--best-mem-std", type=float, default=None)
    p.add_argument("--new-mem-mean", type=float, default=None)
    p.add_argument("--new-mem-std", type=float, default=None)
    p.add_argument("--baseline-mem-std", type=float, default=None)
    p.add_argument("--min-rel-improvement", type=float, default=0.03)
    p.add_argument("--mem-regression-cap", type=float, default=0.10)
    p.add_argument("--mem-branch-reduction", type=float, default=0.25)
    p.add_argument("--runtime-regression-cap", type=float, default=0.05)
    args = p.parse_args()

    reasons = []
    warnings = []

    # ---- Runtime branch -------------------------------------------------
    rt_delta = args.best_runtime_mean - args.new_runtime_mean
    rt_rel = rt_delta / args.best_runtime_mean if args.best_runtime_mean else 0.0
    rt_sig = rt_delta > 2 * args.best_runtime_std
    rt_meaningful = rt_rel > args.min_rel_improvement

    mem_guard_ok = True
    if (args.best_mem_mean is not None and args.new_mem_mean is not None
            and args.baseline_mem_std is not None):
        mem_regression = args.new_mem_mean - args.best_mem_mean
        mem_allowed = args.mem_regression_cap * args.best_mem_mean + 2 * args.baseline_mem_std
        mem_guard_ok = mem_regression <= mem_allowed
        if not mem_guard_ok:
            reasons.append(f"peak-RSS regression {mem_regression:.4g} exceeds allowance {mem_allowed:.4g}")
    else:
        warnings.append("memory stats incomplete; runtime-branch memory guardrail skipped")

    runtime_accept = rt_sig and rt_meaningful and mem_guard_ok
    if not rt_sig:
        reasons.append(f"runtime delta {rt_delta:.4g}s not > 2*best_std ({2 * args.best_runtime_std:.4g}s)")
    if not rt_meaningful:
        reasons.append(f"relative runtime improvement {rt_rel:.2%} not > {args.min_rel_improvement:.0%}")

    # ---- Memory branch --------------------------------------------------
    memory_accept = False
    if (args.best_mem_mean is not None and args.best_mem_std is not None
            and args.new_mem_mean is not None):
        mem_delta = args.best_mem_mean - args.new_mem_mean
        mem_rel = mem_delta / args.best_mem_mean if args.best_mem_mean else 0.0
        mem_sig = mem_delta > 2 * args.best_mem_std
        mem_big = mem_rel >= args.mem_branch_reduction
        rt_regression = args.new_runtime_mean - args.best_runtime_mean
        rt_allowed = args.runtime_regression_cap * args.best_runtime_mean + 2 * args.best_runtime_std
        rt_guard_ok = rt_regression <= rt_allowed
        memory_accept = mem_sig and mem_big and rt_guard_ok
        if not mem_sig:
            reasons.append(f"memory delta {mem_delta:.4g} not > 2*best_mem_std ({2 * args.best_mem_std:.4g})")
        if not mem_big:
            reasons.append(f"relative peak-RSS reduction {mem_rel:.2%} below {args.mem_branch_reduction:.0%}")
        if not rt_guard_ok:
            reasons.append(f"runtime regression {rt_regression:.4g}s exceeds allowance {rt_allowed:.4g}s")
    else:
        reasons.append("memory branch unavailable: incomplete memory statistics")

    if runtime_accept:
        verdict = {"decision": "accept", "branch": "runtime", "memory_driven": False,
                   "runtime_rel_improvement": rt_rel, "reasons": ["runtime branch satisfied"], "warnings": warnings}
    elif memory_accept:
        verdict = {"decision": "accept", "branch": "memory", "memory_driven": True,
                   "runtime_rel_improvement": rt_rel, "reasons": ["memory branch satisfied (MEMORY-DRIVEN)"],
                   "warnings": warnings}
    else:
        verdict = {"decision": "reject", "branch": "none", "memory_driven": False,
                   "runtime_rel_improvement": rt_rel, "reasons": reasons, "warnings": warnings}

    print(json.dumps(verdict, indent=2, sort_keys=True))
    sys.exit(0 if verdict["decision"] == "accept" else 3)


if __name__ == "__main__":
    main()
