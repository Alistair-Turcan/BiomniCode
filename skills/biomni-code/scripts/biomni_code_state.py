#!/usr/bin/env python3
"""State helper for the biomni-code skill (code-repository optimization).

This script is the executable backbone of the biomni-code optimization loop. It does
not optimize code by itself; it maintains durable, append-only ledgers and a
machine-checkable termination gate so that a long optimization run is auditable,
resumable, and cannot be stopped early on a subjective judgment.

Adapted from tusoskill's biomni_method_state.py and re-specialized for code
optimization: the primary metric is wall-clock runtime (lower is better), the
secondary metric is peak RSS (lower is better), and every substantive iteration is
a scoped, correctness-gated patch candidate.

Key contracts (see references/state_and_memory.md):
- Iteration counter = number of substantive records in iterations/results.jsonl
  (decision in {accept, reject, memory-driven}, not flagged non-substantive).
    It is derived from the committed ledger, including at the stop gate.
- One candidate is admitted with begin-iteration; review, disposition and a
  verified checkpoint precede the next admission. Same-ID retries are idempotent.
- stop-check: 0 => budget exhausted and eligible for Phase 6, 3 => continue or
  recover, 4 => sourced stop/block (not completion), 2 => invalid/missing state.
- After mutations changed history files are mirrored to
  $BIOMNICODE_RESULTS_ROOT/<run_id>/history/ (default /mnt/results/biomnicode/...)
  using complete object writes. Checkpoint/full-sync commands check all history.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List
import run_control as control

# ---------------------------------------------------------------------------
    # Default adaptive strategy weights over code-optimization intervention families.
# Floors keep plausible under-tested families alive; see references/strategy_library/.
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "algorithmic": 0.16,
    "vectorization": 0.14,
    "data_structures": 0.12,
    "allocation_memory_layout": 0.11,
    "caching_memoization": 0.10,
    "io_batching": 0.09,
    "parallelism_concurrency": 0.08,
    "serialization": 0.06,
    "compiled_kernels": 0.06,
    "startup_imports": 0.04,
    "build_flags": 0.04,
}
DEFAULT_FLOORS = {
    "algorithmic": 0.05,
    "vectorization": 0.05,
    "data_structures": 0.04,
    "allocation_memory_layout": 0.04,
    "caching_memoization": 0.04,
    "io_batching": 0.03,
    "parallelism_concurrency": 0.03,
    "serialization": 0.02,
    "compiled_kernels": 0.02,
    "startup_imports": 0.02,
    "build_flags": 0.02,
}

# Decisions that represent a real, informative iteration. Anything else
# (infrastructure retries) or an explicit --non-substantive flag does not advance
# the iteration counter.
SUBSTANTIVE_DECISIONS = {"accept", "reject", "memory-driven"}

# Closed list of hard blockers -- the ONLY reasons Phase 5 may end while the
# iteration floor is unreached.
HARD_BLOCKER_KINDS = {"infra", "harness", "user", "repo"}

# Files that together constitute the complete iteration history. Mirrored to
# /mnt/results incrementally; checkpoint/full-sync checks every history file.
HISTORY_FILES = [
    "state/state.json",
    "state/strategy_weights.json",
    "state/diversity_portfolio.json",
    "state/backlog.jsonl",
    "iterations/results.jsonl",
    "iterations/leaderboard.csv",
    "feedback/feedback.jsonl",
    "profiling/hotspots.jsonl",
    "reports/status.json",
    "reports/status.md",
    "bench/phase6_results.csv",
]
HISTORY_GLOBS = ["state/checkpoint_*.json", "logs/*.jsonl"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    control.write(path, value)


def append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    # Logical append, atomic local replacement under the coordinator transaction.
    previous = path.read_text() if path.exists() else ""
    control.atomic_bytes(path, (previous + json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(weights: Dict[str, float], floors: Dict[str, float] | None = None) -> Dict[str, float]:
    floors = floors or {}
    keys = list(weights)
    values = {k: max(float(weights.get(k, 0.0)), float(floors.get(k, 0.0))) for k in keys}
    total = sum(values.values())
    if total <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: v / total for k, v in values.items()}


def state_paths(root: Path) -> Dict[str, Path]:
    return {
        "state": root / "state" / "state.json",
        "weights": root / "state" / "strategy_weights.json",
        "portfolio": root / "state" / "diversity_portfolio.json",
        "backlog": root / "state" / "backlog.jsonl",
        "experiments": root / "iterations" / "results.jsonl",
        "leaderboard": root / "iterations" / "leaderboard.csv",
        "feedback": root / "feedback" / "feedback.jsonl",
        "hotspots": root / "profiling" / "hotspots.jsonl",
        "status": root / "reports" / "status.json",
        "status_md": root / "reports" / "status.md",
    }


def is_substantive(record: Dict[str, Any]) -> bool:
    """A record counts toward the iteration floor iff it has a substantive decision
    and is not explicitly flagged non-substantive / failed-infra."""
    if record.get("non_substantive"):
        return False
    return str(record.get("decision")) in SUBSTANTIVE_DECISIONS


def count_substantive(experiments: List[Dict[str, Any]]) -> int:
    return sum(1 for r in experiments if is_substantive(r))


def phase5_elapsed_hours(state: Dict[str, Any]) -> float | None:
    ts = state.get("phase5_start_ts")
    if not ts:
        return None
    return (time.time() - float(ts)) / 3600.0


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> None:
    root = Path(args.root)
    if args.budget_iterations <= 0:
        raise ValueError("Require a positive authorized substantive-iteration budget")
    for d in ["state", "iterations", "feedback", "profiling", "bench", "patches",
              "candidates", "diagnostics", "reports", "logs", "scaling_suite",
              "correctness_suite", "original_snapshot", "work", "env"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    paths = state_paths(root)
    run_id = args.run_id or f"bc-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    state = read_json(paths["state"], {})
    if not state:
        state = {
            "policy_version": "2.5.3",
            "active_candidate": None,
            "best_iteration_id": None,
            "run_id": run_id,
            "created_at": now(),
            "updated_at": now(),
            "task": "code optimization",
            "repo_url": args.repo_url,
            "commit_hash": args.commit_hash,
            "budget_iterations": args.budget_iterations,
            "phase5_start_ts": None,
            "iteration": 0,
            "fp_tolerance": args.fp_tolerance,
            "fp_tolerance_request": args.fp_tolerance,
            "numerical_policy": None,
            "numerical_policy_sha256": None,
            "workload_class": args.workload_class,
            "machine_id": args.machine_id,
            "build_cmd": args.build_cmd,
            "correctness_cmd": args.correctness_cmd,
            "benchmark_cmd": args.benchmark_cmd,
            "baseline": {
                "runtime_mean": None, "runtime_std": None,
                "mem_mean": None, "mem_std": None,
                "output_digest": "", "determinism": "",
            },
            "current_best": {
                "patch": "", "runtime_mean": None, "runtime_std": None,
                "mem_mean": None, "mem_std": None,
            },
            "benchmark_sigma": None,
            "cumulative_speedup": 1.0,
            "cumulative_mem_delta": 0.0,
            "blocker": None,
        }
        write_json(paths["state"], state)
    elif state.get("repo_url") != args.repo_url or state.get("commit_hash") != args.commit_hash:
        raise ValueError("Existing run identity mismatch; never overwrite or restart it")
    weights = normalize(DEFAULT_WEIGHTS, DEFAULT_FLOORS)
    if not paths["weights"].exists():
        write_json(paths["weights"], {"updated_at": now(), "weights": weights, "floors": DEFAULT_FLOORS})
    if not paths["portfolio"].exists():
        write_json(paths["portfolio"], {"updated_at": now(), "families": []})
    write_status(root)
    print(json.dumps({"root": str(root), "run_id": state["run_id"], "state": str(paths["state"])}))


def cmd_set_baseline(args: argparse.Namespace) -> None:
    """Record Phase 1 baseline metrics and the per-iteration benchmark sigma."""
    root = Path(args.root)
    paths = state_paths(root)
    state = read_json(paths["state"], {})
    if not state:
        sys.exit("no state.json found; run `init` first")
    if control.records(root) or state.get("active_candidate") or state.get("numerical_policy"):
        raise ValueError("Cannot reset baseline after numerical policy freeze or candidate admission")
    from run_checkpoint import freeze_source_policy, source_inventory
    freeze_source_policy(root, state)
    state["best_tree"] = source_inventory(root, state=state)
    if not state["best_tree"]:
        raise ValueError("Baseline source tree is missing")
    state["original_tree"] = source_inventory(root, "original_snapshot", state=state)
    if state["original_tree"] != state["best_tree"]:
        raise ValueError("Baseline working source must match the protected original")
    state["baseline"] = {
        "runtime_mean": args.runtime_mean,
        "runtime_std": args.runtime_std,
        "mem_mean": args.mem_mean,
        "mem_std": args.mem_std,
        "output_digest": args.output_digest,
        "determinism": args.determinism,
    }
    # The baseline is also the initial current best (unmodified code).
    state["current_best"] = {
        "patch": "",
        "runtime_mean": args.runtime_mean,
        "runtime_std": args.runtime_std,
        "mem_mean": args.mem_mean,
        "mem_std": args.mem_std,
    }
    if args.runtime_std is not None:
        state["benchmark_sigma"] = args.runtime_std
    state["updated_at"] = now()
    write_json(paths["state"], state)
    write_status(root)
    print(json.dumps({"baseline_recorded": True, "benchmark_sigma": state["benchmark_sigma"]}))


def cmd_start_phase5(args: argparse.Namespace) -> None:
    """Persist the Phase 5 start timestamp (informational elapsed-time marker;
    nothing gates on it). Idempotent: never resets."""
    root = Path(args.root)
    paths = state_paths(root)
    state = read_json(paths["state"], {})
    if not state:
        sys.exit("no state.json found; run `init` first")
    from numerical_policy import check
    check(root, state)
    if not state.get("phase5_start_ts"):
        state["phase5_start_ts"] = time.time()
        state["phase5_start_iso"] = now()
        state["updated_at"] = now()
        write_json(paths["state"], state)
    print(json.dumps({"phase5_start_ts": state["phase5_start_ts"], "phase5_start_iso": state.get("phase5_start_iso")}))


def runtime_is_better(new: float | None, best: float | None) -> bool:
    if new is None or (isinstance(new, float) and math.isnan(new)):
        return False
    if best is None:
        return True
    return new < best


def update_weights(root: Path, family: str, decision: str) -> None:
    paths = state_paths(root)
    payload = read_json(paths["weights"], {"weights": DEFAULT_WEIGHTS, "floors": DEFAULT_FLOORS})
    weights = dict(payload.get("weights", DEFAULT_WEIGHTS))
    floors = dict(payload.get("floors", DEFAULT_FLOORS))
    if family not in weights:
        weights[family] = 0.03
        floors[family] = 0.01
    if decision == "accept":
        weights[family] *= 1.25
    elif decision == "memory-driven":
        weights[family] *= 1.15
    elif decision == "reject":
        weights[family] *= 0.85
    # infrastructure decisions do not penalize optimization strategy
    payload["weights"] = normalize(weights, floors)
    payload["floors"] = floors
    payload["updated_at"] = now()
    write_json(paths["weights"], payload)


def update_portfolio(root: Path, record: Dict[str, Any]) -> None:
    paths = state_paths(root)
    portfolio = read_json(paths["portfolio"], {"families": []})
    families = portfolio.setdefault("families", [])
    family_name = record.get("family") or "unspecified"
    entry = None
    for fam in families:
        if fam.get("name") == family_name:
            entry = fam
            break
    if entry is None:
        entry = {"name": family_name, "created_at": now(), "attempts": 0, "accepts": 0,
                 "best_iteration_id": None, "best_runtime_mean": None, "notes": []}
        families.append(entry)
    entry["attempts"] = int(entry.get("attempts", 0)) + 1
    if record.get("decision") in {"accept", "memory-driven"}:
        entry["accepts"] = int(entry.get("accepts", 0)) + 1
    rt = record.get("runtime_mean")
    if runtime_is_better(rt, entry.get("best_runtime_mean")):
        entry["best_runtime_mean"] = rt
        entry["best_iteration_id"] = record.get("iteration_id")
    note = record.get("summary") or record.get("hypothesis") or ""
    if note:
        entry.setdefault("notes", []).append({"at": now(), "iteration_id": record.get("iteration_id"),
                                              "decision": record.get("decision"), "note": note[:240]})
        entry["notes"] = entry["notes"][-8:]
    portfolio["updated_at"] = now()
    write_json(paths["portfolio"], portfolio)


def write_leaderboard(root: Path) -> None:
    paths = state_paths(root)
    rows = read_jsonl(paths["experiments"])
    scored = [r for r in rows if r.get("runtime_mean") is not None and r.get("decision") in SUBSTANTIVE_DECISIONS]
    scored.sort(key=lambda r: r.get("runtime_mean"))
    paths["leaderboard"].parent.mkdir(parents=True, exist_ok=True)
    with io.StringIO(newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "iteration_id", "parent_id", "iteration", "family",
                                               "intervention", "hotspot", "decision", "branch",
                                               "runtime_mean", "runtime_std", "mem_mean", "mem_std", "summary"])
        writer.writeheader()
        for i, r in enumerate(scored, 1):
            writer.writerow({
                "rank": i,
                "iteration_id": r.get("iteration_id"),
                "parent_id": r.get("parent_id"),
                "iteration": r.get("iteration"),
                "family": r.get("family"),
                "intervention": r.get("intervention"),
                "hotspot": r.get("hotspot"),
                "decision": r.get("decision"),
                "branch": r.get("branch"),
                "runtime_mean": r.get("runtime_mean"),
                "runtime_std": r.get("runtime_std"),
                "mem_mean": r.get("mem_mean"),
                "mem_std": r.get("mem_std"),
                "summary": r.get("summary", "")[:300],
            })
        control.atomic_bytes(paths["leaderboard"], f.getvalue().encode())


def backlog_items(root: Path) -> List[Dict[str, Any]]:
    """Current idea revisions; admission is separate and completion comes from the ledger."""
    paths = state_paths(root)
    items = {}
    for index, row in enumerate(read_jsonl(paths["backlog"])):
        # Preserve readable 2.4 backlog history without rewriting old rows.
        ident = row.get("backlog_id") or f"legacy-{index}-{control.canonical(row)[:12]}"
        proposal = {**row, "backlog_id": ident}
        items[ident] = {**proposal, "proposal_sha256": control.canonical(proposal)}
    for row in read_jsonl(paths["experiments"]):
        ident = row.get("backlog_id")
        if ident in items and is_substantive(row):
            items[ident].update(status="completed", completed_by=row["iteration_id"])
    active = read_json(paths["state"], {}).get("active_candidate") or {}
    if active.get("backlog_id") in items:
        items[active["backlog_id"]]["status"] = "active"
    return list(items.values())


def backlog_depth_by_family(root: Path) -> Dict[str, int]:
    backlog = backlog_items(root)
    depth: Dict[str, int] = {}
    for item in backlog:
        if item.get("status", "open") != "open":
            continue
        fam = item.get("family") or "unspecified"
        depth[fam] = depth.get(fam, 0) + 1
    return depth


def write_status(root: Path) -> None:
    paths = state_paths(root)
    state = read_json(paths["state"], {})
    experiments = read_jsonl(paths["experiments"])
    feedback = read_jsonl(paths["feedback"])
    elapsed = phase5_elapsed_hours(state)
    last = experiments[-1] if experiments else {}
    status = {
        "updated_at": now(),
        "run_id": state.get("run_id"),
        "repo_url": state.get("repo_url"),
        "commit_hash": state.get("commit_hash"),
        "iteration": count_substantive(experiments),
        "budget_iterations": state.get("budget_iterations"),
        "phase5_elapsed_hours": (round(elapsed, 4) if elapsed is not None else None),
        "baseline": state.get("baseline"),
        "current_best": state.get("current_best"),
        "cumulative_speedup": state.get("cumulative_speedup", 1.0),
        "cumulative_mem_delta": state.get("cumulative_mem_delta", 0.0),
        "benchmark_sigma": state.get("benchmark_sigma"),
        "iterations_recorded": len(experiments),
        "feedback_items": len(feedback),
        "backlog_depth_by_family": backlog_depth_by_family(root),
        "last_decision": last.get("decision"),
        "last_iteration_id": last.get("iteration_id"),
        "blocker": state.get("blocker"),
        "active_candidate": state.get("active_candidate"),
        "recovery": control.control(root).get("recovery"),
        "jobs": control.control(root).get("jobs", {}),
        "retry_counts": {k: len(v["attempts"]) for k, v in control.control(root).get("retries", {}).items()},
        "verified_checkpoint": control.read(root / "state/verified_checkpoint.json"),
        "leaderboard": str(paths["leaderboard"]),
    }
    write_json(paths["status"], status)
    lines = [
        f"# biomni-code run status",
        f"",
        f"- Run ID: {status['run_id']}",
        f"- Repo: {status['repo_url']} @ {status['commit_hash']}",
        f"- Iteration: {status['iteration']} / {status['budget_iterations']}"
        + (f" | Phase 5 elapsed: {status['phase5_elapsed_hours']}h (informational)"
           if status["phase5_elapsed_hours"] is not None else " | Phase 5 not started"),
        f"- Cumulative speedup vs baseline: {status['cumulative_speedup']}",
        f"- Cumulative peak-RSS delta vs baseline: {status['cumulative_mem_delta']}",
        f"- Benchmark sigma: {status['benchmark_sigma']}",
        f"- Last decision: {status['last_decision']} ({status['last_iteration_id']})",
        f"- Open idea backlog (reassess before admission): {json.dumps(status['backlog_depth_by_family'], sort_keys=True)}",
        f"- Blocker: {json.dumps(status['blocker'])}" if status.get("blocker") else "- Blocker: none",
        f"- Active candidate: {json.dumps(status['active_candidate'])}",
        f"- Recovery: {json.dumps(status['recovery'])}",
        f"- Retry counts: {json.dumps(status['retry_counts'])}",
        f"- Updated: {status['updated_at']}",
    ]
    paths["status_md"].parent.mkdir(parents=True, exist_ok=True)
    control.atomic_bytes(paths["status_md"], ("\n".join(lines) + "\n").encode())


def cmd_add_iteration(args: argparse.Namespace) -> None:
    root = Path(args.root)
    paths = state_paths(root)
    state = read_json(paths["state"], {})
    if not state:
        sys.exit("no state.json found; run `init` first")
    # The request identity excludes callables and paths to the review container,
    # but includes its complete content and every result argument.
    request = {k: v for k, v in vars(args).items() if k not in {"func", "root", "evidence_json"}}
    request["evidence"] = control.read(args.evidence_json)
    identity = control.canonical(request)
    existing = [r for r in control.records(root) if r["iteration_id"] == args.iteration_id]
    if existing:
        if existing[0].get("request_sha256") != identity:
            raise ValueError("Conflicting completion replay")
        print(json.dumps({"replayed": True, "recorded": args.iteration_id,
                          "iteration": control.count(root)}))
        return
    admission = dict(state.get("active_candidate") or {})
    non_substantive = bool(getattr(args, "non_substantive", False)) or args.decision not in SUBSTANTIVE_DECISIONS
    family = args.family
    if family == "unspecified" and args.intervention in DEFAULT_WEIGHTS:
        family = args.intervention  # Preserve older --intervention FAMILY calls.
    if not non_substantive and family == "unspecified":
        raise ValueError("Set --family for a substantive result; --intervention names the technique")
    review = control.review(root, args, state)
    if not non_substantive and not args.metric_analysis:
        sys.exit(
            "every substantive iteration requires --metric-analysis: a brief explanation of WHY the "
            "runtime/memory numbers came out as they did. This is the iteration-informs-iteration hook."
        )
    record = {
        "request_sha256": identity,
        "review": review,
        "feedback_id": admission["feedback_id"],
        "feedback_sha256": admission["feedback_sha256"],
        "numerical_policy_sha256": admission["numerical_policy_sha256"],
        "source_policy_sha256": admission["source_policy_sha256"],
        "learning": admission["learning"],
        **{k: admission[k] for k in ("backlog_id", "backlog_sha256") if k in admission},
        "timestamp": now(),
        # `iteration` is assigned from the ledger below; --iteration is advisory only.
        "iteration": control.count(root) + (0 if non_substantive else 1),
        "iteration_id": args.iteration_id,
        "parent_id": args.parent_id,
        "family": family,
        "intervention": args.intervention,
        "hotspot": args.hotspot,
        "hypothesis": admission["hypothesis"],
        "edited_files": args.edited_files or [],
        "patch_file": args.patch_file,
        "correctness": args.correctness,
        "runtime_mean": args.runtime_mean,
        "runtime_std": args.runtime_std,
        "runtime_reps": args.runtime_reps,
        "mem_mean": args.mem_mean,
        "mem_std": args.mem_std,
        "mem_reps": args.mem_reps,
        "output_digest": args.output_digest,
        "decision": args.decision,
        "branch": args.branch,
        "non_substantive": non_substantive,
        "summary": args.summary,
        "metric_analysis": args.metric_analysis,
        "next_action": args.next_action,
    }
    append_jsonl(paths["experiments"], record)
    # Counter := number of substantive records in the ledger (1 substantive iteration = +1).
    experiments = read_jsonl(paths["experiments"])
    counted = count_substantive(experiments)
    record["iteration"] = counted if is_substantive(record) else state.get("iteration", 0)
    state["iteration"] = counted

    # On an accepted change, the cumulative best becomes this patched state.
    if args.decision in {"accept", "memory-driven"}:
        state["current_best"] = {
            "patch": args.patch_file or state.get("current_best", {}).get("patch", ""),
            "runtime_mean": args.runtime_mean,
            "runtime_std": args.runtime_std,
            "mem_mean": args.mem_mean,
            "mem_std": args.mem_std,
        }
        base_rt = (state.get("baseline") or {}).get("runtime_mean")
        base_mem = (state.get("baseline") or {}).get("mem_mean")
        if base_rt and args.runtime_mean:
            state["cumulative_speedup"] = round(base_rt / args.runtime_mean, 6)
        if base_mem and args.mem_mean:
            state["cumulative_mem_delta"] = round((base_mem - args.mem_mean) / base_mem, 6)

    state["updated_at"] = now()
    write_json(paths["state"], state)
    if family != "unspecified":
        update_weights(root, family, args.decision)
    update_portfolio(root, record)
    if args.feedback:
        append_jsonl(paths["feedback"], {"timestamp": now(), "type": "iteration",
                                         "iteration_id": args.iteration_id, "component": args.hotspot,
                                         "evidence": args.feedback, "recommendation": args.next_action})
    # One-line iteration log (the per-iteration heartbeat).
    base_rt = (state.get("baseline") or {}).get("runtime_mean")
    base_mem = (state.get("baseline") or {}).get("mem_mean")
    d_rt = f"{100.0 * (base_rt - args.runtime_mean) / base_rt:+.2f}%" if (base_rt and args.runtime_mean) else "n/a"
    d_mem = f"{100.0 * (base_mem - args.mem_mean) / base_mem:+.2f}%" if (base_mem and args.mem_mean) else "n/a"
    log_line = (f"Iter {record['iteration']} | {family}/{args.intervention} | {args.hotspot} | "
                f"{args.decision} | druntime={d_rt} | dmem={d_mem} | "
                f"{'MEMORY-DRIVEN | ' if args.decision == 'memory-driven' else ''}{args.summary[:160]}")
    append_jsonl(root / "logs" / "iteration_log.jsonl", {"timestamp": now(), "line": log_line})
    write_leaderboard(root)
    write_status(root)
    remaining = max(0, int(state.get("budget_iterations", 0)) - counted)
    print(json.dumps({"recorded": args.iteration_id, "iteration": counted,
                      "substantive": is_substantive(record), "remaining_to_floor": remaining,
                      "decision": args.decision,
                      "cumulative_speedup": state.get("cumulative_speedup"),
                      "cumulative_mem_delta": state.get("cumulative_mem_delta"),
                      "log_line": log_line}))


def cmd_add_backlog(args: argparse.Namespace) -> None:
    root = Path(args.root)
    if read_json(state_paths(root)["state"], {}).get("policy_version") != "2.5.3":
        raise ValueError("Run requires explicit preserved-history policy migration")
    if not math.isfinite(args.priority) or not args.hypothesis.strip():
        raise ValueError("Require a finite priority and a concrete idea")
    ident = args.backlog_id or f"idea-{uuid.uuid4().hex[:12]}"
    previous = next((item for item in backlog_items(root) if item["backlog_id"] == ident), None)
    if previous and previous.get("status") in {"active", "completed"}:
        raise ValueError("Selected/reviewed idea is fixed; use a new ID for a distinct follow-up")
    item = {
        "timestamp": now(), "backlog_id": ident, "priority": args.priority,
        "family": args.family, "intervention": args.intervention,
        "hotspot": args.hotspot, "hypothesis": args.hypothesis,
        "test": args.test, "source": args.source, "status": args.status,
    }
    append_jsonl(state_paths(root)["backlog"], item)
    write_status(root)
    print(json.dumps({"added": item}))


def cmd_begin_iteration(args: argparse.Namespace) -> None:
    root = Path(args.root)
    selected = None
    if args.backlog_id:
        selected = next((item for item in backlog_items(root)
                         if item["backlog_id"] == args.backlog_id), None)
        if not selected or selected.get("status", "open") != "open":
            raise ValueError("Select an open backlog idea; reviewed ideas need a distinct follow-up")
        if args.hypothesis != selected["hypothesis"]:
            raise ValueError("Revise the queued idea before admitting its reassessed hypothesis")
    # Existing current-best, feedback, learning, single-candidate, job, numerical,
    # checkpoint, history and protected-source gates still govern every admission.
    admission = control.begin(root, args)
    if selected:
        admission.update(backlog_id=selected["backlog_id"],
                         backlog_sha256=selected["proposal_sha256"])
        state = read_json(state_paths(root)["state"], {})
        state["active_candidate"] = admission
        write_json(state_paths(root)["state"], state)
    print(json.dumps(admission))


def cmd_add_feedback(args: argparse.Namespace) -> None:
    root = Path(args.root)
    item = {
        "timestamp": now(),
        "type": args.type,
        "component": args.component,
        "evidence": args.evidence,
        "recommendation": args.recommendation,
        "similar_ideas": args.similar_ideas,
        "avoid": args.avoid,
    }
    append_jsonl(state_paths(root)["feedback"], item)
    write_status(root)
    print(json.dumps({"feedback_recorded": True}))


def cmd_add_hotspots(args: argparse.Namespace) -> None:
    """Record a ranked hotspot table from a profiling round (one JSON object per line)."""
    root = Path(args.root)
    entries = []
    for raw in args.entries or []:
        try:
            entries.append(json.loads(raw))
        except Exception:
            entries.append({"raw": raw})
    item = {"timestamp": now(), "profiler": args.profiler, "workload": args.workload, "entries": entries}
    append_jsonl(state_paths(root)["hotspots"], item)
    print(json.dumps({"hotspot_table_recorded": True, "entries": len(entries)}))


def cmd_suggest_next(args: argparse.Namespace) -> None:
    root = Path(args.root)
    paths = state_paths(root)
    state = read_json(paths["state"], {})
    run_control = control.control(root)
    if state.get("policy_version") != "2.5.3":
        raise ValueError("Run requires explicit preserved-history policy migration")
    weights = read_json(paths["weights"], {"weights": DEFAULT_WEIGHTS}).get("weights", DEFAULT_WEIGHTS)
    backlog = [b for b in backlog_items(root) if b.get("status", "open") == "open"]
    portfolio = read_json(paths["portfolio"], {"families": []})
    experiments = read_jsonl(paths["experiments"])
    hotspots = read_jsonl(paths["hotspots"])
    sorted_weights = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    backlog_sorted = sorted(backlog, key=lambda r: float(r.get("priority", 0)), reverse=True)

    # Diversity-floor advisory: if the dominant family accounts for >= 70% of the last
    # 10 substantive iterations, the next iteration should come from a different family
    # or inspect additional profiling evidence before selecting one hypothesis.
    recent = [r for r in experiments if is_substantive(r)][-10:]
    dominant_family, dominant_frac = None, 0.0
    if recent:
        counts: Dict[str, int] = {}
        for r in recent:
            counts[r.get("family") or "unspecified"] = counts.get(r.get("family") or "unspecified", 0) + 1
        dominant_family = max(counts, key=counts.get)
        dominant_frac = counts[dominant_family] / len(recent)
    diversity_advisory = None
    if dominant_frac >= 0.7 and len(recent) >= 5:
        diversity_advisory = (
            f"diversity floor violated: {dominant_family} accounts for {dominant_frac:.0%} of the last "
            f"{len(recent)} substantive iterations. Next iteration MUST come from a different family, "
            "or inspect new profiling / benchmark-calibration evidence before selecting one hypothesis."
        )
    suggestion = {
        "may_inspect_ideas": True,
        "may_begin": False,
        "parent_id": state.get("best_iteration_id"),
        "active_candidate": state.get("active_candidate"),
        "top_families": sorted_weights[:5],
        "top_backlog": backlog_sorted[:5],
        "portfolio_families": portfolio.get("families", [])[:12],
        "recent_iterations": experiments[-5:],
        "latest_hotspot_table": (hotspots[-1] if hotspots else None),
        "latest_reviewed_result": (experiments[-1] if experiments else None),
        "diversity_advisory": diversity_advisory,
        "advice": ("These are ranked ideas, not an execution queue. Reassess the selected idea "
                   "against the current best and latest reviewed result; explain that learning "
                   "and revise its hypothesis if needed before admitting one candidate. Retain "
                   "diverse alternatives and replenish fewer than three open ideas from evidence. "
                   "Finish and review the sole active candidate before implementing another. "
                   "Never run multiple iterations in one command, script, notebook cell, loop, "
                   "or background job. Repetitions for the same candidate remain allowed."),
    }
    if not control.approved(root) or run_control.get("recovery") or state.get("blocker"):
        suggestion["may_inspect_ideas"] = False
        suggestion["next_action"] = "honor_stop_or_reconcile_recovery"
    elif state.get("active_candidate") or control.unfinished_jobs(root):
        suggestion["next_action"] = "finish_or_reconcile_current_candidate"
    elif control.count(root) >= state["budget_iterations"]:
        suggestion["next_action"] = "complete_final_validation"
    else:
        try:
            from numerical_policy import check
            check(root, state)
            control.checkpoint_current(root)
            control.audit_history(root)
            control.protected(root)
            from run_checkpoint import source_inventory
            if source_inventory(root, state=state) != state.get("best_tree"):
                raise ValueError("Working tree differs from current best before admission")
            suggestion.update(control.feedback_context(root))
            suggestion.update(may_begin=True,
                              next_action="reassess_ideas_against_latest_evidence_then_admit_one")
        except ValueError as e:
            suggestion.update(next_action="verify_current_revision", reason=str(e))
    print(json.dumps(suggestion, indent=2, sort_keys=True))


def cmd_checkpoint(args: argparse.Namespace) -> None:
    """Save immutable generation; a separate run_checkpoint verify establishes recovery."""
    root = Path(args.root)
    from run_checkpoint import save
    write_status(root)
    print(json.dumps(save(root, Path(args.store)), indent=2))


def cmd_record_blocker(args: argparse.Namespace) -> None:
    """Record evidenced unavoidable blockage, never successful completion."""
    root = Path(args.root)
    paths = state_paths(root)
    state = read_json(paths["state"], {})
    observed = control.read(args.evidence_json)
    control.evidence(root, observed)
    reason = "user_stop" if args.kind == "user" else observed.get("reason")
    if reason not in {"user_stop", "platform_unavailable", "authorization_block"}:
        raise ValueError("Routine repo/harness/infra failures require recovery, not early completion")
    control.yield_run(root, {**observed, "reason": reason})
    blocker = {
        **observed,
        "kind": args.kind,          # infra | harness | user | repo
        "detail": args.detail,      # exact error / instruction
        "recorded_at": now(),
        "iteration_at_block": int(state.get("iteration", 0)),
    }
    state["blocker"] = blocker
    state["updated_at"] = now()
    write_json(paths["state"], state)
    write_status(root)
    print(json.dumps({"blocker_recorded": blocker}))


def cmd_clear_blocker(args: argparse.Namespace) -> None:
    root = Path(args.root)
    paths = state_paths(root)
    state = read_json(paths["state"], {})
    if not control.approved(root):
        raise ValueError("Original approval or explicit sourced resume required")
    if state.get("blocker"):
        observed = control.read(args.evidence_json)
        control.reconcile(root, observed)
    had = state.pop("blocker", None)
    state["updated_at"] = now()
    write_json(paths["state"], state)
    write_status(root)
    print(json.dumps({"blocker_cleared": bool(had)}))


def cmd_stop_check(args: argparse.Namespace) -> None:
    """Ledger-based gate; actual stop/block is not completion."""
    root = Path(args.root)
    code, verdict = control.stop_verdict(root)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return code


def cmd_render_report(args: argparse.Namespace) -> None:
    """Emit optimization_report.md with the iteration history pre-filled from the ledgers."""
    if control.stop_verdict(Path(args.root))[0] != 0:
        raise ValueError("Phase 6 is not eligible; publish status, not a final report")
    root = Path(args.root)
    paths = state_paths(root)
    state = read_json(paths["state"], {})
    experiments = read_jsonl(paths["experiments"])
    baseline = state.get("baseline") or {}
    best = state.get("current_best") or {}

    def _mb(kb: Any) -> str:
        return f"{float(kb) / 1024.0:.1f}" if isinstance(kb, (int, float)) else str(kb)

    lines = []
    lines.append("# Optimization report\n")
    speedup = state.get("cumulative_speedup", 1.0)
    mem_delta = state.get("cumulative_mem_delta", 0.0)
    n_acc = sum(1 for r in experiments if r.get("decision") == "accept")
    n_mem = sum(1 for r in experiments if r.get("decision") == "memory-driven")
    n_rej = sum(1 for r in experiments if r.get("decision") == "reject")
    lines.append("## Executive summary\n")
    lines.append(
        f"Cumulative speedup vs baseline: **{speedup}x** "
        f"(runtime {baseline.get('runtime_mean')}s -> {best.get('runtime_mean')}s). "
        f"Cumulative peak-RSS change: **{-100.0 * float(mem_delta):.1f}%** "
        f"({_mb(baseline.get('mem_mean'))} MB -> {_mb(best.get('mem_mean'))} MB). "
        f"Accepted {n_acc + n_mem} change(s) ({n_acc} runtime, {n_mem} memory-driven); "
        f"rejected {n_rej}.\n"
    )
    lines.append("_TODO: one paragraph on why the accepted patches won and what remains._\n")
    lines.append("## Target and environment\n")
    lines.append(f"- Repository: {state.get('repo_url', '')}")
    lines.append(f"- Commit: {state.get('commit_hash', '')}")
    lines.append(f"- Run ID: {state.get('run_id', '')}")
    lines.append(f"- Required budget: {state.get('budget_iterations', '')} substantive attempts (default 50)")
    lines.append(f"- FP tolerance: {state.get('fp_tolerance', 'auto')}")
    if state.get("numerical_policy"):
        from numerical_policy import check
        policy = check(root, state)
        lines.append(f"- Numerical policy SHA-256: {state['numerical_policy_sha256']}")
        lines.append(f"- Selection rationale: {policy['rationale']}")
        lines.append(f"- Semantic limits: {policy['semantic_limits']}")
        lines.append(f"- Per-output policy: {json.dumps(policy['outputs'], sort_keys=True)}")
        lines.append("- Report actual original-comparison error maxima and failure counts at every final size; digest-match fraction is only an exact-match diagnostic.")
    lines.append(f"- Workload-class diagnosis: {state.get('workload_class', '')}")
    lines.append(f"- Machine: {state.get('machine_id', '')}\n")
    lines.append("## Baseline\n")
    lines.append(f"- Runtime: {baseline.get('runtime_mean')} +/- {baseline.get('runtime_std')} s")
    lines.append(f"- Peak RSS: {_mb(baseline.get('mem_mean'))} +/- {_mb(baseline.get('mem_std'))} MB")
    lines.append(f"- Output digest (SHA-256): {baseline.get('output_digest', '')}")
    lines.append(f"- Determinism: {baseline.get('determinism', '')}\n")
    lines.append("## Current best\n")
    lines.append(f"- Patch: {best.get('patch', '')}")
    lines.append(f"- Runtime: {best.get('runtime_mean')} +/- {best.get('runtime_std')} s")
    lines.append(f"- Peak RSS: {_mb(best.get('mem_mean'))} +/- {_mb(best.get('mem_std'))} MB")
    lines.append(f"- Cumulative speedup: {state.get('cumulative_speedup', 1.0)}x")
    lines.append(f"- Cumulative peak-RSS delta: {state.get('cumulative_mem_delta', 0.0)}\n")

    lines.append("## Complete iteration history\n")
    lines.append("Every recorded iteration -- accepted, rejected, memory-driven, and infrastructure -- in order.\n")
    if experiments:
        lines.append("| Iter | ID | Family | Intervention | Hotspot | Decision | Branch | Runtime (s) | Peak RSS (MB) | Metric analysis | Summary |")
        lines.append("|---:|---|---|---|---|---|---|---:|---:|---|---|")
        for r in experiments:
            rt = r.get("runtime_mean")
            rt_s = f"{rt:.4g}" if isinstance(rt, (int, float)) else str(rt)
            mem = r.get("mem_mean")
            mem_s = _mb(mem) if isinstance(mem, (int, float)) else str(mem)
            ma = str(r.get("metric_analysis", "")).replace("|", "\\|")
            if len(ma) > 120:
                ma = ma[:117] + "..."
            lines.append(
                f"| {r.get('iteration', '')} | {r.get('iteration_id', '')} | {r.get('family', '')} "
                f"| {r.get('intervention', '')} | {r.get('hotspot', '')} | {r.get('decision', '')} "
                f"| {r.get('branch', '')} | {rt_s} | {mem_s} | {ma} "
                f"| {str(r.get('summary', '')).replace('|', chr(92) + '|')} |"
            )
    else:
        lines.append("_No iterations recorded._")
    lines.append("")

    accepted = [r for r in experiments if r.get("decision") in {"accept", "memory-driven"}]
    rejected = [r for r in experiments if r.get("decision") == "reject"]
    lines.append("## Accepted changes\n")
    if accepted:
        lines.append("| Iter | ID | Hotspot | Patch | Runtime (s) | Peak RSS (MB) | Branch |")
        lines.append("|---:|---|---|---|---:|---:|---|")
        for r in accepted:
            lines.append(f"| {r.get('iteration', '')} | {r.get('iteration_id', '')} | {r.get('hotspot', '')} "
                         f"| {r.get('patch_file', '')} | {r.get('runtime_mean', '')} | {_mb(r.get('mem_mean'))} | {r.get('branch', '')} |")
    else:
        lines.append("_No accepted changes._")
    lines.append("")
    lines.append("## Rejected hypotheses\n")
    if rejected:
        lines.append("| Iter | ID | Hotspot | Hypothesis | Why rejected |")
        lines.append("|---:|---|---|---|---|")
        for r in rejected:
            lines.append(f"| {r.get('iteration', '')} | {r.get('iteration_id', '')} | {r.get('hotspot', '')} "
                         f"| {str(r.get('hypothesis', '')).replace('|', chr(92) + '|')} "
                         f"| {str(r.get('summary', '')).replace('|', chr(92) + '|')} |")
    else:
        lines.append("_No rejected hypotheses._")
    lines.append("")

    phase6_csv = root / "bench" / "phase6_results.csv"
    lines.append("## Scaling Suite results (Phase 6)\n")
    if phase6_csv.exists():
        lines.append(f"Raw measurements: `{phase6_csv}`\n")
        lines.append("| size | variant | reps | runtime mean (s) | runtime std | peak RSS mean (MB) |")
        lines.append("|---|---|---:|---:|---:|---:|")
        import csv as _csv
        agg: Dict[tuple, Dict[str, list]] = {}
        with phase6_csv.open("r", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                key = (row.get("size", ""), row.get("variant", ""))
                agg.setdefault(key, {"rt": [], "rss": []})
                try:
                    agg[key]["rt"].append(float(row.get("runtime_sec", "nan")))
                    agg[key]["rss"].append(float(row.get("maxrss_kb", "nan")))
                except Exception:
                    pass
        for (size, variant), vals in agg.items():
            rt = vals["rt"]
            rss = vals["rss"]
            if rt:
                mean_rt = sum(rt) / len(rt)
                std_rt = (sum((x - mean_rt) ** 2 for x in rt) / max(1, len(rt) - 1)) ** 0.5 if len(rt) > 1 else 0.0
                mean_rss = sum(rss) / len(rss) / 1024.0 if rss else float("nan")
                lines.append(f"| {size} | {variant} | {len(rt)} | {mean_rt:.4g} | {std_rt:.2g} | {mean_rss:.4g} |")
    else:
        lines.append("_TODO: run scripts/scaling_suite.py and paste the comparison table._")
    lines.append("")
    lines.append("## Correctness tests used\n\n_TODO._\n")
    lines.append("## Remaining likely hotspots\n\n_TODO._\n")
    lines.append("## Latest learning and one evidence-grounded next direction\n\n_TODO._\n")
    lines.append("## Blockers, uncertainty, open questions\n\n_TODO._\n")
    lines.append("## Reproduction commands\n\n_TODO: exact commands, environment, commit._\n")

    out = Path(args.out) if args.out else root / "reports" / "optimization_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.absolute().is_relative_to(control.local_root(root)):
        control.atomic_bytes(out, "\n".join(lines).encode())
    else:
        # Explicit external report destinations are derived, non-authoritative outputs.
        # Object-backed destinations do not support local atomic rename/undo.
        out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report_written": str(out), "iterations": len(experiments)}))


def cmd_summary(args: argparse.Namespace) -> None:
    root = Path(args.root)
    paths = state_paths(root)
    state = read_json(paths["state"], {})
    experiments = read_jsonl(paths["experiments"])
    feedback = read_jsonl(paths["feedback"])
    portfolio = read_json(paths["portfolio"], {"families": []})
    print("# biomni-code run summary\n")
    print(f"Run ID: {state.get('run_id')}")
    print(f"Repo: {state.get('repo_url')} @ {state.get('commit_hash')}")
    print(f"Iteration: {state.get('iteration', 0)} / {state.get('budget_iterations')}")
    print(f"Cumulative speedup: {state.get('cumulative_speedup', 1.0)}x | "
          f"peak-RSS delta: {state.get('cumulative_mem_delta', 0.0)}")
    print(f"Iterations recorded: {len(experiments)} | Feedback items: {len(feedback)}\n")
    print("## Optimization families")
    for fam in portfolio.get("families", []):
        print(f"- {fam.get('name')}: attempts={fam.get('attempts')} accepts={fam.get('accepts')} "
              f"best={fam.get('best_iteration_id')} runtime={fam.get('best_runtime_mean')}")
    print("\n## Recent iterations")
    for r in experiments[-10:]:
        print(f"- iter {r.get('iteration')} {r.get('iteration_id')} {r.get('family')}/{r.get('intervention')} "
              f"{r.get('decision')} runtime={r.get('runtime_mean')} -- {r.get('summary')}")


def mirror_history(root: Path, changed=None, full=False, strict=False) -> None:
    """Mirror changed history, or check/full-copy all history on explicit sync.

    The canonical ledgers stay in the run root (append-only); this mirror makes the
    whole history user-facing and durable beyond the shared-workspace lifetime. A
    An ordinary mirror failure cannot undo a committed state command. Explicit
    synchronization uses strict=True so failures are reported as unsuccessful.
    """
    try:
        state = read_json(root / "state" / "state.json", {})
        run_id = state.get("run_id") or root.name
        base = Path(os.environ.get("BIOMNICODE_RESULTS_ROOT", "/mnt/results/biomnicode"))
        target = base / str(run_id) / "history"
        first_sync = not target.exists()
        sources = [root / rel for rel in HISTORY_FILES]
        for pattern in HISTORY_GLOBS:
            sources.extend(sorted(p for p in root.glob(pattern) if p.is_file()))
        copied = []
        def copy_changed(src, dst, rel, force=False):
            if changed is not None and not first_sync and not full and not force and rel not in changed:
                return
            data = src.read_bytes()
            if not full and dst.is_file() and dst.read_bytes() == data:
                return
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)  # Complete object write, never append or rename.
            copied.append(rel)
        for src in sources:
            if not src.is_file():
                continue
            dst = target / src.relative_to(root)
            copy_changed(src, dst, src.relative_to(root).as_posix())
        best = (state.get("current_best") or {}).get("patch")
        patch = control.resolve(root, best) if best else None
        if changed is None or first_sync or full or "state/state.json" in changed:
            if patch and patch.is_file():
                copy_changed(patch, target / "current_best.patch", best, force=True)
            elif not best:
                dst = target / "current_best.patch"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if full or not dst.exists() or dst.read_bytes() != b"":
                    dst.write_bytes(b"")
        for rel in ("state/control.json", "state/verified_checkpoint.json"):
            if (root / rel).is_file():
                dst = target / rel
                copy_changed(root / rel, dst, rel)
        print(json.dumps({"history_mirrored_to": str(target), "files_copied": len(copied)}), file=sys.stderr)
    except Exception as e:
        if strict:
            raise
        print(f"warning: history mirror failed: {e}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="State helper for biomni-code (code optimization)")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("init")
    s.add_argument("--root", required=True)
    s.add_argument("--run-id")
    s.add_argument("--repo-url", default="")
    s.add_argument("--commit-hash", default="")
    s.add_argument("--budget-iterations", type=int, default=50, help="Required substantive attempts (default 50), not an optional ceiling; explicit user override only.")
    s.add_argument("--fp-tolerance", default="auto", help="'auto' (default): reason and freeze with set-tolerance before optimization; or explicit 'bitwise'/rtol-atol setting.")
    s.add_argument("--workload-class", default="", help="Phase 2 diagnosis: cpu-bound | io-bound | syscall-heavy | allocation-heavy.")
    s.add_argument("--machine-id", default="")
    s.add_argument("--build-cmd", default="")
    s.add_argument("--correctness-cmd", default="")
    s.add_argument("--benchmark-cmd", default="")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("set-baseline")
    s.add_argument("--root", required=True)
    s.add_argument("--runtime-mean", type=float, required=True)
    s.add_argument("--runtime-std", type=float, required=True)
    s.add_argument("--mem-mean", type=float, default=None)
    s.add_argument("--mem-std", type=float, default=None)
    s.add_argument("--output-digest", default="")
    s.add_argument("--determinism", default="")
    s.set_defaults(func=cmd_set_baseline)

    s = sub.add_parser("set-tolerance", help="Freeze justified numerical policy before optimization")
    s.add_argument("--root", required=True)
    s.add_argument("--evidence-json", required=True)
    def set_tolerance(args):
        from numerical_policy import resolve
        result = resolve(Path(args.root), args.evidence_json)
        write_status(Path(args.root))
        print(json.dumps(result))
    s.set_defaults(func=set_tolerance)

    s = sub.add_parser("start-phase5")
    s.add_argument("--root", required=True)
    s.set_defaults(func=cmd_start_phase5)

    s = sub.add_parser("add-iteration")
    s.add_argument("--root", required=True)
    s.add_argument("--iteration-id", required=True)
    s.add_argument("--parent-id")
    s.add_argument("--iteration", type=int, help="Advisory only; the counter is derived from the ledger.")
    s.add_argument("--family", default="unspecified", help="Optimization family (see strategy_library/optimization_families.json).")
    s.add_argument("--intervention", required=True, help="Intervention type within the family.")
    s.add_argument("--hotspot", default="", help="Targeted function/region from the ranked hotspot table.")
    s.add_argument("--hypothesis", default="")
    s.add_argument("--edited-files", nargs="*", default=[])
    s.add_argument("--patch-file", default="")
    s.add_argument("--correctness", choices=["pass", "fail", "not-run"], default="not-run")
    s.add_argument("--runtime-mean", type=float)
    s.add_argument("--runtime-std", type=float)
    s.add_argument("--runtime-reps", type=int)
    s.add_argument("--mem-mean", type=float)
    s.add_argument("--mem-std", type=float)
    s.add_argument("--mem-reps", type=int)
    s.add_argument("--output-digest", default="")
    s.add_argument("--decision", choices=["accept", "reject", "memory-driven", "infrastructure"], required=True)
    s.add_argument("--branch", choices=["runtime", "memory", "none"], default="none")
    s.add_argument("--non-substantive", action="store_true",
                   help="Record but do NOT advance the iteration counter (e.g. pure infra retry).")
    s.add_argument("--summary", default="")
    s.add_argument("--metric-analysis", default="",
                   help="REQUIRED for substantive iterations: why the runtime/memory numbers came out as they did.")
    s.add_argument("--next-action", default="")
    s.add_argument("--feedback", default="")
    s.add_argument("--evidence-json", required=True, help="Hash-linked detailed review and disposition evidence")
    s.set_defaults(func=cmd_add_iteration)

    s = sub.add_parser("begin-iteration", help="Reserve the sole optimization candidate")
    s.add_argument("--root", required=True)
    s.add_argument("--iteration-id", required=True)
    s.add_argument("--hypothesis", required=True)
    s.add_argument("--parent-id", default=None)
    s.add_argument("--feedback-id", required=True, help="Immediately preceding reviewed iteration, or baseline")
    s.add_argument("--feedback-sha256", required=True, help="Exact feedback digest from suggest-next")
    s.add_argument("--learning", required=True, help="Reassess this hypothesis against current best and latest reviewed evidence")
    s.add_argument("--backlog-id", help="Optional open idea selected after reassessment; never auto-admits a queue")
    s.set_defaults(func=cmd_begin_iteration)

    s = sub.add_parser("add-backlog", help="Record or revise a ranked idea; does not admit or execute it")
    s.add_argument("--root", required=True)
    s.add_argument("--backlog-id", help="Stable idea ID; omit to create, reuse with full fields to revise")
    s.add_argument("--priority", type=float, default=1.0)
    s.add_argument("--family", default="exploratory")
    s.add_argument("--intervention", default="algorithmic")
    s.add_argument("--hotspot", default="")
    s.add_argument("--hypothesis", required=True)
    s.add_argument("--test", default="")
    s.add_argument("--source", default="manual")
    s.add_argument("--status", choices=["open", "retired"], default="open")
    s.set_defaults(func=cmd_add_backlog)

    s = sub.add_parser("add-feedback")
    s.add_argument("--root", required=True)
    s.add_argument("--type", default="insight")
    s.add_argument("--component", default="")
    s.add_argument("--evidence", required=True)
    s.add_argument("--recommendation", default="")
    s.add_argument("--similar-ideas", default="", help="Related observations or ideas to reassess before selection")
    s.add_argument("--avoid", default="")
    s.set_defaults(func=cmd_add_feedback)

    s = sub.add_parser("add-hotspots")
    s.add_argument("--root", required=True)
    s.add_argument("--profiler", default="")
    s.add_argument("--workload", default="")
    s.add_argument("--entries", nargs="*", default=[],
                   help="One JSON object per hotspot, e.g. '{\"function\":\"f\",\"pct_runtime\":42.1}'.")
    s.set_defaults(func=cmd_add_hotspots)

    s = sub.add_parser("suggest-next")
    s.add_argument("--root", required=True)
    s.set_defaults(func=cmd_suggest_next)

    s = sub.add_parser("checkpoint")
    s.add_argument("--root", required=True)
    s.add_argument("--store", required=True, help="Durable content-addressed checkpoint store")
    s.set_defaults(func=cmd_checkpoint)

    s = sub.add_parser("status")
    s.add_argument("--root", required=True)
    s.set_defaults(func=lambda args: (write_status(Path(args.root)), print(json.dumps({"status_written": True}))))

    s = sub.add_parser("stop-check",
                       help="0: budget exhausted; 3: continue/recover; 4: sourced stop/block; 2: invalid state")
    s.add_argument("--root", required=True)
    s.set_defaults(func=cmd_stop_check)

    s = sub.add_parser("record-blocker", help="Record a hard blocker (closed list) that permits an early stop.")
    s.add_argument("--root", required=True)
    s.add_argument("--kind", choices=sorted(HARD_BLOCKER_KINDS), required=True,
                   help="infra | harness | user | repo")
    s.add_argument("--detail", required=True, help="Exact error message or user instruction justifying the stop.")
    s.add_argument("--evidence-json", required=True)
    s.set_defaults(func=cmd_record_blocker)

    s = sub.add_parser("clear-blocker")
    s.add_argument("--root", required=True)
    s.add_argument("--evidence-json", required=True)
    s.set_defaults(func=cmd_clear_blocker)

    s = sub.add_parser("render-report", help="Emit optimization_report.md pre-filled from the ledgers.")
    s.add_argument("--root", required=True)
    s.add_argument("--out", default="", help="Output path (default: <root>/reports/optimization_report.md).")
    s.set_defaults(func=cmd_render_report)

    s = sub.add_parser("summary")
    s.add_argument("--root", required=True)
    s.set_defaults(func=cmd_summary)

    s = sub.add_parser("mirror-history", help="Synchronize and content-check the complete derived history mirror")
    s.add_argument("--root", required=True)
    s.add_argument("--full", action="store_true", help="Rewrite every history object, including unchanged files")
    s.set_defaults(func=lambda args: mirror_history(Path(args.root), full=args.full, strict=True))
    return p


def main() -> None:
    args = build_parser().parse_args()
    try:
        root = control.local_root(args.root)
        readonly = args.func in {cmd_summary, cmd_suggest_next, cmd_stop_check}
        with control.transaction(root, readonly=readonly) as tx:
            result = args.func(args)
        # Mirror is not the authoritative commit and cannot cause duplicate counting.
        if args.func == cmd_checkpoint:
            mirror_history(root)
        elif tx.before:
            mirror_history(root, changed=set(tx.before))
        if args.func == cmd_stop_check:
            raise SystemExit(result)
    except (ValueError, OSError, KeyError, TypeError) as e:
        print(json.dumps({"error": str(e), "may_finalize": False}))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
