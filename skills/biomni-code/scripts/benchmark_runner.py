#!/usr/bin/env python3
"""Run a benchmark command repeatedly and collect runtime, peak RSS, and output digests.

Purpose-built for biomni-code Phase 1 (baseline), Phase 3 (scaling suite), Phase 5
(per-iteration benchmark), and Phase 6 (final validation).

Per repetition it captures:
- wall-clock seconds (perf_counter)
- peak RSS in KB: parsed from `/usr/bin/time -v` ("Maximum resident set size (kbytes)")
  when available; otherwise falls back to sampling VmRSS of the whole process tree
  from /proc every ~5 ms and taking the maximum (rss_method="proc-vmrss-sampling").
  The sampler is per-rep and tree-wide, so it works on minimal containers without
  GNU time; on non-Linux platforms without /proc it degrades to
  resource.getrusage(RUSAGE_CHILDREN) deltas (rss_method="getrusage-delta", less
  reliable: ru_maxrss is a cumulative max across children, so later reps can report 0).
- %CPU from /usr/bin/time -v when available
- SHA-256 digest of the canonicalized output: either the bytes of --digest-files
  (concatenated in the given order) or, with --digest-stdout, the exact stdout bytes.

The summary reports mean/std/min/max for runtime and peak RSS, plus a determinism
flag (true iff every repetition produced the same digest).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HAVE_GNU_TIME = shutil.which("/usr/bin/time") is not None or Path("/usr/bin/time").exists()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_files(paths: List[str]) -> str:
    h = hashlib.sha256()
    for s in paths:
        p = Path(s)
        if not p.is_file():
            return f"MISSING:{s}"
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def parse_time_v(stderr: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for line in stderr.splitlines():
        line = line.strip()
        if line.startswith("Maximum resident set size (kbytes):"):
            try:
                out["maxrss_kb"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("Percent of CPU this job got:"):
            try:
                out["cpu_pct"] = float(line.split(":", 1)[1].strip().rstrip("%"))
            except ValueError:
                pass
        elif line.startswith("Elapsed (wall clock) time"):
            out["wall_clock_hms"] = line.split(":", 1)[1].strip()
    return out


def _tree_pids(root_pid: int) -> List[int]:
    """All PIDs in the process tree rooted at root_pid (Linux /proc)."""
    pids = [root_pid]
    i = 0
    while i < len(pids):
        try:
            with open(f"/proc/{pids[i]}/task/{pids[i]}/children", encoding="utf-8") as f:
                pids.extend(int(c) for c in f.read().split())
        except (OSError, ValueError):
            pass
        i += 1
    return pids


def _tree_vmrss_kb(root_pid: int) -> int:
    """Sum of VmRSS (kB) across the process tree; 0 if the tree has exited."""
    total = 0
    for pid in _tree_pids(root_pid):
        try:
            with open(f"/proc/{pid}/status", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
                        break
        except (OSError, ValueError):
            pass
    return total


def run_once(command: str, cwd: Optional[str], timeout: Optional[float],
             digest_files: List[str], digest_stdout: bool, use_time_v: bool) -> Dict[str, Any]:
    start = time.perf_counter()
    rss_method = "none"
    maxrss_kb = None
    cpu_pct = None
    if use_time_v:
        proc = subprocess.run(f"/usr/bin/time -v {command}", shell=True, cwd=cwd, text=True,
                              capture_output=True, timeout=timeout)
        parsed = parse_time_v(proc.stderr or "")
        maxrss_kb = parsed.get("maxrss_kb")
        cpu_pct = parsed.get("cpu_pct")
        rss_method = "time-v" if maxrss_kb is not None else "none"
        # /usr/bin/time -v writes its own report to stderr; keep the tail for debugging.
        stderr_tail = (proc.stderr or "")[-4000:]
    elif Path("/proc/self/status").exists():
        # GNU time unavailable: sample tree-wide VmRSS from /proc and take the max.
        proc = subprocess.Popen(command, shell=True, cwd=cwd, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        peak = 0
        deadline = (start + timeout) if timeout else None
        while proc.poll() is None:
            if deadline is not None and time.perf_counter() > deadline:
                proc.kill()
                proc.wait()
                raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)
            peak = max(peak, _tree_vmrss_kb(proc.pid))
            time.sleep(0.005)
        peak = max(peak, _tree_vmrss_kb(proc.pid))  # final sample near exit
        stdout_s, stderr_s = proc.communicate()
        proc.stdout, proc.stderr = stdout_s, stderr_s  # keep the digest/tail code below uniform
        maxrss_kb = peak
        rss_method = "proc-vmrss-sampling"
        stderr_tail = (stderr_s or "")[-4000:]
    else:
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        proc = subprocess.run(command, shell=True, cwd=cwd, text=True, capture_output=True, timeout=timeout)
        after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        maxrss_kb = max(0, after - before)
        rss_method = "getrusage-delta"
        stderr_tail = (proc.stderr or "")[-4000:]
    wall = time.perf_counter() - start

    if digest_files:
        digest = sha256_files(digest_files)
    elif digest_stdout:
        digest = sha256_bytes((proc.stdout or "").encode("utf-8", errors="replace"))
    else:
        digest = ""
    return {
        "command": command,
        "cwd": cwd,
        "returncode": proc.returncode,
        "runtime_sec": wall,
        "maxrss_kb": maxrss_kb,
        "rss_method": rss_method,
        "cpu_pct": cpu_pct,
        "output_digest": digest,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": stderr_tail,
    }


def summarize(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "std": (statistics.stdev(values) if len(values) > 1 else 0.0),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--command", required=True, help="Shell command to benchmark")
    p.add_argument("--out", required=True, help="Output JSON summary path")
    p.add_argument("--jsonl", default="", help="Optional append-only JSONL path for per-rep records")
    p.add_argument("--cwd", default=None)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--warmup", type=int, default=0, help="Warm-up runs whose results are discarded")
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--digest-files", nargs="*", default=[], help="Output files to hash (in order)")
    p.add_argument("--digest-stdout", action="store_true", help="Hash exact stdout bytes")
    p.add_argument("--no-time-v", action="store_true", help="Do not use /usr/bin/time -v even if present")
    p.add_argument("--label", default="")
    args = p.parse_args()

    use_time_v = HAVE_GNU_TIME and not args.no_time_v
    for _ in range(args.warmup):
        subprocess.run(args.command, shell=True, cwd=args.cwd, capture_output=True, timeout=args.timeout)

    results = []
    for i in range(args.repeat):
        try:
            rec = run_once(args.command, args.cwd, args.timeout, args.digest_files, args.digest_stdout, use_time_v)
        except subprocess.TimeoutExpired as e:
            rec = {"command": args.command, "cwd": args.cwd, "returncode": "timeout",
                   "runtime_sec": args.timeout, "maxrss_kb": None, "rss_method": "none",
                   "cpu_pct": None, "output_digest": "",
                   "stdout_tail": (e.stdout or "")[-4000:] if isinstance(e.stdout, str) else "",
                   "stderr_tail": (e.stderr or "")[-4000:] if isinstance(e.stderr, str) else ""}
        rec["repeat_index"] = i
        rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if args.label:
            rec["label"] = args.label
        results.append(rec)
        if args.jsonl:
            Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
            with Path(args.jsonl).open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")

    runtimes = [r["runtime_sec"] for r in results if isinstance(r.get("runtime_sec"), (int, float))]
    rss = [r["maxrss_kb"] for r in results if isinstance(r.get("maxrss_kb"), (int, float))]
    digests = [r.get("output_digest") for r in results if r.get("output_digest")]
    deterministic = len(set(digests)) <= 1 if digests else None
    summary = {
        "command": args.command,
        "label": args.label,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "rss_method": results[0].get("rss_method") if results else "none",
        "runtime": summarize(runtimes),
        "maxrss_kb": summarize(rss),
        "output_digest": (digests[0] if deterministic else ""),
        "deterministic": deterministic,
        "returncodes": [r.get("returncode") for r in results],
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "returncodes": summary["returncodes"],
                      "runtime": summary["runtime"], "maxrss_kb": summary["maxrss_kb"],
                      "deterministic": deterministic, "rss_method": summary["rss_method"]}))


if __name__ == "__main__":
    main()
