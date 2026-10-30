#!/usr/bin/env python3
"""Run the biomni-code Scaling Suite: baseline vs current-best across sized inputs.

Reads a sizes config JSON:
[
  {"name": "s", "command": "python -m pkg --size 1000", "reps": 5,
   "digest_files": ["out_s.txt"]},
  ...
]

For Phase 5, run only the per-iteration (smallest) size. For Phase 6, run every
size for both variants. The combined Phase 6 run is capped at --cap-minutes
(default 15): if the projected total exceeds the cap, the LARGEST sizes are
dropped -- never the repetition count -- and the drop is recorded in the output.

Variants are shell command prefixes: typically "" (baseline, run in the original
snapshot directory) vs a patched working copy. Commands may contain the placeholder
{variant_dir}; it is replaced per run.

Output CSV columns: size, variant, rep, runtime_sec, maxrss_kb, output_digest,
returncode, protected_original (flag for the largest-size protected-original
comparison rows).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Reuse the single-run machinery from benchmark_runner.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_runner import run_once, HAVE_GNU_TIME  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", required=True, help="Path to sizes config JSON")
    p.add_argument("--variants", required=True, nargs="+",
                   help="variant_name=variant_dir pairs, e.g. baseline=/path/orig optimized=/path/work")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--cap-minutes", type=float, default=15.0)
    p.add_argument("--protected-original", default="",
                   help="Variant name that corresponds to the protected original snapshot "
                        "(rows flagged protected_original=true).")
    p.add_argument("--timeout", type=float, default=None)
    args = p.parse_args()

    sizes = json.loads(Path(args.sizes).read_text(encoding="utf-8"))
    variants = {}
    for v in args.variants:
        name, _, directory = v.partition("=")
        if not directory:
            raise SystemExit(f"bad --variants entry {v!r}; expected name=dir")
        variants[name] = directory

    # Project total cost from the smallest size, then drop largest sizes if needed.
    dropped = []
    cap_sec = args.cap_minutes * 60.0
    active = list(sizes)

    rows = []
    start_all = time.time()
    for size in active:
        for vname, vdir in variants.items():
            reps = int(size.get("reps", 3))
            cmd = size["command"].replace("{variant_dir}", vdir)
            digest_files = [str(Path(vdir) / f) for f in size.get("digest_files", [])]
            for rep in range(reps):
                if time.time() - start_all > cap_sec:
                    dropped.append(f"{size.get('name')}:{vname} (cap reached mid-run)")
                    break
                try:
                    rec = run_once(cmd, vdir, args.timeout, digest_files,
                                   bool(size.get("digest_stdout")), HAVE_GNU_TIME)
                except Exception as e:  # keep the suite going; record the failure
                    rec = {"returncode": "error", "runtime_sec": None, "maxrss_kb": None,
                           "output_digest": "", "stderr_tail": str(e)}
                rows.append({
                    "size": size.get("name"),
                    "variant": vname,
                    "rep": rep,
                    "runtime_sec": rec.get("runtime_sec"),
                    "maxrss_kb": rec.get("maxrss_kb"),
                    "output_digest": rec.get("output_digest"),
                    "returncode": rec.get("returncode"),
                    "protected_original": str(vname == args.protected_original).lower(),
                })
            else:
                continue
            break

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["size", "variant", "rep", "runtime_sec", "maxrss_kb",
                                               "output_digest", "returncode", "protected_original"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"out_csv": str(out), "rows": len(rows), "dropped": dropped,
                      "elapsed_sec": round(time.time() - start_all, 2)}))


if __name__ == "__main__":
    main()
