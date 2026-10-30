#!/usr/bin/env python3
"""Validate biomni-code Phase 6 delivery artifacts.

Checks a delivery directory (typically under /mnt/results) for the artifacts the
final-delivery contract requires:
- an optimization report (optimization_report*.md, non-empty)
- current_best.patch (may be empty only if zero changes were accepted; the file
  must still exist)
- bench/phase6_results.csv (or a phase6_results.csv somewhere in the tree)
- the Phase 6 scaling figure phase6_panels.png and phase6_panels.svg
- the per-size summary CSVs runtime.csv, memory.csv, and correctness.csv
- a final history mirror (history/iterations/results.jsonl somewhere in the tree)
- status.json

Usage: python verify_delivery.py --delivery /mnt/results [--repo <name>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--delivery", required=True)
    p.add_argument("--repo", default="", help="Repository name; used to match optimization_report_<repo>.md")
    args = p.parse_args()
    root = Path(args.delivery)
    errors = []

    reports = sorted(root.rglob("optimization_report*.md"))
    if not reports:
        errors.append("missing optimization_report*.md")
    elif all(r.stat().st_size == 0 for r in reports):
        errors.append("optimization report is empty")

    patches = sorted(root.rglob("current_best.patch"))
    if not patches:
        errors.append("missing current_best.patch")

    csvs = sorted(root.rglob("phase6_results.csv"))
    if not csvs:
        errors.append("missing phase6_results.csv")
    else:
        try:
            rows = list(csvs[0].open("r", encoding="utf-8"))
            if len(rows) < 2:
                errors.append("phase6_results.csv has no data rows")
        except Exception as e:
            errors.append(f"cannot read phase6_results.csv: {e}")

    if not list(root.rglob("phase6_panels.png")):
        errors.append("missing phase6_panels.png (Phase 6 scaling figure)")
    if not list(root.rglob("phase6_panels.svg")):
        errors.append("missing phase6_panels.svg (Phase 6 scaling figure)")

    for name in ("runtime.csv", "memory.csv", "correctness.csv"):
        found = sorted(root.rglob(name))
        if not found:
            errors.append(f"missing {name} (per-size Phase 6 summary)")
        else:
            try:
                if len(found[0].open("r", encoding="utf-8").readlines()) < 2:
                    errors.append(f"{name} has no data rows")
            except Exception as e:
                errors.append(f"cannot read {name}: {e}")

    if not list(root.glob("**/history/iterations/results.jsonl")):
        errors.append("missing final history mirror (history/iterations/results.jsonl)")

    statuses = sorted(root.rglob("status.json"))
    if not statuses:
        errors.append("missing status.json")
    else:
        try:
            json.loads(statuses[0].read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"invalid status.json: {e}")

    if len([p for p in root.rglob("*") if p.is_file()]) > 10000:
        errors.append("delivery exceeds 10000 files")

    if errors:
        print("biomni-code delivery validation failed:")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)
    print(f"biomni-code delivery validation passed: {root}")


if __name__ == "__main__":
    main()
