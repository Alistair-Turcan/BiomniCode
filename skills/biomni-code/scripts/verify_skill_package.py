#!/usr/bin/env python3
"""Validate the biomni-code skill package structure."""
from __future__ import annotations

import argparse
import json
import py_compile
import tempfile
from pathlib import Path

REQUIRED_FILES = [
    "SKILL.md",
    "references/README.md",
    "references/VERSION",
    "references/state_and_memory.md",
    "references/optimization_loop.md",
    "references/parallelism_and_machines.md",
    "references/benchmark_calibration.md",
    "references/acceptance_and_correctness.md",
    "references/numerical_tolerance.md",
    "references/final_delivery_contract.md",
    "scripts/biomni_code_state.py",
    "scripts/benchmark_runner.py",
    "scripts/acceptance.py",
    "scripts/scaling_suite.py",
    "scripts/auto_resume.py",
    "scripts/run_control.py",
    "scripts/numerical_policy.py",
    "scripts/run_checkpoint.py",
    "references/interruption_recovery.md",
    "scripts/protected_hashes.py",
    "scripts/create_manifest.py",
    "scripts/safe_zip.py",
    "scripts/verify_delivery.py",
    "references/strategy_library/optimization_families.json",
    "references/strategy_library/intervention_weights.json",
    "references/strategy_library/hotspot_taxonomy.json",
    "templates/optimization_notebook.md",
    "templates/iteration_record.md",
    "templates/benchmark_card.md",
    "templates/final_report.md",
    "templates/reproduce.sh",
    "templates/run_manifest.json",
    "references/languages/python/hints.md",
    "references/languages/r/hints.md",
    "references/languages/c/hints.md",
    "references/languages/c++/hints.md",
    "references/languages/rust/hints.md",
    "references/languages/go/hints.md",
    "references/languages/java/hints.md",
    "references/languages/julia/hints.md",
    "references/languages/javascript/hints.md",
    "references/languages/assembly/hints.md",
    "references/profiler_toolbox.md",
]

REQUIRED_PHRASES = [
    "stop-check",
    "STOP-GATE",
    "substantive",
    "Hard blockers",
    "diversity floor",
    "acceptance rule",
    "bitwise",
    "protected",
    "Phase 6",
    "current_best.patch",
    "metric_analysis",
    "Diverse ideas, sequential experiments and learning",
    "set-tolerance",
    "50 substantive iterations by default",
    "most permissive defensible numerical tolerance",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skill-root", required=True)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()
    root = Path(args.skill_root)
    errors = []
    for rel in REQUIRED_FILES:
        if not (root / rel).exists():
            errors.append(f"missing {rel}")
    skill = (root / "SKILL.md").read_text(encoding="utf-8") if (root / "SKILL.md").exists() else ""
    for phrase in REQUIRED_PHRASES:
        if phrase not in skill:
            errors.append(f"SKILL.md missing required phrase: {phrase}")
    if len([p for p in root.rglob("*") if p.is_file()]) > 10000:
        errors.append("package exceeds 10000 files")
    for json_path in root.rglob("*.json"):
        try:
            json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"invalid JSON {json_path.relative_to(root)}: {e}")
    if args.compile:
        # py_compile writes the .pyc via os.rename, which fails on S3-FUSE
        # mounts (e.g. /mnt/results). Redirect bytecode output to local disk.
        with tempfile.TemporaryDirectory(prefix="biomnicode-pyc-") as pyc_dir:
            for py in root.rglob("*.py"):
                try:
                    cfile = Path(pyc_dir) / (py.stem + ".pyc")
                    py_compile.compile(str(py), cfile=str(cfile), doraise=True)
                except Exception as e:
                    errors.append(f"python compile failed {py.relative_to(root)}: {e}")
    if errors:
        print("biomni-code skill validation failed:")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)
    print(f"biomni-code skill validation passed: {root}")


if __name__ == "__main__":
    main()
