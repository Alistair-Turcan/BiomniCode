#!/usr/bin/env bash
set -euo pipefail

# Fill in commands for a clean reproduction of the optimization run and its
# Phase 6 validation. Example:
#
# git clone <repo> work && cd work && git checkout <commit>
# python -m venv env && env/bin/pip install -r requirements.txt
# python scripts/benchmark_runner.py --command "<benchmark>" --repeat 5 --out bench/baseline.json
# git apply current_best.patch
# python scripts/scaling_suite.py --sizes scaling_suite/sizes.json \
#     --variants baseline=original_snapshot optimized=work --out-csv bench/phase6_results.csv
