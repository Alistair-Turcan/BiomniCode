#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
conda create -y -p "$ROOT/env/toolchain" --file "$ROOT/recipes/conda-explicit.txt"
