#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYSCENIC_TEST_PYTHON:-$ROOT/.venv/bin/python}"
if [ ! -x "$PY" ]; then
  printf '%s\n' "Install first: bash scripts/install.sh" >&2
  exit 2
fi
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$ROOT/.cache/numba}"
"$PY" "$ROOT/scripts/verify_source.py" --imports
"$PY" "$ROOT/scripts/fetch_test_fixtures.py" --check-only
# Both upstream projects define a top-level resources.tests package. Isolate their
# collection processes and prioritize the matching source tree, without editing tests.
PYTHONPATH="$ROOT/pySCENIC/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" -m pytest "$ROOT/pySCENIC/tests" "$@"
PYTHONPATH="$ROOT/ctxcore/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" -m pytest "$ROOT/ctxcore/tests" "$@"
"$PY" "$ROOT/scripts/verify_source.py"
