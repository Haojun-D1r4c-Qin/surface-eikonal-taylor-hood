#!/usr/bin/env bash
set -euo pipefail
REPORT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
"${EIKONAL_PYTHON:-python}" "$REPORT_ROOT/code/run_experiments.py" "$@"
