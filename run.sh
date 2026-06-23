#!/usr/bin/env bash
# ============================================================================
#  FCC Coverage-Change Audit - one-command launcher (macOS / Linux)
#
#  Usage:
#     ./run.sh                    ->  download + analyze ALL providers/techs
#     ./run.sh download           ->  only pre-fetch raw data from the FCC API
#     ./run.sh run --current 2025-12-31 --prior 2025-06-30
#
#  Creates a local virtual environment, installs dependencies once, then runs
#  the pipeline. Re-running reuses the same environment.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYBIN="${PYTHON:-python3}"

# 1) Create the virtual environment on first run.
if [ ! -x ".venv/bin/python" ]; then
    echo "[setup] creating virtual environment..."
    "$PYBIN" -m venv .venv
    echo "[setup] installing dependencies..."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
fi

export PYTHONPATH=src

# 2) Run. With no arguments, do a full national run that deletes each raw file
#    after processing so disk usage stays bounded.
if [ "$#" -eq 0 ]; then
    echo "[run] full pipeline: all providers, all configured technologies"
    .venv/bin/python -m fcc_audit.cli run --cleanup-raw
else
    .venv/bin/python -m fcc_audit.cli "$@"
fi

echo
echo "Done. Outputs are in data/outputs/  (open dashboard/index.html for the map)"
