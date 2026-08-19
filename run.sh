#!/usr/bin/env bash
# ============================================================================
#  FCC Coverage-Change Audit - one-command launcher (macOS / Linux)
#
#  Usage:
#     ./run.sh                    ->  overnight national batches + final build-web
#     ./run.sh --publish          ->  same, then git commit + push web bundle
#     ./run.sh download           ->  only pre-fetch raw data
#     ./run.sh run --states 01,02 ->  pass-through to CLI
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

# 2) No args (or --publish only): crash-safe geographic overnight path,
#    matching Windows run.bat → run_overnight.ps1.
if [ "$#" -eq 0 ] || { [ "$#" -eq 1 ] && [ "$1" = "--publish" ]; }; then
    echo "[run] full national pipeline: geographic batches + validated web build"
    exec bash ./run_overnight.sh "$@"
fi

.venv/bin/python -m fcc_audit.cli "$@"

echo
echo "Done. Outputs are in data/outputs/"
echo "Serve the web app:  python -m fcc_audit.cli serve"
echo "then open http://127.0.0.1:8000"
