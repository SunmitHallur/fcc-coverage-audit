#!/usr/bin/env bash
# Process all state batches unattended, then publish one validated national bundle.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
PY=".venv/bin/python"
LOG="overnight_$(date +%Y%m%d_%H%M%S).log"
if [[ ! -x "$PY" ]]; then
  python3 -m venv .venv
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements.txt
fi

BATCHES=(
  "04,06,15,32"
  "02,16,41,53"
  "08,30,35,49,56"
  "20,31,38,40,46"
  "05,22,28,48"
  "17,19,27,29,55"
  "18,21,26,39,47"
  "01,12,13,37,45"
  "10,11,24,51,54"
  "09,23,25,33,34,36,42,44,50"
)

echo "=== Overnight run started $(date) ===" | tee -a "$LOG"

FAILED_BATCHES=()
for STATES in "${BATCHES[@]}"; do
  echo "" | tee -a "$LOG"
  echo "=== BATCH $STATES @ $(date) ===" | tee -a "$LOG"
  if ! $PY -m fcc_audit.cli run --states "$STATES" --cleanup-raw 2>&1 | tee -a "$LOG"; then
    echo "BATCH FAILED: $STATES (continuing)" | tee -a "$LOG"
    FAILED_BATCHES+=("$STATES")
    continue
  fi
  echo "=== Completed $STATES @ $(date) ===" | tee -a "$LOG"
done

if ((${#FAILED_BATCHES[@]})); then
  echo "NATIONAL RUN INCOMPLETE: refusing final web build; failed batches: ${FAILED_BATCHES[*]}" | tee -a "$LOG"
  exit 1
fi

echo "=== Overnight run finished $(date) ===" | tee -a "$LOG"
$PY -m fcc_audit.cli build-web 2>&1 | tee -a "$LOG"
git add web/public/data
git commit -m "Final overnight web bundle $(date +%Y-%m-%d)" || true
git push origin HEAD 2>&1 | tee -a "$LOG"
