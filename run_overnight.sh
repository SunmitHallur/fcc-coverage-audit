#!/usr/bin/env bash
# Process all state batches unattended. Each batch: run -> build-web -> commit -> push.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
PY=".venv/bin/python"
LOG="/tmp/fcc_overnight.log"

BATCHES=(
  "01,02,03,04,05"
  "06,08,09,10,11"
  "12,13,15,16,17"
  "18,19,20,21,22"
  "23,24,25,26,27"
  "28,29,30,31,32"
  "33,34,35,36,37"
  "38,39,40,41,42"
  "44,45,46,47,48"
  "49,50,51,53,54"
  "55,56"
)

echo "=== Overnight run started $(date) ===" | tee -a "$LOG"

for STATES in "${BATCHES[@]}"; do
  echo "" | tee -a "$LOG"
  echo "=== BATCH $STATES @ $(date) ===" | tee -a "$LOG"
  if ! $PY -m fcc_audit.cli run --states "$STATES" --cleanup-raw --build-web 2>&1 | tee -a "$LOG"; then
    echo "BATCH FAILED: $STATES (continuing)" | tee -a "$LOG"
    continue
  fi
  git add web/public/data src/fcc_audit/
  if git diff --cached --quiet; then
    echo "No web data changes for $STATES" | tee -a "$LOG"
    continue
  fi
  git commit -m "Add batch results for states $STATES" || true
  git push origin HEAD 2>&1 | tee -a "$LOG" || echo "push failed for $STATES" | tee -a "$LOG"
  echo "=== Pushed $STATES @ $(date) ===" | tee -a "$LOG"
done

echo "=== Overnight run finished $(date) ===" | tee -a "$LOG"
$PY -m fcc_audit.cli build-web 2>&1 | tee -a "$LOG"
git add web/public/data
git commit -m "Final overnight web bundle $(date +%Y-%m-%d)" || true
git push origin HEAD 2>&1 | tee -a "$LOG"
