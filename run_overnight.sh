#!/usr/bin/env bash
# Process all state batches unattended, then publish one validated national bundle.
# Usage:
#   ./run_overnight.sh              # analyze + build-web (no git push)
#   ./run_overnight.sh --publish    # same, then commit + push web/public/data
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
PY=".venv/bin/python"
LOG="overnight_$(date +%Y%m%d_%H%M%S).log"
PUBLISH=0
for arg in "$@"; do
  if [[ "$arg" == "--publish" ]]; then
    PUBLISH=1
  fi
done

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

# National Redshift prefetch once (51 FIPS × 2 vintages, parallel shared scans).
# Batch loops then --skip-prefetch so overlapping neighbor states are not re-scanned.
echo "" | tee -a "$LOG"
echo "=== National prefetch (download) @ $(date) ===" | tee -a "$LOG"
if ! $PY -m fcc_audit.cli download 2>&1 | tee -a "$LOG"; then
  echo "NATIONAL PREFETCH FAILED; refusing overnight batches" | tee -a "$LOG"
  exit 1
fi

FAILED_BATCHES=()
BATCH_TIMES=()
BATCH_IDX=0
N_BATCHES=${#BATCHES[@]}
for STATES in "${BATCHES[@]}"; do
  BATCH_IDX=$((BATCH_IDX + 1))
  echo "" | tee -a "$LOG"
  echo "=== BATCH $BATCH_IDX/$N_BATCHES $STATES @ $(date) ===" | tee -a "$LOG"
  T0=$(date +%s)
  # Caches warm from national download; unit-level --workers 6 for CPU analyze.
  if ! $PY -m fcc_audit.cli run --states "$STATES" --workers 6 --skip-prefetch 2>&1 | tee -a "$LOG"; then
    echo "BATCH FAILED: $STATES (continuing)" | tee -a "$LOG"
    FAILED_BATCHES+=("$STATES")
    continue
  fi
  T1=$(date +%s)
  ELAPSED=$((T1 - T0))
  BATCH_TIMES+=("$ELAPSED")
  # Running ETA from mean completed batch duration.
  if ((${#BATCH_TIMES[@]} > 0)); then
    SUM=0
    for t in "${BATCH_TIMES[@]}"; do SUM=$((SUM + t)); done
    AVG=$((SUM / ${#BATCH_TIMES[@]}))
    REMAIN=$((N_BATCHES - BATCH_IDX))
    ETA=$((AVG * REMAIN))
    echo "=== Completed $STATES in ${ELAPSED}s @ $(date) (ETA remaining ~${ETA}s) ===" | tee -a "$LOG"
  fi
done

if ((${#FAILED_BATCHES[@]})); then
  echo "NATIONAL RUN INCOMPLETE: refusing final web build; failed batches: ${FAILED_BATCHES[*]}" | tee -a "$LOG"
  exit 1
fi

echo "=== Overnight run finished $(date) ===" | tee -a "$LOG"
$PY -m fcc_audit.cli build-web 2>&1 | tee -a "$LOG"

echo "Done. Serve with: cd web && python3 -m http.server 8000" | tee -a "$LOG"

if [[ "$PUBLISH" -eq 1 ]]; then
  echo "=== --publish: committing and pushing web bundle ===" | tee -a "$LOG"
  git add web/public/data
  git commit -m "Final overnight web bundle $(date +%Y-%m-%d)" || true
  git push origin HEAD 2>&1 | tee -a "$LOG"
else
  echo "Skipping git publish (pass --publish to commit + push web/public/data)." | tee -a "$LOG"
fi
