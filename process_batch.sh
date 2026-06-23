#!/usr/bin/env bash
# Process one state batch, rebuild the web bundle, and print next steps.
# Usage: ./process_batch.sh "01,02"   or   ./process_batch.sh 01,02,48
set -euo pipefail
cd "$(dirname "$0")"
STATES="${1:-01,02}"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
export PYTHONPATH=src
echo "=== Processing states: $STATES ==="
.venv/bin/python -m fcc_audit.cli run --states "$STATES" --cleanup-raw --build-web
echo ""
echo "=== Done. Next steps ==="
echo "  git add web/public/data config/pipeline.yaml"
echo "  git commit -m \"Add batch results for states $STATES\""
echo "  git push   # Vercel auto-deploys web/"
