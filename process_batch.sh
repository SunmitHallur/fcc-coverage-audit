#!/usr/bin/env bash
# Process one state batch and print next steps.
# Does NOT rebuild the web bundle (that would wipe other batches' site data).
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
.venv/bin/python -m fcc_audit.cli run --states "$STATES" --workers 6
echo ""
echo "=== Done. Next steps ==="
echo "  # After ALL national batches succeed:"
echo "  python -m fcc_audit.cli build-web"
echo "  # Preview partial: python -m fcc_audit.cli build-web --allow-incomplete"
echo "  python -m fcc_audit.cli serve"
