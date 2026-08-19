#!/usr/bin/env bash
# Copy cook ingredients to Mike's Linux box. Does NOT copy data/raw (warehouse dumps).
#
# Usage:
#   ./deploy/sync-ingredients.sh user@host [/var/www/fcc-coverage-audit]
#
# Then on the server:
#   cd /var/www/fcc-coverage-audit && docker compose up -d --build
set -euo pipefail

HOST="${1:?usage: $0 user@host [dest]}"
DEST="${2:-/var/www/fcc-coverage-audit}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_REL="$(PYTHONPATH=src python3 -c "
from fcc_audit.config import load_config
from fcc_audit.serve import run_dir_for
cfg = load_config()
print(run_dir_for(cfg).relative_to(cfg.project_root))
")"
GPKG="$ROOT/data/interim/tl_us_county.gpkg"

if [ ! -d "$ROOT/$RUN_REL" ]; then
    echo "missing ingredients: $ROOT/$RUN_REL  (run the overnight pipeline first)" >&2
    exit 1
fi
if [ ! -f "$GPKG" ]; then
    echo "missing $GPKG (county GeoPackage)" >&2
    exit 1
fi

echo "syncing cook ingredients → $HOST:$DEST"
echo "  run dir: $RUN_REL"

ssh "$HOST" "mkdir -p '$DEST/data/interim' '$DEST/$RUN_REL'"

rsync -avz Dockerfile docker-compose.yml requirements-serve.txt "$HOST:$DEST/"
rsync -avz src/ "$HOST:$DEST/src/"
rsync -avz config/ "$HOST:$DEST/config/"
rsync -avz --exclude 'public/data/details' web/ "$HOST:$DEST/web/"
rsync -avz "$GPKG" "$HOST:$DEST/data/interim/tl_us_county.gpkg"
rsync -avz --delete "$ROOT/$RUN_REL/" "$HOST:$DEST/$RUN_REL/"

echo "done. on the server: cd $DEST && docker compose up -d --build"
