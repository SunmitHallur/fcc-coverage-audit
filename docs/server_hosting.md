# Hosting on Mike's Linux server (Docker cook)

Overnight on the laptop **preps ingredients**. A Docker container on the
server is the **cook**: when someone clicks a county, Python reads those
files and returns JSON for that county only.

```
Laptop (hours)                         Server (seconds)
──────────────                         ────────────────
Redshift → coverage parquet            browser clicks Reno
         → sites parquet         →     GET /api/county?...
         → scored parquet              cook slices one county
TIGER GeoPackage                       returns JSON
build-web → map shell (records,
            counties, meta, towers)
```

Towers stay overnight. A click never talks to Redshift and never re-runs
site inference. Node-RED is a different app; this is its own container on
the same Docker host.

Apache still terminates TLS and reverse-proxies to `127.0.0.1:8000` — the
pattern Mike already validated.

## What to copy (ingredients, not the warehouse)

Do **not** copy `data/raw/` (tens of GB). Copy:

| Path | Role |
|------|------|
| `web/` | Map UI. Skip `web/public/data/details/` — the cook builds those on click. |
| `web/public/data/records/`, `meta.json`, `counties.geojson`, `towers/` | Choropleth + tower overlay (small). |
| `data/processed/<backend>_<current>_vs_<prior>/coverage/` | Per-state hex parquet. |
| `data/processed/.../sites/` | Precomputed towers. |
| `data/processed/.../scored/` | Rank / tower counts. |
| `data/interim/tl_us_county.gpkg` | County polygons. |
| `src/`, `config/`, `Dockerfile`, `docker-compose.yml` | The cook. |

From the laptop, after a national run:

```bash
python -m fcc_audit.cli build-web --no-details
./deploy/sync-ingredients.sh user@mike-host /var/www/fcc-coverage-audit
```

`--no-details` skips writing thousands of pre-plated `details/*.json` files.
Vercel/static hosting still wants those files; omit `--no-details` for that path.

## Docker (same idea as Mike's other containers)

On the server, in the project directory:

```bash
cd /var/www/fcc-coverage-audit
docker compose up -d --build
docker compose logs -f cook
```

Compose publishes **only** `127.0.0.1:8000`. It is not on the public NIC.
Apache (or another reverse proxy) is the front door.

```bash
curl -s http://127.0.0.1:8000/api/health
curl -s 'http://127.0.0.1:8000/api/county?geoid=20155&provider=130077&service=5G-NR%207/1' | head
```

Without Docker, the same cook is:

```bash
export PYTHONPATH=src
python -m fcc_audit.cli serve --host 127.0.0.1 --port 8000
```

## Apache reverse proxy

Mike's validated pattern — proxy all paths to the cook:

```apache
<VirtualHost *:443>
    ServerName coverage.example.com

    ProxyPreserveHost On
    ProxyPass        / http://127.0.0.1:8000/
    ProxyPassReverse / http://127.0.0.1:8000/

    <IfModule mod_deflate.c>
        AddOutputFilterByType DEFLATE application/json
        AddOutputFilterByType DEFLATE application/geo+json
    </IfModule>
</VirtualHost>
```

```bash
sudo a2enmod proxy proxy_http deflate
sudo systemctl reload apache2
```

## Verify

Open the site through Apache and confirm:

- Map loads with four-tier county colors
- Clicking a county shows hex detail (cook JSON; `source: "api"`)
- `/api/health` reports `"ingredients": true`

If ingredients are missing, the UI falls back to static `details/*.json`
when those files exist.

## Notes

- Hex drawing is still in the browser (`web/vendor/h3-js.js` + MapLibre).
  The container does not need to talk to the FCC or to Redshift.
- Basemap tiles still need outbound HTTPS to OSM / CARTO.
- After each overnight run: `build-web --no-details`, rsync ingredients,
  `docker compose restart` (rebuild only if `requirements-serve.txt` changed).
- Keep this container off Node-RED's compose file. Different image, same host.
