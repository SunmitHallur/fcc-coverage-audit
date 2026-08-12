# Hosting the web bundle on a Linux server

The pipeline writes large raw/processed data under `data/` (tens of GB nationwide).
**Only `web/` needs to be deployed** to Mike's server (~7 GB budget). With the default
`--top-n 250` cap, per-county detail JSON stays under ~2 GB for Big 4 × 3 services.

## 1. Build locally

From the project root, after batches have been scored:

```bash
python -m fcc_audit.cli build-web
```

Optional flags:

- `--top-n 250` — max counties per provider×service with detail files and tier colors (default 250)
- `--render-pngs` — also emit server-side PNG maps (much larger; usually omit)

The bundle lands in `web/public/data/`:

| Path | Purpose |
|------|---------|
| `meta.json` | vintages, providers, services |
| `counties.geojson` | county boundaries for the map |
| `records/<pid>/<svc>.json` | per-county metrics (lazy-loaded; no monolithic `records.json`) |
| `details/<pid>/<svc>/<geoid>.json` | hex/site detail for top-N counties only |
| `towers/<pid>.json` | tower overlay points |

## 2. Upload only `web/`

```bash
rsync -avz --delete web/ user@server:/var/www/fcc-coverage-audit/web/
```

Use Mike's SSH user/host. `--delete` removes stale detail files after a rebuild with a lower `--top-n`.

## 3. Serve with Python on localhost:8000

The viewer is a zero-build static app: ES modules under `web/js/`, styles under
`web/css/`, and vendored MapLibre + h3 under `web/vendor/` (no CDN required).
Serve from the `web/` directory so relative imports resolve:

```bash
cd /var/www/fcc-coverage-audit/web
python3 -m http.server 8000 --bind 127.0.0.1
```

Bind to `127.0.0.1` only — Apache terminates TLS and reverse-proxies to this port.

### systemd unit (survives reboot)

`/etc/systemd/system/fcc-coverage-audit.service`:

```ini
[Unit]
Description=FCC Coverage Audit static web server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/fcc-coverage-audit/web
ExecStart=/usr/bin/python3 -m http.server 8000 --bind 127.0.0.1
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now fcc-coverage-audit
```

## 4. Apache reverse proxy

Mike's validated pattern (hello-world test) — proxy all paths to the Python server:

```apache
<VirtualHost *:443>
    ServerName coverage.example.com

    ProxyPreserveHost On
    ProxyPass        / http://127.0.0.1:8000/
    ProxyPassReverse / http://127.0.0.1:8000/

    # Compress JSON hex payloads (~6–10× over the wire)
    <IfModule mod_deflate.c>
        AddOutputFilterByType DEFLATE application/json
        AddOutputFilterByType DEFLATE application/geo+json
    </IfModule>
</VirtualHost>
```

Enable modules if needed:

```bash
sudo a2enmod proxy proxy_http deflate
sudo systemctl reload apache2
```

## 5. Verify on the server

```bash
curl -sI http://127.0.0.1:8000/ | head
curl -s http://127.0.0.1:8000/public/data/meta.json | python3 -m json.tool | head
```

Open the site through Apache and confirm:

- Map loads with four-tier county colors (top 50 / 51–100 / 101–150 / all others)
- Clicking a top-tier county opens hex detail; others show "Coverage detail not available"

## Notes

- Server-side Python **does not** need the `h3` package for static hosting; hex
  rendering is client-side via vendored `web/vendor/h3-js.js` + MapLibre layers
  (no CDN, no deck.gl).
- Upload the entire `web/` tree (`index.html`, `css/`, `js/`, `vendor/`, `public/`).
  Relative ES-module imports require serving from the `web/` directory root.
- Basemap tiles still need outbound HTTPS to OpenStreetMap / CARTO; if the
  server network blocks those, county fills still work but the basemap is blank.
- Rebuild and `rsync` after each pipeline run; do not copy `data/` to the server.
- To shrink further, lower `--top-n` (e.g. 100) before `build-web`.
