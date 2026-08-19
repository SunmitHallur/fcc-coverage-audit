# Agent changelog (coverage-audit)

Living log of intentional codebase changes so future agents can learn what
changed, why, and what was verified. Append new dated sections at the top.

---

## 2026-08-19 — Overnight crash on NaN tower counts

### Why

Western national batches finished hours of normalize, then died in
`explain_row`: `int(nan)` on `prior_towers`. Units with 0 inferred sites
omitted the column; concat with T-Mobile/Verizon filled NaN. `nan or 0`
is still NaN.

### What changed

- Always fill tower-count columns with 0 before score/explain.
- `_int0` in `explain_row`; save scores even if explanations fail.
- Cache per-unit sites/coverage parquet so a later crash can resume maps.

---

## 2026-08-18 — Ingredient-cook county API + Docker for Mike's host

### Why

Ship a cook, not thousands of pre-plated `details/*.json` files. Overnight
still infers towers and writes parquet; a click is `GET /api/county` that
slices one county from coverage parquet + TIGER GeoPackage. Mike already
runs apps in Docker (Node-RED is a separate container). Apache stays the
TLS front door on `127.0.0.1:8000`.

### What changed

- `src/fcc_audit/serve.py`: cook HTTP server (`/api/county`, `/api/health`).
- `python -m fcc_audit.cli serve` and `python -m fcc_audit.serve`.
- `web/js/app.js`: try `/api/county`, fall back to static details JSON.
- `build-web --no-details`: map shell only (records/meta/towers/counties).
- `Dockerfile`, `docker-compose.yml`, `requirements-serve.txt`,
  `deploy/sync-ingredients.sh`.
- Docs: README + `docs/server_hosting.md`.

### Verify

```bash
PYTHONPATH=src pytest tests/test_serve.py tests/test_web_bundle.py -q
python -m fcc_audit.cli serve   # click a county; Network tab shows /api/county
# on Mike's box, after rsync:
docker compose up -d --build
curl -s http://127.0.0.1:8000/api/health
```

---

## 2026-08-12 — Redshift mrgd + real minsignal (292/291)

### Why

Warehouse tables `bbmap_mob_bb_mrgd_hex9_inter_<build>` already expose per-provider
rows with real **`minsignal`** (dBm). The pipeline had been reading binary
`bbmap_mobile_bb_tech_hex9s_*` and `_write_hex_cache` forced `signal_dbm=0.0`, so
detail maps used distance-estimated pseudo-RSRP.

Process ids (Broadband Map Processing): **292 = D25**, **291 = J25** (replacing
older 277/279 for the same filing windows).

### What changed

- `config/pipeline.yaml`: `hex_table_format: mrgd_inter`,
  `hex_table_prefix: bbmap_mob_bb_mrgd_hex9_inter_`, `service_mrgd_keys`,
  vintages `current: "292"` / `prior: "291"`.
- `RedshiftSource`: direct + shared prefetch SELECT `h3index, minsignal` (and
  fan-out columns); caches persist real `signal_dbm`; max minsignal on hex dupes;
  provider discovery prefers `DISTINCT providerid` from mrgd (merged_all fallback).
- `probe_redshift_schema` / doctor: probes mrgd columns; merged table not required
  for mrgd format.
- Tests updated for mrgd SQL; `_write_hex_cache` signal round-trip covered.
- Docs: README, methodology, coverage viewer memo.

Legacy `hex_table_format: tech_hex9s` path remains for binary snapshots.

### Verify (FCC laptop)

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
Remove-Item -Recurse -Force .\data\processed\* -ErrorAction SilentlyContinue
python -m fcc_audit.cli --backend redshift doctor
python -m fcc_audit.cli --backend redshift download --states 39 --prefetch-workers 2
python -m fcc_audit.cli --backend redshift run --states 39 --workers 2 --skip-prefetch --build-web
```

Confirm a top county detail JSON has varied hex signal encodings (not one flat
value) and the UI does not show the “estimated from distance” note. Old
`data/raw/277|279` caches are unused; new runs write `data/raw/291|292`.

---

## 2026-08-12 — Fix unreadable “inverted” web theme (`:root` missing)

### Cause

`web/css/styles.css` lost the opening `:root {` when styles were split out of
`index.html`. CSS variables (`--ink`, `--bg`, `--panel`, …) never applied, so
hardcoded dark panels (e.g. `#detail`) kept dark backgrounds while text fell
back to browser default black → unreadable “inverted” look. National data was
fine; this was CSS only.

### Fix

Restore `:root { … }`, set `color-scheme: dark`, and force `#detail { color: var(--ink) }`.

### Laptop apply (no re-run needed)

Replace `web/css/styles.css` from git `master`, hard-refresh the browser
(`Ctrl+Shift+R`).

---

## 2026-08-12 — Windows/OneDrive build-web rmtree lock

### Context

National overnight on FCC Windows laptop completed all 10 batches (27,933
provider-county rows, 390 flagged) then failed at final `build-web` with:

`PermissionError: [WinError 5] Access is denied: ...\web\public\data\records\130077`

Project lived under OneDrive (`...\OneDrive - FCC\Downloads\...`).

### Fix

- `webbundle.write_web_bundle` now deletes `records/` / `details/` / `towers/`
  via `_rmtree_retry` (chmod + retry on `PermissionError`).
- Clearer error if locks persist after retries.

### User recovery (no re-analyze needed)

Batches already wrote scored parquet; only re-run:

```powershell
$env:PYTHONPATH = "src"
python -m fcc_audit.cli --backend redshift build-web
```

If still locked: pause OneDrive, close Explorer preview of `web\public\data`,
manually delete `web\public\data\records`, `details`, `towers`, retry.

---

## 2026-08-11 — Pre-commit QA of uncommitted modular web + Redshift overnight

### Context

User built an uncommitted working tree (modular `web/` viewer + pipeline
hardening) and asked for full QA before commit: UI/UX, code capability,
Redshift download path, and Linux server deploy readiness.

### What the uncommitted tree already changed (pre-QA)

- Split monolithic `web/index.html` into `web/css/styles.css`, `web/js/*.js`
  ES modules, and local `web/vendor/` (MapLibre + h3) — no unpkg/Google Fonts
  CDN for app libs.
- Pipeline/Redshift hardening across `acquire.py`, `cli.py`, scoring, webbundle,
  overnight launchers, RHEL 3.9 lockfile path, docs.

### QA findings (verified)

| Area | Result |
|------|--------|
| `pytest` | **85 passed**, 1 skipped |
| Static web assets via `python3 -m http.server` from `web/` | All core paths **200** |
| Split records integrity | AT&T/Verizon OK; **T-Mobile `5G-NR35-3` missing** (404) in current bundle |
| Bundle completeness | `meta.incomplete: true`, states `"20"` only, **no UScellular / no 4G LTE**, stale `generated_at` 2026-06-26 |
| Redshift live warehouse | **Not tested** — no `.env` on this machine |
| Linux Apache deploy | Docs updated; static layout OK; basemap tiles still need outbound OSM/CARTO |

### Bugs fixed in this QA pass

1. **`run_overnight.sh` / `run_overnight.ps1` never passed `--backend redshift`**
   while config defaults to `files`. Overnight “National Redshift prefetch”
   silently used the files backend. Fixed: default `FCC_AUDIT_BACKEND=redshift`
   (override with env / `-Backend`).
2. **`cmd_download` returned 0 when 0 files were cached** (files backend), so
   overnight could proceed after a hollow prefetch. Now exits `1` if `n_files==0`.
3. **Viewer CDN error copy** still mentioned unpkg after vendoring.
4. **National “Show inferred towers” colors** used purple/cyan; legend + detail
   maps use green/orange/slate. Unified via `towerColor()` / `TOWER_COLORS`.
5. **ASR note dropped** when detail JSON refreshed (`setTowerCompare` missing
   `rec` arg).
6. **Partial-run banner wiped** by `fillDropdowns()` overwriting `#vintage-label`.
7. **Missing split service files** (e.g. T-Mobile 5G-NR35-3) still appeared in
   the dropdown when `meta.provider_services` absent. Probe with `HEAD` and
   cache into `meta.provider_services`.
8. **`docs/server_hosting.md`** incorrectly said hex rendering used deck.gl;
   updated for vendored h3 + MapLibre and full `web/` upload notes.
9. **README** claimed “Redshift backend (default)” while YAML defaults to
   `files`; corrected and noted overnight forces Redshift.

### Files touched by this QA pass

- `web/js/app.js`
- `run_overnight.sh`
- `run_overnight.ps1`
- `src/fcc_audit/cli.py` (`cmd_download` fail-fast)
- `tests/test_batch_integrity.py` (overnight launcher contract expects `--backend`)
- `docs/server_hosting.md`
- `README.md`
- `docs/AGENT_CHANGELOG.md` (this file)

### Still open / needs human input

1. **Live Redshift**: copy `.env.example` → `.env` with warehouse creds, then:
   `python -m fcc_audit.cli --backend redshift doctor`
   `python -m fcc_audit.cli --backend redshift download --states 20 --prefetch-workers 2`
2. **Stale/partial web bundle**: rebuild after a real batch (`build-web`) so
   `provider_services`, Big 4, and all services land in `meta.json`.
3. **Windows overnight**: same backend fix applied in `.ps1`; not re-run on Windows here.
4. **Basemap offline**: locked-down servers that block OSM/CARTO will show blank
   basemap (choropleth/hex still work).
5. Soft skips on missing filings (`skipped_no_filing`) can still yield “successful”
   empty analysis units — intentional for sparse providers, but dangerous if *all*
   units skip; watch overnight logs.
6. Vintage process-id ordering (`292` current / D25 vs `291` prior / J25) must
   match warehouse metadata; confirm with `list-vintages` once creds exist.

### Deploy checklist (Linux / Mike’s server)

```bash
# build on analysis machine
python -m fcc_audit.cli --backend redshift build-web   # or after overnight

# upload ONLY web/
rsync -avz --delete web/ user@server:/var/www/fcc-coverage-audit/web/

# on server
cd /var/www/fcc-coverage-audit/web
python3 -m http.server 8000 --bind 127.0.0.1
# Apache reverse-proxies :443 → 127.0.0.1:8000 (see server_hosting.md)

curl -sI http://127.0.0.1:8000/ | head
curl -s http://127.0.0.1:8000/public/data/meta.json | python3 -m json.tool | head
```

Verify in browser: vendor libs load (no CDN error), provider/service dropdowns,
county click → hex compare, triage keys `j/k/a/r/n`, tower toggle colors match
legend, incomplete/partial banner if applicable.

### Feature inventory exercised (viewer)

- Provider / service selects (split lazy load)
- County search, flagged-only toggle, show-all list
- National choropleth + click-to-detail
- Detail: stats, tower compare, ASR note, flag-math, bullets/rec
- Hex prior/current MapLibre maps (no PNGs in this bundle)
- Swipe/download toolbar only when raster PNGs present (correctly hidden here)
- Triage accept/reject/needs-info + localStorage + export JSON + print
- Keyboard nav / Esc close
- Collapsible map legend
- Incomplete / partial-run header text
