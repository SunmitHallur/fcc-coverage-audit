# FCC Mobile Coverage-Change Audit Pipeline

A re-runnable pipeline that compares two 6-month vintages of FCC Broadband Data
Collection (BDC) **mobile coverage — 3G, 4G LTE, and 5G-NR — for every provider**,
infers approximate cell-site locations, attributes coverage growth to **new
towers** vs **expanded existing towers**, and **flags the provider × county ×
service pairs the FCC should physically test** because growth looks like it may
be gamed (large coverage jumps claimed from existing sites without a
corresponding build-out).

Each `(technology, speed tier)` is analyzed as its own unit — because the FCC
selects per service type (e.g. "Verizon 5G 7/1") — so a flag is always tied to a
specific provider, county, technology, and speed tier.

**Why it exists:** today this county-by-county selection is done manually by
consultants billed at ~$170–200/hour. This pipeline automates that selection - it
produces, every 6 months when new data drops, the ranked list of provider/county
pairs to review (`data/outputs/selected_counties_*.csv`), the deliverable the
consultants currently hand-produce.

The geography mirrors the FCC's own mobile-audit process (H3 resolution-8
hexagons within a county), so outputs map directly onto how the FCC already
verifies coverage on the ground.

---

## Why this approach

The FCC publishes the real underlying coverage as **vector polygons** with a
modeled signal-strength ("heat map") attribute - not just the green map image.
So the pipeline analyzes the actual geometry/signal data (accurate), and only
uses the rendered green tiles as a **cross-check**, falling back to computer
vision on the image where the two disagree.

All analysis is **local**. Site inference uses coverage geometry, and ranking
uses a deterministic monotone score with a maximum 0.25 influence from any
single feature. Core scoring has **no external LLM calls**. Optional
`case-files --llm` narratives (local Ollama / Gemini) are off by default.

```
acquire ─► normalize ─► (reconcile) ─► change-detect ─► infer sites
        ─► attribute (new vs expanded) ─► score / rank ─► CSV + report + dashboard
```

Formulas for each stage, why they are used, how the knobs were chosen, and the
papers those methods come from: [Mathematics of each step](#mathematics-of-each-step)
and the longer [docs/formulas.md](docs/formulas.md).

---

## Quick start

Supports **Python 3.9 through 3.14**. Same install command everywhere — pins are
version-gated in `requirements.txt`. Verified on macOS/Windows (3.13/3.14) and
with a real 3.9.25 venv for RHEL 9 handoff.

### One-time (clone / first machine)

Do this once after `git clone` or the first checkout. Skip on later days — you
do **not** recreate the venv or rewrite `.env` after a normal `git pull`.

#### macOS / Windows / modern Linux (Python 3.11+)

```bash
git clone <repo-url>
cd fcc-coverage-audit
python3 -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env                 # Windows: copy .env.example .env
# fill REDSHIFT_* only if you will use --backend redshift
```

#### RHEL 9 / EL9 (stock Python 3.9.25)

Use the locked tree so you get the exact dependency versions tested for handoff:

```bash
git clone <repo-url>
cd fcc-coverage-audit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-py39.lock.txt
cp .env.example .env   # then fill REDSHIFT_* from your warehouse INI
```

Map a typical Redshift INI to `.env`:

```text
REDSHIFT_HOST=redshift-cluster-….amazonaws.com
REDSHIFT_DB=db_fcc_bdc
REDSHIFT_USER=…
REDSHIFT_PASSWORD=…
```

**Offline wheelhouse** (if the server cannot reach PyPI — still one-time):

```bash
# on a connected machine with the same Python 3.9
pip download -r requirements-py39.lock.txt -d wheelhouse \
  --python-version 3.9 --platform manylinux2014_x86_64 --only-binary=:all:
# copy wheelhouse/ + requirements-py39.lock.txt to the server, then:
pip install --no-index --find-links wheelhouse -r requirements-py39.lock.txt
```

### Every time (after `git pull`)

```bash
git pull
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt      # no-op if unchanged; picks up new pins
                                     # RHEL 9: pip install -r requirements-py39.lock.txt
export PYTHONPATH=src                # Windows: $env:PYTHONPATH="src"
```

Then pick the command you actually need:

| Goal | Command |
|------|---------|
| Preflight | `python -m fcc_audit.cli doctor` (add `--backend redshift` on the warehouse) |
| Smoke one state | `--backend files run --states 20 --workers 2` (cached parquet) or `--backend redshift run --states 20 --workers 2` |
| Full national | `./run_overnight.sh` (Linux) or `.\run_overnight.ps1` (Windows) |
| View the map | `python -m fcc_audit.cli serve` → http://127.0.0.1:8000 |

`doctor` checks the interpreter, heavy imports, the **active** backend
(`files` / `fcc` / `redshift` / `fixture`), CPU count (suggests `--workers`),
and free disk. For Redshift it also probes schema access — distinguishing
SQLSTATE `3F000` ("schema does not exist", usually wrong `REDSHIFT_DB`) from a
missing vintage table — and verifies the mrgd hex table columns
(`h3index`, `providerid`, `technology`, `mindown`, `minsignal`, …) are
readable. Schema `bdc_dataplatform` lives in **`db_fcc_bdc`**, not
`db_fcc_bdp_ext`.

You do **not** need to re-run the national pipeline after every pull. Pull +
`pip install` is enough if you only want the latest code; `run` / overnight
only when you want new scores. `serve` cooks one county on click from processed
parquet (see [docs/server_hosting.md](docs/server_hosting.md) for Docker on
Mike's box).

### 1. Try it offline first (no FCC / Redshift access needed)

**Default backend is `files`** — it reads hex parquet under `data/raw/` that an
FCC machine previously prefetched with `--backend redshift`. Ship `data/raw/`
and the public can run with no warehouse credentials.

For a fully synthetic demo (no real coverage files):

```bash
python -m fcc_audit.cli --backend fixture make-fixtures
python -m fcc_audit.cli --backend fixture run
```

Expected result: **T-Mobile / Charlie County is the #1 flagged** county
("100% of growth claimed from existing sites; coverage up 643%"), while AT&T's
genuine new-tower build in the same county is **not** flagged.

### 2. No API key needed

Downloads use the **same public endpoint the broadbandmap.fcc.gov "Download"
buttons call** (`/nbm/map/api/getNBMDataDownloadFile/...`), so **no account or
API token is required** — just open network access to `broadbandmap.fcc.gov`.
The pipeline sends the browser-like `User-Agent` / `Referer` / `Origin` headers
(configured in `config/pipeline.yaml`) that the FCC server expects.

> If you previously pasted an FCC token anywhere (env vars or `pipeline.yaml`),
> you can remove it — it is no longer used.

### 3. Run on real FCC data

**Downloading is fully automated — you never fetch files by hand.** Backends:

| Backend | Who | What it reads |
|---------|-----|---------------|
| `files` (default) | Public / offline | Cached hex parquet under `data/raw/` |
| `redshift` | FCC staff | Warehouse `bdc_dataplatform` hex tables |
| `fcc` | Anyone with internet | Public National Broadband Map downloads |
| `fixture` | CI / demos | Synthetic GeoJSON under `data/fixtures/` |

With `providers` set to the Big 4 (AT&T, T-Mobile, Verizon, UScellular) by default,
the pipeline analyzes each `(provider, service)` across the configured states.

**Easiest (one command, handles venv + install + run):**

```bash
./run.sh                 # macOS/Linux        |   run.bat   (Windows: double-click)
```

With no arguments the launcher runs the **overnight national path** (10 state
batches + final `build-web`). It does **not** pass `--cleanup-raw` — Redshift
neighbor-state caches must be kept across batches. Pass `--cleanup-raw`
explicitly only for FCC polygon runs when you need to bound disk. You can also
pass through any subcommand, e.g. `./run.sh download` or
`./run.sh run --states 01,02`.

**Or drive the CLI directly:**

```bash
python -m fcc_audit.cli list-vintages          # available vintages
python -m fcc_audit.cli download                 # PRE-FETCH all raw files only (resumable)
python -m fcc_audit.cli run                      # download + analyze, Big 4 + all services
python -m fcc_audit.cli run --states 01,02       # one state batch (keep Redshift caches)
python -m fcc_audit.cli run --states 01,02 --cleanup-raw   # FCC only: bound disk
python -m fcc_audit.cli build-web                # rebuild web bundle from ALL batches
python -m fcc_audit.cli build-web --no-details   # map shell only; clicks cooked by serve
python -m fcc_audit.cli serve                    # local map + on-demand county JSON
python -m fcc_audit.cli run --current "292" --prior "291"   # Redshift build ids (D25/J25)
```

Vintages depend on backend: Redshift uses hex-table build ids (e.g. `"292"`);
FCC uses `filing_subtype` labels (e.g. `"December 31, 2025"`). `download` and
`run` are both resumable — already-downloaded files and interim parquet are
cached and skipped, so an interrupted run picks up where it left off.

**National completeness** means **50 states + DC (51 FIPS)**. Puerto Rico and
other territories are out of scope unless you add them explicitly.

### Start small — validate before the national run

Real mobile coverage is **per state × provider × service**. On the FCC polygon
backend, downloads are rate-limited (~10/min) so a full national run is
**multi-day**. On Redshift (`--backend redshift`), prefer the overnight
launcher for a full national run targeting under ~10 wall-clock hours.
Confirm everything works on a small scope first by editing
`config/pipeline.yaml`:

```yaml
analysis:
  services:
    - { label: "5G-NR 7/1", desc: "5G-NR (7/1 Mbps)" }   # one service
  providers:
    - { id: 131425, name: "Verizon" }                     # one provider
  states: ["48"]                                           # Texas only
```

Then `python -m fcc_audit.cli run --states 48`. Once that produces a sane `selected_counties_*.csv`,
widen `services`/`states` for the full run.

Or use the batch helper (analyzes one batch; does **not** rebuild the web site):

```bash
./process_batch.sh "01,02"          # macOS/Linux
process_batch.bat 01,02             # Windows
# After ALL batches succeed:
python -m fcc_audit.cli build-web
```

### Incremental processing → live website

Process data in manageable batches, then build the national web bundle once:

```bash
# 1. Process a batch (downloads, analyzes, saves parquet — does NOT wipe web/)
./process_batch.sh "01,02,04,05"

# 2. Repeat with remaining state batches…

# 3. After all batches succeed, merge into the web bundle:
python -m fcc_audit.cli build-web
#    → web/public/data/meta.json
#    → web/public/data/records/<provider_id>/<service>.json
#    → web/public/data/details/…

# 4. Optionally publish (overnight --publish, or manually):
git add web/public/data
git commit -m "National web bundle"
git push
```

**What goes in git vs what doesn't:**

| In git (small) | Not in git (huge) |
|---|---|
| Code, config, `web/public/data/` bundle | Raw FCC downloads (`data/raw/`) |
| Simplified county boundaries in the bundle | Interim hex parquet (`data/interim/`) |
| | Batch scored parquet (`data/processed/`) |

Anyone can regenerate raw data with `python -m fcc_audit.cli download` — no API token needed.

> Note: the FCC public API filters requests by User-Agent and rate-limits
> aggressive clients; both are handled in config (`source.fcc`).

#### Data volume & time — read this before a national run

**FCC polygon backend:** thousands of file downloads at ~10/minute → multi-day,
and **0.5–1 TB** of raw files. **Redshift backend** (opt-in via
`--backend redshift` or `./run_overnight.sh`, which defaults to Redshift): shared
`(vintage, state)` hex scans — the recommended path for a full national run
under ~10 hours. Config `source.backend` defaults to `files` for offline/public
use of prefetched parquet. Options, in order of preference:

1. **Redshift (best).** Query server-side and never download raw geometry.
   Use `./run.sh` / `run_overnight.sh` (passes `--backend redshift`; national
   `download` once, then 10 geo batches + `--workers 6 --skip-prefetch`).
2. **`--cleanup-raw` + scope down (FCC only).** Deletes big raw files after each
   service so peak disk stays small. Do **not** use on Redshift overnight
   (neighbor-state caches must persist).
3. **Run per-state / per-provider in batches.** The interim parquet cache makes
   re-runs cheap and the download is resumable.

If you only care about the FCC's current funding focus, keep it to **5G-NR**
(both tiers) + **4G LTE**; that is a fraction of the full volume.

### 4. View results

```bash
# Ranked priority list + summary
open data/outputs/priority_ranking_*.csv
open data/outputs/summary_*.md

# Interactive web app — cook one county on click (parquet + GeoPackage)
python -m fcc_audit.cli serve
# then open http://127.0.0.1:8000
# static files only: cd web && python3 -m http.server 8000

# Legacy dashboard (point markers)
cd dashboard && python3 -m http.server 8000
```

### 5. Deploy

**Mike's Linux box (recommended):** ingredient-cook in Docker. Overnight on the
laptop writes parquet; a container cooks one county on click. Same Apache
reverse-proxy pattern as his other Docker apps. See
[docs/server_hosting.md](docs/server_hosting.md).

```bash
python -m fcc_audit.cli build-web --no-details
./deploy/sync-ingredients.sh user@host
# on the server: docker compose up -d --build
```

**Vercel (static only):** the `web/` folder can still be hosted as files. County
clicks then need pre-plated `details/*.json` (`build-web` without `--no-details`).

1. Push this repo to GitHub (or your FCC private repo).
2. Go to [vercel.com](https://vercel.com) → **Add New Project** → import the repo.
3. Leave **Root Directory** blank (repo root). The root `vercel.json` sets
   `outputDirectory` to `web` automatically. No build command needed.
4. Every `git push` that updates `web/public/data/` auto-redeploys the site.

```bash
npx vercel --prod    # from repo root (vercel.json points at web/)
```

The site loads `public/data/counties.geojson` (county boundaries),
`public/data/meta.json`, and split `public/data/records/<provider_id>/<service>.json`
(provider × service × county metrics + explanations; see `meta.use_split_records`).
Select a provider from the dropdown to see coverage-change shading and flagged
counties highlighted in red.

### 6. Work laptop (local website, no Vercel)

On a locked-down work machine, skip Vercel. After overnight, view the map with
`python -m fcc_audit.cli serve` (a click asks Python to slice one county from
parquet + the TIGER GeoPackage). Mike's public site uses the same cook inside
Docker — see [docs/server_hosting.md](docs/server_hosting.md).

#### One-time (first clone only)

```powershell
git clone <your-repo-url>
cd fcc-coverage-audit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env          # fill in Redshift creds (see below)
```

**Redshift via DBeaver:** DBeaver is for browsing the warehouse and testing SQL.
The pipeline connects with the same credentials using `redshift-connector` (now
in `requirements.txt`). In DBeaver, open your connection → *Edit* and copy host,
database, user, and password into `.env`:

```
REDSHIFT_HOST=...
REDSHIFT_DB=...
REDSHIFT_USER=...
REDSHIFT_PASSWORD=...
```

The backend reads the warehouse's **per-provider H3 res-9 intersection tables**
(`<schema>.bbmap_mob_bb_mrgd_hex9_inter_<build>`) — one row per
`(h3index, providerid, technology, mindown, environmnt)` with real
**`minsignal`** (dBm). Cached parquet is `h3` + `signal_dbm` (= `minsignal`).
Since the warehouse already did the H3 indexing, the pipeline **skips polygon
polyfill**. Detail maps use warehouse dBm bands; distance-based signal estimate
is only a fallback when caches are still flat (legacy `tech_hex9s`).

Configure `source.redshift` in `config/pipeline.yaml`: `schema`,
`hex_table_format: mrgd_inter`, `hex_table_prefix`, `environment` (0/1), and
`service_mrgd_keys` (e.g. `"5G-NR (7/1 Mbps)" → technology 500 / mindown 7`).
The table-name suffix is a Broadband Map Processing **process id** and serves as
the vintage token — default `analysis.vintages` are **`292` (D25)** and
**`291` (J25)**. Set `hex_table_format: tech_hex9s` only if you need the older
binary 0/1 + `_prov` snapshots.

#### Every time (after `git pull`)

```powershell
git pull
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m fcc_audit.cli --backend redshift doctor
```

Then only the step you need:

```powershell
# Re-run analysis (hours). Skip if you only pulled code / CSS.
python -m fcc_audit.cli --backend redshift run --build-web
# or double-click run.bat  (full national overnight)

# View the map (seconds). Needs processed parquet from a prior run.
python -m fcc_audit.cli serve
# open http://127.0.0.1:8000
```

VS Code: open the folder, let it pick up `.vscode/settings.json` (sets
`PYTHONPATH=src`), then **Terminal → Run Task** for *Run pipeline (Redshift)*,
*Build web bundle*, or *Serve website locally*.

Click any county to see **before/after coverage maps** and **tower counts**.
`serve` cooks that JSON on click. Static `web/public/data/details/` is only
needed for Vercel / plain `http.server`.

---

## Outputs

| File | What it is |
|------|------------|
| `data/outputs/selected_counties_<cur>_vs_<prior>.csv` | **The automated selection list** - only the flagged provider/county pairs, ranked. This is the deliverable that replaces the manual consultant selection. |
| `data/outputs/priority_ranking_<cur>_vs_<prior>.csv` | One row per provider x county (all, not just flagged), ranked by priority, with risk features and plain-language explanation. |
| `data/outputs/summary_<cur>_vs_<prior>.md` | Human-readable top-25 review list. |
| `web/public/data/` | **Static web bundle** for Vercel: county GeoJSON, records JSON, meta, tower files. |
| `web/index.html` | Production MapLibre app: county choropleth, provider/service filters, flagged highlights, click-to-explain detail panel. |
| `dashboard/index.html` | Legacy point-marker dashboard. |

**Key flag signal:** `same_site_growth_share` - the fraction of new coverage
attributed to *existing* sites. High values mean a provider is claiming big
coverage gains without building, which is the primary gaming pattern.

---

## Configuration

Everything is in [`config/pipeline.yaml`](config/pipeline.yaml): providers,
technologies/speed tiers, vintages, H3 resolutions, the signal threshold and
match radius for site inference, reconciliation thresholds, and scoring weights.

**Scope.** Default config targets the **Big 4** providers. Set `analysis.providers: all`
to auto-discover every mobile provider from the catalog. `analysis.services` lists the
FCC mobile datasets to analyze, each identified by its catalog `desc`
(`technology_code_desc`). Use `--states 01,02` on the CLI to scope a batch without
editing the YAML, or set `analysis.states` to a FIPS list (e.g. `["48"]` for Texas).

### Switching to Redshift later

The data layer is pluggable. Once your AWS Redshift access is granted:

1. `pip install -r requirements.txt` (`redshift-connector` is included).
2. Copy `.env.example` → `.env` and fill `REDSHIFT_*` (same values as your DBeaver connection).
3. Set `source.redshift.schema` / `hex_table_format` / `hex_table_prefix` /
   `environment` / `service_mrgd_keys` in `config/pipeline.yaml`, and set two
   process ids in `analysis.vintages` (suffix on `..._mrgd_hex9_inter_<build>`,
   e.g. `292` / `291`).
4. Set `source.backend: redshift` (or run with `--backend redshift`).
5. Wipe stale `data/raw/277|279` (or re-download) so caches are under `291|292`
   with real `signal_dbm`.

Nothing else in the pipeline changes. **This is the recommended path for an
all-providers / all-technologies national run**, because it avoids pulling ~1 TB
of raw geometry onto the laptop — the heavy spatial filtering happens in the
warehouse and only the per-layer results come back.

---

## Testing

```bash
python -m pytest tests/ -q
```

The test suite runs the full pipeline offline and asserts the gaming case is
flagged #1 and the legitimate buildout is not.

## Validation against the FCC's own decisions

The FCC shared labeled example counties (J25 vs D25, 5G 7/1 Mbps) marked
*Selected* / *Not Selected* for manual review. They are encoded in
`config/pipeline.yaml` under `benchmark` and documented in
[`docs/validation_benchmark.md`](docs/validation_benchmark.md). Once real data is
available, check the pipeline reproduces those decisions:

```bash
python -m fcc_audit.cli benchmark
```

The scoring logic was refined from these examples: a county must add meaningful
*in-county* coverage (`scoring.min_added_km2_to_flag`) before it can be flagged,
which excludes near-empty counties (Edmunds, SD) and non-area-increasing signal
shifts (Menard, TX), matching the FCC's choices.

---

## Delivery / portability

- The code is tiny (well under 1 MB) and **zip-emailable**. `data/` and `.venv/`
  are git-ignored and must not be shipped (they can be tens of GB).
- If corporate mail strips the zip, push to **Bitbucket** and `git clone`.
- On the target laptop, just create a venv and `pip install -r requirements.txt`.

---

## Mathematics of each step

This is what the program actually computes. Full derivations, knobs, and paper
links: [`docs/formulas.md`](docs/formulas.md). Distances and areas use
**NAD83 / Conus Albers** ([EPSG:5070](https://epsg.io/5070)), not lat/lng.

**Papers vs this project.** H3, Albers, connected components, the discrete
distance transform, non-maximum suppression, k-d trees, and Hungarian
assignment are standard methods (citations below). Relative-core inference,
cloverleaf merge, same-site implausibility gates, and the bounded monotone
score are engineering choices calibrated on live Verizon 5G-NR 7/1 J25→D25
filings (382 counties), preferring missed flags over false flags. They are
not a published fraud model and not frozen to old FCC slide labels.

### 1. Index coverage on H3

Warehouse tables are already H3 resolution 9. Mean cell area
\(A_{\mathrm{hex}}=0.105332513\,\mathrm{km}^2\)
([H3 restable](https://h3geo.org/docs/core-library/restable/)). A polygon
backend includes a cell iff its **centroid** is inside the polygon
([`polygonToCells`](https://h3geo.org/docs/api/regions/)).

- Sahr, White & Kimerling (2003), geodesic DGGS: <https://doi.org/10.1559/152304003100011090>
- Uber H3 (Brodsky 2018): <https://www.uber.com/blog/h3/>
- FCC BDC: <https://www.fcc.gov/BroadbandData>
- Why meters in Albers: Snyder, USGS PP 1395, <https://doi.org/10.3133/pp1395> · PDF <https://pubs.usgs.gov/pp/1395/report.pdf>

### 2. Change detection

For hex sets prior \(P\) and current \(C\), with signal \(s\) in dBm:

\[
\mathrm{new}=C\setminus P,\quad
\mathrm{lost}=P\setminus C,\quad
\mathrm{upgraded}=\{h\in P\cap C: s_C-s_P\ge 5\,\mathrm{dB}\}.
\]

County \(g\):

\[
\mathrm{added\_km}^2 = (|C_g|-|P_g|)\,A_{\mathrm{hex}},\qquad
f_{\mathrm{added}}=\mathrm{added\_km}^2/\mathrm{area}(g),\qquad
\rho=\mathrm{added\_km}^2/\mathrm{prior\_km}^2
\]

(\(\rho=+\infty\) if prior area is 0 and something was added). The 5 dB upgrade
gate is an engineering noise floor, not a 3GPP spec. The **flag** uses this
**net** `added_km2`; growth *shares* below count only `new` and `upgraded`
hexes.

### 3. Infer sites from coverage shape

Sites are not published. This is **not** Okumura–Hata path loss; the filing
already is a heatmap. We invert lobe shape → pin.

**Relative core.** Walk this layer’s dBm bands from hottest to weakest until
\(\approx 35\%\) of hexes are kept (clamped to 18–60%). Binary layers use the
whole footprint and split on shape.

**Blobs.** 6-connected components on H3 (`grid_disk` radius 1) via DFS/BFS.
Drop blobs smaller than 35 hexes at res 9 (\(\approx 3.7\,\mathrm{km}^2\)).

- Rosenfeld (1970), connectivity in digital pictures: <https://doi.org/10.1145/321556.321570>

**Depth (flat/binary signal).** Discrete distance to the blob edge (BFS).
Interior maxima are candidate masts.

- Rosenfeld & Pfaltz (1966), distance transform: <https://doi.org/10.1145/321356.321357>
- Borgefors (1986): <https://doi.org/10.1016/S0734-189X(86)80047-0>

**Peaks.** Local maxima, then greedy non-maximum suppression:

\[
d_{\mathrm{NMS}}=\begin{cases}500\,\mathrm{m} & \text{signal peaks}\\ 2000\,\mathrm{m} & \text{depth peaks on blobs }\ge 4000\text{ hexes.}\end{cases}
\]

- Neubeck & Van Gool (2006), NMS: <https://doi.org/10.1109/ICPR.2006.479>
  (greedy distance NMS; not their 2-D scan algorithm)

A peak is dropped if the saddle toward a stronger peak is shallower than 3
(rings or dB) — overlap shoulders, not true sector petals. 2-/3-sector
cloverleafs are merged back to one hub (project rule; sectorization is
standard macro architecture, e.g. Rappaport *Wireless Communications*). The
pin sits on the **peak cell**, not the mass centroid.

### 4. Attribute growth to new vs same site

Each `new` or `upgraded` hex goes to the nearest pin in Albers if
\(d\le \mathrm{lobe\_reach}\) (95th percentile of that site’s hex distances,
floor \(3\,\mathrm{km}\); else \(1.6\times\) core reach). `stable_site`
hexes are not counted as expanded growth.

- Bentley (1975), k-d trees: <https://doi.org/10.1145/361002.361007>

\[
\mathrm{same\_site}=\frac{A_{\mathrm{expanded}}}{A_+},\quad
\mathrm{new\_site}=\frac{A_{\mathrm{new}}}{A_+},\quad
\mathrm{unattributed}=\frac{A_{\mathrm{un}}}{A_+}.
\]

Fallback vintage matching (if joint inference is unavailable): linear
assignment with cost \(d\) inside \(2\,\mathrm{km}\), else a large sentinel.

- Kuhn (1955), assignment problem: <https://doi.org/10.1002/nav.3800020109>
- Crouse (2016), solver SciPy actually runs: <https://doi.org/10.1109/TAES.2016.140952>

**ASR snap.** Unique registered mast within \(750\,\mathrm{m}\) (no second
ASR inside 750 m unless \(\ge 1.5\times\) farther). Missing ASR does **not**
flag (`asr_no_new_structure` weight 0). Bulk file:
<https://data.fcc.gov/download/pub/uls/complete/r_tower.zip>

**Boundary snap.** Share of new hexes within \(1.5\,\mathrm{km}\) of the
county outline.

### 5. Score and flag

Relative jump, squashed and absolute-gated so a tiny fill cannot look like
\(+\infty\%\):

\[
\tilde{\rho}=\frac{\rho}{1+\rho},\qquad
\mathrm{magnitude}=\tilde{\rho}\cdot\min(f_{\mathrm{added}}/0.05,\,1).
\]

Blanket fill-in:

\[
\mathrm{blanket}=(f_C-f_P)_+\cdot(1-f_P).
\]

Each feature is \(\hat{x}=\mathrm{clip}(x/r,0,1)\) (e.g. \(r=0.15\) for
added fraction of county). Weights \(|w|\le 0.25\). Let
\(D=\max(\sum|w|,1)\):

\[
c_f=\begin{cases} w\hat{x}/D & w\ge 0\\ |w|(1-\hat{x})/D & w<0 \end{cases},\qquad
S=\sum c_f.
\]

Default weights: added fraction \(+0.25\), magnitude \(+0.10\), blanket
\(+0.14\), same-site \(+0.22\), unattributed \(0\), boundary \(+0.08\),
new-site \(-0.22\), ASR \(0\).

**Binary flag is not a percentile.** `flag_percentile` is a severity badge
only. Flag iff:

\[
\begin{aligned}
&\mathrm{added\_km}^2\ge 10 \;\land\; \mathrm{same\_site}\ge 0.50\\
&\quad\land\; (f_{\mathrm{added}}\ge 0.075 \lor \mathrm{blanket}\ge 0.20)\\
&\quad\land\; \text{not majority new-site build}\\
&\quad\land\; \text{not failed inference}.
\end{aligned}
\]

Majority new-site: \(\mathrm{new\_site\_share}\ge 0.50\) with \(\ge 1\) new
tower, or \(\ge 0.35\) with \(\ge 1\) new tower of which at least one is
cross-border. Gates were set so Middlesex-scale urban fill (\(\approx 7.8\%\)
of county) can flag and 3–6% modest growth does not.

This is a **monotone scorecard**, not Isolation Forest and not a trained
classifier. Reviewers can read \(c_f\) as “how much this feature moved the
score.”

Optional vector-vs-tile check (off overnight) uses Jaccard
\(\mathrm{IoU}=|A\cap B|/|A\cup B|\)
([Jaccard 1901](https://doi.org/10.5169/seals-266450)).

---

## Limitations & honest caveats


- **Tower locations are inferred**, not published. They come from contiguous
  blobs of high modeled signal and are approximate - use them to target field
  tests, not as ground truth. Dense urban areas may merge nearby towers.
- Coverage is **modeled** (propagation), not measured; the whole point of the
  flag list is to direct real-world measurement where the model looks suspicious.
- `historical_volatility` scoring needs >2 vintages and is off by default.
- **Nationwide, all providers × all technologies × 2 vintages is ~0.5–1 TB of
  raw downloads.** Use Redshift, `--cleanup-raw`, narrowed scope, or per-state
  batches (see "Data volume" above). Start with one state/provider to validate
  before a full national run.
- The in-file signal column is auto-detected (`normalize._SIGNAL_COLUMNS`); if a
  vintage names it differently, add the alias there. If a file has no signal
  column, coverage is treated as a flat band (tower inference still works from
  coverage geometry).
- No FCC API token is needed: the pipeline downloads via the same public endpoint
  the website's own "Download" buttons use. It only needs network access to
  `broadbandmap.fcc.gov` and the browser-like headers set in `pipeline.yaml`.
