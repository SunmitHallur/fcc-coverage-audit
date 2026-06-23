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
consultants billed at ~$2k/hour. This pipeline automates that selection - it
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

All machine learning is **local** (connected-component site inference +
scikit-learn IsolationForest for anomaly ranking). There are **no external
LLM/API calls**, which keeps approvals minimal.

```
acquire ─► normalize ─► (reconcile) ─► change-detect ─► infer sites
        ─► attribute (new vs expanded) ─► score / rank ─► CSV + report + dashboard
```

---

## Quick start

Requires Python 3.13 or 3.14 (verified). From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=src                 # Windows: $env:PYTHONPATH="src"
```

### 1. Try it offline first (no FCC access needed)

This generates synthetic data with a planted gaming scenario and runs the whole
pipeline, so you can confirm everything works before touching real data:

```bash
python -m fcc_audit.cli --backend fixture make-fixtures
python -m fcc_audit.cli --backend fixture run
```

Expected result: **T-Mobile / Charlie County is the #1 flagged** county
("100% of growth claimed from existing sites; coverage up 643%"), while AT&T's
genuine new-tower build in the same county is **not** flagged.

### 2. Get a free FCC API token (required for downloads)

The FCC catalog is public, but **downloading the coverage files requires a free
FCC API token**. One-time setup:

1. Create an account at https://broadbandmap.fcc.gov (Sign In → Create account).
2. Log in, click your **username** (top-right) → **Manage API Access** → **Generate**. Copy the token.
3. Set two environment variables before running (use the same email + token):

```powershell
# Windows PowerShell
$env:FCC_API_USERNAME = "you@example.com"
$env:FCC_API_TOKEN    = "<your-44-char-token>"
```
```bash
# macOS/Linux
export FCC_API_USERNAME="you@example.com"
export FCC_API_TOKEN="<your-44-char-token>"
```

The launchers (`run.bat`/`run.sh`) and CLI read these automatically. Without
them, downloads return `401 Unauthorized` (the catalog/`list-vintages` still work).

### 3. Run on real FCC data

**Downloading is fully automated — you never fetch files by hand.** The default
backend (`source.backend: fcc`) pulls coverage from the National Broadband Map.
With `providers: all` it discovers every mobile provider, and for each
`(provider, service)` it downloads the per-state coverage files and merges them.

**Easiest (one command, handles venv + install + run):**

```bash
./run.sh                 # macOS/Linux        |   run.bat   (Windows: double-click)
```

With no arguments the launcher does a full national run with `--cleanup-raw`. You
can also pass through any subcommand, e.g. `./run.sh download` or
`run.bat run --current 2025-12-31 --prior 2025-06-30`.

**Or drive the CLI directly:**

```bash
python -m fcc_audit.cli list-vintages          # available vintages (no token needed)
python -m fcc_audit.cli download                 # PRE-FETCH all raw files only (resumable)
python -m fcc_audit.cli run                      # download + analyze, ALL providers/services
python -m fcc_audit.cli run --cleanup-raw        # delete each raw download after use
python -m fcc_audit.cli run --current "December 31, 2025" --prior "June 30, 2025"
```

Vintages are the FCC `filing_subtype` labels (e.g. `"December 31, 2025"`), not
ISO dates. `download` and `run` are both resumable — already-downloaded files and
interim parquet are cached and skipped, so an interrupted run picks up where it
left off.

### Start small — validate before the national run

Real mobile coverage is **per state × provider × service**, downloaded at the
FCC's ~10-requests/minute limit, so a full national, all-provider run is an
**overnight (many-hour) job**. Confirm everything works on a small scope first by
editing `config/pipeline.yaml`:

```yaml
analysis:
  services:
    - { label: "5G-NR 7/1", desc: "5G-NR (7/1 Mbps)" }   # one service
  providers:
    - { id: 131425, name: "Verizon" }                     # one provider
  states: ["48"]                                           # Texas only
```

Then `python -m fcc_audit.cli run`. Once that produces a sane `selected_counties_*.csv`,
widen `services`/`providers`/`states` (or set them back to `all`) for the full run.

> Note: the FCC public API filters requests by User-Agent and rate-limits
> aggressive clients; both are handled in config (`source.fcc`).

#### Data volume & time — read this before a national run

Mobile coverage is per state × provider × service, so a full national run is
**thousands of file downloads at ~10/minute → many hours / overnight**, and
**0.5–1 TB** of raw files cumulatively. Options, in order of preference:

1. **Redshift (best).** Once your AWS access lands, query server-side and never
   download the raw geometry (see below). The right path for "all data".
2. **`--cleanup-raw` + scope down.** The pipeline keeps only a compact per-layer
   hex parquet (`data/interim/`) and deletes the big raw files after each service
   when you pass `--cleanup-raw`, so peak disk stays small. Combine with dropping
   `3G` (being retired) and unneeded states/providers in config.
3. **Run per-state / per-provider in batches.** The interim parquet cache makes
   re-runs cheap and the download is resumable.

If you only care about the FCC's current funding focus, keep it to **5G-NR**
(both tiers) + **4G LTE**; that is a fraction of the full volume.

### 4. View results

```bash
# Ranked priority list + summary
open data/outputs/priority_ranking_*.csv
open data/outputs/summary_*.md

# Interactive map (must be served, not opened via file://)
cd dashboard && python3 -m http.server 8000
# then open http://localhost:8000
```

---

## Outputs

| File | What it is |
|------|------------|
| `data/outputs/selected_counties_<cur>_vs_<prior>.csv` | **The automated selection list** - only the flagged provider/county pairs, ranked. This is the deliverable that replaces the manual consultant selection. |
| `data/outputs/priority_ranking_<cur>_vs_<prior>.csv` | One row per provider x county (all, not just flagged), ranked by priority, with the risk features and a plain-language flag reason. |
| `data/outputs/summary_<cur>_vs_<prior>.md` | Human-readable top-25 review list. |
| `data/outputs/dashboard_data_<...>.json` + `dashboard/data.json` | Map payload. |
| `dashboard/index.html` | MapLibre app: flagged counties + inferred new/expanded towers. |

**Key flag signal:** `same_site_growth_share` - the fraction of new coverage
attributed to *existing* sites. High values mean a provider is claiming big
coverage gains without building, which is the primary gaming pattern.

---

## Configuration

Everything is in [`config/pipeline.yaml`](config/pipeline.yaml): providers,
technologies/speed tiers, vintages, H3 resolutions, the signal threshold and
match radius for site inference, reconciliation thresholds, and scoring weights.

**Scope.** `analysis.providers: all` auto-discovers every mobile provider (or use
an explicit `{id, name}` list). `analysis.services` lists the FCC mobile datasets
to analyze, each identified by its catalog `desc` (`technology_code_desc`) — the
FCC ships each 5G speed tier as a separate file, so `5G-NR (7/1 Mbps)` and
`5G-NR (35/3 Mbps)` are distinct services. `analysis.states` is `all` or a FIPS
list (e.g. `["48"]` for Texas) to scope a run. Drop services/providers/states to
cut download volume and time.

### Switching to Redshift later

The data layer is pluggable. Once your AWS Redshift access is granted:

1. `pip install redshift-connector` (uncomment in `requirements.txt`).
2. Fill `source.redshift` in the config (host/db/user/password come from env
   vars: `REDSHIFT_HOST`, `REDSHIFT_DB`, `REDSHIFT_USER`, `REDSHIFT_PASSWORD`).
3. Adjust `coverage_query` to your warehouse schema. `provider_id`, `technology`,
   and the vintage are templated in; `list_providers` and `fetch` already issue
   per-provider/technology queries.
4. Set `source.backend: redshift` (or run with `--backend redshift`).

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
- Downloads require a free FCC API token (`FCC_API_USERNAME`/`FCC_API_TOKEN`); the
  public catalog (`list-vintages`, provider discovery) needs no token.
