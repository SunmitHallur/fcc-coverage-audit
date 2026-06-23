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

### 2. Run on real FCC data

**Downloading is fully automated — you never fetch files by hand.** The default
backend (`source.backend: fcc`) pulls coverage straight from the National
Broadband Map API. With `providers: all` it first discovers every mobile provider
in the catalog for the vintage, then fetches one file per `(provider, technology)`.

**Easiest (one command, handles venv + install + run):**

```bash
./run.sh                 # macOS/Linux        |   run.bat   (Windows: double-click)
```

With no arguments the launcher does a full national run with `--cleanup-raw`. You
can also pass through any subcommand, e.g. `./run.sh download` or
`run.bat run --current 2025-12-31 --prior 2025-06-30`.

**Or drive the CLI directly:**

```bash
python -m fcc_audit.cli list-vintages          # see available 6-month vintages
python -m fcc_audit.cli download                 # PRE-FETCH all raw files only (resumable)
python -m fcc_audit.cli run                      # download + analyze, ALL providers/techs
python -m fcc_audit.cli run --cleanup-raw        # delete each raw download after use
python -m fcc_audit.cli run --current 2025-12-31 --prior 2025-06-30
```

`download` and `run` are both resumable — already-downloaded raw files and
already-built interim parquet are cached and skipped, so an interrupted run just
picks up where it left off. Use `download` if you want to stage all the raw data
in one pass (e.g. onto an external drive) and analyze later offline; use plain
`run` to stream-download-and-process (best with `--cleanup-raw` to bound disk).

> Note: the FCC public API filters requests by User-Agent and rate-limits
> aggressive clients; both are handled in config (`source.fcc`).

#### Data volume — read this before a national run

Downloading **every provider × every technology × 2 vintages nationwide** is
large — realistically **0.5–1 TB of raw shapefiles**, which usually will not fit
a laptop. Options, in order of preference:

1. **Redshift (best).** Once your AWS access lands, query server-side and never
   download the raw geometry to the laptop (see below). This is the right path
   for "all data".
2. **`--cleanup-raw` + scope down.** The pipeline keeps only a compact per-layer
   hex parquet (`data/interim/`) and deletes the big raw file after each provider
   when you pass `--cleanup-raw`, so peak disk ≈ one provider's download, not all
   of them. Combine with narrowing scope in config (e.g. comment out `3G`, which
   is being retired, and run `4G-LTE` + `5G-NR` only).
3. **Run per-state / per-provider in batches.** Process a subset, archive the
   outputs, then move on. The interim parquet cache makes re-runs cheap.

If you only care about the FCC's current funding focus, keep it to **5G-NR**
(both tiers) + **4G-LTE**; that is a fraction of the full volume.

### 3. View results

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

**Scope (providers & technologies).** `analysis.providers: all` auto-discovers
every mobile provider; or replace it with an explicit list of `{id, name}` to
narrow. `analysis.technologies` maps each technology to its FCC speed tiers
(`mindown`/`minup` in Mbps); comment out a technology or tier to drop it. The
normalize stage filters each downloaded file by tier (mindown/minup) and, when
`combine_environments: false`, by environment (`environmnt`). Verify the
`environment_codes` against a real file the first time — code meanings can vary
by vintage; the default (combined) path does not depend on them.

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
- Speed-tier filtering assumes the BDC `mindown`/`minup` attributes are present;
  if a vintage names them differently, add the alias in `normalize._MINDOWN_COLUMNS`
  / `_MINUP_COLUMNS`.
