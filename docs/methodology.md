# FCC Coverage-Audit Pipeline — Methodology & Defensibility

> Version: 2.0 | Last updated: June 2026  
> Status: **Active** — used in production FCC coverage-change audits

---

## Purpose

This document explains how the pipeline detects potentially gamed FCC Broadband Data Collection (BDC) filings, what every feature and threshold means, and where honest limitations exist.

The goal is **defensibility**: any flag the system produces should be explainable in plain language, reproducible from the committed inputs, and corroborated by at least one independent data source.

---

## 1. What "gaming" means in this context

Carriers self-report coverage maps to the FCC. Overclaiming — filing coverage for areas that aren't actually served — inflates subsidy eligibility and distorts competition data.

Common patterns:
1. **Blanket fill-in**: a new filing suddenly shows solid coverage across an entire county with no new construction.
2. **Implausible same-site expansion**: an existing tower is claimed to now cover 5× its previous footprint with no change in antennas.
3. **Phantom construction**: new towers appear in the filing that have no corresponding FCC Antenna Structure Registration.
4. **Paper coverage**: coverage is claimed in areas where no independent speed tests or measurements exist.

---

## 2. Data sources

| Source | Type | Ingested by |
|--------|------|-------------|
| Cached hex parquet under `data/raw/` | Carrier-reported | `acquire.py` (`files` backend — **default**) |
| FCC BDC mrgd H3 tables (Redshift) | Carrier-reported + `minsignal` dBm | `acquire.py` (`redshift` — opt-in for FCC staff) |
| FCC public NBM downloads | Carrier-reported | `acquire.py` (`fcc`) |
| FCC Antenna Structure Registration (ASR) | Independent federal register | `groundtruth_asr.py` (wired into `run` when `groundtruth.asr.enabled`) |
| Ookla Open Data (speed tests) | Independent field measurements | `groundtruth_measured.py` (library; **not** wired into default `run`) |
| FCC Speed Test app data | Independent field measurements | `groundtruth_measured.py` (optional) |

**Independence**: ASR is fetched with a local cache and merged into features / site
anchors during `run` (fails soft if the download is unavailable). Measurement
enrichment remains optional. Tile-CV reconcile helpers in `reconcile.py` stay
off the default path (boundary-snap distance threshold is still used).

### Redshift mrgd tables (default warehouse path)

Staff runs use `bbmap_mob_bb_mrgd_hex9_inter_<process_id>` (Broadband Map
Processing process ids: **292 = D25**, **291 = J25**). Each row is already
resolved to `(providerid, technology, mindown, environmnt)` with warehouse
**`minsignal`** (dBm). The acquire stage maps analysis services via
`source.redshift.service_mrgd_keys` (e.g. 5G-NR 7/1 → technology 500 / mindown 7)
and writes `data/raw/<vintage>/<provider_id>/*.parquet` with columns `h3` and
`signal_dbm`. Site inference and detail heatmaps therefore see real signal bands;
the web bundle’s distance-based estimate runs only when unique signal values are
still flat (legacy binary caches).

---

## 3. Pipeline stages

```
BDC hex files
    │
    ├─ normalize.py     H3 indexing, signal-strength normalization, state/county join
    │
    ├─ changedetect.py  Hex-level diff: added/removed per (provider, service, county)
    │
    ├─ towers.py        Joint site inference on the union of both vintages
    │                   (shared identity; per-vintage hex counts classify new/expanded/stable)
    │
    ├─ attribute.py     Attribution: assign gained hexes to nearest site; classify as
    │                   same-site / new-site / unattributed
    │
    ├─ score.py         Feature engineering → monotone bounded score → flag
    │
    ├─ groundtruth_asr.py + groundtruth_measured.py   (optional corroboration)
    │
    ├─ explain.py       Plain-language explanation per flagged county
    │
    ├─ casefile.py      Structured case-file generation (Claim → Evidence → Contradiction → Recommendation)
    │
    └─ webbundle.py     Static JSON for the reviewer cockpit web app
```

---

## 4. Feature definitions

### 4a. Core geometry features

| Feature | Description | Suspicious direction |
|---------|-------------|----------------------|
| `added_km2` | Net new coverage area added in the vintage window | Large absolute increase |
| `added_frac_of_county` | `added_km2` ÷ county area | High fraction (>15%) |
| `prior_km2` | Coverage in the prior vintage | Context only |
| `pct_change` | `added_km2 / prior_km2` | Very large % jump |
| `coverage_density` | Added area ÷ county area | High density |

### 4b. Attribution features (derived from site inference)

| Feature | Description | Suspicious direction |
|---------|-------------|----------------------|
| `same_site_growth_share` | Fraction of new hexes attributed to towers that already existed | Not suspicious alone; suspicious **in combination** with implausibility gates (see §5) |
| `unattributed_share` | Fraction of new hexes not attributable to any inferred site | High (>20%) — hexes "appearing from nowhere" |
| `blanket_fillin` | Spatial uniformity score: simultaneous whole-county fill-in pattern | High (>20%) |
| `boundary_snap_share` | Fraction of new hexes near county boundaries | High — can indicate border artifact inflation |
| `new_towers` | Number of newly inferred towers (new construction signal) | Low relative to growth area |

### 4c. Site-level plausibility features

| Feature | Description | Suspicious direction |
|---------|-------------|----------------------|
| `lobe_reach_m` | Empirical propagation radius derived from the full coverage lobe (high percentile of assigned-hex distances) | Not directly scored; used to calibrate attribution |
| `reach_m` | Core strong-signal radius of each inferred site | Very large |

### 4d. Ground-truth corroboration features

| Feature | Description | Suspicious direction |
|---------|-------------|----------------------|
| `asr_no_new_structure` | 1 if the FCC ASR database shows no new antenna structure was registered in the county during the filing window, 0 otherwise | 1 (no structure found) |
| `measurement_gap` | Fraction of claimed new coverage area with no speed-test measurements in the period | High (>20%) |

---

## 5. Flagging logic

### 5a. Deterministic monotone priority score

Each configured feature is mapped to a fixed [0, 1] scale and combined with
auditable weights. Suspicious features can only increase the score, while
legitimacy features can only decrease it. Each feature's maximum possible score
swing is 0.25, and the score for a county does not change when unrelated
counties are added to or removed from the run.

Flagging threshold: the `flag_percentile` percentile of all priority scores across the run (default: top 10%). This is intentionally conservative: we prefer fewer high-confidence flags over many uncertain ones.

### 5b. Implausibility gates for same-site growth

A high `same_site_growth_share` alone is **not** flagged. Normal 6-month organic growth often comes entirely from existing towers (antenna upgrades, carrier aggregation, software changes). The flag fires only when same-site growth is **combined with at least one implausibility gate**:

| Gate | Threshold | Meaning |
|------|-----------|---------|
| `added_frac_of_county` | ≥ 15% | Growth covers most of the county |
| `blanket_fillin` | ≥ 20% | Coverage appeared simultaneously everywhere |
| `unattributed_share` | ≥ 20% | Hexes not attributable to any real site |

Configuration in `config/pipeline.yaml`:
```yaml
scoring:
  suspicious_same_site_growth: 0.50
  suspicious_same_site_min_county_frac: 0.15
  suspicious_same_site_min_blanket: 0.20
  suspicious_same_site_min_unattributed: 0.20
```

### 5c. Ground truth amplification

When ASR and/or measured coverage data is available, the anomaly features include:
- `asr_no_new_structure` (weight 0.12): the combination of large growth + no registered tower construction is a strong signal.
- `measurement_gap` (weight 0.10): coverage claimed but not observed in field data is suspicious.

These do **not** independently flag a county — they amplify the score of an already-anomalous filing.

---

## 6. Threshold rationale

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `flag_percentile` | 0.90 | Top 10% of anomaly scores; keeps false-positive rate manageable for a small review team |
| `min_added_km2_to_flag` | 10.0 km² | Ignore trivial / near-empty county changes (FCC-validated) |
| `suspicious_same_site_growth` | 0.50 | 50% same-site share is not itself suspicious; only the combination matters |
| `min_site_hexes` | 35 (at H3 res 9) | ~3.7 km² minimum blob to infer a site at res 9; prevents noise |
| `peak_separation_m` | 500 | Minimum spacing between accepted *signal* peaks within one footprint (NMS); independent of cross-vintage match. Depth NMS on blobs ≥4000 hexes uses 2 km so county-scale fill does not tile at 500 m |
| `site_match_radius_m` | 2000 | Cross-vintage site identity match radius |
| Feature weight: `asr_no_new_structure` | 0.12 | Active when ASR labels merge successfully; soft-disabled if ASR fetch fails |

---

## 7. H3 resolution

Coverage hexes use Uber H3 for geographic indexing:

| Resolution | Hex edge length | Avg hex area | Use |
|------------|-----------------|--------------|-----|
| 8 | ~461 m | ~0.74 km² | Legacy (retired) |
| 9 | ~174 m | ~0.11 km² | County + site analysis default (matches Redshift hex tables) |
| 10 | ~65 m | ~0.015 km² | Optional finer site inference for FCC polygon backend only |
| 11 | ~24 m | ~0.002 km² | Optional fine detail |

Higher resolution → more hexes → more compute. **`--workers N` parallelises across
providers** (process pool), not counties. Prefetch Redshift slices **before**
workers so analyze processes do not thrash warehouse WLM. Keep
`site_h3_resolution: 9` for Redshift — setting 10 forces a 7× child expansion and
can OOM at national scale. The pipeline auto-scales hex-count-based knobs
(e.g. `min_site_hexes`) when the actual data resolution differs from the
configured default.

**Small cohorts (< 50 scored rows):** percentile flagging is disabled; only the
absolute same-site implausibility gates can flag. This avoids over-flagging
one-state pilots where `quantile` would otherwise be meaningless.

**Cross-border context:** the Redshift path expands each batch with a ~50 km
neighbor-state halo so towers just across a state line can explain in-county
growth. The FCC polygon path does not fetch neighbor states today — border
counties may look more "unattributed" on that backend.

---

## 8. Honest limitations

### 8a. ASR ↔ carrier join caveat

The Antenna Structure Registration (ASR) database records structures by geographic location. The pipeline joins ASR structures to counties by FIPS code and date window. **Known limitations**:

- Some carriers do not individually register every antenna (small cells, DAS systems).
- A carrier may build on a pre-existing structure owned by another party — this would appear as "no new structure" even though new equipment was installed.
- The FCC registration date may lag the actual construction date by weeks.

**Implication**: `asr_no_new_structure = 1` (no structure found) is **suggestive but not conclusive**. It raises suspicion; it does not prove fraud. The pipeline treats it as a moderate-weight corroborating signal (weight 0.12), not a binary gate.

### 8b. Ookla measurement coverage

- Ookla collects speed tests from consenting app users — sampling is uneven and favors urban/suburban areas.
- Low measurement counts in rural areas may produce a high `measurement_gap` even for legitimate coverage.
- The pipeline requires a minimum density threshold before using the measurement gap as a signal.

### 8c. Site inference from the same BDC data

Towers are inferred from the carrier's own BDC hex map (blob detection + centroid). This creates circularity: if the map is gamed, the inferred towers will also be wrong.

**Mitigation**: the ASR and Ookla features are derived from independent sources and are not affected by this circularity. Highly anomalous attribution (e.g. no blob structure, uniform blanket fill) is itself a flag signal.

### 8d. 6-month vintage cadence

The BDC is filed semi-annually. One large jump may represent two half-year builds, or a correction of a prior under-filing. The pipeline compares adjacent vintages and does not look further back. This can produce false positives for carriers correcting historical under-filings.

**Mitigation**: the benchmark block in `config/pipeline.yaml` allows marking known legitimate large changes as must-not-flag cases.

### 8e. No supervised labels (yet)

The monotone score is deterministic rather than learned. Once the `validate`
subcommand has accumulated enough human-reviewed accept/reject decisions, a
supervised model can be trained and benchmarked against it. Until then,
precision/recall are estimated against ASR/measurement ground truth only.

---

## 9. Reproducibility

The full pipeline is reproducible from a clean clone:

```bash
git clone <repo>
pip install -r requirements.txt
python -m fcc_audit run --config config/pipeline.yaml --vintage 2024-06-30 --state KS
python -m fcc_audit build-web
# Open web/index.html in a browser
```

All intermediate data is cached in `data/interim/` keyed by `(vintage, provider, service, resolution)`. Re-running with the same inputs produces bit-identical outputs (determinism gate: `random_state=42` throughout).

---

## 10. Deliverable statement

> "Across N states and 3 technologies of real FCC data, the app flags gamed coverage with precision X / recall Y against independent FCC tower-registration and measured-coverage records, and gives reviewers an auditable accept/reject workflow that outputs field-test work orders."

Fill in X and Y by running: `python -m fcc_audit validate`
