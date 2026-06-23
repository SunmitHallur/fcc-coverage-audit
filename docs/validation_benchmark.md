# Validation benchmark: FCC labeled counties (J25 vs D25)

The FCC shared example slides comparing **J25 (June 2025)** vs **D25 (December
2025)** mobile **5G 7/1 Mbps** coverage, labeling counties as *Selected* (chosen
for manual on-the-ground review) or *Not Selected*. **All labeled examples are
for Verizon (provider 131425).** These are ground truth for tuning and
validating the pipeline.

| County | FIPS | FCC decision | What the maps show |
|--------|------|--------------|--------------------|
| Pearl River County, MS | 28109 | **Selected** | Sparse scattered tower lobes in J25 -> dense, near-complete county fill with many lobes in D25. Very large in-county coverage-area increase. |
| Middlesex County, MA | 25017 | **Selected** | Already dense (urban) in J25 -> noticeably more saturated/filled in D25. Large increase. |
| Edmunds County, SD | 46045 | Not selected | County interior is essentially uncovered in both J25 and D25 (coverage sits outside the boundary). Negligible in-county coverage and change. |
| Menard County, TX | 48327 | Not selected | Strong (green) signal over a wide area in J25 -> mostly weaker (orange/red) in D25. Signal degraded / no covered-area increase, not a suspicious gain. |

### Additional Verizon 7/1 5G examples

| County | FIPS | FCC decision |
|--------|------|--------------|
| Lewis County, NY | 36049 | **Selected** |
| Sullivan County, NY | 36105 | **Selected** |
| Mills County, TX | 48307 | **Selected** |
| San Saba County, TX | 48411 | **Selected** |
| Lamar County, TX | 48277 | **Selected** |
| Yoakum County, TX | 48501 | **Selected** |
| Palo Pinto County, TX | 48363 | **Selected** |
| Grant County, OK | 40053 | **Selected** |
| Oneida County, ID | 16071 | **Selected** |
| Crowley County, CO | 08025 | Not selected |
| Rio Grande County, CO | 08105 | Not selected |
| Conejos County, CO | 08021 | Not selected |

The Colorado "not selected" cluster (Crowley / Rio Grande / Conejos) is a useful
true-negative set: nearby rural counties that did *not* warrant review, to guard
against over-flagging.

## Decision logic inferred from these examples

A county is selected for review when there is a **large increase in 5G coverage
*within the county boundary***. It is **not** selected when:

- in-county coverage and its change are negligible (Edmunds), or
- the change is a signal-strength shift that does not add covered area, or is a
  degradation (Menard).

## Encoded selection model (verified with FCC)

The patterns below were confirmed as the selection drivers and are encoded as the
scoring baseline in `scoring.feature_weights`:

| Pattern | Feature | Type | Weight |
|---------|---------|------|:------:|
| Large in-county coverage-area increase (absolute, normalized by county size) | `added_frac_of_county` | PRIMARY | 0.28 |
| Large relative jump (D25 / J25) | `coverage_increase_magnitude` | PRIMARY | 0.20 |
| Rapid blanket fill-in from a low baseline (rural implausibility, P5) | `blanket_fillin` | secondary | 0.14 |
| Growth claimed from existing towers, not new builds (P6) | `same_site_growth_share` | secondary | 0.16 |
| Coverage far from any inferred tower / flat polygons (P6) | `unattributed_share` | secondary | 0.10 |
| New coverage hugging the county boundary (snapping) | `boundary_snap_share` | secondary | 0.08 |
| Growth from genuine new towers (legitimacy) | `new_site_share` | reduces | -0.10 |

A county must also clear an absolute in-county area floor
(`scoring.min_added_km2_to_flag`) before it can be flagged, which encodes the
Edmunds/Menard "no meaningful in-county increase" cases.

> Not yet encoded (need extra data): true 5G-Fund *eligibility*-boundary snapping
> (needs the eligibility polygon layer) and a population/ruralness signal for P5
> (currently proxied by county-area-normalized fill-in). Both are noted in the
> code as future inputs.

## How the pipeline encodes this

- **In-county measurement:** hexes are assigned to a county by centroid, and all
  change metrics are computed per county (`changedetect.county_change`).
- **Magnitude gate:** `scoring.min_added_km2_to_flag` requires a county to add
  meaningful in-county coverage before it can be flagged - this excludes the
  Edmunds and Menard patterns.
- **Direction:** only signal/area *increases* are counted; downgrades are ignored
  (`changedetect.hex_change`).
- **Signal convention:** green = stronger signal (near tower), red = weaker
  (coverage fringe). Because the pipeline reads the actual vector dBm values
  (`minsignal`), the rendered tile color polarity does not affect correctness.

## Running the benchmark

Once the real J25 and D25 data are downloaded (FCC or Redshift backend):

```bash
python -m fcc_audit.cli benchmark
```

This runs D25 vs J25 for the Big-4 and prints, per labeled county, whether the
pipeline's selected/not-selected decision matches the FCC's. Use it to tune
`min_added_km2_to_flag`, `flag_percentile`, and `suspicious_same_site_growth`.

> Note: the FCC's full selection methodology may include factors beyond coverage
> change (e.g. population, prior challenges). If a complete list of selected
> counties for J25->D25 is available, add them here to tune precision/recall.
