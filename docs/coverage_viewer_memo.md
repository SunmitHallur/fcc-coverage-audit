# FCC Mobile Coverage-Change Viewer — How It Works

**Audience:** Engineers and reviewers evaluating the automated mobile coverage audit  
**Attachment:** `sample_coverage_demo.html` (standalone demo; illustrative data only)  
**Production system:** `fcc-coverage-audit` pipeline + web cockpit (`web/index.html`)

---

## 1. What this tool does

The FCC requires mobile providers to file modeled coverage every six months. Reviewers must decide **which provider × county × service combinations** deserve a physical drive test because the claimed growth looks implausible (e.g., a large coverage jump with no new towers).

This project automates that selection:

1. **Compare** two vintages of coverage (e.g., June 2025 vs December 2025).
2. **Detect** where coverage grew, shrank, or stayed the same — at H3 resolution-9 hex granularity (~0.1 km² per cell), matching FCC audit geography.
3. **Infer** approximate tower locations from contiguous high-signal blobs.
4. **Attribute** new coverage to **new towers**, **expanded existing towers**, or **unattributed** growth (no nearby tower).
5. **Score and rank** every provider × county × service row, and **flag** the top anomalies for review.

The **deliverable** is a ranked CSV (`selected_counties_*.csv`) — the automated equivalent of the manual consultant selection list.

---

## 2. The demo HTML (`sample_coverage_demo.html`)

### How to open it

- **No install required.** Double-click the file, or drag it into Chrome / Edge / Firefox.
- **Internet required** for the basemap tiles and map libraries (loaded from CDN). No server, no login, no Python.
- **Safe to email** as a single attachment (~40 KB).

### Layout

| Area | Purpose |
|------|---------|
| **County dropdown** | Switch between illustrative counties. |
| **Prior map (left)** | Coverage footprint at the earlier vintage. |
| **Current map (right)** | Coverage footprint at the later vintage. |
| **Stats bar** | Priority score, added km², % increase, tower counts (prior → current, new towers). |
| **Explanation line** | Plain-language reason the county was or was not flagged. |
| **Legend (bottom-right)** | Signal color ramp and symbol key. **Click the header to collapse** so it does not cover the map or stats. |

### Controls

- **Highlight newly-covered hexes** — hexes that appear in *current* but not *prior* are drawn in **green** on the current map. This is the primary visual for “where did coverage grow?”
- **Show towers** — colored dots on the maps:
  - **Blue** — existing / stable tower
  - **Green** — new tower
  - **Purple** — expanded tower (matched prior site with materially more hexes)

### Signal colors (when signal data is present)

Hex fill color reflects **modeled signal strength** (weak → strong):

`red / orange → yellow → green → teal`

In the **demo file**, most counties use synthetic signal gradients for illustration. In **production** runs sourced from Redshift hex tables, coverage is often a **0/1 flag** (covered vs not), so hexes may appear as a uniform band; the important signal is **footprint change**, not per-hex dBm.

---

## 3. The four demo counties (what to look for)

| County | Flagged? | Story |
|--------|----------|--------|
| **Blanketed County** | Yes | Coverage jumps ~640% but **tower count unchanged** — growth attributed to existing sites. Classic gaming pattern: blanket fill without build-out. |
| **New-Tower County** | No | New coverage radiates from a **genuinely new tower** (green dot). Growth is consistent with real infrastructure. |
| **Steady County** | No | Little change between vintages — not a review priority. |
| **Signal Gradient (demo)** | No | Shows the **signal color ramp** from a single tower (strong at core, weak at fringe). |

Use **Blanketed** vs **New-Tower** as the contrast: same kind of map, opposite conclusions.

---

## 4. How the full production web app differs

The demo is a **minimal, self-contained** viewer. The production app (`web/index.html`, built by `python -m fcc_audit.cli build-web`) adds:

- **National county choropleth** — shading by coverage change for the selected provider and service.
- **Provider and service filters** — e.g., AT&T, 5G-NR 7/1.
- **Flagged counties highlighted** on the national map.
- **Click any county** for the same prior/current hex view, tower overlay, and written explanation.
- **Flag-math panel** — shows which risk features drove the score (same-site growth share, blanket fill-in, boundary snapping, etc.).

Data is loaded from `web/public/data/` (records JSON, county GeoJSON, per-county detail files). Deployed as a static site (e.g., Vercel); no backend at view time.

---

## 5. Pipeline overview (how data gets into the viewer)

```
Redshift hex tables (or FCC download)
        ↓
  Normalize to H3 + tag counties
        ↓
  Change-detect (prior vs current hexes)
        ↓
  Infer towers + attribute growth (new / expanded / unattributed)
        ↓
  Score + rank + flag top anomalies
        ↓
  CSV deliverables + optional web bundle
```

**Data source (current deployment):** BDC Redshift pre-aggregated H3 res-9 tables (`bbmap_mobile_bb_tech_hex9s_<build>`), filtered per provider via the warehouse’s comma-delimited provider list. Vintages are **build IDs** (e.g., 277 = D25, 279 = J25).

**Primary flag logic (simplified):** A county is suspicious when it adds meaningful in-county coverage **and** a large share of that growth is claimed from **existing towers** without a corresponding new build — especially when combined with blanket fill-in or unattributed area. Legitimate **new-tower** buildouts reduce the score.

---

## 6. Key outputs (besides the map)

| File | Contents |
|------|----------|
| `selected_counties_<cur>_vs_<prior>.csv` | Flagged pairs only — the automated selection list |
| `priority_ranking_*.csv` | All counties ranked, with features and explanations |
| `summary_*.md` | Human-readable top-25 review list |

---

## 7. Limitations (worth stating in a review)

- **Tower locations are inferred**, not ASR ground truth — use them to target tests, not as legal evidence of structure.
- Coverage is **modeled** (propagation), not measured — the tool directs *where* to measure.
- Hex-level signal from Redshift tech tables is **binary** (covered / not); graded signal requires polygon sources with `minsignal`.
- Flags are **prioritization**, not proof of fraud — they narrow the field for field verification.

---

## 8. Questions / next steps

- Run the full pipeline for additional providers or services via `config/pipeline.yaml` and `run_overnight.ps1`.
- Swap vintage build IDs when newer Redshift snapshots are available (e.g., 283 for D25 when permitted).
- For a **real-data** sample of a few counties (not synthetic), use `tools/make_sample_coverage.py` on a machine that has run the pipeline and saved coverage parquet.

---

*Generated for the FCC mobile coverage-change audit project. Demo data in `sample_coverage_demo.html` is synthetic and for illustration only.*
