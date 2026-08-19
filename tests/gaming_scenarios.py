"""Named June→December coverage-change physics — not screenshot FIPS.

Each scenario is a fully specified feature row plus whether a reviewer should
see a flag. Families:

- review: same-site growth that is physically implausible (urban lobe-merge /
  rural blanket / huge in-county fill without new macros).
- skip: ordinary antenna/software growth, new-tower construction, area loss,
  tiny jumps, inference failure, or ASR/measurement gaps by themselves.

A large ordinary background population is generated so percentile ranking
cannot be gamed by a 4-row cohort.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_BASE = {
    "provider_id": 131425,
    "provider_name": "Verizon",
    "technology": "5G-NR 7/1",
    "county_name": "Synthetic",
    "unattributed_share": 0.02,
    "boundary_snap_share": 0.0,
    "asr_no_new_structure": 0.0,
    "measurement_gap": 0.0,
    "new_towers": 0,
    "new_towers_cross_border": 0,
    "inference_insufficient": False,
    "coverage_increase_magnitude": 0.15,
}


def _row(geoid: str, expect_flag: bool, family: str, why: str, **kw: Any) -> dict:
    out = dict(_BASE)
    out.update(kw)
    same = float(out.get("same_site_growth_share", 0.0))
    new = float(out.get("new_site_share", 0.0))
    unattr = float(out.get("unattributed_share", 0.0))
    leftover = max(0.0, 1.0 - same - new - unattr)
    if "new_site_share" not in kw:
        out["new_site_share"] = leftover
    out["county_geoid"] = geoid
    out["case"] = geoid
    out["expect_flag"] = expect_flag
    out["family"] = family
    out["why"] = why
    if "added_km2" not in out:
        out["added_km2"] = max(12.0, float(out.get("added_frac_of_county", 0.0)) * 1200.0)
    return out


# ---------------------------------------------------------------------------
# REVIEW: implausible same-site change (should flag)
# ---------------------------------------------------------------------------
REVIEW: list[dict] = [
    _row("R01", True, "review",
         "Urban existing lobes expand until they merge (Middlesex-style).",
         added_km2=180, added_frac_of_county=0.085, same_site_growth_share=0.97,
         blanket_fillin=0.04, new_site_share=0.01, coverage_increase_magnitude=0.35),
    _row("R02", True, "review",
         "Already-dense urban county adds another ~10% from existing sites.",
         added_km2=220, added_frac_of_county=0.10, same_site_growth_share=0.94,
         blanket_fillin=0.02, new_site_share=0.04, prior_cov_frac=0.78,
         coverage_increase_magnitude=0.40),
    _row("R03", True, "review",
         "Rural sparse lobes become near-complete county fill, no new macros.",
         added_km2=420, added_frac_of_county=0.20, same_site_growth_share=0.99,
         blanket_fillin=0.16, new_site_share=0.0, coverage_increase_magnitude=0.70),
    _row("R04", True, "review",
         "Prairie county jumps from nearly empty to majority covered, same sites.",
         added_km2=1100, added_frac_of_county=0.42, same_site_growth_share=0.92,
         blanket_fillin=0.38, new_site_share=0.06, coverage_increase_magnitude=0.95),
    _row("R05", True, "review",
         "Low baseline → simultaneous whole-county fill (blanket gate, modest %).",
         added_km2=280, added_frac_of_county=0.06, same_site_growth_share=0.88,
         blanket_fillin=0.28, new_site_share=0.10, coverage_increase_magnitude=0.55),
    _row("R06", True, "review",
         "Four existing macros double their radius in a small county.",
         added_km2=95, added_frac_of_county=0.18, same_site_growth_share=1.0,
         blanket_fillin=0.12, new_site_share=0.0),
    _row("R07", True, "review",
         "Single rural lobe explodes to cover half a small county.",
         added_km2=160, added_frac_of_county=0.48, same_site_growth_share=0.99,
         blanket_fillin=0.35, new_site_share=0.0),
    _row("R08", True, "review",
         "Fill-in between existing urban sites, 9% of county, zero new towers.",
         added_km2=140, added_frac_of_county=0.09, same_site_growth_share=0.96,
         blanket_fillin=0.05, new_site_share=0.0, new_towers=0),
    _row("R09", True, "review",
         "Same-site 15% fill hugging the county line (boundary corroboration).",
         added_km2=200, added_frac_of_county=0.15, same_site_growth_share=0.80,
         blanket_fillin=0.08, boundary_snap_share=0.55, new_site_share=0.15),
    _row("R10", True, "review",
         "Mountain basin: 25% of county appears from three existing sites.",
         added_km2=640, added_frac_of_county=0.25, same_site_growth_share=0.91,
         blanket_fillin=0.22, new_site_share=0.07),
    _row("R11", True, "review",
         "Suburban ring: 8% same-site just above the area gate.",
         added_km2=75, added_frac_of_county=0.078, same_site_growth_share=0.93,
         blanket_fillin=0.03, new_site_share=0.05),
    _row("R12", True, "review",
         "Two-county lookalike: large in-county fill, all expanded_site.",
         added_km2=510, added_frac_of_county=0.33, same_site_growth_share=0.87,
         blanket_fillin=0.19, new_site_share=0.11, new_towers=0),
    _row("R13", True, "review",
         "High same-site + high blanket even though new_site is 20% (not majority).",
         added_km2=300, added_frac_of_county=0.16, same_site_growth_share=0.72,
         blanket_fillin=0.21, new_site_share=0.26, new_towers=1),
    _row("R14", True, "review",
         "Desert county 30% same-site fill, ASR missing (must still flag on physics).",
         added_km2=900, added_frac_of_county=0.30, same_site_growth_share=0.95,
         blanket_fillin=0.24, asr_no_new_structure=1.0, new_site_share=0.03),
    _row("R15", True, "review",
         "Large absolute km² (500) at 12% of a mid-size county, same sites.",
         added_km2=500, added_frac_of_county=0.12, same_site_growth_share=0.88,
         blanket_fillin=0.07, new_site_share=0.10),
]

# ---------------------------------------------------------------------------
# SKIP: not gaming (must not flag) — varied lookalikes
# ---------------------------------------------------------------------------
SKIP: list[dict] = [
    _row("S01", False, "skip",
         "Ordinary 2% same-site lobe growth (antenna/software).",
         added_km2=28, added_frac_of_county=0.02, same_site_growth_share=0.90,
         blanket_fillin=0.015, new_site_share=0.08),
    _row("S02", False, "skip",
         "Ordinary 4% same-site growth in a large county.",
         added_km2=55, added_frac_of_county=0.04, same_site_growth_share=0.85,
         blanket_fillin=0.03, new_site_share=0.12),
    _row("S03", False, "skip",
         "5.8% same-site urban bump — below the 7.5% gate (prefer FN).",
         added_km2=150, added_frac_of_county=0.058, same_site_growth_share=0.90,
         blanket_fillin=0.04, new_site_share=0.08),
    _row("S04", False, "skip",
         "One tower grew 3 rings in a huge county (~3% of area).",
         added_km2=70, added_frac_of_county=0.03, same_site_growth_share=1.0,
         blanket_fillin=0.02, new_site_share=0.0),
    _row("S05", False, "skip",
         "Added area below the 10 km² floor.",
         added_km2=6, added_frac_of_county=0.12, same_site_growth_share=0.99,
         blanket_fillin=0.10, new_site_share=0.0),
    _row("S06", False, "skip",
         "Zero net added coverage.",
         added_km2=0, added_frac_of_county=0.0, same_site_growth_share=0.0,
         blanket_fillin=0.0, new_site_share=0.0, coverage_increase_magnitude=0.0),
    _row("S07", False, "skip",
         "Coverage loss / recoding, no area gain.",
         added_km2=-40, added_frac_of_county=0.0, same_site_growth_share=0.99,
         blanket_fillin=0.0, new_site_share=0.0, coverage_increase_magnitude=0.0),
    _row("S08", False, "skip",
         "Signal dropped 20 dB on the same footprint.",
         added_km2=0, added_frac_of_county=0.0, same_site_growth_share=1.0,
         blanket_fillin=0.0, new_site_share=0.0, coverage_increase_magnitude=0.0),
    _row("S09", False, "skip",
         "75% of added area is new macros (real construction).",
         added_km2=1500, added_frac_of_county=0.53, same_site_growth_share=0.22,
         blanket_fillin=0.40, new_site_share=0.75, new_towers=4,
         coverage_increase_magnitude=0.90),
    _row("S10", False, "skip",
         "60% new-site share with two in-county towers.",
         added_km2=400, added_frac_of_county=0.18, same_site_growth_share=0.35,
         blanket_fillin=0.12, new_site_share=0.60, new_towers=2),
    _row("S11", False, "skip",
         "Exactly 50% new-site with a new tower (majority-or-tie is buildout).",
         added_km2=220, added_frac_of_county=0.14, same_site_growth_share=0.45,
         blanket_fillin=0.10, new_site_share=0.50, new_towers=1),
    _row("S12", False, "skip",
         "Cross-border new macro explains the in-county gain.",
         added_km2=180, added_frac_of_county=0.16, same_site_growth_share=0.40,
         blanket_fillin=0.11, new_site_share=0.55, new_towers=1,
         new_towers_cross_border=1),
    _row("S13", False, "skip",
         "Cross-border at the 35% new-site special case.",
         added_km2=120, added_frac_of_county=0.11, same_site_growth_share=0.58,
         blanket_fillin=0.08, new_site_share=0.38, new_towers=1,
         new_towers_cross_border=1),
    _row("S14", False, "skip",
         "Huge relative jump from zero coverage but only 0.8% of the county.",
         added_km2=18, added_frac_of_county=0.008, same_site_growth_share=0.70,
         blanket_fillin=0.008, new_site_share=0.25, coverage_increase_magnitude=1.0),
    _row("S15", False, "skip",
         "Missing ASR on otherwise ordinary 3% growth (rooftops).",
         added_km2=40, added_frac_of_county=0.03, same_site_growth_share=0.40,
         blanket_fillin=0.02, new_site_share=0.45, asr_no_new_structure=1.0,
         new_towers=2),
    _row("S16", False, "skip",
         "Measurement gap alone, modest same-site growth.",
         added_km2=35, added_frac_of_county=0.03, same_site_growth_share=0.55,
         blanket_fillin=0.02, new_site_share=0.40, measurement_gap=0.80),
    _row("S17", False, "skip",
         "Site inference failed; unattributed 100% is an artifact.",
         added_km2=200, added_frac_of_county=0.15, same_site_growth_share=0.0,
         blanket_fillin=0.12, new_site_share=0.0, unattributed_share=1.0,
         inference_insufficient=True),
    _row("S18", False, "skip",
         "Mixed 45/45 new vs expanded, only 4% of county.",
         added_km2=48, added_frac_of_county=0.04, same_site_growth_share=0.45,
         blanket_fillin=0.03, new_site_share=0.45, new_towers=1),
    _row("S19", False, "skip",
         "High boundary-snap share but tiny added area.",
         added_km2=8, added_frac_of_county=0.20, same_site_growth_share=0.80,
         blanket_fillin=0.15, boundary_snap_share=0.90, new_site_share=0.10),
    _row("S20", False, "skip",
         "High boundary snap on a 3% ordinary bump.",
         added_km2=36, added_frac_of_county=0.03, same_site_growth_share=0.70,
         blanket_fillin=0.02, boundary_snap_share=0.70, new_site_share=0.25),
    _row("S21", False, "skip",
         "70% same-site but 2% of a huge county.",
         added_km2=80, added_frac_of_county=0.02, same_site_growth_share=0.70,
         blanket_fillin=0.015, new_site_share=0.25),
    _row("S22", False, "skip",
         "Two new macros filling an empty county (legitimate 5G build).",
         added_km2=800, added_frac_of_county=0.22, same_site_growth_share=0.10,
         blanket_fillin=0.20, new_site_share=0.88, new_towers=2),
    _row("S23", False, "skip",
         "Same-site 49% (below 50% gate) even at 20% of county.",
         added_km2=240, added_frac_of_county=0.20, same_site_growth_share=0.49,
         blanket_fillin=0.12, new_site_share=0.48, new_towers=2),
    _row("S24", False, "skip",
         "Identical vintages: no change.",
         added_km2=0, added_frac_of_county=0.0, same_site_growth_share=0.0,
         blanket_fillin=0.0, new_site_share=0.0, unattributed_share=0.0,
         coverage_increase_magnitude=0.0),
    _row("S25", False, "skip",
         "Small-cell / rooftop infill explained as new sites.",
         added_km2=90, added_frac_of_county=0.09, same_site_growth_share=0.20,
         blanket_fillin=0.04, new_site_share=0.72, new_towers=8),
    _row("S26", False, "skip",
         "Carrier-aggregation bump: 1.5% same-site, 20 km².",
         added_km2=20, added_frac_of_county=0.015, same_site_growth_share=0.95,
         blanket_fillin=0.01, new_site_share=0.03),
    _row("S27", False, "skip",
         "Neighbor-county spill: 38% new, one cross-border tower, 8% frac.",
         added_km2=95, added_frac_of_county=0.08, same_site_growth_share=0.55,
         blanket_fillin=0.05, new_site_share=0.38, new_towers=1,
         new_towers_cross_border=1),
    _row("S28", False, "skip",
         "Unattributed 40% but inference ran; same-site 30% — not the gaming shape.",
         added_km2=60, added_frac_of_county=0.05, same_site_growth_share=0.30,
         blanket_fillin=0.04, new_site_share=0.30, unattributed_share=0.40),
    _row("S29", False, "skip",
         "Just-below-gate 7.4% same-site (prefer FN over a 5% hair-trigger).",
         added_km2=110, added_frac_of_county=0.074, same_site_growth_share=0.92,
         blanket_fillin=0.05, new_site_share=0.06),
    _row("S30", False, "skip",
         "ASR absent + large new-site build (construction not in ASR yet).",
         added_km2=350, added_frac_of_county=0.17, same_site_growth_share=0.20,
         blanket_fillin=0.10, new_site_share=0.78, new_towers=3,
         asr_no_new_structure=1.0),
]


def ordinary_background(n: int = 120, seed: int = 7) -> list[dict]:
    """Typical 6-month organic growth: same-site, 1–4.5% of county, ≥10 km²."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        frac = float(rng.uniform(0.010, 0.045))
        same = float(rng.uniform(0.58, 0.96))
        new = float(max(0.0, min(0.35, 1.0 - same - rng.uniform(0.0, 0.08))))
        unattr = float(max(0.0, 1.0 - same - new))
        rows.append(_row(
            f"B{i:03d}", False, "ordinary",
            "Seeded ordinary organic growth.",
            added_km2=float(rng.uniform(12, 60)),
            added_frac_of_county=frac,
            same_site_growth_share=same,
            new_site_share=new,
            unattributed_share=unattr,
            blanket_fillin=float(rng.uniform(0.005, 0.06)),
            coverage_increase_magnitude=float(rng.uniform(0.05, 0.25)),
            asr_no_new_structure=float(rng.random() < 0.4),
        ))
    return rows


def all_scenarios() -> list[dict]:
    return REVIEW + SKIP


def catalog_frame(background_n: int = 120) -> pd.DataFrame:
    rows = all_scenarios() + ordinary_background(background_n)
    return pd.DataFrame(rows)
