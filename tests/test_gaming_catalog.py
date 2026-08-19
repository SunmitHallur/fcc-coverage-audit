"""Extensive gaming-physics catalog: review vs skip vs ordinary background.

These cases are generated from coverage-change physics (lobe merge, new macros,
modest antenna growth, recoding, inference failure). They are *not* the 16
screenshot FIPS — those are too few to pin weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h3
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(TESTS))

from fcc_audit.attribute import attribute_changes  # noqa: E402
from fcc_audit.changedetect import county_change, hex_change  # noqa: E402
from fcc_audit.config import load_config  # noqa: E402
from fcc_audit.score import build_features, score  # noqa: E402
from fcc_audit.towers import infer_sites_joint  # noqa: E402

from gaming_scenarios import (  # noqa: E402
    REVIEW,
    SKIP,
    catalog_frame,
    ordinary_background,
)


@pytest.fixture
def cfg():
    return load_config()


def _score_catalog(cfg, background_n=120):
    feats = catalog_frame(background_n)
    return score(feats, cfg), feats


def test_every_review_scenario_flags_in_a_realistic_cohort(cfg):
    scored, _ = _score_catalog(cfg)
    by = scored.set_index("county_geoid")
    missed = [r["county_geoid"] for r in REVIEW if not bool(by.loc[r["county_geoid"], "flag_for_review"])]
    assert missed == [], f"review physics missed: {missed}"


def test_every_skip_scenario_does_not_flag_in_a_realistic_cohort(cfg):
    scored, _ = _score_catalog(cfg)
    by = scored.set_index("county_geoid")
    false_pos = [r["county_geoid"] for r in SKIP if bool(by.loc[r["county_geoid"], "flag_for_review"])]
    assert false_pos == [], f"skip physics false-flagged: {false_pos}"


def test_ordinary_organic_background_is_not_flagged(cfg):
    scored, raw = _score_catalog(cfg)
    bg = raw[raw["family"] == "ordinary"]["county_geoid"]
    flagged = scored[scored["county_geoid"].isin(bg) & scored["flag_for_review"]]
    assert flagged.empty, (
        f"{len(flagged)} ordinary organic counties flagged; "
        f"examples={flagged['county_geoid'].head(8).tolist()}"
    )


def test_all_ordinary_state_does_not_mint_percentile_flags(cfg):
    """A batch of only modest same-site growth must not flag the top 5%."""
    rows = ordinary_background(200, seed=21)
    scored = score(pd.DataFrame(rows), cfg)
    assert int(scored["flag_for_review"].sum()) == 0


@pytest.mark.parametrize("scenario", REVIEW + SKIP, ids=lambda s: s["county_geoid"])
def test_each_named_scenario_alone_matches_expectation(cfg, scenario):
    """Small-cohort path (implausibility only) agrees with the label."""
    scored = score(pd.DataFrame([scenario]), cfg)
    got = bool(scored.iloc[0]["flag_for_review"])
    assert got is scenario["expect_flag"], scenario["why"]


def test_same_area_same_site_scores_higher_than_new_site_build(cfg):
    same = {
        "case": "same", "county_geoid": "90001", "added_km2": 400.0,
        "added_frac_of_county": 0.20, "coverage_increase_magnitude": 0.6,
        "blanket_fillin": 0.15, "same_site_growth_share": 0.95,
        "unattributed_share": 0.02, "boundary_snap_share": 0.0,
        "new_site_share": 0.03, "asr_no_new_structure": 0.0,
        "new_towers": 0, "new_towers_cross_border": 0, "provider_id": 1,
        "provider_name": "X", "technology": "5G-NR 7/1", "county_name": "A",
    }
    build = dict(same)
    build.update({
        "case": "build", "county_geoid": "90002",
        "same_site_growth_share": 0.15, "new_site_share": 0.82, "new_towers": 3,
        "blanket_fillin": 0.15,
    })
    scored = score(pd.DataFrame([same, build]), cfg)
    s_same = float(scored.loc[scored["case"] == "same", "priority_score"].iloc[0])
    s_build = float(scored.loc[scored["case"] == "build", "priority_score"].iloc[0])
    assert s_same > s_build
    assert bool(scored.loc[scored["case"] == "same", "flag_for_review"].iloc[0])
    assert not bool(scored.loc[scored["case"] == "build", "flag_for_review"].iloc[0])


def test_rural_blanket_ranks_above_modest_urban_bump(cfg):
    scored, _ = _score_catalog(cfg)
    by = scored.set_index("county_geoid")
    rural = float(by.loc["R04", "priority_score"])
    modest = float(by.loc["S03", "priority_score"])
    assert rural > modest


def test_review_scores_are_all_above_ordinary_median(cfg):
    scored, raw = _score_catalog(cfg)
    med = float(scored.loc[scored["county_geoid"].isin(
        raw.loc[raw["family"] == "ordinary", "county_geoid"]
    ), "priority_score"].median())
    review_geoids = [r["county_geoid"] for r in REVIEW]
    lows = scored.loc[
        scored["county_geoid"].isin(review_geoids)
        & (scored["priority_score"] <= med),
        "county_geoid",
    ].tolist()
    assert lows == [], f"review cases at/below ordinary median: {lows} (median={med:.3f})"


def test_catalog_precision_and_recall_against_physics_labels(cfg):
    scored, raw = _score_catalog(cfg)
    labeled = raw[raw["family"].isin(["review", "skip"])]
    merged = labeled.merge(
        scored[["county_geoid", "flag_for_review"]], on="county_geoid", how="left"
    )
    expect = merged["expect_flag"].astype(bool)
    got = merged["flag_for_review"].astype(bool)
    tp = int((got & expect).sum())
    fp = int((got & ~expect).sum())
    fn = int((~got & expect).sum())
    assert fp == 0, merged.loc[got & ~expect, ["county_geoid", "why"]].to_dict("records")
    assert fn == 0, merged.loc[~got & expect, ["county_geoid", "why"]].to_dict("records")
    assert tp == len(REVIEW)


def test_ordinary_high_score_is_ranked_not_flagged(cfg):
    """Percentile badges still exist; they must not become review flags."""
    scored = score(pd.DataFrame(ordinary_background(80, seed=3)), cfg)
    assert int(scored["flag_for_review"].sum()) == 0
    assert int((scored["severity"] == "Top 5%").sum()) >= 1
    row = next(s for s in SKIP if s["county_geoid"] == "S17")
    scored = score(pd.DataFrame(ordinary_background(60) + [row]), cfg)
    hit = scored[scored["county_geoid"] == "S17"].iloc[0]
    assert not bool(hit["flag_for_review"])


# ---------------------------------------------------------------------------
# Hex geometry grid: inference → attribution → score (not feature stubs)
# ---------------------------------------------------------------------------

_EXISTING = [
    (30.52, -89.68),
    (30.58, -89.62),
    (30.55, -89.75),
    (30.45, -89.70),
    (30.62, -89.78),
    (30.48, -89.55),
]
_NEW = [
    (30.72, -89.48),
    (30.38, -89.88),
    (30.34, -89.58),
]


def _disk(lat, lng, rings, geoid="28109", signal=-80.0):
    origin = h3.latlng_to_cell(lat, lng, 9)
    rows = []
    for c in h3.grid_disk(origin, rings):
        d = h3.grid_distance(origin, c)
        rows.append({
            "h3": c,
            "signal_dbm": float(signal) - 2.0 * d,
            "county_geoid": geoid,
            "county_name": "Geom",
            "state_fips": geoid[:2],
        })
    return pd.DataFrame(rows)


def _merge(parts):
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values("signal_dbm", ascending=False)
        .drop_duplicates("h3")
        .reset_index(drop=True)
    )


def _hex_score(cfg, prior, current, county_km2):
    cfg.raw["geography"]["site_h3_resolution"] = 9
    cfg.raw["towers"]["min_site_hexes"] = 25
    _ps, cs = infer_sites_joint(prior, current, cfg)
    ch = hex_change(prior, current)
    cc = county_change(ch, 9, {"28109": county_km2})
    attr = attribute_changes(ch, cs, 9)
    feats = build_features(cc, attr)
    scored = score(feats, cfg)
    row = scored[scored["county_geoid"].astype(str) == "28109"].iloc[0]
    return row, cs


# prior_rings, current_rings, n_existing, n_new, county_km2, expect_flag, note
_HEX_CASES = [
    (8, 22, 3, 0, 400, True, "three lobes merge into a blanket"),
    (6, 16, 4, 0, 600, True, "four macros double radius in a small county"),
    (8, 24, 1, 0, 350, True, "one rural lobe covers half a small county"),
    (8, 24, 1, 0, 5000, False, "same lobe growth is ~3% of a huge county"),
    (10, 12, 2, 0, 2500, False, "modest two-tower antenna growth"),
    (8, 9, 6, 0, 3000, False, "six towers grow one ring"),
    (12, 12, 2, 0, 2000, False, "identical footprints"),
    (10, 10, 1, 2, 2800, False, "two new distant macros, old lobe unchanged"),
    (8, 10, 2, 2, 2200, False, "mostly new blooms plus slight old growth"),
    (10, 11, 3, 0, 1800, False, "three towers grow one ring in a large county"),
    (7, 20, 2, 0, 500, True, "two-site rural fill of a small county"),
    (9, 9, 1, 0, 2000, False, "stable single site"),
    (16, 16, 1, 0, 2300, False, "same footprint, ignore signal recoding in hex test"),
    (4, 8, 1, 0, 2800, False, "one lobe grows in a huge county, well under 7.5%"),
]


@pytest.mark.parametrize(
    "pr,cr,n_ex,n_new,km2,expect,note",
    _HEX_CASES,
    ids=[c[-1] for c in _HEX_CASES],
)
def test_hex_geometry_grid(pr, cr, n_ex, n_new, km2, expect, note):
    cfg = load_config()
    cfg.raw["geography"]["site_h3_resolution"] = 9
    cfg.raw["towers"]["min_site_hexes"] = 25
    prior = _merge([_disk(*t, pr) for t in _EXISTING[:n_ex]])
    current = _merge(
        [_disk(*t, cr) for t in _EXISTING[:n_ex]]
        + [_disk(*t, 10) for t in _NEW[:n_new]]
    )
    row, cs = _hex_score(cfg, prior, current, km2)
    assert bool(row["flag_for_review"]) is expect, (
        f"{note}: flag={bool(row['flag_for_review'])} expect={expect} "
        f"frac={float(row['added_frac_of_county']):.3f} "
        f"same={float(row['same_site_growth_share']):.2f} "
        f"new={float(row['new_site_share']):.2f} "
        f"added_km2={float(row['added_km2']):.1f} sites={len(cs)}"
    )


def test_new_blooms_are_classified_new_site_not_expanded():
    cfg = load_config()
    cfg.raw["geography"]["site_h3_resolution"] = 9
    cfg.raw["towers"]["min_site_hexes"] = 25
    prior = _disk(*_EXISTING[0], 10)
    current = _merge([_disk(*_EXISTING[0], 10), _disk(*_NEW[0], 10), _disk(*_NEW[1], 10)])
    row, cs = _hex_score(cfg, prior, current, 2800.0)
    assert int((cs["site_class"] == "new_site").sum()) >= 2
    assert float(row["new_site_share"]) >= 0.45
    assert not bool(row["flag_for_review"])


def test_fill_between_existing_towers_is_same_site():
    cfg = load_config()
    cfg.raw["geography"]["site_h3_resolution"] = 9
    cfg.raw["towers"]["min_site_hexes"] = 25
    towers = _EXISTING[:3]
    prior = _merge([_disk(*t, 7) for t in towers])
    current = _merge([_disk(*t, 18) for t in towers])
    row, cs = _hex_score(cfg, prior, current, 450.0)
    assert int(row["new_towers"] or 0) <= 1
    assert float(row["same_site_growth_share"]) >= 0.70
    assert bool(row["flag_for_review"])
