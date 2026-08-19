"""June→December gaming patterns from the engineer slides — physics, not labels.

Pearl River / Middlesex: existing lobes expand or blanket-fill without new
sites → review. Edmunds: negligible in-county area → skip. Menard: signal
shift, no added area → skip. New distinct blooms → legitimate build, skip.
Missing ASR alone must not flag (rooftops/small cells are unregistered).
"""
from __future__ import annotations

import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fcc_audit.attribute import (  # noqa: E402
    anchor_sites_to_asr,
    attribute_changes,
)
from fcc_audit.changedetect import county_change, hex_change  # noqa: E402
from fcc_audit.config import load_config  # noqa: E402
from fcc_audit.score import build_features, score  # noqa: E402
from fcc_audit.towers import infer_sites_joint  # noqa: E402

GEOID = "28109"


@pytest.fixture
def cfg():
    c = load_config()
    c.raw["geography"]["site_h3_resolution"] = 9
    c.raw["towers"]["min_site_hexes"] = 25
    return c


def _disk(lat, lng, rings, signal=-80.0):
    origin = h3.latlng_to_cell(lat, lng, 9)
    rows = []
    for c in h3.grid_disk(origin, rings):
        d = h3.grid_distance(origin, c)
        rows.append({
            "h3": c,
            "signal_dbm": float(signal) - 2.0 * d,
            "county_geoid": GEOID,
            "county_name": "Pearl River",
            "state_fips": "28",
        })
    return pd.DataFrame(rows)


def _merge_disks(parts):
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values("signal_dbm", ascending=False)
        .drop_duplicates("h3")
        .reset_index(drop=True)
    )


def _score_pair(prior, current, cfg, county_km2=800.0, extra=None):
    prior_sites, current_sites = infer_sites_joint(prior, current, cfg)
    ch = hex_change(prior, current)
    cc = county_change(ch, 9, {GEOID: county_km2})
    attr = attribute_changes(ch, current_sites, 9)
    feats = build_features(cc, attr)
    if extra:
        for k, v in extra.items():
            feats[k] = v
    scored = score(feats, cfg)
    row = scored[scored["county_geoid"].astype(str) == GEOID]
    assert not row.empty
    return row.iloc[0], current_sites, prior_sites


def test_asr_snap_moves_pin_onto_mast():
    sites = pd.DataFrame({
        "site_id": ["A"],
        "lat": [30.5250],
        "lng": [-89.6770],
        "n_hexes": [40],
        "county_geoid": [GEOID],
        "site_class": ["stable_site"],
        "reach_m": [8000.0],
    })
    asr = pd.DataFrame({"lat": [30.5254], "lng": [-89.6773]})  # ~50 m
    out = anchor_sites_to_asr(sites, asr, radius_m=2000, snap_radius_m=500)
    assert bool(out.iloc[0]["asr_snapped"])
    assert abs(float(out.iloc[0]["lat"]) - 30.5254) < 1e-9
    assert abs(float(out.iloc[0]["lng"]) + 89.6773) < 1e-9
    assert float(out.iloc[0]["asr_distance_m"]) < 80


def test_two_peaks_on_one_mast_collapse():
    """Cloverleaf petals near one ASR become one reviewer pin."""
    sites = pd.DataFrame({
        "site_id": ["p1", "p2"],
        "lat": [30.5250, 30.5265],
        "lng": [-89.6770, -89.6760],
        "n_hexes": [80, 30],
        "county_geoid": [GEOID, GEOID],
        "site_class": ["stable_site", "stable_site"],
        "reach_m": [8000.0, 8000.0],
    })
    asr = pd.DataFrame({"lat": [30.5254], "lng": [-89.6773]})
    out = anchor_sites_to_asr(sites, asr, radius_m=2000, snap_radius_m=500)
    assert len(out) == 1
    assert bool(out.iloc[0]["asr_snapped"])


def test_ambiguous_two_masts_does_not_steal_neighbor():
    """Peak between two macros must not snap onto the slightly closer one."""
    sites = pd.DataFrame({
        "site_id": ["A"],
        "lat": [30.5240],
        "lng": [-89.6770],
        "n_hexes": [40],
        "county_geoid": [GEOID],
        "site_class": ["stable_site"],
        "reach_m": [8000.0],
    })
    asr = pd.DataFrame({
        "lat": [30.5255, 30.5228],
        "lng": [-89.6770, -89.6770],
    })
    out = anchor_sites_to_asr(sites, asr, radius_m=2000, snap_radius_m=500)
    assert bool(out.iloc[0]["asr_matched"])
    assert not bool(out.iloc[0]["asr_snapped"])
    assert abs(float(out.iloc[0]["lat"]) - 30.5240) < 1e-9


def test_modest_same_site_lobe_growth_does_not_flag(cfg):
    """Ordinary 6-month antenna/software growth is not a review case."""
    towers = [(30.52, -89.68), (30.58, -89.62)]
    prior = _merge_disks([_disk(*t, 10) for t in towers])
    current = _merge_disks([_disk(*t, 12) for t in towers])
    row, _, _ = _score_pair(prior, current, cfg, county_km2=2500.0)
    assert float(row["same_site_growth_share"]) >= 0.50
    assert float(row["added_frac_of_county"]) < 0.05
    assert not bool(row["flag_for_review"])


def test_far_asr_is_matched_but_not_snapped():
    sites = pd.DataFrame({
        "site_id": ["A"],
        "lat": [30.52],
        "lng": [-89.68],
        "n_hexes": [40],
        "county_geoid": [GEOID],
        "site_class": ["stable_site"],
        "reach_m": [8000.0],
    })
    asr = pd.DataFrame({"lat": [30.53], "lng": [-89.67]})  # ~1.4 km
    out = anchor_sites_to_asr(sites, asr, radius_m=2000, snap_radius_m=500)
    assert bool(out.iloc[0]["asr_matched"])
    assert not bool(out.iloc[0]["asr_snapped"])
    assert abs(float(out.iloc[0]["lat"]) - 30.52) < 1e-9


def test_middlesex_style_blanket_from_existing_lobes_can_flag(cfg):
    towers = [(30.52, -89.68), (30.58, -89.62), (30.55, -89.75)]
    prior = _merge_disks([_disk(*t, 8) for t in towers])
    current = _merge_disks([_disk(*t, 22) for t in towers])
    row, cur_sites, _ = _score_pair(prior, current, cfg, county_km2=400.0)
    assert int(row["new_towers"] or 0) <= 1
    assert float(row["same_site_growth_share"]) >= 0.50
    assert float(row["added_km2"]) >= 10
    assert bool(row["flag_for_review"])


def test_edmunds_style_new_blooms_are_new_sites_not_same_site_gaming(cfg):
    prior = _disk(30.52, -89.68, 10)
    current = _merge_disks([
        _disk(30.52, -89.68, 10),
        _disk(30.62, -89.55, 10),
        _disk(30.40, -89.80, 10),
    ])
    row, cur_sites, _ = _score_pair(prior, current, cfg, county_km2=2800.0)
    assert int((cur_sites["site_class"] == "new_site").sum()) >= 2
    assert float(row["same_site_growth_share"]) < 0.50 or not bool(row["flag_for_review"])


def test_menard_style_signal_drop_without_area_gain_does_not_flag(cfg):
    prior = _disk(30.52, -89.68, 16, signal=-70.0)
    current = _disk(30.52, -89.68, 16, signal=-100.0)
    row, _, _ = _score_pair(prior, current, cfg, county_km2=2300.0)
    assert float(row["added_km2"]) < 10 or not bool(row["flag_for_review"])
    assert not bool(row["flag_for_review"])


def test_empty_county_below_area_floor_does_not_flag(cfg):
    prior = pd.DataFrame(columns=["h3", "signal_dbm", "county_geoid", "county_name", "state_fips"])
    current = _disk(30.52, -89.68, 4)
    row, _, _ = _score_pair(prior, current, cfg, county_km2=2800.0)
    assert float(row["added_km2"]) < 10
    assert not bool(row["flag_for_review"])


def test_in_county_new_towers_do_not_flag_in_large_cohort(cfg):
    """Edmunds-style: huge area gain from new macros must not rank as gaming."""
    rows = []
    for i in range(60):
        rows.append({
            "case": f"c{i}",
            "provider_id": 131425,
            "provider_name": "Verizon",
            "technology": "5G-NR 7/1",
            "county_geoid": f"{i:05d}",
            "county_name": "X",
            "added_km2": 20.0 if i else 1500.0,
            "added_frac_of_county": 0.02 if i else 0.53,
            "coverage_increase_magnitude": 0.10 if i else 1.0,
            "blanket_fillin": 0.02 if i else 0.43,
            "same_site_growth_share": 0.20 if i else 0.25,
            "unattributed_share": 0.05,
            "boundary_snap_share": 0.0,
            "new_site_share": 0.30 if i else 0.75,
            "asr_no_new_structure": 0.0,
            "new_towers": 0 if i else 4,
            "new_towers_cross_border": 0,
        })
    scored = score(pd.DataFrame(rows), cfg)
    target = scored[scored["county_geoid"] == "00000"].iloc[0]
    assert not bool(target["flag_for_review"])


def test_rural_peak_650m_from_unique_mast_snaps():
    """Rural macros sit ~650 m from inferred peaks; 750 m snap is still unique."""
    sites = pd.DataFrame({
        "site_id": ["A"],
        "lat": [30.5200],
        "lng": [-89.6800],
        "n_hexes": [40],
        "county_geoid": [GEOID],
        "site_class": ["stable_site"],
        "reach_m": [8000.0],
    })
    # ~650 m north of the peak (1 deg lat ≈ 111 km).
    asr = pd.DataFrame({"lat": [30.52585], "lng": [-89.6800]})
    out = anchor_sites_to_asr(sites, asr, radius_m=2000, snap_radius_m=750)
    assert bool(out.iloc[0]["asr_snapped"])
    assert abs(float(out.iloc[0]["lat"]) - 30.52585) < 1e-9


def test_missing_asr_alone_does_not_flag(cfg):
    """Urban rooftops are often unregistered. ASR absence is not a flag."""
    rows = []
    for i in range(60):
        rows.append({
            "case": f"c{i}",
            "provider_id": 131425,
            "provider_name": "Verizon",
            "technology": "5G-NR 7/1",
            "county_geoid": f"{i:05d}",
            "county_name": "X",
            "added_km2": 20.0,
            "added_frac_of_county": 0.01 if i == 0 else 0.04,
            "coverage_increase_magnitude": 0.05 if i == 0 else 0.15,
            "blanket_fillin": 0.02,
            "same_site_growth_share": 0.25,
            "unattributed_share": 0.05,
            "boundary_snap_share": 0.0,
            "new_site_share": 0.40,
            "asr_no_new_structure": 1.0 if i == 0 else 0.0,
            "new_towers": 2,
            "new_towers_cross_border": 0,
        })
    scored = score(pd.DataFrame(rows), cfg)
    target = scored[scored["county_geoid"] == "00000"].iloc[0]
    assert not bool(target["flag_for_review"])
