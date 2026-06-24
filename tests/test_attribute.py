"""Tests for cross-county tower attribution."""
from __future__ import annotations

import sys
from pathlib import Path

import h3
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fcc_audit import attribute  # noqa: E402
from fcc_audit.config import load_config  # noqa: E402
from fcc_audit.score import build_features, score  # noqa: E402

CHARLIE = "90003"
ALPHA = "90001"


def _site(lat: float, lng: float, *, geoid: str, site_class: str, reach_m: float = 12000.0):
    from pyproj import Transformer

    fwd = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x, y = fwd.transform(lng, lat)
    return {
        "site_id": f"{geoid}-{lat}",
        "lat": lat,
        "lng": lng,
        "x_m": x,
        "y_m": y,
        "reach_m": reach_m,
        "n_hexes": 50,
        "county_geoid": geoid,
        "site_class": site_class,
    }


def test_serving_towers_includes_cross_border():
    lat, lng = 39.05, -98.80
    cell = h3.latlng_to_cell(lat, lng, 8)
    hex_df = pd.DataFrame([{"h3": cell, "county_geoid": CHARLIE, "signal_dbm": -90.0}])
    sites = pd.DataFrame([_site(38.92, -98.80, geoid=ALPHA, site_class="new_site")])

    srv = attribute.serving_towers_by_county(hex_df, sites)
    row = srv.loc[srv["county_geoid"] == CHARLIE].iloc[0]
    assert row["towers_serving"] == 1
    assert row["towers_in_county"] == 0
    assert row["towers_cross_border"] == 1


def test_new_tower_in_neighbor_counted_for_growth():
    lat, lng = 39.05, -98.80
    cell = h3.latlng_to_cell(lat, lng, 8)
    change = pd.DataFrame([{
        "h3": cell,
        "county_geoid": CHARLIE,
        "status": "new",
        "signal_dbm": -90.0,
    }])
    sites = pd.DataFrame([_site(38.92, -98.80, geoid=ALPHA, site_class="new_site")])

    attr = attribute.attribute_changes(change, sites, resolution=8)
    row = attr.iloc[0]
    assert row["added_km2_new_site"] > 0
    assert row["new_towers"] == 1
    assert row["new_towers_in_county"] == 0
    assert row["new_towers_cross_border"] == 1


def test_cross_border_build_not_flagged_when_growth_explained():
    cfg = load_config()
    county_change = pd.DataFrame([{
        "county_geoid": CHARLIE,
        "county_name": "Charlie County",
        "added_km2": 100.0,
        "added_frac_of_county": 0.15,
        "pct_increase": 1.5,
        "prior_cov_frac": 0.1,
        "current_cov_frac": 0.25,
        "mean_signal_delta": 0.0,
    }])
    attr = pd.DataFrame([{
        "county_geoid": CHARLIE,
        "added_km2_new_site": 90.0,
        "added_km2_expanded_site": 5.0,
        "added_km2_unattributed": 5.0,
        "new_towers": 1,
        "new_towers_in_county": 0,
        "new_towers_cross_border": 1,
    }])
    feats = build_features(county_change, attr)
    feats["provider_id"] = 130077
    feats["provider_name"] = "AT&T"
    feats["technology"] = "5G-NR 7/1"
    feats["current_towers_cross_border"] = 1
    scored = score(feats, cfg)
    assert bool(scored.iloc[0]["flag_for_review"]) is False
