"""Smoke tests for web bundle structure (no browser required)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fcc_audit.report import build_county_detail, write_county_details  # noqa: E402


@pytest.fixture
def web_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "web" / "public" / "data"


def test_web_meta_and_records_exist(web_data_dir):
    meta = json.loads((web_data_dir / "meta.json").read_text())
    records = json.loads((web_data_dir / "records.json").read_text())
    assert meta.get("providers")
    assert records


def test_county_detail_fixture_structure():
    import pandas as pd

    coverage = pd.DataFrame([
        {"county_geoid": "90003", "vintage": "prior", "h3": "8826e64247fffff", "signal_dbm": -95.0},
        {"county_geoid": "90003", "vintage": "current", "h3": "8826e64335fffff", "signal_dbm": -85.0},
    ])
    sites = pd.DataFrame([
        {"county_geoid": "90003", "vintage": "prior", "lat": 39.0, "lng": -98.0, "site_class": "prior_site", "n_hexes": 5},
        {"county_geoid": "90003", "vintage": "current", "lat": 39.1, "lng": -98.1, "site_class": "new_site", "n_hexes": 8},
    ])
    detail = build_county_detail("90003", coverage, sites, {"prior": "2025-06-30", "current": "2025-12-31"})
    assert len(detail["prior_hexes"]) == 1
    assert len(detail["current_hexes"]) == 1
    assert detail["sites_prior"][0]["site_class"] == "prior_site"


def test_write_county_details_creates_files(tmp_path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    scored = pd.DataFrame([{
        "provider_id": 130403,
        "technology": "5G-NR 7/1",
        "county_geoid": "90003",
        "prior_towers": 1,
        "current_towers": 2,
        "new_towers": 1,
    }])
    coverage = pd.DataFrame([
        {"provider_id": 130403, "technology": "5G-NR 7/1", "county_geoid": "90003",
         "vintage": "prior", "h3": "8826e64247fffff", "signal_dbm": -95.0},
    ])
    sites = pd.DataFrame()
    counties = gpd.GeoDataFrame([{
        "county_geoid": "90003",
        "county_name": "Charlie County",
        "state_fips": "90",
        "geometry": box(-99.0, 39.0, -98.5, 39.5),
    }], crs="EPSG:4326")
    data_dir = tmp_path / "data"
    n = write_county_details(
        scored, coverage, sites, data_dir, {"prior": "a", "current": "b"}, counties=counties,
    )
    assert n == 1
    out = data_dir / "details" / "130403" / "5G-NR7-1" / "90003.json"
    assert out.exists()
    blob = json.loads(out.read_text())
    assert blob["towers_prior"] == 1
    assert blob["towers_current"] == 2
    # PNG rendering is opt-in (render_pngs=False by default); prior_map key should be absent.
    assert blob.get("prior_map") is None
