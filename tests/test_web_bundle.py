"""Smoke tests for web bundle structure (no browser required)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fcc_audit.report import build_county_detail, write_county_details  # noqa: E402
from fcc_audit.webbundle import (  # noqa: E402
    assign_record_tiers,
    build_web_meta,
    build_web_records,
    write_web_bundle,
)


@pytest.fixture
def web_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "web" / "public" / "data"


def test_web_meta_and_records_exist(web_data_dir):
    meta = json.loads((web_data_dir / "meta.json").read_text())
    assert meta.get("providers")
    records_dir = web_data_dir / "records"
    if records_dir.exists():
        split_files = list(records_dir.glob("*/*.json"))
        assert split_files, "expected per-provider split record files"
    else:
        # Legacy bundles may still ship monolithic records.json
        assert (web_data_dir / "records.json").exists()


def test_assign_record_tiers():
    import pandas as pd
    from fcc_audit.webbundle import _tier_for_rank

    assert _tier_for_rank(1) == "red"
    assert _tier_for_rank(50) == "red"
    assert _tier_for_rank(51) == "orange"
    assert _tier_for_rank(100) == "orange"
    assert _tier_for_rank(101) == "yellow"
    assert _tier_for_rank(150) == "yellow"
    assert _tier_for_rank(151) == "green"
    assert _tier_for_rank(250) == "green"
    assert _tier_for_rank(251) is None

    scored = pd.DataFrame([
        {"provider_id": 1, "technology": "5G", "county_geoid": "A", "priority_score": 0.9, "added_km2": 10},
        {"provider_id": 1, "technology": "5G", "county_geoid": "B", "priority_score": 0.8, "added_km2": 20},
        {"provider_id": 1, "technology": "5G", "county_geoid": "C", "priority_score": 0.8, "added_km2": 5},
        {"provider_id": 1, "technology": "5G", "county_geoid": "D", "priority_score": 0.1, "added_km2": 1},
    ])
    tiered = assign_record_tiers(scored, top_n=3)
    by_geoid = tiered.set_index("county_geoid")["tier"].to_dict()
    assert by_geoid["A"] == "red"
    assert by_geoid["B"] == "red"
    assert by_geoid["C"] == "red"
    assert by_geoid["D"] is None or (isinstance(by_geoid["D"], float) and pd.isna(by_geoid["D"]))


def test_build_web_records_includes_tier():
    import pandas as pd

    scored = assign_record_tiers(pd.DataFrame([{
        "provider_id": 130403,
        "provider_name": "AT&T",
        "technology": "5G-NR 7/1",
        "county_geoid": "20001",
        "county_name": "Allen",
        "state_fips": "20",
        "priority_score": 0.95,
        "added_km2": 12.0,
        "flag_for_review": True,
    }]), top_n=250)
    records = build_web_records(scored)
    rec = records["130403"]["5G-NR 7/1"]["20001"]
    assert rec["tier"] == "red"


def test_write_web_bundle_removes_stale_snapshot_files(tmp_path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    web_dir = tmp_path / "web"
    data_dir = web_dir / "public" / "data"
    (data_dir / "records" / "old-provider").mkdir(parents=True)
    (data_dir / "records" / "old-provider" / "old-service.json").write_text("{}")
    (data_dir / "details" / "old-provider").mkdir(parents=True)
    (data_dir / "details" / "old-provider" / "old.json").write_text("{}")
    (data_dir / "towers").mkdir(parents=True)
    (data_dir / "towers" / "old-provider.json").write_text("[]")
    (data_dir / "records.json").write_text('{"stale": true}')

    scored = pd.DataFrame([{
        "provider_id": 130077,
        "provider_name": "AT&T",
        "technology": "5G-NR 7/1",
        "county_geoid": "20001",
        "county_name": "Allen",
        "state_fips": "20",
        "priority_score": 0.5,
        "flag_for_review": False,
        "added_km2": 0.0,
    }])
    counties = gpd.GeoDataFrame([{
        "county_geoid": "20001",
        "county_name": "Allen",
        "state_fips": "20",
        "geometry": box(-96.0, 37.5, -95.5, 38.0),
    }], crs="EPSG:4326")

    write_web_bundle(
        scored, pd.DataFrame(), counties, web_dir,
        {"current": "277", "prior": "279", "states_processed": "20"},
    )

    assert not (data_dir / "records.json").exists()
    assert not (data_dir / "records" / "old-provider").exists()
    assert not (data_dir / "details" / "old-provider").exists()
    assert not (data_dir / "towers" / "old-provider.json").exists()
    assert (data_dir / "records" / "130077" / "5G-NR7-1.json").exists()
    meta = json.loads((data_dir / "meta.json").read_text())
    assert meta["provider_services"] == {"130077": ["5G-NR 7/1"]}


def test_build_web_meta_lists_services_by_provider():
    import pandas as pd

    scored = pd.DataFrame([
        {"provider_id": 1, "provider_name": "One", "technology": "5G-NR 7/1", "flag_for_review": False},
        {"provider_id": 1, "provider_name": "One", "technology": "5G-NR 35/3", "flag_for_review": False},
        {"provider_id": 2, "provider_name": "Two", "technology": "5G-NR 7/1", "flag_for_review": False},
    ])
    meta = build_web_meta(scored, {"current": "277", "prior": "279"})
    assert meta["provider_services"] == {
        "1": ["5G-NR 35/3", "5G-NR 7/1"],
        "2": ["5G-NR 7/1"],
    }


def test_estimate_signal_from_sites_flat_only():
    import h3

    from fcc_audit.webbundle import _encode_signal, _estimate_signal_from_sites

    center = h3.latlng_to_cell(38.5, -98.5, 9)
    cells = list(h3.grid_disk(center, 12))
    flat = [[c, _encode_signal(0.0)] for c in cells]
    site = [{"lat": 38.5, "lng": -98.5}]

    out, estimated = _estimate_signal_from_sites(flat, site)
    assert estimated
    # Gradient present: many distinct levels, decaying with distance.
    assert len({enc for _c, enc in out}) > 5
    by_cell = dict(out)
    assert by_cell[str(center)] > min(enc for _c, enc in out)

    # Varied signal passes through untouched.
    varied = [[cells[0], 10], [cells[1], 40]]
    same, estimated2 = _estimate_signal_from_sites(varied, site)
    assert not estimated2 and same == varied

    # Flat but no sites: leave as-is (nothing to estimate from).
    same2, estimated3 = _estimate_signal_from_sites(flat, [])
    assert not estimated3 and same2 == flat


def test_write_county_details_caps_to_tiered(tmp_path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    rows = []
    for i in range(5):
        rows.append({
            "provider_id": 130403,
            "technology": "5G-NR 7/1",
            "county_geoid": f"90{i:03d}",
            "priority_score": 1.0 - i * 0.1,
            "added_km2": float(i),
            "prior_towers": 0,
            "current_towers": 0,
            "new_towers": 0,
        })
    scored = assign_record_tiers(pd.DataFrame(rows), top_n=2)
    coverage = pd.DataFrame([
        {"provider_id": 130403, "technology": "5G-NR 7/1", "county_geoid": g,
         "vintage": "prior", "h3": "8826e64247fffff", "signal_dbm": -95.0}
        for g in ["90000", "90001", "90002", "90003", "90004"]
    ])
    sites = pd.DataFrame()
    counties = gpd.GeoDataFrame([
        {"county_geoid": g, "county_name": f"C{g}", "state_fips": "90",
         "geometry": box(-99.0, 39.0, -98.5, 39.5)}
        for g in ["90000", "90001", "90002", "90003", "90004"]
    ], crs="EPSG:4326")
    data_dir = tmp_path / "data"
    n = write_county_details(
        scored, coverage, sites, data_dir, {"prior": "a", "current": "b"}, counties=counties,
    )
    assert n == 2
    detail_dir = data_dir / "details" / "130403" / "5G-NR7-1"
    assert (detail_dir / "90000.json").exists()
    assert (detail_dir / "90001.json").exists()
    assert not (detail_dir / "90002.json").exists()


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


def test_build_web_meta_marks_incomplete_and_flag_threshold():
    import pandas as pd

    scored = pd.DataFrame([{
        "provider_id": 130403,
        "provider_name": "T-Mobile",
        "technology": "5G-NR 7/1",
        "county_geoid": "20001",
        "flag_for_review": True,
        "priority_score": 0.9,
    }])
    meta = build_web_meta(
        scored,
        {
            "current": "277",
            "prior": "279",
            "states_processed": "20",
            "incomplete": True,
            "flag_threshold": 0.42,
            "feature_weights": {"added_frac_of_county": 0.25},
        },
    )
    assert meta["incomplete"] is True
    assert meta["flag_threshold"] == 0.42
    assert meta["feature_weights"]["added_frac_of_county"] == 0.25


def test_flag_math_prefers_score_contributions():
    import pandas as pd

    scored = pd.DataFrame([{
        "provider_id": 130403,
        "provider_name": "T-Mobile",
        "technology": "5G-NR 7/1",
        "county_geoid": "20001",
        "county_name": "Allen",
        "state_fips": "20",
        "priority_score": 0.55,
        "flag_for_review": True,
        "added_frac_of_county": 0.2,
        "score_contribution_added_frac_of_county": 0.11,
    }])
    records = build_web_records(
        scored, threshold=0.4, weights={"added_frac_of_county": 0.25},
    )
    fm = records["130403"]["5G-NR 7/1"]["20001"]["flag_math"]
    assert fm["flag_threshold"] == 0.4
    feat = next(f for f in fm["features"] if f["name"] == "added_frac_of_county")
    assert feat["contribution"] == 0.11
