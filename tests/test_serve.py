"""On-demand county extract: parquet hexes + GeoPackage boundary → JSON."""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fcc_audit.config import load_config  # noqa: E402
from fcc_audit.serve import batch_paths_for_state, county_slice, run_dir_for  # noqa: E402


def test_batch_paths_for_state_matches_overnight_keys(tmp_path):
    sites = tmp_path / "sites"
    sites.mkdir()
    (sites / "sites_20-31-38.parquet").write_bytes(b"")
    (sites / "sites_01-02.parquet").write_bytes(b"")
    (sites / "sites_all.parquet").write_bytes(b"")
    names = {p.name for p in batch_paths_for_state(sites, "sites_", "20")}
    assert "sites_20-31-38.parquet" in names
    assert "sites_all.parquet" in names
    assert "sites_01-02.parquet" not in names


def test_county_slice_extracts_hexes_from_state_parquet(tmp_path):
    cfg = load_config()
    cfg.project_root = tmp_path
    cfg.raw["source"]["backend"] = "redshift"
    cfg.raw["analysis"]["vintages"]["current"] = "277"
    cfg.raw["analysis"]["vintages"]["prior"] = "279"

    run_dir = run_dir_for(cfg)
    cov_dir = run_dir / "coverage"
    cov_dir.mkdir(parents=True)
    pd.DataFrame({
        "h3": ["8928308280fffff", "8928308281fffff", "other"],
        "signal_dbm": [0.0, 0.0, 0.0],
        "county_geoid": ["20155", "20155", "20001"],
        "provider_id": [130077, 130077, 130077],
        "technology": ["5G-NR 7/1", "5G-NR 7/1", "5G-NR 7/1"],
        "vintage": ["current", "prior", "current"],
    }).to_parquet(cov_dir / "coverage_20.parquet", index=False)

    counties = gpd.GeoDataFrame(
        {
            "county_geoid": ["20155"],
            "county_name": ["Reno"],
            "state_fips": ["20"],
            "geometry": [box(-99, 37, -97, 39)],
        },
        crs="EPSG:4326",
    )

    detail = county_slice(
        cfg, "20155", 130077, "5G-NR 7/1", counties=counties,
    )
    assert detail["geoid"] == "20155"
    assert detail["source"] == "api"
    assert "county_boundary" in detail
    hex_ids = {row[0] for row in (detail["prior_hexes"] + detail["current_hexes"])}
    assert "8928308280fffff" in hex_ids or "8928308281fffff" in hex_ids
    assert "other" not in hex_ids
