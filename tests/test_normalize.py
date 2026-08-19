"""County tagging / clip behaviour for the normalize stage."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import box

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fcc_audit.normalize import assign_counties, clip_hexes_to_target_buffer  # noqa: E402


def _cell(lat: float, lng: float, res: int = 9) -> str:
    return h3.latlng_to_cell(lat, lng, res)


def _counties_for(*points: tuple[str, str, str, float, float], pad: float = 0.4):
    """Tiny synthetic counties: a box around each (lat, lng) centroid."""
    rows = []
    for geoid, name, state, lat, lng in points:
        rows.append({
            "county_geoid": geoid,
            "county_name": name,
            "state_fips": state,
            "geometry": box(lng - pad, lat - pad, lng + pad, lat + pad),
        })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def test_assign_counties_tags_by_centroid():
    sf = _cell(37.77, -122.42)
    counties = _counties_for(("06075", "San Francisco", "06", 37.77, -122.42))
    hex_df = pd.DataFrame({"h3": [sf], "signal_dbm": [-90.0]})
    out = assign_counties(hex_df, counties)
    assert len(out) == 1
    assert out.iloc[0]["county_geoid"] == "06075"
    assert out.iloc[0]["state_fips"] == "06"
    assert out.iloc[0]["signal_dbm"] == -90.0


def test_assign_counties_clips_distant_hexes_to_target_buffer():
    ca = _cell(37.77, -122.42)
    ny = _cell(40.71, -74.01)
    counties = _counties_for(
        ("06075", "San Francisco", "06", 37.77, -122.42),
        ("36061", "New York", "36", 40.71, -74.01),
    )
    hex_df = pd.DataFrame({"h3": [ca, ny], "signal_dbm": [-90.0, -95.0]})
    out = assign_counties(hex_df, counties, clip_to_states=["06"], buffer_m=50_000.0)
    assert set(out["h3"]) == {ca}
    assert out.iloc[0]["county_geoid"] == "06075"


def test_clip_hexes_to_target_buffer_keeps_in_state_hex():
    ca = _cell(37.77, -122.42)
    ny = _cell(40.71, -74.01)
    counties = _counties_for(
        ("06075", "San Francisco", "06", 37.77, -122.42),
        ("36061", "New York", "36", 40.71, -74.01),
    )
    hex_df = pd.DataFrame({"h3": [ca, ny], "signal_dbm": [-90.0, -95.0]})
    out = clip_hexes_to_target_buffer(hex_df, counties, ["06"], buffer_m=50_000.0)
    assert set(out["h3"]) == {ca}


def test_assign_counties_empty_input():
    counties = _counties_for(("06075", "San Francisco", "06", 37.77, -122.42))
    empty = pd.DataFrame(columns=["h3", "signal_dbm"])
    out = assign_counties(empty, counties)
    assert out.empty
    assert "county_geoid" in out.columns


def test_assign_counties_vectorized_path_is_fast():
    """Guard the polygon/iterrows regression: thousands of hexes must be cheap."""
    origin = _cell(39.05, -95.69)  # Kansas
    cells = list(h3.grid_disk(origin, 25))
    counties = _counties_for(("20177", "Shawnee", "20", 39.05, -95.69), pad=2.0)
    hex_df = pd.DataFrame({"h3": cells, "signal_dbm": [-90.0] * len(cells)})
    t0 = time.perf_counter()
    out = assign_counties(hex_df, counties, clip_to_states=["20"], buffer_m=50_000.0)
    elapsed = time.perf_counter() - t0
    assert len(out) == len(cells)
    assert elapsed < 8.0, f"centroid join too slow ({elapsed:.1f}s for {len(cells)} hexes)"
