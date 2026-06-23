"""Normalize raw coverage vectors into a county-tagged H3 hex coverage table.

Output schema (one row per occupied hex):
    h3            : H3 res-N cell id (str)
    signal_dbm    : strongest modeled signal band in that hex (float, dBm)
    county_geoid  : 5-digit county FIPS (str)
    county_name   : county name (str)
    state_fips    : 2-digit state FIPS (str)

We index to H3 res-8 (the geography the FCC's own mobile audits use) for county
reporting, and can re-index to res-9 for finer tower clustering.
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd

from .acquire import CoverageFile
from .config import Config

log = logging.getLogger(__name__)

# Candidate attribute names for the modeled signal-strength band across vintages.
_SIGNAL_COLUMNS = ["minsignal", "min_signal", "signal", "sig_strength", "signalstr"]
# Some vintages encode signal as an ordinal band code rather than dBm.
_BAND_CODE_TO_DBM = {1: -105.0, 2: -95.0, 3: -85.0}
# Attribute names for speed tier and environment (case-insensitive lookup).
_MINDOWN_COLUMNS = ["mindown", "min_down", "min_dl"]
_MINUP_COLUMNS = ["minup", "min_up", "min_ul"]
_ENV_COLUMNS = ["environmnt", "environment", "env"]


def _find_col(gdf: gpd.GeoDataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in gdf.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def filter_by_tier(gdf: gpd.GeoDataFrame, tier_spec: dict) -> gpd.GeoDataFrame:
    """Keep polygons matching this speed tier (mindown/minup). No-op if the file
    has no speed columns (already a single-tier file)."""
    import numpy as np

    md = _find_col(gdf, _MINDOWN_COLUMNS)
    mu = _find_col(gdf, _MINUP_COLUMNS)
    if md is None or mu is None:
        return gdf
    keep = np.isclose(gdf[md].astype(float), float(tier_spec["mindown"]), atol=0.06) & \
        np.isclose(gdf[mu].astype(float), float(tier_spec["minup"]), atol=0.06)
    return gdf[keep]


def filter_by_environment(gdf: gpd.GeoDataFrame, env_codes: list[int] | None) -> gpd.GeoDataFrame:
    """Keep polygons whose environment is in env_codes. No-op when env_codes is
    None (combined) or the file has no environment column."""
    if not env_codes:
        return gdf
    col = _find_col(gdf, _ENV_COLUMNS)
    if col is None:
        return gdf
    return gdf[gdf[col].isin(env_codes)]


def load_coverage_gdf(path: Path) -> gpd.GeoDataFrame:
    """Read a coverage layer (zipped shapefile / gpkg / geojson) in EPSG:4326."""
    suffix = path.suffix.lower()
    if suffix == ".zip":
        # geopandas/pyogrio can read a zipped shapefile directly.
        gdf = gpd.read_file(f"zip://{path}")
    else:
        gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def detect_signal_column(gdf: gpd.GeoDataFrame) -> str | None:
    lower = {c.lower(): c for c in gdf.columns}
    for cand in _SIGNAL_COLUMNS:
        if cand in lower:
            return lower[cand]
    return None


def _to_dbm(value: float) -> float:
    """Coerce a signal value to dBm, treating small ints as band codes."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if v in _BAND_CODE_TO_DBM:  # ordinal band code
        return _BAND_CODE_TO_DBM[v]
    return v


def coverage_to_hex(
    gdf: gpd.GeoDataFrame, resolution: int, signal_col: str | None
) -> pd.DataFrame:
    """Fill polygons with H3 cells, keeping the strongest signal band per hex."""
    best: dict[str, float] = {}
    for geom, sig in zip(
        gdf.geometry, (gdf[signal_col] if signal_col else [None] * len(gdf))
    ):
        if geom is None or geom.is_empty:
            continue
        dbm = _to_dbm(sig) if sig is not None else 0.0
        try:
            cells = h3.geo_to_cells(geom, resolution)
        except Exception:  # noqa: BLE001 - h3 raises on odd geometries
            # Fall back to filling the geometry's representative point.
            pt = geom.representative_point()
            cells = [h3.latlng_to_cell(pt.y, pt.x, resolution)]
        for c in cells:
            if c not in best or dbm > best[c]:
                best[c] = dbm
    if not best:
        return pd.DataFrame(columns=["h3", "signal_dbm"])
    return pd.DataFrame({"h3": list(best.keys()), "signal_dbm": list(best.values())})


def load_counties(cfg: Config) -> gpd.GeoDataFrame:
    """Load (and cache/download) TIGER county boundaries in EPSG:4326."""
    cache = cfg.path("interim") / "tl_us_county.gpkg"
    if cache.exists():
        return gpd.read_file(cache)

    url = cfg.geography["counties_url"]
    raw = cfg.path("raw") / "tl_us_county.zip"
    if not raw.exists():
        import requests

        log.info("downloading county boundaries: %s", url)
        resp = requests.get(url, timeout=300, headers={"user-agent": "fcc-coverage-audit/0.1"})
        resp.raise_for_status()
        raw.write_bytes(resp.content)
    # Validate it is a real zip (helps when behind a proxy returning HTML).
    if not zipfile.is_zipfile(raw):
        raise RuntimeError(f"County download is not a zip archive: {raw}")

    gdf = gpd.read_file(f"zip://{raw}")[["GEOID", "NAME", "STATEFP", "geometry"]]
    gdf = gdf.rename(columns={"GEOID": "county_geoid", "NAME": "county_name", "STATEFP": "state_fips"})
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(cache, driver="GPKG")
    return gdf


def county_areas_km2(counties: gpd.GeoDataFrame, equal_area_crs: str = "EPSG:5070") -> dict[str, float]:
    """Map county GEOID -> land area in km^2 (equal-area projection)."""
    proj = counties.to_crs(equal_area_crs)
    areas = proj.geometry.area / 1e6  # m^2 -> km^2
    return dict(zip(counties["county_geoid"].astype(str), areas))


def boundary_snap_share(
    change_df: pd.DataFrame,
    counties: gpd.GeoDataFrame,
    threshold_m: float = 1500.0,
    equal_area_crs: str = "EPSG:5070",
) -> pd.DataFrame:
    """Per-county share of newly-covered hexes that hug the county boundary.

    A high share means new coverage aligns to the administrative boundary rather
    than radiating from towers - a classic gaming tell (coverage drawn to match
    an eligibility/county outline). Returns columns [county_geoid, boundary_snap_share].

    Note: this checks the *county* boundary as a proxy. True 5G-Fund eligibility
    boundaries would be more precise; add that polygon layer here when available.
    """
    import h3

    gained = change_df[change_df["status"] == "new"].dropna(subset=["county_geoid"]).copy()
    if gained.empty:
        return pd.DataFrame(columns=["county_geoid", "boundary_snap_share"])

    centers = [h3.cell_to_latlng(c) for c in gained["h3"]]
    pts = gpd.GeoSeries(
        gpd.points_from_xy([lng for _la, lng in centers], [la for la, _lng in centers]),
        crs="EPSG:4326",
    ).to_crs(equal_area_crs)
    gained = gained.assign(_geom=pts.values)

    boundaries = counties.to_crs(equal_area_crs)
    boundary_by_geoid = {
        str(r["county_geoid"]): r.geometry.boundary for _, r in boundaries.iterrows()
    }

    rows = []
    for geoid, grp in gained.groupby("county_geoid"):
        b = boundary_by_geoid.get(str(geoid))
        if b is None:
            share = 0.0
        else:
            dists = gpd.GeoSeries(grp["_geom"].values, crs=equal_area_crs).distance(b)
            share = float((dists <= threshold_m).mean())
        rows.append({"county_geoid": str(geoid), "boundary_snap_share": share})
    return pd.DataFrame(rows)


def assign_counties(hex_df: pd.DataFrame, counties: gpd.GeoDataFrame) -> pd.DataFrame:
    """Attach county attributes via each hex centroid (point-in-polygon join)."""
    if hex_df.empty:
        return hex_df.assign(county_geoid=None, county_name=None, state_fips=None)
    centers = [h3.cell_to_latlng(c) for c in hex_df["h3"]]
    pts = gpd.GeoDataFrame(
        hex_df.copy(),
        geometry=gpd.points_from_xy(
            [lng for _lat, lng in centers], [lat for lat, _lng in centers]
        ),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, counties, how="left", predicate="within")
    return pd.DataFrame(
        joined[["h3", "signal_dbm", "county_geoid", "county_name", "state_fips"]]
    )


def normalize_layer(
    cfg: Config,
    cov: CoverageFile,
    counties: gpd.GeoDataFrame,
    resolution: int,
    tier_label: str,
    tier_spec: dict,
    env_label: str = "all",
    env_codes: list[int] | None = None,
) -> pd.DataFrame:
    """Normalize one (provider, tech, tier, env-group) layer to county-tagged
    hexes, filtering the source file by speed tier and environment. Cached."""
    cache = (
        cfg.path("interim")
        / f"hex_{cov.vintage}_{cov.provider_id}_{cov.technology}"
        f"_{tier_label}_{env_label}_r{resolution}.parquet"
    )
    if cache.exists():
        return pd.read_parquet(cache)

    gdf = load_coverage_gdf(cov.local_path)
    gdf = filter_by_tier(gdf, tier_spec)
    gdf = filter_by_environment(gdf, env_codes)
    signal_col = detect_signal_column(gdf)
    if signal_col is None:
        log.warning("no signal column found in %s; using flat band", cov.local_path.name)
    hex_df = coverage_to_hex(gdf, resolution, signal_col)
    hex_df = assign_counties(hex_df, counties)
    hex_df["provider_id"] = cov.provider_id
    hex_df["technology"] = cov.technology
    hex_df["speed_tier"] = tier_label
    hex_df["environment"] = env_label
    hex_df["vintage"] = cov.vintage
    hex_df.to_parquet(cache, index=False)
    return hex_df
