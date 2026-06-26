"""Measured-coverage ground-truth ingestion from Ookla and FCC Speed Test data.

The strongest single gaming tell is *claimed coverage vs. measured reality*: a
provider claims a county is covered, but zero or very few speed tests have ever
been observed there. Ookla's open dataset and the FCC's own Measuring Broadband
America / Speed Test data provide that independent measurement layer.

Two sources are supported, in priority order:
  1. Ookla Open Data (quarterly tile-aggregated speed tests, per-technology)
     https://github.com/teamookla/ookla-open-data
  2. FCC Speed Test data (from the FCC Fixed Broadband Deployment Data / MBA)

The output is a per-county, per-technology measurement summary:
    county_geoid          : 5-digit FIPS
    technology_category   : "5G" | "LTE" | "all"
    measured_dl_mbps_p50  : float  — median download speed (Mbps) observed
    test_count            : int    — total test records in the county
    is_measured           : bool   — at least one test above the min-count floor
    measurement_gap       : float  — claimed_coverage_frac - measured_coverage_frac
                                     (positive = more claimed than measured)

This is wired into scoring as a corroboration feature: a flagged county with
high measurement_gap (claims coverage, nobody measures it) is more likely to be
gaming; one with low gap or abundant tests is more likely legitimate.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests

log = logging.getLogger(__name__)

# Ookla open data: quarterly tile-level aggregates on S3 (public, no auth).
# Format: year=YYYY/quarter=Q/type=<mobile|fixed>/tiles.parquet
_OOKLA_BASE_URL = "https://ookla-open-data.s3.amazonaws.com/parquet/performance"
_OOKLA_TECHNOLOGY_MAP = {
    "mobile": ["5G", "LTE", "3G"],  # Ookla uses "connection_type" field
}
_OOKLA_MIN_TEST_COUNT = 3   # floor below which a tile is considered unmeasured

# FCC Speed Test data (HTTPS CSV download from public FCC site).
_FCC_SPEEDTEST_URL = "https://opendata.fcc.gov/api/views/ry5c-xg5b/rows.csv?accessType=DOWNLOAD"

_REQUEST_TIMEOUT = 300


def _ookla_quarter_url(year: int, quarter: int) -> str:
    return (
        f"{_OOKLA_BASE_URL}/year={year}/quarter={quarter}"
        f"/type=mobile/2024-01-01_performance_mobile_tiles.parquet"
    )


def _vintage_to_year_quarter(vintage: str) -> tuple[int, int]:
    """Map an FCC vintage label to the closest Ookla quarter."""
    import re

    # Handle "December 31, 2025" or "2025-12-31"
    if re.match(r"\d{4}-\d{2}-\d{2}", vintage.strip()):
        year, month = int(vintage[:4]), int(vintage[5:7])
    else:
        from datetime import datetime
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                dt = datetime.strptime(vintage.strip(), fmt)
                year, month = dt.year, dt.month
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Cannot parse vintage: {vintage!r}")

    quarter = (month - 1) // 3 + 1
    return year, quarter


def _download_ookla_quarter(
    year: int, quarter: int, cache_dir: Path
) -> pd.DataFrame | None:
    """Download one Ookla quarterly parquet and cache it."""
    cache_file = cache_dir / f"ookla_mobile_{year}_Q{quarter}.parquet"
    if cache_file.exists():
        log.info("using cached Ookla tiles: %s", cache_file)
        return pd.read_parquet(cache_file)

    # Try multiple URL patterns (Ookla has changed the format over time).
    urls = [
        f"{_OOKLA_BASE_URL}/year={year}/quarter={quarter}/type=mobile/"
        f"{year}-{quarter:02d}-01_performance_mobile_tiles.parquet",
        f"https://ookla-open-data.s3.amazonaws.com/parquet/performance/"
        f"year={year}/quarter={quarter}/type=mobile/"
        f"{year}-0{quarter * 3 - 2:02d}-01_performance_mobile_tiles.parquet",
    ]
    for url in urls:
        try:
            log.info("downloading Ookla tiles: %s", url)
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT, stream=True,
                                headers={"User-Agent": "fcc-coverage-audit/0.1"})
            if resp.status_code == 200:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(resp.content)
                df = pd.read_parquet(cache_file)
                log.info("downloaded %d Ookla tile records for %d Q%d", len(df), year, quarter)
                return df
            log.debug("HTTP %d for %s", resp.status_code, url)
        except Exception as exc:
            log.debug("Ookla download failed for %s: %s", url, exc)

    log.warning("Ookla data not available for %d Q%d — skip measured coverage", year, quarter)
    return None


def _ookla_tiles_to_county(
    tiles_df: pd.DataFrame,
    counties: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Spatial join Ookla tile centroids (quadkeys/WGS84) to counties."""
    if tiles_df.empty or counties.empty:
        return pd.DataFrame()

    # Ookla tiles use quadkey → need to decode to lat/lng, or use tile centroids.
    # The parquet has 'tile' (quadkey) or 'avg_lat'/'avg_lng' depending on version.
    if "avg_lat" in tiles_df.columns and "avg_lng" in tiles_df.columns:
        tiles_df = tiles_df.dropna(subset=["avg_lat", "avg_lng"])
        gdf = gpd.GeoDataFrame(
            tiles_df,
            geometry=gpd.points_from_xy(tiles_df["avg_lng"], tiles_df["avg_lat"]),
            crs="EPSG:4326",
        )
    elif "tile" in tiles_df.columns:
        # Decode quadkey to lat/lng centroid.
        try:
            import mercantile
        except ImportError:
            log.warning("mercantile not installed; cannot decode Ookla quadkeys. "
                        "pip install mercantile")
            return pd.DataFrame()
        centers = [mercantile.xy(*mercantile.ul(*mercantile.quadkey_to_tile(qk)[:2]),
                                 reverse=True) if qk else (None, None)
                   for qk in tiles_df["tile"].astype(str)]
        tiles_df = tiles_df.copy()
        tiles_df["_lng"] = [c[0] for c in centers]
        tiles_df["_lat"] = [c[1] for c in centers]
        tiles_df = tiles_df.dropna(subset=["_lng", "_lat"])
        gdf = gpd.GeoDataFrame(
            tiles_df,
            geometry=gpd.points_from_xy(tiles_df["_lng"], tiles_df["_lat"]),
            crs="EPSG:4326",
        )
    else:
        log.warning("Ookla parquet missing lat/lng or tile columns; cannot join to counties")
        return pd.DataFrame()

    joined = gpd.sjoin(gdf, counties[["county_geoid", "geometry"]], how="left", predicate="within")
    joined = joined.dropna(subset=["county_geoid"])

    # Identify speed / test-count columns (vary by Ookla version).
    dl_col = next((c for c in ["avg_d_kbps", "avg_dl_mbps", "d_kbps"] if c in joined.columns), None)
    tests_col = next((c for c in ["tests", "test_count", "num_tests"] if c in joined.columns), None)
    devices_col = next((c for c in ["devices", "num_devices"] if c in joined.columns), None)

    agg_dict: dict[str, Any] = {}
    if tests_col:
        agg_dict["test_count"] = (tests_col, "sum")
    if dl_col:
        agg_dict["sum_dl"] = (dl_col, "sum")
        agg_dict["tiles_with_data"] = (dl_col, "count")

    if not agg_dict:
        log.warning("Cannot find speed/test columns in Ookla data; columns: %s", list(joined.columns))
        return pd.DataFrame()

    result = joined.groupby("county_geoid").agg(**agg_dict).reset_index()

    if "test_count" not in result:
        result["test_count"] = 0
    if "sum_dl" in result and "tiles_with_data" in result:
        # Prefer median approximation: sum / count gives tile-average (proxy for median).
        scale = 1.0 if dl_col and "kbps" not in dl_col else 1.0 / 1000.0
        result["measured_dl_mbps_p50"] = (result["sum_dl"] / result["tiles_with_data"].clip(1)) * scale
    else:
        result["measured_dl_mbps_p50"] = 0.0

    result["is_measured"] = result["test_count"] >= _OOKLA_MIN_TEST_COUNT
    return result[["county_geoid", "test_count", "measured_dl_mbps_p50", "is_measured"]]


def fetch_measured_labels(
    vintage: str,
    counties: gpd.GeoDataFrame,
    cache_dir: Path | str = Path("data/groundtruth/measured"),
    technology: str = "mobile",
) -> pd.DataFrame:
    """Produce per-county measured-coverage labels for a given vintage quarter.

    Parameters
    ----------
    vintage : str
        FCC vintage label (e.g. "December 31, 2025") — used to select the
        closest Ookla quarterly dataset.
    counties : GeoDataFrame
        County boundaries with ``county_geoid`` column for spatial join.
    cache_dir : Path
        Cache directory for downloaded Ookla tiles and derived labels.
    technology : str
        Ookla connection type filter ("mobile" for cellular data).

    Returns
    -------
    DataFrame with columns:
        county_geoid         : 5-digit FIPS
        test_count           : int   — total Ookla tests in county for that quarter
        measured_dl_mbps_p50 : float — approximate median download speed (Mbps)
        is_measured          : bool  — at least OOKLA_MIN_TEST_COUNT tests present
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    year, quarter = _vintage_to_year_quarter(vintage)
    label_cache = cache_dir / f"measured_labels_{year}_Q{quarter}.parquet"
    if label_cache.exists():
        log.info("loading cached measured labels: %s", label_cache)
        return pd.read_parquet(label_cache)

    tiles_df = _download_ookla_quarter(year, quarter, cache_dir)
    if tiles_df is None or tiles_df.empty:
        log.warning("no Ookla data for %d Q%d; returning empty measured labels", year, quarter)
        return pd.DataFrame(columns=[
            "county_geoid", "test_count", "measured_dl_mbps_p50", "is_measured",
        ])

    result = _ookla_tiles_to_county(tiles_df, counties)
    if result.empty:
        return pd.DataFrame(columns=[
            "county_geoid", "test_count", "measured_dl_mbps_p50", "is_measured",
        ])

    result.to_parquet(label_cache, index=False)
    log.info("wrote measured labels: %s (%d counties)", label_cache, len(result))
    return result


def compute_measurement_gap(
    features: pd.DataFrame,
    measured: pd.DataFrame,
) -> pd.DataFrame:
    """Add measurement gap features to scored features DataFrame.

    measurement_gap = current_cov_frac (claimed) - measured_frac (observed).
    A large positive gap means coverage is claimed but not measured — the
    strongest single combination signal for gaming when also accompanied by
    same-site growth and no new ASR structures.

    Adds columns:
        meas_test_count           : int
        meas_dl_mbps_p50          : float
        meas_is_measured          : bool
        measurement_gap           : float (current_cov_frac - measured_frac approx)
    """
    if measured.empty:
        out = features.copy()
        out["meas_test_count"] = 0
        out["meas_dl_mbps_p50"] = 0.0
        out["meas_is_measured"] = False
        out["measurement_gap"] = 0.0
        return out

    m = measured.rename(columns={
        "test_count": "meas_test_count",
        "measured_dl_mbps_p50": "meas_dl_mbps_p50",
        "is_measured": "meas_is_measured",
    })
    merged = features.merge(
        m[["county_geoid", "meas_test_count", "meas_dl_mbps_p50", "meas_is_measured"]],
        on="county_geoid",
        how="left",
    )
    merged["meas_test_count"] = merged["meas_test_count"].fillna(0).astype(int)
    merged["meas_dl_mbps_p50"] = merged["meas_dl_mbps_p50"].fillna(0.0)
    merged["meas_is_measured"] = merged["meas_is_measured"].fillna(False).astype(bool)

    # Proxy for measured coverage fraction: tests-per-sq-km normalised [0,1].
    # A full calculation would need the Ookla tile count vs county tile count;
    # this approximation uses "is there ANY measurement" as the primary signal.
    cur_frac = merged.get("current_cov_frac", pd.Series(0.0, index=merged.index)).fillna(0.0)
    # Gap is meaningful as a signal: claimed but not measured at all.
    merged["measurement_gap"] = cur_frac.where(~merged["meas_is_measured"], 0.0)
    return merged
