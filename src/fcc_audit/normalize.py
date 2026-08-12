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

from .acquire import CoverageFile, safe_service_name as safe
from .config import Config

log = logging.getLogger(__name__)

# Candidate attribute names for the modeled signal-strength band across vintages.
_SIGNAL_COLUMNS = ["minsignal", "min_signal", "signal", "sig_strength", "signalstr"]
# Some vintages encode signal as an ordinal band code rather than dBm.
_BAND_CODE_TO_DBM = {1: -105.0, 2: -95.0, 3: -85.0}


# Column names a WKT geometry might arrive under (e.g. DBeaver CSV exports).
_WKT_COLUMNS = ["geometry_wkt", "wkt", "geom_wkt", "the_geom", "geometry", "geom", "shape"]


def _read_wkt_table(path: Path) -> gpd.GeoDataFrame:
    """Read a CSV/TSV whose geometry is stored as WKT text (EPSG:4326).

    Lets you export a Redshift coverage query straight from DBeaver to CSV and
    feed it to the pipeline without a live database connection. One column must
    hold WKT polygons (e.g. ``geometry_wkt``); all other columns are kept as
    attributes (a ``minsignal`` column, if present, drives the signal heatmap).
    """
    from shapely import wkt

    sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    df = pd.read_csv(path, sep=sep)
    lower = {c.lower(): c for c in df.columns}
    geom_col = next((lower[c] for c in _WKT_COLUMNS if c in lower), None)
    if geom_col is None:
        raise RuntimeError(
            f"{path.name}: no WKT geometry column found. Expected one of "
            f"{_WKT_COLUMNS}. Alias your geometry as `geometry_wkt` in the query."
        )
    geom = df[geom_col].map(lambda v: wkt.loads(v) if isinstance(v, str) and v else None)
    gdf = gpd.GeoDataFrame(df.drop(columns=[geom_col]), geometry=geom, crs="EPSG:4326")
    return gdf[~gdf.geometry.isna()].reset_index(drop=True)


def load_coverage_gdf(path: Path) -> gpd.GeoDataFrame:
    """Read a coverage layer in EPSG:4326.

    Supports zipped shapefiles, GeoPackage, GeoJSON, and CSV/TSV exports that
    carry geometry as WKT text (the easiest format to export from DBeaver).
    """
    suffix = path.suffix.lower()
    if suffix == ".zip":
        # geopandas/pyogrio can read a zipped shapefile directly.
        gdf = gpd.read_file(f"zip://{path}")
    elif suffix in (".csv", ".tsv", ".tab"):
        gdf = _read_wkt_table(path)
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
    """Fill polygons with H3 cells, keeping the strongest signal band per hex.

    This is the most CPU-intensive stage: it polyfills every coverage polygon
    (FCC mobile files run into the millions). Progress is logged periodically so
    a long run is visibly making progress rather than appearing hung.
    """
    import time

    total = len(gdf)
    log_every = 250_000
    start = time.monotonic()

    def _cells_for(geom) -> list[str]:
        try:
            return h3.geo_to_cells(geom, resolution)
        except Exception:  # noqa: BLE001 - h3 raises on odd geometries
            pt = geom.representative_point()
            return [h3.latlng_to_cell(pt.y, pt.x, resolution)]

    # Flat-coverage fast path: with no signal column only the SET of covered
    # cells matters, so we union C-level cell lists instead of doing a per-cell
    # Python comparison. Much faster on large national files.
    if signal_col is None:
        covered: set[str] = set()
        for i, geom in enumerate(gdf.geometry):
            if i and i % log_every == 0:
                elapsed = time.monotonic() - start
                rate = i / elapsed if elapsed else 0.0
                eta = (total - i) / rate if rate else 0.0
                log.info(
                    "    H3-indexing %s/%s polygons (%.0f/s, ~%.0fs left, %s hexes so far)",
                    f"{i:,}", f"{total:,}", rate, eta, f"{len(covered):,}",
                )
            if geom is None or geom.is_empty:
                continue
            covered.update(_cells_for(geom))
        if not covered:
            return pd.DataFrame(columns=["h3", "signal_dbm"])
        return pd.DataFrame({"h3": list(covered), "signal_dbm": 0.0})

    best: dict[str, float] = {}
    best_get = best.get
    for i, (geom, sig) in enumerate(zip(gdf.geometry, gdf[signal_col])):
        if i and i % log_every == 0:
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed else 0.0
            eta = (total - i) / rate if rate else 0.0
            log.info(
                "    H3-indexing %s/%s polygons (%.0f/s, ~%.0fs left, %s hexes so far)",
                f"{i:,}", f"{total:,}", rate, eta, f"{len(best):,}",
            )
        if geom is None or geom.is_empty:
            continue
        dbm = _to_dbm(sig)
        for c in _cells_for(geom):
            cur = best_get(c)
            if cur is None or dbm > cur:
                best[c] = dbm
    if not best:
        return pd.DataFrame(columns=["h3", "signal_dbm"])
    return pd.DataFrame({"h3": list(best.keys()), "signal_dbm": list(best.values())})


def _is_fixture_county_cache(gdf: gpd.GeoDataFrame) -> bool:
    """True when the cached counties file is synthetic fixture geography."""
    if len(gdf) < 100:
        return True
    states = gdf["state_fips"].astype(str).unique()
    return len(states) == 1 and states[0] == "90"


def load_counties(cfg: Config) -> gpd.GeoDataFrame:
    """Load (and cache/download) TIGER county boundaries in EPSG:4326."""
    cache = cfg.path("interim") / "tl_us_county.gpkg"
    if cache.exists():
        gdf = gpd.read_file(cache)
        # Fixture runs write a 4-county synthetic layer; replace it for real FCC runs.
        if cfg.backend != "fixture" and _is_fixture_county_cache(gdf):
            log.info("replacing fixture county cache with TIGER/Line boundaries")
            cache.unlink()
        else:
            return gdf

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


def assign_counties(
    hex_df: pd.DataFrame,
    counties: gpd.GeoDataFrame,
    *,
    clip_to_states=None,
    buffer_m: float = 50_000.0,
    equal_area_crs: str = "EPSG:5070",
) -> pd.DataFrame:
    """Attach county attributes via hex-polygon overlap (largest-area wins).

    Each H3 cell becomes its boundary polygon and is joined to counties with
    ``predicate="intersects"``. A hex straddling a boundary is assigned to the
    county with the largest intersection area — avoiding centroid mis-tags.

    When *clip_to_states* is set, hexes whose polygons do not intersect a
    *buffer_m* fringe of those states are dropped (polygon, not centroid).
    """
    if hex_df.empty:
        return hex_df.assign(county_geoid=None, county_name=None, state_fips=None)

    from shapely.geometry import Polygon

    base = hex_df[["h3", "signal_dbm"]].copy().reset_index(drop=True)
    cell_ids = base["h3"].astype(str).tolist()
    polys = []
    for c in cell_ids:
        try:
            boundary = h3.cell_to_boundary(c)  # [(lat, lng), ...]
            polys.append(Polygon([(lng, lat) for lat, lng in boundary]))
        except Exception:
            polys.append(None)
    gdf = gpd.GeoDataFrame(base, geometry=polys, crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notna()].copy().reset_index(drop=True)
    if gdf.empty:
        return base.assign(county_geoid=None, county_name=None, state_fips=None)

    if clip_to_states not in (None, "all") and clip_to_states:
        wanted = {str(s).zfill(2) for s in clip_to_states}
        state_col = counties["state_fips"].astype(str).str.zfill(2)
        target = counties.loc[state_col.isin(wanted)]
        if not target.empty:
            geom = target.to_crs(equal_area_crs).geometry.union_all().buffer(float(buffer_m))
            keep = gdf.to_crs(equal_area_crs).geometry.intersects(geom).to_numpy()
            if not bool(keep.all()):
                n_before = len(gdf)
                gdf = gdf.loc[keep].reset_index(drop=True)
                log.info(
                    "  clipped hexes to target buffer: %s -> %s (states=%s)",
                    f"{n_before:,}", f"{len(gdf):,}", ",".join(sorted(wanted)),
                )
            if gdf.empty:
                return pd.DataFrame(columns=[
                    "h3", "signal_dbm", "county_geoid", "county_name", "state_fips",
                ])

    counties_use = counties[["county_geoid", "county_name", "state_fips", "geometry"]].copy()
    counties_use = counties_use.reset_index(drop=True)
    try:
        joined = gpd.sjoin(gdf, counties_use, how="left", predicate="intersects")
    except Exception:
        cents = gdf.copy()
        cents.geometry = cents.geometry.centroid
        joined = gpd.sjoin(cents, counties_use, how="left", predicate="within")

    # Largest-area county wins when a hex intersects multiple counties.
    joined = joined.reset_index(names="_hex_i")
    if "index_right" in joined.columns and joined["h3"].duplicated().any():
        areas = []
        hex_geoms = gdf.geometry
        county_geoms = counties_use.geometry
        for _, row in joined.iterrows():
            right = row.get("index_right")
            hi = int(row["_hex_i"])
            if pd.isna(right):
                areas.append(0.0)
                continue
            try:
                inter = hex_geoms.iloc[hi].intersection(county_geoms.iloc[int(right)])
                areas.append(float(inter.area) if not inter.is_empty else 0.0)
            except Exception:
                areas.append(0.0)
        joined = joined.assign(_area=areas)
        joined = joined.sort_values("_area", ascending=False).drop_duplicates(
            subset=["h3"], keep="first"
        )
        joined = joined.drop(columns=["_area"], errors="ignore")
    else:
        joined = joined.drop_duplicates(subset=["h3"], keep="first")
    joined = joined.drop(columns=["_hex_i"], errors="ignore")

    joined = joined.drop(columns=["index_right", "geometry"], errors="ignore")
    for col in ["county_geoid", "county_name", "state_fips"]:
        right = f"{col}_right"
        if right in joined.columns:
            joined[col] = joined[right]
            joined = joined.drop(columns=[right], errors="ignore")
        elif col not in joined.columns:
            joined[col] = None
    # Drop any remaining join suffixes.
    drop_cols = [c for c in joined.columns if c.endswith("_left") or c.endswith("_right")]
    if drop_cols:
        joined = joined.drop(columns=drop_cols, errors="ignore")
    keep_cols = ["h3", "signal_dbm", "county_geoid", "county_name", "state_fips"]
    for c in keep_cols:
        if c not in joined.columns:
            joined[c] = None
    return pd.DataFrame(joined[keep_cols]).reset_index(drop=True)


def filter_counties_to_states(counties: gpd.GeoDataFrame, states) -> gpd.GeoDataFrame:
    """Keep only counties in the requested state list (or all if unrestricted)."""
    if states == "all" or not states:
        return counties
    wanted = {str(s).zfill(2) for s in states}
    mask = counties["state_fips"].astype(str).str.zfill(2).isin(wanted)
    return counties.loc[mask].copy()


def clip_hexes_to_target_buffer(
    hex_df: pd.DataFrame,
    counties: gpd.GeoDataFrame,
    target_states,
    buffer_m: float = 50_000.0,
    equal_area_crs: str = "EPSG:5070",
) -> pd.DataFrame:
    """Drop neighbor-state hexes farther than *buffer_m* from target states.

    Prefer :func:`assign_counties` with ``clip_to_states=`` (single centroid pass).
    Kept for callers that only need spatial filtering.
    """
    if hex_df.empty or target_states == "all" or not target_states:
        return hex_df
    wanted = {str(s).zfill(2) for s in target_states}
    state_col = counties["state_fips"].astype(str).str.zfill(2)
    target = counties.loc[state_col.isin(wanted)]
    if target.empty:
        return hex_df
    import numpy as np

    geom = target.to_crs(equal_area_crs).geometry.union_all().buffer(float(buffer_m))
    cell_ids = hex_df["h3"].astype(str).tolist()
    lats = np.empty(len(cell_ids), dtype=float)
    lngs = np.empty(len(cell_ids), dtype=float)
    for i, c in enumerate(cell_ids):
        lat, lng = h3.cell_to_latlng(c)
        lats[i] = lat
        lngs[i] = lng
    pts = gpd.GeoSeries(
        gpd.points_from_xy(lngs, lats),
        crs="EPSG:4326",
    ).to_crs(equal_area_crs)
    keep = pts.intersects(geom)
    if bool(keep.all()):
        return hex_df
    out = hex_df.loc[keep.to_numpy()].reset_index(drop=True)
    log.info(
        "  clipped hexes to target buffer: %s -> %s (states=%s)",
        f"{len(hex_df):,}", f"{len(out):,}", ",".join(sorted(wanted)),
    )
    return out


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

    # Only compute against counties that actually gained coverage.
    geoids = set(gained["county_geoid"].astype(str))
    counties = counties[counties["county_geoid"].astype(str).isin(geoids)]
    if counties.empty:
        return pd.DataFrame(columns=["county_geoid", "boundary_snap_share"])

    centers = [h3.cell_to_latlng(c) for c in gained["h3"].tolist()]
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


def _derive_parent_hexes(
    fine_df: pd.DataFrame, fine_res: int, parent_res: int
) -> pd.DataFrame:
    """Derive a coarse-resolution hex table from a finer one, EXACTLY matching a
    direct polyfill at ``parent_res``.

    H3 polyfill selects a cell iff the cell's *center* lies in the polygon. A
    coarse cell's center coincides with its "center child" at any finer
    resolution, so a coarse cell would be selected by a direct polyfill iff its
    center-child fine cell was selected here. Keeping exactly those parents (with
    their center-child's signal) reproduces the direct-polyfill result cell-for-
    cell and signal-for-signal -- just without re-polyfilling the source polygons.
    """
    if fine_df.empty:
        return pd.DataFrame(columns=["h3", "signal_dbm"])
    out_h3: list[str] = []
    out_sig: list[float] = []
    for cell, sig in zip(fine_df["h3"], fine_df["signal_dbm"]):
        parent = h3.cell_to_parent(cell, parent_res)
        if h3.cell_to_center_child(parent, fine_res) == cell:
            out_h3.append(parent)
            out_sig.append(sig)
    return pd.DataFrame({"h3": out_h3, "signal_dbm": out_sig})


def _hex_layer_at_resolution(cov: CoverageFile, resolution: int) -> pd.DataFrame:
    """Read a pre-indexed hex coverage file and return it at ``resolution``.

    The source parquet (written by the Redshift backend) holds columns
    ``h3`` + ``signal_dbm`` at ``cov.hex_resolution``. If the requested
    resolution differs, roll cells up to their H3 parents (strongest signal
    wins) or expand down to their children (signal carried unchanged). When the
    resolution matches, the cells are used as-is.
    """
    base = pd.read_parquet(cov.local_path)
    src_res = cov.hex_resolution or resolution
    if resolution == src_res:
        return base[["h3", "signal_dbm"]].copy()
    if resolution > src_res:
        child_h3: list[str] = []
        child_sig: list[float] = []
        for cell, sig in zip(base["h3"], base["signal_dbm"]):
            for child in h3.cell_to_children(cell, resolution):
                child_h3.append(child)
                child_sig.append(sig)
        return pd.DataFrame({"h3": child_h3, "signal_dbm": child_sig})
    parents = base["h3"].map(lambda c: h3.cell_to_parent(c, resolution))
    rolled = pd.DataFrame({"h3": parents, "signal_dbm": base["signal_dbm"]})
    return rolled.groupby("h3", as_index=False)["signal_dbm"].max()


def _normalize_scope_key(cfg: Config) -> str:
    """Cache token for state query scope, including target when clipped."""
    scope = cfg.states_scope_key()
    targets = cfg.target_states
    if (
        targets not in (None, "all")
        and cfg.states not in (None, "all")
        and set(str(s).zfill(2) for s in targets) != set(str(s).zfill(2) for s in cfg.states)
    ):
        return f"{scope}_t{'-'.join(sorted(str(s).zfill(2) for s in targets))}"
    return scope


def normalize_layers(
    cfg: Config,
    cov: CoverageFile,
    counties: gpd.GeoDataFrame,
    county_res: int,
    site_res: int,
    service_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize one (provider, service) file into BOTH resolutions in one pass.

    Polyfilling polygons to H3 is the pipeline's most expensive step. Instead of
    indexing the source twice (once per resolution), we polyfill once at the finer
    ``site_res`` and derive the coarser ``county_res`` table by rolling each fine
    cell up to its H3 parent (strongest signal wins). County tags are still
    assigned per resolution via centroid point-in-polygon, so semantics are
    unchanged. Each resolution caches to parquet for instant resumes.

    Returns ``(county_res_df, site_res_df)``.
    """
    if site_res == county_res:
        # This must precede the two-cache fast path and the pre-indexed branch:
        # both cache paths are identical at one resolution, and reading each
        # separately doubles national Redshift memory for no analytical benefit.
        df = normalize_layer(cfg, cov, counties, county_res, service_label)
        return df, df

    safe_svc = safe(service_label)
    scope = _normalize_scope_key(cfg)
    backend = cfg.backend
    # Include backend so fixture caches never poison fcc/redshift national runs.
    cache_c = (
        cfg.path("interim")
        / f"hex_{backend}_{cov.vintage}_{cov.provider_id}_{safe_svc}_{scope}_r{county_res}.parquet"
    )
    cache_s = (
        cfg.path("interim")
        / f"hex_{backend}_{cov.vintage}_{cov.provider_id}_{safe_svc}_{scope}_r{site_res}.parquet"
    )
    if cache_c.exists() and cache_s.exists():
        return pd.read_parquet(cache_c), pd.read_parquet(cache_s)

    # Pre-indexed hex source (Redshift): the warehouse already resolved coverage
    # to H3 cells, so there is nothing to polyfill. Build each resolution's
    # county-tagged table directly from the hex list (normalize_layer handles
    # any resolution change via H3 parent/child rollup).
    if getattr(cov, "is_hex", False):
        return (
            normalize_layer(cfg, cov, counties, county_res, service_label),
            normalize_layer(cfg, cov, counties, site_res, service_label),
        )

    # Parent rollup requires site_res to be strictly finer than county_res.
    # Otherwise fall back to indexing each resolution independently.
    if site_res < county_res:
        return (
            normalize_layer(cfg, cov, counties, county_res, service_label),
            normalize_layer(cfg, cov, counties, site_res, service_label),
        )

    gdf = load_coverage_gdf(cov.local_path)
    signal_col = detect_signal_column(gdf)
    if signal_col is None:
        log.warning("no signal column in %s; treating coverage as flat band", cov.local_path.name)
    log.info(
        "  normalize %s provider %s %s: H3-indexing %s polygons (r%d, deriving r%d)",
        cov.vintage, cov.provider_id, service_label, f"{len(gdf):,}", site_res, county_res,
    )
    fine = coverage_to_hex(gdf, site_res, signal_col)
    coarse = _derive_parent_hexes(fine, site_res, county_res)

    def _finish(hex_df: pd.DataFrame, cache: Path) -> pd.DataFrame:
        out = assign_counties(hex_df, counties)
        out["provider_id"] = cov.provider_id
        out["technology"] = service_label
        out["vintage"] = cov.vintage
        out.to_parquet(cache, index=False)
        return out

    site_df = _finish(fine, cache_s)
    county_df = _finish(coarse, cache_c)
    return county_df, site_df


def normalize_layer(
    cfg: Config,
    cov: CoverageFile,
    counties: gpd.GeoDataFrame,
    resolution: int,
    service_label: str,
) -> pd.DataFrame:
    """Normalize one (provider, service) coverage file to county-tagged hexes.

    Each FCC mobile file is already a single technology/speed tier, so no tier
    filtering is needed. If the file has a signal column it's kept (strongest per
    hex); otherwise coverage is treated as a flat band. Cached to parquet.
    """
    scope = _normalize_scope_key(cfg)
    backend = cfg.backend
    cache = (
        cfg.path("interim")
        / f"hex_{backend}_{cov.vintage}_{cov.provider_id}_{safe(service_label)}_{scope}_r{resolution}.parquet"
    )
    if cache.exists():
        return pd.read_parquet(cache)

    if getattr(cov, "is_hex", False):
        hex_df = _hex_layer_at_resolution(cov, resolution)
        targets = cfg.target_states
        query = cfg.states
        clip_to = None
        if (
            targets not in (None, "all")
            and query not in (None, "all")
            and set(str(s).zfill(2) for s in targets) != set(str(s).zfill(2) for s in query)
        ):
            clip_to = targets
        hex_df = assign_counties(
            hex_df, counties,
            clip_to_states=clip_to,
            buffer_m=50_000.0,
            equal_area_crs=cfg.geography.get("equal_area_crs", "EPSG:5070"),
        )
        hex_df["provider_id"] = cov.provider_id
        hex_df["technology"] = service_label
        hex_df["vintage"] = cov.vintage
        hex_df.to_parquet(cache, index=False)
        return hex_df

    gdf = load_coverage_gdf(cov.local_path)
    signal_col = detect_signal_column(gdf)
    if signal_col is None:
        log.warning("no signal column in %s; treating coverage as flat band", cov.local_path.name)
    log.info(
        "  normalize %s provider %s %s: H3-indexing %s polygons (r%d)",
        cov.vintage, cov.provider_id, service_label, f"{len(gdf):,}", resolution,
    )
    hex_df = coverage_to_hex(gdf, resolution, signal_col)
    hex_df = assign_counties(hex_df, counties)
    hex_df["provider_id"] = cov.provider_id
    hex_df["technology"] = service_label
    hex_df["vintage"] = cov.vintage
    hex_df.to_parquet(cache, index=False)
    return hex_df
