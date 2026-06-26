"""Web data assembly: county GeoJSON, hex records, detail JSON, tower overlays.

This module owns all web-delivery data construction. Hot loops (.iterrows)
are vectorized via DataFrame.to_dict('records') and batch operations.

The flag-math tooltip payload is also exported here: each scored record carries
a ``flag_math`` dict with the exact numeric threshold, per-feature values, and
which gates fired — consumed by the reviewer cockpit's info tooltip.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import mapping

from .explain import explain_row, add_explanations
from . import attribute
from . import map_render

log = logging.getLogger(__name__)

_METRIC_KEYS = [
    "priority_score", "flag_for_review", "added_km2", "added_frac_of_county",
    "pct_increase", "same_site_growth_share", "new_site_share",
    "unattributed_share", "boundary_snap_share", "blanket_fillin", "new_towers",
    "new_towers_in_county", "new_towers_cross_border",
    "prior_towers", "current_towers", "prior_towers_in_county", "current_towers_in_county",
    "prior_towers_cross_border", "current_towers_cross_border",
    "prior_cov_frac", "current_cov_frac", "flag_reason",
    # Ground-truth corroboration (present when GT data loaded)
    "asr_has_new_structure", "asr_new_structure_count",
    "meas_test_count", "meas_dl_mbps_p50", "meas_is_measured", "measurement_gap",
]

# Features shown in the flag-math tooltip, with display labels.
_TOOLTIP_FEATURES = [
    ("added_frac_of_county",        "+% of county covered"),
    ("coverage_increase_magnitude", "coverage increase magnitude"),
    ("blanket_fillin",              "blanket fill-in"),
    ("same_site_growth_share",      "same-site growth share"),
    ("unattributed_share",          "unattributed share"),
    ("boundary_snap_share",         "boundary snap share"),
    ("new_site_share",              "new-site share"),
    ("asr_no_new_structure",        "no new ASR structure (GT)"),
    ("measurement_gap",             "claimed-vs-measured gap (GT)"),
]


def _finite_or_none(value: Any, ndigits: int = 3) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, ndigits)


def _safe_service_key(service: str) -> str:
    return service.replace("/", "-").replace(" ", "")


# ---------------------------------------------------------------------------
# Flag-math tooltip export
# ---------------------------------------------------------------------------

def _build_flag_math(row: pd.Series, threshold: float, weights: dict[str, float]) -> dict[str, Any]:
    """Build the flag-math dict for the reviewer cockpit info tooltip.

    Returns:
        priority_score   : float
        flag_threshold   : float  — the percentile cutoff value
        flag             : bool
        features         : list[{name, label, value, weight, contribution, gates_fired}]
    """
    feats = []
    for name, label in _TOOLTIP_FEATURES:
        val = _finite_or_none(row.get(name, 0.0))
        w = weights.get(name, 0.0)
        contrib = _finite_or_none((val or 0.0) * w)
        feats.append({
            "name": name,
            "label": label,
            "value": val,
            "weight": round(w, 4),
            "contribution": contrib,
        })
    return {
        "priority_score": _finite_or_none(row.get("priority_score")),
        "flag_threshold": round(threshold, 4),
        "flag": bool(row.get("flag_for_review", False)),
        "features": feats,
    }


# ---------------------------------------------------------------------------
# Dashboard payload (vectorized)
# ---------------------------------------------------------------------------

def build_dashboard_payload(
    scored: pd.DataFrame, sites: pd.DataFrame, counties: gpd.GeoDataFrame
) -> dict[str, Any]:
    centroids = counties.copy()
    centroids["geometry"] = centroids.geometry.representative_point()
    cen: dict[str, tuple[float, float]] = {
        row["county_geoid"]: (row.geometry.y, row.geometry.x)
        for _, row in centroids.iterrows()
    }

    county_features = []
    # Vectorized: convert to list-of-dicts once, avoid per-row Series overhead.
    for r in scored.to_dict("records"):
        latlng = cen.get(str(r.get("county_geoid", "")))
        if not latlng:
            continue
        expl = explain_row(pd.Series(r))
        county_features.append({
            "geoid": str(r["county_geoid"]),
            "name": r.get("county_name", ""),
            "provider": r.get("provider_name", str(r.get("provider_id"))),
            "technology": r.get("technology", ""),
            "lat": latlng[0],
            "lng": latlng[1],
            "priority": _finite_or_none(r.get("priority_score")),
            "flag": bool(r.get("flag_for_review", False)),
            "reason": r.get("flag_reason", ""),
            "plain_explanation": expl["headline"],
            "pct_increase": _finite_or_none(r.get("pct_increase")),
            "same_site_growth_share": _finite_or_none(r.get("same_site_growth_share", 0)),
        })

    site_features: list[dict[str, Any]] = []
    if not sites.empty:
        for s in sites.to_dict("records"):
            site_features.append({
                "lat": float(s["lat"]),
                "lng": float(s["lng"]),
                "provider": s.get("provider_name", str(s.get("provider_id"))),
                "site_class": s.get("site_class", "site"),
                "n_hexes": int(s.get("n_hexes", 0)),
            })
    return {"counties": county_features, "sites": site_features}


# ---------------------------------------------------------------------------
# Web records (vectorized)
# ---------------------------------------------------------------------------

def _record_from_dict(r: dict[str, Any], threshold: float, weights: dict[str, float]) -> dict[str, Any]:
    """Compact web record for one provider × service × county. Vectorized version."""
    row_series = pd.Series(r)
    expl = explain_row(row_series)
    metrics: dict[str, Any] = {}
    for k in _METRIC_KEYS:
        if k == "flag_for_review":
            metrics[k] = bool(r.get(k, False))
        else:
            metrics[k] = _finite_or_none(r[k]) if k in r else None
    if "pct_increase" in r:
        metrics["pct_increase"] = _finite_or_none(r["pct_increase"])
    return {
        "geoid": str(r["county_geoid"]),
        "name": str(r.get("county_name", "")),
        "state_fips": str(r.get("state_fips", "")),
        "provider_id": int(r["provider_id"]),
        "provider_name": str(r.get("provider_name", r["provider_id"])),
        "service": str(r.get("technology", "")),
        "priority": _finite_or_none(r.get("priority_score")),
        "flag": bool(r.get("flag_for_review", False)),
        "metrics": metrics,
        "explanation": expl,
        "flag_math": _build_flag_math(row_series, threshold, weights),
    }


def build_web_records(
    scored: pd.DataFrame,
    threshold: float = 0.0,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build nested lookup: provider_id -> service -> geoid -> record.

    Vectorized: a single .to_dict('records') call replaces .iterrows() to avoid
    per-row Series construction overhead.
    """
    if scored.empty:
        return {}
    if weights is None:
        weights = {}
    lookup: dict[str, dict[str, dict[str, Any]]] = {}
    for r in scored.to_dict("records"):
        pid = str(int(r["provider_id"]))
        svc = str(r.get("technology", ""))
        geoid = str(r["county_geoid"])
        lookup.setdefault(pid, {}).setdefault(svc, {})[geoid] = _record_from_dict(r, threshold, weights)
    return lookup


def build_web_meta(scored: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    from datetime import datetime, timezone
    providers = []
    if not scored.empty and "provider_id" in scored.columns:
        for pid in sorted(scored["provider_id"].unique()):
            name = scored.loc[scored["provider_id"] == pid, "provider_name"].iloc[0]
            providers.append({"id": int(pid), "name": str(name)})
    services = sorted(scored["technology"].unique().tolist()) if "technology" in scored.columns else []
    flagged = int(scored["flag_for_review"].sum()) if "flag_for_review" in scored.columns else 0
    web_meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_vintage": meta.get("current"),
        "prior_vintage": meta.get("prior"),
        "providers": providers,
        "services": services,
        "total_records": len(scored),
        "flagged_count": flagged,
        "states_processed": meta.get("states_processed", "all"),
    }
    if meta.get("default_provider_id") is not None:
        web_meta["default_provider_id"] = int(meta["default_provider_id"])
    if meta.get("default_county_geoid"):
        web_meta["default_county_geoid"] = str(meta["default_county_geoid"])
    return web_meta


# ---------------------------------------------------------------------------
# County GeoJSON
# ---------------------------------------------------------------------------

def _county_boundary_feature(
    counties: gpd.GeoDataFrame,
    geoid: str,
    *,
    simplify_tolerance: float | None = 0.0002,
) -> dict[str, Any] | None:
    if counties.empty:
        return None
    mask = counties["county_geoid"].astype(str) == str(geoid)
    if not mask.any():
        return None
    gdf = counties.loc[mask].copy()
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    tol = simplify_tolerance
    if tol is not None and len(counties) <= 12:
        tol = None
    if tol is not None:
        gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
    row = gdf.iloc[0]
    return {
        "type": "Feature",
        "properties": {
            "geoid": str(geoid),
            "name": str(row.get("county_name", "")),
            "state": str(row.get("state_fips", "")),
        },
        "geometry": mapping(row.geometry),
    }


def build_counties_geojson(
    counties: gpd.GeoDataFrame,
    simplify_tolerance: float = 0.001,
    geoids: set[str] | None = None,
) -> dict[str, Any]:
    gdf = counties.copy()
    if geoids is not None:
        gdf = gdf[gdf["county_geoid"].astype(str).isin(geoids)]
    if gdf.empty:
        return {"type": "FeatureCollection", "features": []}
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    if simplify_tolerance is not None and len(gdf) > 12:
        gdf["geometry"] = gdf.geometry.simplify(simplify_tolerance, preserve_topology=True)
    gdf = gdf.rename(columns={"county_geoid": "geoid", "county_name": "name", "state_fips": "state"})
    return json.loads(gdf[["geoid", "name", "state", "geometry"]].to_json())


# ---------------------------------------------------------------------------
# County detail JSON (before/after hexes + sites)
# ---------------------------------------------------------------------------

def _encode_signal(signal_dbm: float) -> int:
    """Encode signal_dbm as int8 in the range [-128, 127].

    FCC BDC signals range from roughly -140 to 0 dBm.
    We shift by +100 so -100 → 0, -40 → 60, etc., clamped to int8 range.
    The JS client decodes with: dBm = raw - 100
    """
    return int(np.clip(round(signal_dbm) + 100, -128, 127))


def _hexes_for_county(df: pd.DataFrame, geoid: str, vintage: str) -> list[list]:
    if df.empty:
        return []
    mask = (df["county_geoid"].astype(str) == str(geoid)) & (df["vintage"] == vintage)
    sub = df.loc[mask, ["h3", "signal_dbm"]]
    # Encode signal as int8 (5-10x smaller than float JSON).
    h3s = sub["h3"].astype(str).tolist()
    sigs = [_encode_signal(s) for s in sub["signal_dbm"].tolist()]
    return [[h, s] for h, s in zip(h3s, sigs)]


def _context_hexes_for_bbox(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    vintage: str,
) -> list[list]:
    if df.empty or "vintage" not in df.columns:
        return []
    sub = df[df["vintage"] == vintage][["h3", "signal_dbm"]].drop_duplicates(subset=["h3"])
    if sub.empty:
        return []
    minx, miny, maxx, maxy = bbox
    out: list[list] = []
    for row in sub.itertuples(index=False):
        try:
            lat, lng = h3.cell_to_latlng(str(row.h3))
        except Exception:
            continue
        if minx <= lng <= maxx and miny <= lat <= maxy:
            out.append([str(row.h3), _encode_signal(float(row.signal_dbm))])
    return out


def _sites_for_county(sites: pd.DataFrame, geoid: str, vintage: str) -> list[dict[str, Any]]:
    if sites.empty:
        return []
    mask = (
        sites["county_geoid"].astype(str) == str(geoid)
    ) & (sites.get("vintage", pd.Series("current", index=sites.index)) == vintage)
    out = []
    for s in sites.loc[mask].to_dict("records"):
        out.append({
            "lat": float(s["lat"]),
            "lng": float(s["lng"]),
            "site_class": str(s.get("site_class", "site")),
            "n_hexes": int(s.get("n_hexes", 0)),
            "in_county": True,
            "home_county": str(s.get("county_geoid", geoid)),
        })
    return out


def _sites_serving_county(
    sites: pd.DataFrame,
    coverage: pd.DataFrame,
    geoid: str,
    vintage: str,
) -> list[dict[str, Any]]:
    if sites.empty:
        return []
    sub_sites = sites[sites.get("vintage", pd.Series("current", index=sites.index)) == vintage].reset_index(drop=True)
    if sub_sites.empty:
        return []

    cov = coverage
    if not cov.empty and "vintage" in cov.columns:
        cov = cov[cov["vintage"] == vintage]
    if cov.empty:
        return _sites_for_county(sites, geoid, vintage)

    idxs = attribute.site_indices_serving_county(cov, sub_sites, geoid)
    if len(idxs) == 0:
        return _sites_for_county(sites, geoid, vintage)

    geoid_s = str(geoid)
    out = []
    for i in idxs:
        s = sub_sites.iloc[int(i)]
        home = str(s.get("county_geoid", ""))
        out.append({
            "lat": float(s["lat"]),
            "lng": float(s["lng"]),
            "site_class": str(s.get("site_class", "site")),
            "n_hexes": int(s.get("n_hexes", 0)),
            "in_county": home == geoid_s,
            "home_county": home,
        })
    return out


def build_county_detail(
    geoid: str,
    coverage: pd.DataFrame,
    sites: pd.DataFrame,
    meta: dict[str, Any],
    counties: gpd.GeoDataFrame | None = None,
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "geoid": str(geoid),
        "prior_vintage": meta.get("prior"),
        "current_vintage": meta.get("current"),
        "prior_hexes": _hexes_for_county(coverage, geoid, "prior"),
        "current_hexes": _hexes_for_county(coverage, geoid, "current"),
        "sites_prior": _sites_serving_county(sites, coverage, geoid, "prior"),
        "sites_current": _sites_serving_county(sites, coverage, geoid, "current"),
    }
    if counties is not None:
        boundary = _county_boundary_feature(counties, geoid)
        if boundary:
            detail["county_boundary"] = boundary
    return detail


def write_county_details(
    scored: pd.DataFrame,
    coverage: pd.DataFrame,
    sites: pd.DataFrame,
    data_dir: Path,
    meta: dict[str, Any],
    counties: gpd.GeoDataFrame | None = None,
    *,
    render_pngs: bool = False,
) -> int:
    details_dir = data_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    if scored.empty:
        return 0

    keys = scored[["provider_id", "technology", "county_geoid"]].drop_duplicates()
    n = 0
    for key_row in keys.to_dict("records"):
        pid = int(key_row["provider_id"])
        svc = str(key_row["technology"])
        geoid = str(key_row["county_geoid"])
        svc_dir = details_dir / str(pid) / _safe_service_key(svc)
        geoid_dir = svc_dir / geoid
        geoid_dir.mkdir(parents=True, exist_ok=True)

        cov = coverage
        if not coverage.empty and "provider_id" in coverage.columns:
            cov = coverage[coverage["provider_id"] == pid]
        if not coverage.empty and "technology" in coverage.columns:
            cov = cov[cov["technology"] == svc]

        st = sites
        if not sites.empty and "provider_id" in sites.columns:
            st = sites[sites["provider_id"] == pid]
        if not sites.empty and "technology" in sites.columns:
            st = st[st["technology"] == svc]

        detail = build_county_detail(geoid, cov, st, meta, counties=counties)
        sr = scored[
            (scored["provider_id"] == pid)
            & (scored["technology"] == svc)
            & (scored["county_geoid"].astype(str) == geoid)
        ]
        if not sr.empty:
            row = sr.iloc[0]
            detail["towers_prior"] = int(row.get("prior_towers", len(detail["sites_prior"])))
            detail["towers_current"] = int(row.get("current_towers", len(detail["sites_current"])))
            detail["new_towers"] = int(row.get("new_towers", 0))
            detail["prior_towers_cross_border"] = int(row.get("prior_towers_cross_border", 0))
            detail["current_towers_cross_border"] = int(row.get("current_towers_cross_border", 0))
            detail["new_towers_cross_border"] = int(row.get("new_towers_cross_border", 0))
        else:
            detail["towers_prior"] = len(detail["sites_prior"])
            detail["towers_current"] = len(detail["sites_current"])
            detail["new_towers"] = max(0, detail["towers_current"] - detail["towers_prior"])

        # Render server-side PNGs only when explicitly requested (--render-pngs).
        # By default the cockpit renders hexes client-side via deck.gl H3HexagonLayer,
        # which is ~1.4 MB/county cheaper and works at res 9/10.
        if render_pngs:
            all_sites_list = (detail.get("sites_prior") or []) + (detail.get("sites_current") or [])
            render_extent = map_render.compute_render_extent(
                detail.get("county_boundary"), all_sites_list,
            )
            context: dict[str, list] = {}
            if render_extent is not None:
                context = {
                    "prior_hexes": _context_hexes_for_bbox(cov, render_extent, "prior"),
                    "current_hexes": _context_hexes_for_bbox(cov, render_extent, "current"),
                }
            map_refs = map_render.render_county_compare_maps(detail, geoid_dir, context=context)
            detail.update(map_refs)

        out = svc_dir / f"{geoid}.json"
        out.write_text(json.dumps(detail, allow_nan=False), encoding="utf-8")
        n += 1
    log.info("wrote %d county detail files under %s", n, details_dir)
    return n


# ---------------------------------------------------------------------------
# Tower overlay per provider
# ---------------------------------------------------------------------------

def build_towers_by_provider(sites: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
    if sites.empty:
        return {}
    out: dict[int, list[dict[str, Any]]] = {}
    for s in sites.to_dict("records"):
        pid = int(s.get("provider_id", 0))
        out.setdefault(pid, []).append({
            "lat": float(s["lat"]),
            "lng": float(s["lng"]),
            "service": str(s.get("technology", "")),
            "site_class": str(s.get("site_class", "site")),
            "vintage": str(s.get("vintage", "current")),
            "county_geoid": str(s.get("county_geoid", "")),
            "n_hexes": int(s.get("n_hexes", 0)),
        })
    return out


# ---------------------------------------------------------------------------
# Full web bundle writer
# ---------------------------------------------------------------------------

def write_web_bundle(
    scored: pd.DataFrame,
    sites: pd.DataFrame,
    counties: gpd.GeoDataFrame,
    web_dir: Path,
    meta: dict[str, Any],
    *,
    simplify_tolerance: float = 0.001,
    coverage: pd.DataFrame | None = None,
    threshold: float = 0.0,
    weights: dict[str, float] | None = None,
    render_pngs: bool = False,
) -> dict[str, Path]:
    """Write static web data bundle under ``web/public/data/``."""
    data_dir = web_dir / "public" / "data"
    towers_dir = data_dir / "towers"
    data_dir.mkdir(parents=True, exist_ok=True)
    towers_dir.mkdir(parents=True, exist_ok=True)

    counties_path = data_dir / "counties.geojson"
    records_path = data_dir / "records.json"
    meta_path = data_dir / "meta.json"

    geoids = set(scored["county_geoid"].astype(str).unique()) if not scored.empty else None
    geo = build_counties_geojson(counties, simplify_tolerance, geoids=geoids)
    if not geo.get("features") and geoids:
        cache = web_dir.parent / "data" / "interim" / "tl_us_county.gpkg"
        if cache.exists():
            syn = gpd.read_file(cache)
            geo = build_counties_geojson(syn, simplify_tolerance, geoids=geoids)
            log.info("used synthetic county cache for web bundle (%d features)", len(geo.get("features", [])))
    counties_path.write_text(json.dumps(geo), encoding="utf-8")

    records = build_web_records(scored, threshold=threshold, weights=weights or {})
    records_path.write_text(json.dumps(records, allow_nan=False), encoding="utf-8")

    # Also write per-provider split files for lazy loading:
    # data/records/<pid>/<svc_key>.json — tiny index per provider+service
    records_split_dir = data_dir / "records"
    records_split_dir.mkdir(parents=True, exist_ok=True)
    for pid, svc_map in records.items():
        pid_dir = records_split_dir / str(pid)
        pid_dir.mkdir(exist_ok=True)
        for svc, geoid_map in svc_map.items():
            svc_path = pid_dir / f"{_safe_service_key(svc)}.json"
            svc_path.write_text(json.dumps(geoid_map, allow_nan=False), encoding="utf-8")

    web_meta = build_web_meta(scored, meta)
    meta_path.write_text(json.dumps(web_meta, indent=2), encoding="utf-8")

    towers_by_provider = build_towers_by_provider(sites)
    tower_paths: dict[str, Path] = {}
    for pid, feats in towers_by_provider.items():
        tp = towers_dir / f"{pid}.json"
        tp.write_text(json.dumps(feats, allow_nan=False), encoding="utf-8")
        tower_paths[str(pid)] = tp

    detail_count = 0
    if coverage is not None and not coverage.empty:
        detail_count = write_county_details(
            scored, coverage, sites, data_dir, meta, counties=counties,
            render_pngs=render_pngs,
        )

    log.info(
        "wrote web bundle: %d records, %d providers, %d tower files, %d county details",
        len(scored), len(web_meta["providers"]), len(tower_paths), detail_count,
    )
    return {
        "counties": counties_path,
        "records": records_path,
        "meta": meta_path,
        "towers_dir": towers_dir,
    }
