"""Outputs: ranked priority CSV, a human-readable summary, dashboard data, and web bundle.

The dashboard payload (JSON) is consumed by ``dashboard/index.html`` (legacy) and
``web/index.html`` (production MapLibre choropleth). The web bundle is a compact
static set of files suitable for Vercel deployment.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import mapping

from .explain import add_explanations, explain_row
from . import attribute
from . import map_render

log = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "rank", "provider_name", "provider_id", "technology",
    "county_geoid", "county_name",
    "state_fips", "priority_score", "flag_for_review", "flag_reason", "plain_explanation",
    "added_km2", "added_frac_of_county", "pct_increase", "blanket_fillin",
    "same_site_growth_share", "new_site_share", "unattributed_share",
    "boundary_snap_share", "new_towers",     "prior_towers", "current_towers", "prior_towers_cross_border", "current_towers_cross_border",
    "new_towers_in_county", "new_towers_cross_border", "new_hexes", "upgraded_hexes",
]

_METRIC_KEYS = [
    "priority_score", "flag_for_review", "added_km2", "added_frac_of_county",
    "pct_increase", "same_site_growth_share", "new_site_share",
    "unattributed_share", "boundary_snap_share", "blanket_fillin", "new_towers",
    "new_towers_in_county", "new_towers_cross_border",
    "prior_towers", "current_towers", "prior_towers_in_county", "current_towers_in_county",
    "prior_towers_cross_border", "current_towers_cross_border",
    "prior_cov_frac", "current_cov_frac", "flag_reason",
]


def _finite_or_none(value: Any, ndigits: int = 3) -> float | None:
    """Coerce inf/nan to None so the dashboard JSON stays valid."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, ndigits)


def write_ranking_csv(scored: pd.DataFrame, path: Path) -> Path:
    df = scored.copy()
    df.insert(0, "rank", range(1, len(df) + 1))
    cols = [c for c in _CSV_COLUMNS if c in df.columns]
    df[cols].to_csv(path, index=False)
    log.info("wrote ranking: %s (%d rows)", path, len(df))
    return path


def write_selected_list(scored: pd.DataFrame, path: Path) -> Path:
    """Emit the automated selection list - the deliverable that replaces the
    manual consultant county-selection process. Only flagged rows, ranked."""
    sel = scored[scored["flag_for_review"]].copy() if "flag_for_review" in scored else scored.copy()
    sel = sel.sort_values("priority_score", ascending=False)
    sel.insert(0, "select_rank", range(1, len(sel) + 1))
    cols = [c for c in _CSV_COLUMNS if c in sel.columns and c != "rank"]
    sel[["select_rank", *cols]].to_csv(path, index=False)
    log.info("wrote selection list: %s (%d counties selected)", path, len(sel))
    return path


def write_summary_md(scored: pd.DataFrame, path: Path, meta: dict[str, Any]) -> Path:
    flagged = scored[scored["flag_for_review"]] if "flag_for_review" in scored else scored
    lines = [
        "# FCC Mobile Coverage-Change Audit - Priority Review",
        "",
        f"- Current vintage: **{meta.get('current')}**",
        f"- Prior vintage: **{meta.get('prior')}**",
        f"- Providers analyzed: {meta.get('providers')}",
        f"- Services analyzed: {meta.get('technologies', '')}",
        f"- Provider-county-service rows scored: **{len(scored)}**",
        f"- Flagged for manual review: **{len(flagged)}**",
        "",
        "## Top priorities for on-the-ground testing",
        "",
        "| Rank | Provider | Service | County | State | Priority | Why |",
        "|-----:|----------|---------|--------|:-----:|:--------:|-----|",
    ]
    for i, (_, r) in enumerate(flagged.head(25).iterrows(), start=1):
        svc = str(r.get("technology", "")).strip()
        why = r.get("plain_explanation") or r.get("flag_reason", "")
        lines.append(
            f"| {i} | {r.get('provider_name', r.get('provider_id'))} "
            f"| {svc} | {r.get('county_name', '')} | {r.get('state_fips', '')} "
            f"| {r['priority_score']:.3f} | {why} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "- **same_site_growth_share** high = provider claims big new coverage from "
        "*existing* towers (no new build) - the primary gaming signal.",
        "- **unattributed_share** high = new coverage that does not radiate from any "
        "inferred tower (physically implausible).",
        "- Tower locations are *inferred* from coverage structure and are approximate; "
        "use them to target field tests, not as ground truth.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote summary: %s", path)
    return path


def build_dashboard_payload(
    scored: pd.DataFrame, sites: pd.DataFrame, counties: gpd.GeoDataFrame
) -> dict[str, Any]:
    centroids = counties.copy()
    centroids["geometry"] = centroids.geometry.representative_point()
    cen = {
        row["county_geoid"]: (row.geometry.y, row.geometry.x)
        for _, row in centroids.iterrows()
    }
    county_features = []
    for _, r in scored.iterrows():
        latlng = cen.get(str(r["county_geoid"]))
        if not latlng:
            continue
        expl = explain_row(r)
        county_features.append(
            {
                "geoid": str(r["county_geoid"]),
                "name": r.get("county_name", ""),
                "provider": r.get("provider_name", str(r.get("provider_id"))),
                "technology": r.get("technology", ""),
                "lat": latlng[0],
                "lng": latlng[1],
                "priority": _finite_or_none(r["priority_score"]),
                "flag": bool(r.get("flag_for_review", False)),
                "reason": r.get("flag_reason", ""),
                "plain_explanation": expl["headline"],
                "pct_increase": _finite_or_none(r.get("pct_increase")),
                "same_site_growth_share": _finite_or_none(r.get("same_site_growth_share", 0)),
            }
        )
    site_features = [
        {
            "lat": float(s["lat"]),
            "lng": float(s["lng"]),
            "provider": s.get("provider_name", str(s.get("provider_id"))),
            "site_class": s.get("site_class", "site"),
            "n_hexes": int(s.get("n_hexes", 0)),
        }
        for _, s in sites.iterrows()
    ] if not sites.empty else []
    return {"counties": county_features, "sites": site_features}


def _record_from_row(r: pd.Series) -> dict[str, Any]:
    """Compact web record for one provider x service x county."""
    expl = explain_row(r)
    metrics = {
        k: _finite_or_none(r[k]) if k in r and k != "flag_for_review" else None
        for k in _METRIC_KEYS
    }
    metrics["flag_for_review"] = bool(r.get("flag_for_review", False))
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
    }


def build_web_records(scored: pd.DataFrame) -> dict[str, Any]:
    """Build nested lookup: provider_id -> service -> geoid -> record."""
    if scored.empty:
        return {}
    lookup: dict[str, dict[str, dict[str, Any]]] = {}
    for _, r in scored.iterrows():
        pid = str(int(r["provider_id"]))
        svc = str(r.get("technology", ""))
        geoid = str(r["county_geoid"])
        lookup.setdefault(pid, {}).setdefault(svc, {})[geoid] = _record_from_row(r)
    return lookup


def build_web_meta(scored: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    """Site-wide metadata for the web app."""
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


def _county_boundary_feature(
    counties: gpd.GeoDataFrame,
    geoid: str,
    *,
    simplify_tolerance: float | None = 0.0002,
) -> dict[str, Any] | None:
    """GeoJSON Feature for one county (used in compare maps)."""
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
    """Simplify county boundaries for web delivery (~1-3 MB GeoJSON)."""
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


def _safe_service_key(service: str) -> str:
    return service.replace("/", "-").replace(" ", "")


def _hexes_for_county(df: pd.DataFrame, geoid: str, vintage: str) -> list[list]:
    """Compact [h3, signal_dbm] pairs for one county and vintage."""
    if df.empty:
        return []
    mask = (
        df["county_geoid"].astype(str) == str(geoid)
    ) & (df["vintage"] == vintage)
    sub = df.loc[mask, ["h3", "signal_dbm"]]
    return [
        [str(row.h3), round(float(row.signal_dbm), 1)]
        for row in sub.itertuples(index=False)
    ]


def _context_hexes_for_bbox(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    vintage: str,
) -> list[list]:
    """All coverage hexes whose centroids fall inside *bbox* (WGS84 minx,miny,maxx,maxy)."""
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
            out.append([str(row.h3), round(float(row.signal_dbm), 1)])
    return out


def _sites_for_county(sites: pd.DataFrame, geoid: str, vintage: str) -> list[dict[str, Any]]:
    if sites.empty:
        return []
    mask = (
        sites["county_geoid"].astype(str) == str(geoid)
    ) & (sites.get("vintage", "current") == vintage)
    out = []
    for _, s in sites.loc[mask].iterrows():
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
    """All inferred sites whose lobes cover hexes in this county (incl. neighbors)."""
    if sites.empty:
        return []
    sub_sites = sites[sites.get("vintage", "current") == vintage].reset_index(drop=True)
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

    out = []
    geoid_s = str(geoid)
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
    """One county's before/after coverage hexes and inferred tower sites."""
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
) -> int:
    """Write per-county JSON for before/after map comparison on click."""
    details_dir = data_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    if scored.empty:
        return 0

    keys = scored[["provider_id", "technology", "county_geoid"]].drop_duplicates()
    n = 0
    for _, row in keys.iterrows():
        pid = int(row["provider_id"])
        svc = str(row["technology"])
        geoid = str(row["county_geoid"])
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

        all_sites = (detail.get("sites_prior") or []) + (detail.get("sites_current") or [])
        render_extent = map_render.compute_render_extent(
            detail.get("county_boundary"), all_sites,
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


def build_towers_by_provider(sites: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
    """Group tower features by provider_id for lazy loading."""
    if sites.empty:
        return {}
    out: dict[int, list[dict[str, Any]]] = {}
    for _, s in sites.iterrows():
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


def write_web_bundle(
    scored: pd.DataFrame,
    sites: pd.DataFrame,
    counties: gpd.GeoDataFrame,
    web_dir: Path,
    meta: dict[str, Any],
    *,
    simplify_tolerance: float = 0.001,
    coverage: pd.DataFrame | None = None,
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

    records = build_web_records(scored)
    records_path.write_text(json.dumps(records, allow_nan=False), encoding="utf-8")

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


def write_outputs(
    scored: pd.DataFrame,
    sites: pd.DataFrame,
    counties: gpd.GeoDataFrame,
    outputs_dir: Path,
    dashboard_dir: Path,
    meta: dict[str, Any],
) -> dict[str, Path]:
    scored = add_explanations(scored)
    tag = f"{meta.get('current')}_vs_{meta.get('prior')}"
    csv_path = write_ranking_csv(scored, outputs_dir / f"priority_ranking_{tag}.csv")
    selected_path = write_selected_list(scored, outputs_dir / f"selected_counties_{tag}.csv")
    md_path = write_summary_md(scored, outputs_dir / f"summary_{tag}.md", meta)

    payload = build_dashboard_payload(scored, sites, counties)
    blob = json.dumps(payload, allow_nan=False)  # guarantee browser-valid JSON
    data_path = outputs_dir / f"dashboard_data_{tag}.json"
    data_path.write_text(blob, encoding="utf-8")
    # Also drop a copy the static dashboard reads by default.
    (dashboard_dir / "data.json").write_text(blob, encoding="utf-8")
    log.info("wrote dashboard data: %s (%d counties, %d sites)",
             data_path, len(payload["counties"]), len(payload["sites"]))
    return {
        "selected_list": selected_path,
        "ranking_csv": csv_path,
        "summary": md_path,
        "dashboard_data": data_path,
    }


def load_accumulated_scored(scored_dir: Path) -> pd.DataFrame:
    """Load and merge all batch parquet files from ``data/processed/scored/``."""
    if not scored_dir.exists():
        return pd.DataFrame()
    parts = sorted(scored_dir.glob("scored_*.parquet"))
    if not parts:
        return pd.DataFrame()
    dfs = [pd.read_parquet(p) for p in parts]
    combined = pd.concat(dfs, ignore_index=True)
    # Drop synthetic fixture counties if present from earlier test runs.
    combined = combined[~combined["county_geoid"].astype(str).str.startswith("900")]
    # De-duplicate on provider + service + county, keeping the latest batch.
    if "batch_ts" in combined.columns:
        combined = combined.sort_values("batch_ts").drop_duplicates(
            subset=["provider_id", "technology", "county_geoid"], keep="last"
        )
    else:
        combined = combined.drop_duplicates(
            subset=["provider_id", "technology", "county_geoid"], keep="last"
        )
    return combined.reset_index(drop=True)


def load_accumulated_coverage(coverage_dir: Path) -> pd.DataFrame:
    """Load and merge all batch coverage snapshot parquet files."""
    if not coverage_dir.exists():
        return pd.DataFrame()
    parts = sorted(coverage_dir.glob("coverage_*.parquet"))
    if not parts:
        return pd.DataFrame()
    combined = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    dedup = ["provider_id", "technology", "county_geoid", "vintage", "h3"]
    present = [c for c in dedup if c in combined.columns]
    if "batch_ts" in combined.columns:
        combined = combined.sort_values("batch_ts").drop_duplicates(subset=present, keep="last")
    elif present:
        combined = combined.drop_duplicates(subset=present, keep="last")
    return combined.reset_index(drop=True)


def save_batch_scored(
    scored: pd.DataFrame,
    scored_dir: Path,
    *,
    service_label: str,
    states: list[str],
    meta: dict[str, Any],
) -> Path:
    """Persist one batch's scored rows for later web bundle assembly."""
    scored_dir.mkdir(parents=True, exist_ok=True)
    states_key = "-".join(sorted(states)) if states else "all"
    safe_svc = service_label.replace("/", "-").replace(" ", "")
    path = scored_dir / f"scored_{safe_svc}_{states_key}.parquet"
    df = scored.copy()
    df["batch_ts"] = datetime.now(timezone.utc).isoformat()
    df["batch_states"] = ",".join(states) if states else "all"
    df["batch_current"] = meta.get("current")
    df["batch_prior"] = meta.get("prior")
    df.to_parquet(path, index=False)
    log.info("saved batch scored: %s (%d rows)", path.name, len(df))
    return path
