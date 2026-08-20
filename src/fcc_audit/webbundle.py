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
import shutil
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import mapping

from .explain import explain_row, add_explanations
from . import attribute

log = logging.getLogger(__name__)


def _rmtree_retry(path: Path, *, attempts: int = 6, delay_s: float = 0.5) -> None:
    """Remove a directory tree, retrying Windows/OneDrive lock races.

    OneDrive and Explorer often hold a brief exclusive handle on empty dirs
    right after file deletes, which surfaces as WinError 5 / PermissionError
    on ``os.rmdir``. Retrying is enough; ignore_errors alone can leave a
    half-deleted tree that breaks the next write.
    """
    if not path.exists():
        return

    def _onexc(func, p, exc_info):  # noqa: ANN001 — shutil callback signature
        err = exc_info[1] if isinstance(exc_info, tuple) else exc_info
        if isinstance(err, PermissionError):
            try:
                Path(p).chmod(0o700)
            except OSError:
                pass
            try:
                func(p)
                return
            except PermissionError:
                pass
        raise err

    last: Exception | None = None
    for i in range(attempts):
        try:
            # Python 3.12+ prefers onexc; older 3.9 still has onerror.
            try:
                shutil.rmtree(path, onexc=_onexc)
            except TypeError:
                shutil.rmtree(path, onerror=_onexc)
            if not path.exists():
                return
        except PermissionError as exc:
            last = exc
            log.warning(
                "retrying delete of %s after lock (%d/%d): %s",
                path, i + 1, attempts, exc,
            )
            time.sleep(delay_s * (i + 1))
    if path.exists():
        raise PermissionError(
            f"Could not remove {path} after {attempts} attempts (OneDrive/Explorer "
            f"lock?). Close anything using web/public/data, pause OneDrive sync, "
            f"delete the folder manually, then re-run build-web. Last error: {last}"
        )


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
    ("added_frac_of_county",        "absolute area added (% of county)"),
    ("coverage_increase_magnitude", "relative jump (absolute-gated)"),
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


def _tier_for_rank(rank: int, top_n: int = 250) -> str | None:
    """Map 1-based rank within a provider×service group to a display tier."""
    if rank < 1 or rank > top_n:
        return None
    if rank <= 50:
        return "red"
    if rank <= 100:
        return "orange"
    if rank <= 150:
        return "yellow"
    return "green"


def assign_record_tiers(scored: pd.DataFrame, top_n: int = 250) -> pd.DataFrame:
    """Rank counties per provider×service and attach a ``tier`` for the top *top_n*."""
    if scored.empty:
        return scored
    out = scored.copy()
    tiers = pd.Series([None] * len(out), index=out.index, dtype=object)
    for (_pid, _svc), grp in out.groupby(["provider_id", "technology"], sort=False):
        ranked = grp.sort_values(
            ["priority_score", "added_km2"],
            ascending=[False, False],
            na_position="last",
        )
        for rank, idx in enumerate(ranked.index[:top_n], start=1):
            tiers.loc[idx] = _tier_for_rank(rank, top_n)
    out["tier"] = tiers
    return out


# ---------------------------------------------------------------------------
# Flag-math tooltip export
# ---------------------------------------------------------------------------

def _build_flag_math(row: pd.Series, threshold: float, weights: dict[str, float]) -> dict[str, Any]:
    """Build the flag-math dict for the reviewer cockpit info tooltip.

    Prefers exported ``score_contribution_*`` columns from the scoring path so
    the UI matches the bounded / exculpatory math in ``score.score``.
    """
    feats = []
    for name, label in _TOOLTIP_FEATURES:
        val = _finite_or_none(row.get(name, 0.0))
        w = float(weights.get(name, 0.0) or 0.0)
        contrib_key = f"score_contribution_{name}"
        if contrib_key in row.index and pd.notna(row.get(contrib_key)):
            contrib = _finite_or_none(row.get(contrib_key))
        else:
            # Fallback only when contributions were not persisted (legacy rows).
            contrib = None
        feats.append({
            "name": name,
            "label": label,
            "value": val,
            "weight": round(w, 4),
            "contribution": contrib,
        })
    return {
        "priority_score": _finite_or_none(row.get("priority_score")),
        "flag_threshold": round(float(threshold), 4),
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
    rec: dict[str, Any] = {
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
    tier = r.get("tier")
    if tier is not None and not (isinstance(tier, float) and math.isnan(tier)):
        rec["tier"] = str(tier)
    return rec


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
    provider_services: dict[str, list[str]] = {}
    if not scored.empty and "provider_id" in scored.columns:
        for pid in sorted(scored["provider_id"].unique()):
            name = scored.loc[scored["provider_id"] == pid, "provider_name"].iloc[0]
            providers.append({"id": int(pid), "name": str(name)})
            provider_services[str(int(pid))] = sorted(
                scored.loc[scored["provider_id"] == pid, "technology"].astype(str).unique().tolist()
            )
    services = sorted(scored["technology"].unique().tolist()) if "technology" in scored.columns else []
    flagged = int(scored["flag_for_review"].sum()) if "flag_for_review" in scored.columns else 0
    web_meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_vintage": meta.get("current"),
        "prior_vintage": meta.get("prior"),
        "providers": providers,
        "services": services,
        "provider_services": provider_services,
        "total_records": len(scored),
        "flagged_count": flagged,
        "states_processed": meta.get("states_processed", "all"),
    }
    if meta.get("incomplete") or meta.get("allow_incomplete"):
        web_meta["incomplete"] = True
    if meta.get("flag_threshold") is not None:
        try:
            web_meta["flag_threshold"] = float(meta["flag_threshold"])
        except (TypeError, ValueError):
            pass
    if meta.get("feature_weights"):
        web_meta["feature_weights"] = {
            str(k): float(v) for k, v in dict(meta["feature_weights"]).items()
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
    """Legacy rectangular context (kept for PNG render extent fallback)."""
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


_CONTEXT_BUFFER_M = 8_000.0  # feather coverage ~8 km past the county boundary


def _context_hexes_for_county_ring(
    df: pd.DataFrame,
    counties: gpd.GeoDataFrame,
    geoid: str,
    vintage: str,
    buffer_m: float = _CONTEXT_BUFFER_M,
    equal_area_crs: str = "EPSG:5070",
) -> list[list]:
    """Hexes whose centroids fall inside the county polygon buffered by *buffer_m*.

    Replaces the rectangular lat/lng box that produced straight-edged crops at
    the edge of detail maps. Coverage feathers around the county instead.
    """
    if df.empty or "vintage" not in df.columns or counties is None or counties.empty:
        return []
    sub_c = counties[counties["county_geoid"].astype(str) == str(geoid)]
    if sub_c.empty:
        return []
    try:
        geom = sub_c.to_crs(equal_area_crs).geometry.union_all().buffer(float(buffer_m))
    except Exception:
        return []
    sub = df[df["vintage"] == vintage][["h3", "signal_dbm"]].drop_duplicates(subset=["h3"])
    if sub.empty:
        return []
    out: list[list] = []
    # Project centroids in batches via GeoSeries for speed.
    cells = sub["h3"].astype(str).tolist()
    sigs = sub["signal_dbm"].tolist()
    lats, lngs = [], []
    for c in cells:
        try:
            lat, lng = h3.cell_to_latlng(c)
        except Exception:
            lat, lng = float("nan"), float("nan")
        lats.append(lat)
        lngs.append(lng)
    pts = gpd.GeoSeries(gpd.points_from_xy(lngs, lats), crs="EPSG:4326").to_crs(equal_area_crs)
    keep = pts.intersects(geom)
    for cell, sig, ok in zip(cells, sigs, keep.to_numpy()):
        if bool(ok):
            out.append([cell, _encode_signal(float(sig))])
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


def _estimate_signal_from_sites(
    hexes: list[list],
    site_list: list[dict[str, Any]],
) -> tuple[list[list], bool]:
    """Replace a FLAT per-hex signal with a distance-to-tower estimate.

    Binary coverage sources (the Redshift 0/1 hex snapshots) carry no signal,
    so every hex encodes the same value and the heat map renders one solid
    color. For display only, synthesize a plausible RSRP via log-distance path
    loss from the nearest inferred site: −65 dBm at the tower falling to
    −120 dBm at the far fringe. Analysis never sees these values.

    Returns ``(hexes, estimated)`` — untouched when signal already varies or
    when there are no sites to measure distance from.
    """
    if not hexes or not site_list:
        return hexes, False
    values = {enc for _cell, enc in hexes}
    if len(values) > 1:
        return hexes, False

    site_lat = np.array([float(s["lat"]) for s in site_list])
    site_lng = np.array([float(s["lng"]) for s in site_list])
    out: list[list] = []
    for cell, _enc in hexes:
        try:
            lat, lng = h3.cell_to_latlng(str(cell))
        except Exception:
            out.append([cell, _encode_signal(-95.0)])
            continue
        # Equirectangular distance is plenty accurate at county scale.
        dx = (site_lng - lng) * 111.32 * math.cos(math.radians(lat))
        dy = (site_lat - lat) * 110.57
        d_km = float(np.min(np.hypot(dx, dy)))
        dbm = max(-120.0, -65.0 - 25.0 * math.log10(1.0 + d_km))
        out.append([cell, _encode_signal(dbm)])
    return out, True


def apply_scored_tower_counts(detail: dict[str, Any], row: Any | None) -> None:
    """Copy scored serving counts onto a county detail payload.

    When inference dropped every lobe, scored ``prior_towers``/``current_towers``
    are 0 even if the map still lists home-county pins. Prefer those list
    lengths so the header is not stuck at ``0 → 0``.
    """
    n_prior = len(detail.get("sites_prior") or [])
    n_cur = len(detail.get("sites_current") or [])

    def _as_int(val: Any, default: int = 0) -> int:
        if val is None:
            return default
        try:
            if pd.isna(val):
                return default
        except (TypeError, ValueError):
            pass
        try:
            n = int(val)
        except (TypeError, ValueError):
            return default
        return n

    if row is None:
        detail["towers_prior"] = n_prior
        detail["towers_current"] = n_cur
        detail["new_towers"] = max(0, n_cur - n_prior)
        return

    getter = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    scored_prior = _as_int(getter("prior_towers"))
    scored_cur = _as_int(getter("current_towers"))
    used_fallback = False
    if scored_prior <= 0 and n_prior > 0:
        prior = n_prior
        used_fallback = True
    else:
        prior = scored_prior
    if scored_cur <= 0 and n_cur > 0:
        current = n_cur
        used_fallback = True
    else:
        current = scored_cur
    detail["towers_prior"] = prior
    detail["towers_current"] = current
    if used_fallback:
        detail["new_towers"] = max(0, current - prior)
    else:
        detail["new_towers"] = _as_int(getter("new_towers"), max(0, current - prior))
    detail["prior_towers_cross_border"] = _as_int(getter("prior_towers_cross_border"))
    detail["current_towers_cross_border"] = _as_int(getter("current_towers_cross_border"))
    detail["new_towers_cross_border"] = _as_int(getter("new_towers_cross_border"))


def build_county_detail(
    geoid: str,
    coverage: pd.DataFrame,
    sites: pd.DataFrame,
    meta: dict[str, Any],
    counties: gpd.GeoDataFrame | None = None,
) -> dict[str, Any]:
    # Prefer a buffered-county ring of hexes so detail maps don't end at a
    # hard rectangular crop. Fall back to in-county-only when counties missing.
    if counties is not None and not counties.empty and not coverage.empty:
        prior_hexes = _context_hexes_for_county_ring(coverage, counties, geoid, "prior")
        current_hexes = _context_hexes_for_county_ring(coverage, counties, geoid, "current")
        if not prior_hexes and not current_hexes:
            prior_hexes = _hexes_for_county(coverage, geoid, "prior")
            current_hexes = _hexes_for_county(coverage, geoid, "current")
    else:
        prior_hexes = _hexes_for_county(coverage, geoid, "prior")
        current_hexes = _hexes_for_county(coverage, geoid, "current")

    detail: dict[str, Any] = {
        "geoid": str(geoid),
        "prior_vintage": meta.get("prior"),
        "current_vintage": meta.get("current"),
        "prior_hexes": prior_hexes,
        "current_hexes": current_hexes,
        "sites_prior": _sites_serving_county(sites, coverage, geoid, "prior"),
        "sites_current": _sites_serving_county(sites, coverage, geoid, "current"),
        "fit_mode": "county_buffer",
    }
    # ASR anchors on sites (when present) for the UI tower panel.
    for key in ("sites_prior", "sites_current"):
        for s in detail[key]:
            # Pass through asr fields if the sites frame carried them.
            pass
    if not sites.empty and "asr_matched" in sites.columns:
        vmap = {"sites_prior": "prior", "sites_current": "current"}
        for key, vintage in vmap.items():
            sub = sites[sites.get("vintage", pd.Series("", index=sites.index)) == vintage]
            if sub.empty:
                continue
            by_coord = {
                (round(float(r["lat"]), 5), round(float(r["lng"]), 5)): r
                for r in sub.to_dict("records")
            }
            for s in detail[key]:
                hit = by_coord.get((round(float(s["lat"]), 5), round(float(s["lng"]), 5)))
                if hit is not None:
                    s["asr_matched"] = bool(hit.get("asr_matched", False))
                    s["asr_snapped"] = bool(hit.get("asr_snapped", False))
                    dist = hit.get("asr_distance_m")
                    s["asr_distance_m"] = (
                        float(dist) if dist is not None and math.isfinite(float(dist)) else None
                    )

    detail["prior_hexes"], est_p = _estimate_signal_from_sites(
        detail["prior_hexes"], detail["sites_prior"],
    )
    detail["current_hexes"], est_c = _estimate_signal_from_sites(
        detail["current_hexes"], detail["sites_current"],
    )
    if est_p or est_c:
        detail["signal_estimated"] = True
    else:
        prior_vals = {h[1] for h in detail["prior_hexes"]} if detail["prior_hexes"] else set()
        current_vals = {h[1] for h in detail["current_hexes"]} if detail["current_hexes"] else set()
        # Binary / flat Redshift coverage with no per-hex signal variation and no
        # sites to distance-estimate from — UI must not paint this as max green.
        if len(prior_vals | current_vals) <= 1:
            detail["signal_flat"] = True
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

    # Pre-index scored rows and county boundaries by key so the per-county loop
    # below never rescans a large frame. Without this, a nationwide run rescans
    # the ~10^8-row coverage table (and the county GeoDataFrame) once PER county,
    # which is O(counties x rows) and takes hours.
    scored_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for r in scored.assign(county_geoid=scored["county_geoid"].astype(str)).to_dict("records"):
        scored_by_key[(int(r["provider_id"]), str(r["technology"]), str(r["county_geoid"]))] = r

    counties_by_geoid: dict[str, gpd.GeoDataFrame] = {}
    if counties is not None and not counties.empty:
        for i, g in enumerate(counties["county_geoid"].astype(str)):
            counties_by_geoid.setdefault(g, counties.iloc[[i]])

    # Only emit per-county detail JSON for tiered (top-N) counties — main storage win.
    detail_scored = scored
    if "tier" in scored.columns:
        detail_scored = scored[scored["tier"].notna()]
    keys = detail_scored[["provider_id", "technology", "county_geoid"]].drop_duplicates()
    n = 0
    # Filter the big coverage/sites tables ONCE per (provider, service), then
    # group coverage by county so each county is an O(1) dict lookup.
    for (pid_raw, svc_raw), key_grp in keys.groupby(["provider_id", "technology"], sort=False):
        pid = int(pid_raw)
        svc = str(svc_raw)
        svc_dir = details_dir / str(pid) / _safe_service_key(svc)

        cov_ps = coverage
        if not coverage.empty and "provider_id" in coverage.columns:
            cov_ps = cov_ps[cov_ps["provider_id"] == pid]
        if not coverage.empty and "technology" in coverage.columns:
            cov_ps = cov_ps[cov_ps["technology"] == svc]
        cov_by_county: dict[str, pd.DataFrame] = {}
        if not cov_ps.empty:
            for g, sub in cov_ps.groupby(cov_ps["county_geoid"].astype(str), sort=False):
                cov_by_county[str(g)] = sub
        empty_cov = cov_ps.iloc[0:0]

        st = sites
        if not sites.empty and "provider_id" in sites.columns:
            st = st[st["provider_id"] == pid]
        if not sites.empty and "technology" in sites.columns:
            st = st[st["technology"] == svc]

        for geoid in key_grp["county_geoid"].astype(str):
            geoid_dir = svc_dir / geoid
            geoid_dir.mkdir(parents=True, exist_ok=True)

            cov = cov_by_county.get(geoid, empty_cov)
            cty = counties_by_geoid.get(geoid)
            detail = build_county_detail(geoid, cov, st, meta, counties=cty)
            row = scored_by_key.get((pid, svc, geoid))
            apply_scored_tower_counts(detail, row)

            # Render server-side PNGs only when explicitly requested (--render-pngs).
            # By default the cockpit renders hexes client-side via deck.gl H3HexagonLayer,
            # which is ~1.4 MB/county cheaper and works at res 9/10.
            if render_pngs:
                from . import map_render

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


def write_county_details_from_parquets(
    scored: pd.DataFrame,
    coverage_paths: list[Path],
    sites: pd.DataFrame,
    data_dir: Path,
    meta: dict[str, Any],
    counties: gpd.GeoDataFrame | None = None,
    *,
    render_pngs: bool = False,
) -> int:
    """Stream state coverage partitions when building national county details."""
    total = 0
    for path in coverage_paths:
        coverage = pd.read_parquet(path)
        if coverage.empty:
            continue
        coverage = coverage[
            ~coverage["county_geoid"].astype(str).str.startswith("90")
        ].reset_index(drop=True)
        geoids = set(coverage["county_geoid"].astype(str))
        state_scored = scored[scored["county_geoid"].astype(str).isin(geoids)]
        if state_scored.empty:
            continue
        total += write_county_details(
            state_scored,
            coverage,
            sites,
            data_dir,
            meta,
            counties=counties,
            render_pngs=render_pngs,
        )
    return total


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
    coverage_paths: list[Path] | None = None,
    threshold: float = 0.0,
    weights: dict[str, float] | None = None,
    render_pngs: bool = False,
    top_n: int = 250,
    write_details: bool = True,
) -> dict[str, Path]:
    """Write static web data bundle under ``web/public/data/``."""
    data_dir = web_dir / "public" / "data"
    towers_dir = data_dir / "towers"
    data_dir.mkdir(parents=True, exist_ok=True)

    # A build is a snapshot, not an append. Remove generated data from older
    # snapshots so a partial run cannot leave stale provider/service records or
    # county details behind. In particular, older bundles wrote records.json;
    # if it remains, the browser can load that stale monolith instead of the
    # current split files.
    for generated_dir in (data_dir / "records", data_dir / "details", towers_dir):
        _rmtree_retry(generated_dir)
    try:
        (data_dir / "records.json").unlink(missing_ok=True)
    except PermissionError:
        time.sleep(1.0)
        (data_dir / "records.json").unlink(missing_ok=True)

    towers_dir.mkdir(parents=True, exist_ok=True)

    counties_path = data_dir / "counties.geojson"
    meta_path = data_dir / "meta.json"

    scored = assign_record_tiers(scored, top_n=top_n)

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

    # Per-provider split files for lazy loading (monolithic records.json omitted):
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
    web_meta["top_n"] = top_n
    web_meta["use_split_records"] = True
    if threshold:
        web_meta["flag_threshold"] = float(threshold)
    if weights:
        web_meta["feature_weights"] = {str(k): float(v) for k, v in weights.items()}
    meta_path.write_text(json.dumps(web_meta, indent=2), encoding="utf-8")

    towers_by_provider = build_towers_by_provider(sites)
    tower_paths: dict[str, Path] = {}
    for pid, feats in towers_by_provider.items():
        tp = towers_dir / f"{pid}.json"
        tp.write_text(json.dumps(feats, allow_nan=False), encoding="utf-8")
        tower_paths[str(pid)] = tp

    detail_count = 0
    if write_details:
        if coverage_paths:
            detail_count = write_county_details_from_parquets(
                scored, coverage_paths, sites, data_dir, meta, counties=counties,
                render_pngs=render_pngs,
            )
        elif coverage is not None and not coverage.empty:
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
        "records_dir": records_split_dir,
        "meta": meta_path,
        "towers_dir": towers_dir,
    }
