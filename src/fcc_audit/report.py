"""Outputs: ranked priority CSV, a human-readable summary, and dashboard data.

The dashboard payload (JSON) is consumed by ``dashboard/index.html`` (MapLibre),
which renders flagged counties and inferred new/expanded sites on a map.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


def _finite_or_none(value: Any, ndigits: int = 3) -> float | None:
    """Coerce inf/nan to None so the dashboard JSON stays valid."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, ndigits)

log = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "rank", "provider_name", "provider_id", "technology", "speed_tier",
    "environment", "county_geoid", "county_name",
    "state_fips", "priority_score", "flag_for_review", "flag_reason",
    "added_km2", "added_frac_of_county", "pct_increase", "blanket_fillin",
    "same_site_growth_share", "new_site_share", "unattributed_share",
    "boundary_snap_share", "new_towers", "new_hexes", "upgraded_hexes",
]


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
        svc = f"{r.get('technology', '')} {r.get('speed_tier', '')}".strip()
        lines.append(
            f"| {i} | {r.get('provider_name', r.get('provider_id'))} "
            f"| {svc} | {r.get('county_name', '')} | {r.get('state_fips', '')} "
            f"| {r['priority_score']:.3f} | {r.get('flag_reason', '')} |"
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
        county_features.append(
            {
                "geoid": str(r["county_geoid"]),
                "name": r.get("county_name", ""),
                "provider": r.get("provider_name", str(r.get("provider_id"))),
                "technology": r.get("technology", ""),
                "speed_tier": r.get("speed_tier", ""),
                "lat": latlng[0],
                "lng": latlng[1],
                "priority": _finite_or_none(r["priority_score"]),
                "flag": bool(r.get("flag_for_review", False)),
                "reason": r.get("flag_reason", ""),
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


def write_outputs(
    scored: pd.DataFrame,
    sites: pd.DataFrame,
    counties: gpd.GeoDataFrame,
    outputs_dir: Path,
    dashboard_dir: Path,
    meta: dict[str, Any],
) -> dict[str, Path]:
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
