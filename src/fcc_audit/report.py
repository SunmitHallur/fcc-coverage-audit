"""Backward-compatible re-export shim for the report module.

The 650-line monolithic report.py has been split into:
  serialize.py  — CSV and Markdown outputs
  webbundle.py  — web data assembly (records, counties, details, towers)
  persist.py    — batch parquet IO

All public names are re-exported here so existing imports keep working.
New code should import from the split modules directly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from .explain import add_explanations
from .persist import (
    save_batch_scored,
    load_accumulated_scored,
    load_accumulated_coverage,
)
from .serialize import (
    write_ranking_csv,
    write_selected_list,
    write_summary_md,
    _CSV_COLUMNS,
)
from .webbundle import (
    _METRIC_KEYS,
    _finite_or_none,
    _safe_service_key,
    _hexes_for_county,
    _context_hexes_for_bbox,
    _sites_for_county,
    _sites_serving_county,
    _county_boundary_feature,
    build_dashboard_payload,
    build_web_records,
    build_web_meta,
    build_counties_geojson,
    build_county_detail,
    write_county_details,
    build_towers_by_provider,
    write_web_bundle,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy combined write_outputs (kept for CLI backward compat)
# ---------------------------------------------------------------------------

def write_outputs(
    scored: pd.DataFrame,
    sites: pd.DataFrame,
    counties: gpd.GeoDataFrame,
    outputs_dir: Path,
    dashboard_dir: Path,
    meta: dict[str, Any],
    *,
    threshold: float = 0.0,
    weights: dict[str, float] | None = None,
) -> dict[str, Path]:
    scored = add_explanations(scored)
    tag = f"{meta.get('current')}_vs_{meta.get('prior')}"
    csv_path = write_ranking_csv(scored, outputs_dir / f"priority_ranking_{tag}.csv")
    selected_path = write_selected_list(scored, outputs_dir / f"selected_counties_{tag}.csv")
    md_path = write_summary_md(scored, outputs_dir / f"summary_{tag}.md", meta)

    payload = build_dashboard_payload(scored, sites, counties)
    blob = json.dumps(payload, allow_nan=False)
    data_path = outputs_dir / f"dashboard_data_{tag}.json"
    data_path.write_text(blob, encoding="utf-8")
    (dashboard_dir / "data.json").write_text(blob, encoding="utf-8")
    log.info("wrote dashboard data: %s (%d counties, %d sites)",
             data_path, len(payload["counties"]), len(payload["sites"]))
    return {
        "selected_list": selected_path,
        "ranking_csv": csv_path,
        "summary": md_path,
        "dashboard_data": data_path,
    }


# Keep the old _record_from_row name available for any code that imported it directly.
def _record_from_row(r: pd.Series) -> dict[str, Any]:
    from .webbundle import _record_from_dict
    return _record_from_dict(r.to_dict(), threshold=0.0, weights={})
