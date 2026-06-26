"""CSV and Markdown serializers for audit pipeline outputs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .explain import add_explanations

log = logging.getLogger(__name__)

_CSV_COLUMNS = [
    "rank", "provider_name", "provider_id", "technology",
    "county_geoid", "county_name",
    "state_fips", "priority_score", "flag_for_review", "flag_reason", "plain_explanation",
    "added_km2", "added_frac_of_county", "pct_increase", "blanket_fillin",
    "same_site_growth_share", "new_site_share", "unattributed_share",
    "boundary_snap_share", "new_towers", "prior_towers", "current_towers",
    "prior_towers_cross_border", "current_towers_cross_border",
    "new_towers_in_county", "new_towers_cross_border", "new_hexes", "upgraded_hexes",
]


def write_ranking_csv(scored: pd.DataFrame, path: Path) -> Path:
    df = scored.copy()
    df.insert(0, "rank", range(1, len(df) + 1))
    cols = [c for c in _CSV_COLUMNS if c in df.columns]
    df[cols].to_csv(path, index=False)
    log.info("wrote ranking: %s (%d rows)", path, len(df))
    return path


def write_selected_list(scored: pd.DataFrame, path: Path) -> Path:
    """Emit the automated selection list — replaces manual consultant selection."""
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
        "*existing* towers (no new build) — the primary gaming signal.",
        "- **unattributed_share** high = new coverage that does not radiate from any "
        "inferred tower (physically implausible).",
        "- **asr_no_new_structure** = no FCC-registered structure built in window "
        "(independent corroboration from ASR database).",
        "- **measurement_gap** high = coverage claimed but no field measurements observed "
        "(Ookla/FCC Speed Test).",
        "- Tower locations are *inferred* from coverage structure and are approximate; "
        "use them to target field tests, not as ground truth.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote summary: %s", path)
    return path


def write_outputs_csv_md(
    scored: pd.DataFrame,
    outputs_dir: Path,
    meta: dict[str, Any],
) -> dict[str, Path]:
    """Write CSV + markdown outputs; return paths dict."""
    scored = add_explanations(scored)
    tag = f"{meta.get('current')}_vs_{meta.get('prior')}"
    return {
        "ranking_csv": write_ranking_csv(scored, outputs_dir / f"priority_ranking_{tag}.csv"),
        "selected_list": write_selected_list(scored, outputs_dir / f"selected_counties_{tag}.csv"),
        "summary": write_summary_md(scored, outputs_dir / f"summary_{tag}.md", meta),
    }
