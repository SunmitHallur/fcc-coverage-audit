"""Plain-language explanations for flagged counties.

Turns technical risk features into headlines, bullet points, and field-test
recommendations suitable for non-expert reviewers (e.g. managers, FCC staff).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _pct(value: Any, *, cap: float = 5.0) -> str:
    """Format a fraction as a readable percentage."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(v):
        return "a large amount"
    if v >= cap:
        return f"{cap * 100:.0f}%+"
    return f"{v * 100:.0f}%"


def _km2(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if v >= 100:
        return f"{v:.0f} km²"
    if v >= 10:
        return f"{v:.1f} km²"
    return f"{v:.2f} km²"


def severity_label(priority: float) -> str:
    """Map priority score to a human severity bucket."""
    if priority >= 0.85:
        return "Critical"
    if priority >= 0.70:
        return "High"
    if priority >= 0.50:
        return "Moderate"
    return "Low"


def explain_row(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Build a structured plain-language explanation for one scored county row."""
    r = row if isinstance(row, dict) else row.to_dict()
    provider = str(r.get("provider_name") or r.get("provider_id") or "Provider")
    county = str(r.get("county_name") or "this county")
    service = str(r.get("technology") or "mobile service")
    flagged = bool(r.get("flag_for_review", False))
    priority = float(r.get("priority_score") or 0.0)

    added_frac = float(r.get("added_frac_of_county") or 0.0)
    added_km2 = float(r.get("added_km2") or 0.0)
    pct_inc = r.get("pct_increase")
    same_site = float(r.get("same_site_growth_share") or 0.0)
    new_site = float(r.get("new_site_share") or 0.0)
    unattributed = float(r.get("unattributed_share") or 0.0)
    boundary = float(r.get("boundary_snap_share") or 0.0)
    blanket = float(r.get("blanket_fillin") or 0.0)
    new_towers = int(r.get("new_towers") or 0)
    prior_towers = int(r.get("prior_towers") or 0)
    current_towers = int(r.get("current_towers") or 0)
    prior_cross = int(r.get("prior_towers_cross_border") or 0)
    current_cross = int(r.get("current_towers_cross_border") or 0)
    new_cross = int(r.get("new_towers_cross_border") or 0)

    # --- headline (one sentence) ---
    if flagged and same_site >= 0.5:
        headline = (
            f"{provider} reported a large {service} coverage increase in {county} "
            f"but attributed most of it to existing towers — not new construction."
        )
    elif flagged and unattributed >= 0.3:
        headline = (
            f"{provider}'s new {service} coverage in {county} does not line up with "
            f"any inferred tower locations, which is physically unusual."
        )
    elif flagged and added_frac >= 0.10:
        headline = (
            f"{provider} added {service} coverage across {_pct(added_frac)} of {county} "
            f"since the last filing — an unusually large jump worth verifying."
        )
    elif flagged:
        headline = (
            f"{provider}'s {service} coverage change in {county} scored unusually high "
            f"compared to other counties nationwide."
        )
    elif added_km2 > 0 and new_site >= 0.5:
        if new_cross > 0 and new_towers > new_cross:
            headline = (
                f"{provider} expanded {service} in {county} primarily through "
                f"new towers — including {new_cross} just outside the county border."
            )
        elif new_cross > 0:
            headline = (
                f"{provider} expanded {service} in {county} from "
                f"{new_cross} newly inferred tower{'s' if new_cross != 1 else ''} "
                f"in a neighboring county."
            )
        else:
            headline = (
                f"{provider} expanded {service} in {county} primarily through "
                f"{new_towers} new tower{'s' if new_towers != 1 else ''} — a typical build pattern."
            )
    elif added_km2 > 0:
        headline = (
            f"{provider} reported a modest {service} coverage change in {county} "
            f"that does not raise major red flags."
        )
    else:
        headline = f"No meaningful {service} coverage change detected for {provider} in {county}."

    # --- bullets (2-4 supporting facts) ---
    bullets: list[str] = []
    if added_km2 > 0:
        bullets.append(f"New coverage area: approximately {_km2(added_km2)} ({_pct(added_frac)} of the county).")
    if pct_inc is not None and math.isfinite(float(pct_inc)) and float(pct_inc) > 0.05:
        bullets.append(f"Relative increase vs. prior map: {_pct(pct_inc)}.")
    if prior_towers or current_towers:
        tower_line = (
            f"Inferred towers covering this county: {prior_towers} prior → {current_towers} current"
        )
        if new_towers:
            tower_line += f" ({new_towers} new"
            if new_cross:
                tower_line += f", {new_cross} outside county"
            tower_line += ")"
        tower_line += "."
        bullets.append(tower_line)
    if current_cross or prior_cross:
        bullets.append(
            f"{current_cross or prior_cross} neighboring-county tower"
            f"{'s' if max(current_cross, prior_cross) != 1 else ''} "
            f"{'cover' if max(current_cross, prior_cross) != 1 else 'covers'} into {county} "
            f"and {'are' if max(current_cross, prior_cross) != 1 else 'is'} included in these counts."
        )
    if same_site >= 0.3:
        bullets.append(
            f"{_pct(same_site)} of the new coverage is attributed to towers that already existed "
            f"— the provider claims existing sites got stronger, not that new ones were built."
        )
    if new_site >= 0.3:
        if new_cross:
            bullets.append(
                f"{_pct(new_site)} of growth is attributed to {new_towers} newly inferred tower"
                f"{'s' if new_towers != 1 else ''} ({new_cross} outside this county)."
            )
        else:
            bullets.append(
                f"{_pct(new_site)} of growth comes from {new_towers} newly inferred tower"
                f"{'s' if new_towers != 1 else ''}."
            )
    if unattributed >= 0.2:
        bullets.append(
            f"{_pct(unattributed)} of new coverage sits far from any inferred tower, "
            f"which is hard to explain with normal radio propagation."
        )
    if boundary >= 0.3:
        bullets.append(
            f"{_pct(boundary)} of new coverage hugs the county boundary rather than "
            f"radiating from tower clusters — a pattern sometimes seen in map gaming."
        )
    if blanket >= 0.3:
        bullets.append(
            "The county went from very little coverage to near-complete fill-in in one filing period."
        )
    if not bullets:
        bullets.append("Coverage metrics are within normal ranges for this comparison period.")

    # --- recommendation ---
    if flagged:
        recommendation = (
            f"Schedule on-the-ground drive tests for {service} in {county} to confirm "
            f"the reported coverage matches what consumers actually receive."
        )
    elif added_km2 > 0:
        recommendation = (
            f"No immediate action required, but spot-check {service} in {county} "
            f"if resources allow."
        )
    else:
        recommendation = "No field testing recommended for this county at this time."

    return {
        "headline": headline,
        "bullets": bullets[:4],
        "recommendation": recommendation,
        "severity": severity_label(priority),
    }


def add_explanations(scored: pd.DataFrame) -> pd.DataFrame:
    """Add explanation columns to a scored DataFrame."""
    if scored.empty:
        return scored
    df = scored.copy()
    explanations = df.apply(explain_row, axis=1)
    df["plain_explanation"] = explanations.apply(lambda e: e["headline"])
    df["explanation_json"] = explanations.apply(lambda e: __import__("json").dumps(e))
    return df
