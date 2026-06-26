"""Gaming-risk feature engineering, anomaly ranking, and prioritization.

Produces one row per (provider, county) with interpretable risk features, an
IsolationForest anomaly score, a composite priority score, and a flag for the
counties the FCC should manually test. Designed so a reviewer can see *why* a
county was flagged, not just that it was.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import Config

_EPS = 1e-9
# Cap runaway percentages (e.g. coverage from 0 -> something => inf).
_PCT_CAP = 5.0
# A large RELATIVE jump only matters when the ABSOLUTE area gained is also
# meaningful: going from 0% to 1% of a county is a huge percentage but a trivial
# absolute change and is not a concern. The relative-jump feature is dampened by
# a saturating ramp on the added fraction of county, reaching full strength once
# a county gains this fraction of its area in new coverage.
_REL_JUMP_ABS_REF = 0.05

# Map config feature names to the columns built in build_features().
# Order reflects the FCC-verified selection patterns (primary first).
_RISK_FEATURES = [
    "added_frac_of_county",          # PRIMARY: in-county area increase (absolute, normalized)
    "coverage_increase_magnitude",   # PRIMARY: relative jump (D25/J25)
    "blanket_fillin",                # SECONDARY: low baseline -> near-complete fill (rural)
    "same_site_growth_share",        # SECONDARY: growth from existing towers
    "unattributed_share",            # SECONDARY: coverage far from any inferred tower
    "boundary_snap_share",           # SECONDARY: new coverage hugging county boundary
    "new_site_share",                # legitimacy (negative weight)
    "signal_jump_implausibility",    # available; default weight 0
    # Ground-truth corroboration signals (available when ASR/measured data loaded).
    "asr_no_new_structure",          # GT: no registered tower built in window (positive = suspicious)
    "measurement_gap",               # GT: claimed coverage with no field measurement (suspicious)
]


def build_features(
    county_change: pd.DataFrame,
    attribution: pd.DataFrame,
    boundary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge change + attribution + boundary into one row per county with the
    FCC-verified risk features."""
    if county_change.empty:
        return pd.DataFrame()
    df = county_change.merge(attribution, on="county_geoid", how="left")
    if boundary is not None and not boundary.empty:
        df = df.merge(boundary, on="county_geoid", how="left")
    for col in ["added_km2_new_site", "added_km2_expanded_site", "added_km2_unattributed"]:
        if col not in df:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    total_added = (
        df["added_km2_new_site"]
        + df["added_km2_expanded_site"]
        + df["added_km2_unattributed"]
    ).clip(lower=_EPS)

    # PRIMARY drivers.
    df["added_frac_of_county"] = df.get("added_frac_of_county", 0.0)
    df["added_frac_of_county"] = df["added_frac_of_county"].clip(lower=0.0).fillna(0.0)
    # Relative jump (D25/J25), capped, then ABSOLUTE-GATED: a near-zero-base
    # percentage (e.g. 0% -> 1% of county) is huge in relative terms but trivial
    # in absolute terms, so it must not inflate the score. Dampen by a saturating
    # ramp on the added fraction of county; a county that adds a negligible
    # absolute share contributes a near-zero relative-jump signal regardless of
    # how large the percentage looks.
    rel_jump = (
        df["pct_increase"]
        .replace([np.inf, -np.inf], _PCT_CAP)
        .clip(upper=_PCT_CAP)
        .fillna(0.0)
    )
    abs_damp = (df["added_frac_of_county"] / _REL_JUMP_ABS_REF).clip(upper=1.0)
    df["coverage_increase_magnitude"] = rel_jump * abs_damp

    # SECONDARY: rural blanket fill-in (low prior fraction -> high current fraction).
    prior_frac = df.get("prior_cov_frac", pd.Series(np.nan, index=df.index)).fillna(0.0).clip(0, 1)
    cur_frac = df.get("current_cov_frac", pd.Series(np.nan, index=df.index)).fillna(0.0).clip(0, 1)
    df["blanket_fillin"] = ((cur_frac - prior_frac).clip(lower=0.0) * (1.0 - prior_frac)).clip(0, 1)

    # SECONDARY: attribution-based gaming signals.
    df["same_site_growth_share"] = df["added_km2_expanded_site"] / total_added
    df["unattributed_share"] = df["added_km2_unattributed"] / total_added
    df["new_site_share"] = df["added_km2_new_site"] / total_added

    # SECONDARY: boundary snapping (0 if not computed).
    if "boundary_snap_share" not in df:
        df["boundary_snap_share"] = 0.0
    df["boundary_snap_share"] = df["boundary_snap_share"].fillna(0.0)

    df["signal_jump_implausibility"] = df.get("mean_signal_delta", pd.Series(0.0, index=df.index)).clip(lower=0.0).fillna(0.0)
    if "new_towers" not in df:
        df["new_towers"] = 0
    df["new_towers"] = df["new_towers"].fillna(0).astype(int)
    for col in ["new_towers_in_county", "new_towers_cross_border"]:
        if col not in df:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    # Ground-truth corroboration features (populated when ASR / Ookla data is
    # merged in via groundtruth_asr.merge_asr_into_features and
    # groundtruth_measured.compute_measurement_gap). Default to 0 (no signal)
    # when not available so the deterministic core is unchanged without GT data.
    if "asr_has_new_structure" in df:
        # Invert: "no new structure" is the suspicious direction.
        df["asr_no_new_structure"] = (~df["asr_has_new_structure"].astype(bool)).astype(float)
    else:
        df["asr_no_new_structure"] = 0.0
    if "measurement_gap" not in df:
        df["measurement_gap"] = 0.0
    df["measurement_gap"] = df["measurement_gap"].fillna(0.0).clip(lower=0.0)

    return df


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < _EPS:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def score(features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add anomaly score, composite priority score, and review flag."""
    if features.empty:
        return features
    df = features.copy()
    weights: dict[str, float] = cfg.scoring["feature_weights"]

    present = [f for f in _RISK_FEATURES if f in df and f in weights]
    norm = pd.DataFrame({f: _minmax(df[f]) for f in present})

    # Composite weighted risk (rescaled to 0..1 over the positive weight range).
    weighted = sum(norm[f] * weights[f] for f in present)
    df["risk_score"] = _minmax(weighted)

    # Unsupervised anomaly score over the same features (robust to weighting).
    if len(df) >= 8 and present:
        iso = IsolationForest(random_state=0, contamination="auto")
        iso.fit(norm[present].to_numpy())
        anom = -iso.score_samples(norm[present].to_numpy())  # higher = more anomalous
        df["anomaly_score"] = _minmax(pd.Series(anom, index=df.index))
    else:
        df["anomaly_score"] = df["risk_score"]

    # Rescale to [0, 1] so the score is always on a consistent absolute scale
    # regardless of how compressed the input features are (small batches in
    # particular can produce a weighted sum that only spans e.g. 0.00–0.10).
    df["priority_score"] = _minmax(0.7 * df["risk_score"] + 0.3 * df["anomaly_score"])

    flag_pct = float(cfg.scoring["flag_percentile"])
    threshold = df["priority_score"].quantile(flag_pct) if len(df) > 1 else 0.0
    susp = float(cfg.scoring["suspicious_same_site_growth"])
    min_added = float(cfg.scoring.get("min_added_km2_to_flag", 0.0))

    # Implausibility gate thresholds (configurable). High same-site share alone
    # is NORMAL 6-month growth (antenna upgrades, carrier adds, software
    # improvements at existing towers). It only becomes suspicious when paired
    # with at least one implausibility signal: a large absolute area jump, rapid
    # blanket fill-in from a low baseline, or coverage orphaned from all inferred
    # towers (unattributed). This prevents flagging normal single-tower expansion.
    min_county_frac = float(cfg.scoring.get("suspicious_same_site_min_county_frac", 0.15))
    min_blanket = float(cfg.scoring.get("suspicious_same_site_min_blanket", 0.20))
    min_unattributed = float(cfg.scoring.get("suspicious_same_site_min_unattributed", 0.20))

    # Magnitude gate (learned from FCC examples): only counties that actually
    # added meaningful in-county coverage are eligible to flag. This excludes
    # near-empty counties and non-area-increasing signal shifts.
    eligible = df["added_km2"].fillna(0.0) >= min_added

    # "Same-site implausible": high same-site share AND at least one
    # implausibility gate fires. A county where one tower modestly expanded its
    # lobe (normal physics) will not meet any of the implausibility thresholds.
    same_site_implausible = (
        (df["same_site_growth_share"].fillna(0.0) >= susp)
        & (
            (df["added_frac_of_county"].fillna(0.0) >= min_county_frac)
            | (df["blanket_fillin"].fillna(0.0) >= min_blanket)
            | (df["unattributed_share"].fillna(0.0) >= min_unattributed)
        )
    )
    suspicious = (df["priority_score"] >= threshold) | same_site_implausible

    # Growth explained by new towers outside this county is usually legitimate.
    cross_border_build = (
        (df["new_site_share"].fillna(0.0) >= 0.35)
        & (df["new_towers"].fillna(0).astype(int) >= 1)
        & (df.get("new_towers_cross_border", 0).fillna(0).astype(int) >= 1)
    )
    df["flag_for_review"] = eligible & suspicious & ~cross_border_build
    df["flag_reason"] = df.apply(_reason, axis=1, susp=susp)
    return df.sort_values("priority_score", ascending=False).reset_index(drop=True)


def _reason(row: pd.Series, susp: float) -> str:
    reasons = []
    if row.get("added_frac_of_county", 0) >= 0.10:
        reasons.append(f"+{row['added_frac_of_county']:.0%} of county newly covered")
    if row.get("blanket_fillin", 0) >= 0.30:
        reasons.append("rapid blanket fill-in from low baseline")
    if row.get("same_site_growth_share", 0) >= susp:
        reasons.append(
            f"{row['same_site_growth_share']:.0%} of growth claimed from existing sites"
        )
    if row.get("unattributed_share", 0) >= 0.3:
        reasons.append(
            f"{row['unattributed_share']:.0%} of new coverage far from any inferred tower"
        )
    if row.get("boundary_snap_share", 0) >= 0.4:
        reasons.append(f"{row['boundary_snap_share']:.0%} of new coverage hugs county boundary")
    if row.get("coverage_increase_magnitude", 0) >= 1.0:
        pct = row.get("pct_increase", 0)
        if not np.isfinite(pct):
            reasons.append("coverage in previously-uncovered area")
        else:
            reasons.append(f"coverage up {pct:.0%}")
    # Ground-truth corroboration reasons (only shown when GT data is loaded).
    if row.get("asr_no_new_structure", 0) >= 0.5:
        reasons.append("no new FCC-registered structure in county during this window (ASR)")
    if row.get("measurement_gap", 0) >= 0.20:
        reasons.append(
            f"claimed coverage not reflected in field measurements "
            f"(gap: {row['measurement_gap']:.0%})"
        )
    return "; ".join(reasons) if reasons else "ranked by composite anomaly"
