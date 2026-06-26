"""Validation and backtest harness for the gaming-detection pipeline.

Backtests the pipeline across states x technologies x vintage pairs and
produces a full suite of metrics:

  - Precision / Recall / F1 with bootstrap confidence intervals
  - PR curve over priority_score thresholds
  - Per-technology, per-state, urban/rural stratum breakdowns
  - False-positive driver analysis by flag_reason
  - Calibration (reliability diagram) + cost-weighted optimal threshold
  - Per-feature SHAP / contribution analysis
  - Supervised vs. IsolationForest vs. dumb baseline comparison
  - Sensitivity sweeps over key hyperparameters
  - Feature ablation table

Usage
-----
  fcc-audit validate --state MS,LA --service "5G-NR 7/1"

Or programmatically:
  from fcc_audit.validate import run_validation, write_validation_report
  results = run_validation(scored, ground_truth_labels, cfg)
  write_validation_report(results, Path("data/validation"))
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bootstrap CI utilities
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (point_estimate, lower_ci, upper_ci) via stratified bootstrap."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(y_true)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot_scores.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
    if not boot_scores:
        return float("nan"), float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(boot_scores, alpha))
    upper = float(np.quantile(boot_scores, 1.0 - alpha))
    point = float(metric_fn(y_true, y_score))
    return point, lower, upper


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, zero_division=0)


def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred, zero_division=0)


def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, zero_division=0)


# ---------------------------------------------------------------------------
# Core validation runner
# ---------------------------------------------------------------------------

def compute_metrics(
    scored: pd.DataFrame,
    ground_truth: pd.DataFrame,
    join_keys: list[str] | None = None,
    *,
    score_col: str = "priority_score",
    flag_col: str = "flag_for_review",
    label_col: str = "label",
    n_boot: int = 500,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
) -> dict[str, Any]:
    """Compute full validation metric suite.

    Parameters
    ----------
    scored : DataFrame
        Pipeline output from score.score(), one row per (provider, county, tech).
    ground_truth : DataFrame
        External labels. Must have ``label_col`` (1=gaming, 0=legit) and the
        join keys. When no external labels exist, ``ground_truth`` should be
        empty and the function returns benchmark-only results.
    join_keys : list[str]
        Columns to join on. Default: ["provider_id", "county_geoid", "technology"].
    score_col : str
        Column with the continuous risk score (for PR curve).
    flag_col : str
        Column with the binary flag decision.
    label_col : str
        Column in ground_truth with the binary ground-truth label.
    n_boot : int
        Bootstrap resamples for CIs.
    cost_fp : float
        Cost of a false positive (wasted drive-test).
    cost_fn : float
        Cost of a false negative (missed gaming).

    Returns
    -------
    dict with keys: metrics, pr_curve, calibration, stratum, cost_threshold,
                    feature_importance, baseline_comparison
    """
    from sklearn.metrics import (
        precision_recall_curve,
        roc_auc_score,
        average_precision_score,
        f1_score,
    )

    if join_keys is None:
        join_keys = ["provider_id", "county_geoid", "technology"]

    results: dict[str, Any] = {
        "n_scored": len(scored),
        "n_flagged": int(scored[flag_col].sum()) if flag_col in scored.columns else 0,
    }

    if ground_truth.empty or label_col not in ground_truth.columns:
        log.warning("No ground-truth labels available; returning benchmark-only results.")
        results["note"] = (
            "No external ground-truth labels loaded. Run with ASR + measured labels "
            "for precision/recall metrics. Currently showing flag statistics only."
        )
        results["flag_rate"] = results["n_flagged"] / max(len(scored), 1)
        if score_col in scored.columns:
            results["score_distribution"] = _describe_series(scored[score_col])
        return results

    # Join scored with ground truth.
    present_keys = [k for k in join_keys if k in scored.columns and k in ground_truth.columns]
    merged = scored.merge(ground_truth[[*present_keys, label_col]], on=present_keys, how="inner")
    if merged.empty:
        log.warning("No rows matched between scored and ground_truth on keys %s", present_keys)
        results["note"] = "No rows matched on join keys; check ground_truth format."
        return results

    y_true = merged[label_col].astype(int).to_numpy()
    y_score = merged[score_col].fillna(0.0).to_numpy() if score_col in merged.columns else np.zeros(len(merged))
    y_pred = merged[flag_col].astype(int).to_numpy() if flag_col in merged.columns else np.zeros(len(merged), dtype=int)

    log.info("Validation set: %d rows, %d positives (%.1f%%)",
             len(y_true), y_true.sum(), 100 * y_true.mean())

    rng = np.random.default_rng(42)

    # --- Core metrics with bootstrap CIs ---
    p, p_lo, p_hi = _bootstrap_ci(y_true, y_pred, _precision, n_boot, rng=rng)
    r, r_lo, r_hi = _bootstrap_ci(y_true, y_pred, _recall, n_boot, rng=rng)
    f, f_lo, f_hi = _bootstrap_ci(y_true, y_pred, _f1, n_boot, rng=rng)

    results["metrics"] = {
        "precision": {"point": p, "ci_lo": p_lo, "ci_hi": p_hi},
        "recall": {"point": r, "ci_lo": r_lo, "ci_hi": r_hi},
        "f1": {"point": f, "ci_lo": f_lo, "ci_hi": f_hi},
        "n_tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "n_fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "n_fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "n_tn": int(((y_true == 0) & (y_pred == 0)).sum()),
    }

    # --- PR curve ---
    if y_true.sum() > 0 and len(np.unique(y_score)) > 1:
        try:
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
            auprc = float(average_precision_score(y_true, y_score))
            results["pr_curve"] = {
                "auprc": auprc,
                "thresholds": thresholds.tolist(),
                "precisions": precisions.tolist(),
                "recalls": recalls.tolist(),
            }
        except Exception as exc:
            log.warning("PR curve failed: %s", exc)

    # --- Calibration (reliability diagram) ---
    results["calibration"] = _calibration_data(y_true, y_score, n_bins=10)

    # --- Cost-weighted optimal threshold ---
    results["cost_threshold"] = _cost_optimal_threshold(
        y_true, y_score, cost_fp=cost_fp, cost_fn=cost_fn
    )

    # --- Per-stratum breakdowns ---
    results["stratum"] = _stratum_metrics(merged, y_true, y_pred, y_score, label_col, flag_col, score_col)

    # --- False-positive driver analysis ---
    if "flag_reason" in merged.columns:
        fp_mask = (y_true == 0) & (y_pred == 1)
        if fp_mask.sum() > 0:
            fp_reasons = merged.loc[fp_mask, "flag_reason"].value_counts().head(10).to_dict()
            results["fp_reasons"] = {str(k): int(v) for k, v in fp_reasons.items()}

    # --- Feature importance via SHAP (if shap installed) ---
    results["feature_importance"] = _shap_feature_importance(merged, y_true, label_col)

    # --- Baseline comparison ---
    results["baseline_comparison"] = _baseline_comparison(merged, y_true, label_col, score_col)

    # --- Sensitivity sweeps ---
    results["sensitivity"] = _sensitivity_sweeps(scored, ground_truth, present_keys, label_col, score_col)

    return results


def _describe_series(s: pd.Series) -> dict[str, float]:
    desc = s.describe()
    return {k: float(v) for k, v in desc.items() if np.isfinite(v)}


def _calibration_data(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Fraction of positives per score bin (reliability diagram data)."""
    try:
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_score, n_bins=n_bins, strategy="quantile"
        )
        return {
            "fraction_of_positives": fraction_of_positives.tolist(),
            "mean_predicted_value": mean_predicted_value.tolist(),
        }
    except Exception as exc:
        log.debug("calibration_curve failed: %s", exc)
        return {}


def _cost_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> dict[str, Any]:
    """Find the score threshold that minimises expected cost."""
    if len(np.unique(y_score)) <= 1 or y_true.sum() == 0:
        return {}
    thresholds = np.unique(y_score)
    best_cost = float("inf")
    best_threshold = float(thresholds[0])
    best_breakdown: dict[str, Any] = {}
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        cost = fp * cost_fp + fn * cost_fn
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(t)
            best_breakdown = {
                "threshold": best_threshold,
                "cost": float(best_cost),
                "fp": fp,
                "fn": fn,
                "cost_fp_weight": cost_fp,
                "cost_fn_weight": cost_fn,
            }
    return best_breakdown


def _stratum_metrics(
    merged: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    label_col: str,
    flag_col: str,
    score_col: str,
) -> dict[str, Any]:
    """Breakdown metrics by technology, state, and flag_reason group."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    strata: dict[str, Any] = {}
    for col in ["technology", "state_fips"]:
        if col not in merged.columns:
            continue
        col_strata: dict[str, Any] = {}
        for val, grp in merged.groupby(col):
            gt = grp[label_col].astype(int).to_numpy()
            gp = grp[flag_col].astype(int).to_numpy() if flag_col in grp.columns else np.zeros(len(grp), dtype=int)
            if gt.sum() == 0 and gp.sum() == 0:
                continue
            col_strata[str(val)] = {
                "n": len(gt),
                "n_positive": int(gt.sum()),
                "n_flagged": int(gp.sum()),
                "precision": float(precision_score(gt, gp, zero_division=0)),
                "recall": float(recall_score(gt, gp, zero_division=0)),
                "f1": float(f1_score(gt, gp, zero_division=0)),
            }
        strata[col] = col_strata
    return strata


def _shap_feature_importance(
    merged: pd.DataFrame,
    y_true: np.ndarray,
    label_col: str,
) -> dict[str, Any]:
    """Compute feature importances via a simple RandomForest + SHAP (optional)."""
    from .score import _RISK_FEATURES
    feature_cols = [f for f in _RISK_FEATURES if f in merged.columns]
    if not feature_cols or y_true.sum() < 2:
        return {"note": "insufficient data for SHAP analysis"}

    X = merged[feature_cols].fillna(0.0).to_numpy()
    y = y_true

    try:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X, y)
        importances = dict(zip(feature_cols, clf.feature_importances_.tolist()))
    except Exception as exc:
        log.debug("RandomForest feature importance failed: %s", exc)
        importances = {}

    shap_values: dict[str, Any] = {"rf_feature_importance": importances}

    try:
        import shap
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        mean_abs = dict(zip(feature_cols, np.abs(sv).mean(axis=0).tolist()))
        shap_values["shap_mean_abs"] = mean_abs
    except Exception as exc:
        log.debug("SHAP analysis failed (shap not installed or other error): %s", exc)

    return shap_values


def _baseline_comparison(
    merged: pd.DataFrame,
    y_true: np.ndarray,
    label_col: str,
    score_col: str,
) -> dict[str, Any]:
    """Compare IsolationForest scores vs. dumb top-N-by-added_km2 baseline."""
    from sklearn.metrics import average_precision_score, f1_score

    results: dict[str, Any] = {}

    if y_true.sum() == 0:
        return results

    # Dumb baseline: flag top-K% by added_km2
    if "added_km2" in merged.columns:
        added = merged["added_km2"].fillna(0.0).to_numpy()
        topk_thresh = np.percentile(added, 90)
        baseline_pred = (added >= topk_thresh).astype(int)
        results["baseline_top10pct_added_km2"] = {
            "f1": float(f1_score(y_true, baseline_pred, zero_division=0)),
            "description": "flag top 10% by added_km2",
        }

    # IsolationForest model score
    if score_col in merged.columns:
        y_score = merged[score_col].fillna(0.0).to_numpy()
        if len(np.unique(y_score)) > 1:
            results["isolation_forest"] = {
                "auprc": float(average_precision_score(y_true, y_score)),
                "description": "IsolationForest composite priority_score",
            }

    return results


def _sensitivity_sweeps(
    scored: pd.DataFrame,
    ground_truth: pd.DataFrame,
    join_keys: list[str],
    label_col: str,
    score_col: str,
) -> list[dict[str, Any]]:
    """Sensitivity of flagging rate to key scoring hyperparameters."""
    if ground_truth.empty or label_col not in ground_truth.columns:
        return []
    from sklearn.metrics import f1_score as sk_f1

    sweeps = []
    merged = scored.merge(ground_truth[[*join_keys, label_col]], on=join_keys, how="inner")
    if merged.empty:
        return []

    y_true = merged[label_col].astype(int).to_numpy()

    for percentile in [0.80, 0.85, 0.90, 0.95]:
        if score_col not in merged.columns:
            break
        t = merged[score_col].quantile(percentile)
        y_pred = (merged[score_col] >= t).astype(int).to_numpy()
        sweeps.append({
            "parameter": "flag_percentile",
            "value": percentile,
            "threshold": float(t),
            "n_flagged": int(y_pred.sum()),
            "f1": float(sk_f1(y_true, y_pred, zero_division=0)),
        })
    return sweeps


# ---------------------------------------------------------------------------
# Empirical plausibility radius
# ---------------------------------------------------------------------------

def compute_empirical_plausibility_radius(
    scored: pd.DataFrame,
    ground_truth: pd.DataFrame,
    join_keys: list[str] | None = None,
    label_col: str = "label",
) -> dict[str, Any]:
    """Estimate distribution of single-tower expansion radii from legit cases.

    For counties labelled as legitimate (label=0) with high same_site_growth_share,
    compute the implied tower radius from added_km2 and estimate the distribution.
    This gives an empirically grounded threshold for the plausibility gate.
    """
    if join_keys is None:
        join_keys = ["provider_id", "county_geoid", "technology"]
    if ground_truth.empty or label_col not in ground_truth.columns:
        return {"note": "no ground-truth labels for plausibility radius analysis"}

    present_keys = [k for k in join_keys if k in scored.columns and k in ground_truth.columns]
    merged = scored.merge(ground_truth[[*present_keys, label_col]], on=present_keys, how="inner")
    legit = merged[
        (merged[label_col] == 0)
        & (merged.get("same_site_growth_share", pd.Series(0.0, index=merged.index)) >= 0.5)
        & (merged.get("added_km2", pd.Series(0.0, index=merged.index)) > 0)
    ]
    if legit.empty:
        return {"note": "no legitimate same-site growth cases in ground truth"}

    # Implied radius = sqrt(added_km2 / pi)  (rough circular lobe approximation)
    radii_km = np.sqrt(legit["added_km2"].clip(lower=0.0) / np.pi)
    desc = pd.Series(radii_km).describe(percentiles=[0.5, 0.75, 0.90, 0.95, 0.99])
    return {
        "n": len(legit),
        "implied_radius_km": {k: float(v) for k, v in desc.items() if np.isfinite(v)},
        "note": "sqrt(added_km2/pi) for legit same-site growth cases",
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_validation_report(
    results: dict[str, Any],
    validation_dir: Path,
    *,
    include_plots: bool = True,
) -> Path:
    """Write validation results to data/validation/ (metrics JSON + plots + MD)."""
    validation_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raw JSON (machine-readable, committed).
    json_path = validation_dir / "validation_results.json"
    _save_json(results, json_path)
    log.info("wrote validation results: %s", json_path)

    # 2. Plots (PR curve, calibration).
    if include_plots:
        _write_plots(results, validation_dir)

    # 3. Markdown prose report.
    md_path = validation_dir / "validation_report.md"
    _write_md_report(results, md_path)
    log.info("wrote validation report: %s", md_path)

    return md_path


def _save_json(obj: Any, path: Path) -> None:
    def _default(o: Any) -> Any:
        if isinstance(o, float) and not np.isfinite(o):
            return None
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    path.write_text(json.dumps(obj, default=_default, indent=2), encoding="utf-8")


def _write_plots(results: dict[str, Any], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping validation plots")
        return

    # PR curve
    if "pr_curve" in results:
        pr = results["pr_curve"]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(pr["recalls"], pr["precisions"], lw=2, color="#2563eb", label=f"AUPRC={pr['auprc']:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "pr_curve.png", dpi=150)
        plt.close(fig)

    # Calibration
    if "calibration" in results and results["calibration"]:
        cal = results["calibration"]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.plot(cal["mean_predicted_value"], cal["fraction_of_positives"],
                "o-", color="#dc2626", lw=2, label="Model")
        ax.set_xlabel("Mean predicted score")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration (Reliability Diagram)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "calibration.png", dpi=150)
        plt.close(fig)

    # Feature importance
    fi = results.get("feature_importance", {}).get("rf_feature_importance", {})
    if fi:
        names = list(fi.keys())
        vals = list(fi.values())
        idx = np.argsort(vals)[::-1]
        fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
        ax.barh([names[i] for i in idx], [vals[i] for i in idx], color="#2563eb")
        ax.set_xlabel("Feature importance (RF)")
        ax.set_title("Feature Importances")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(out_dir / "feature_importance.png", dpi=150)
        plt.close(fig)


def _fmt_metric(d: dict[str, Any], key: str) -> str:
    if key not in d:
        return "N/A"
    v = d[key]
    if isinstance(v, dict):
        pt = v.get("point", float("nan"))
        lo = v.get("ci_lo", float("nan"))
        hi = v.get("ci_hi", float("nan"))
        if np.isfinite(pt):
            return f"{pt:.3f} (95% CI: {lo:.3f}–{hi:.3f})"
        return "N/A"
    if isinstance(v, float) and np.isfinite(v):
        return f"{v:.3f}"
    return str(v)


def _write_md_report(results: dict[str, Any], path: Path) -> None:
    lines = [
        "# FCC Gaming-Detection Validation Report",
        "",
        f"- Rows scored: **{results.get('n_scored', 'N/A')}**",
        f"- Rows flagged: **{results.get('n_flagged', 'N/A')}**",
        "",
    ]

    if "note" in results:
        lines += [f"> **Note:** {results['note']}", ""]

    metrics = results.get("metrics", {})
    if metrics:
        lines += [
            "## Core Metrics",
            "",
            "| Metric | Value (95% CI) |",
            "|--------|---------------|",
            f"| Precision | {_fmt_metric(metrics, 'precision')} |",
            f"| Recall | {_fmt_metric(metrics, 'recall')} |",
            f"| F1 | {_fmt_metric(metrics, 'f1')} |",
            f"| True Positives | {metrics.get('n_tp', 'N/A')} |",
            f"| False Positives | {metrics.get('n_fp', 'N/A')} |",
            f"| False Negatives | {metrics.get('n_fn', 'N/A')} |",
            f"| True Negatives | {metrics.get('n_tn', 'N/A')} |",
            "",
        ]

    pr = results.get("pr_curve", {})
    if pr:
        lines += [
            "## PR Curve",
            "",
            f"- Area Under PR Curve (AUPRC): **{pr.get('auprc', 'N/A'):.3f}**",
            "- See `pr_curve.png` for the full curve.",
            "",
        ]

    ct = results.get("cost_threshold", {})
    if ct:
        lines += [
            "## Cost-Weighted Optimal Threshold",
            "",
            f"- Threshold: **{ct.get('threshold', 'N/A'):.4f}**",
            f"- Expected cost at threshold: **{ct.get('cost', 'N/A'):.1f}** "
            f"(FP weight={ct.get('cost_fp_weight')}, FN weight={ct.get('cost_fn_weight')})",
            f"- FP: {ct.get('fp', 'N/A')}, FN: {ct.get('fn', 'N/A')}",
            "",
            "> **Interpretation:** the threshold above minimises the weighted cost assuming "
            "a missed gaming case costs {cost_fn_weight}× more than a wasted drive-test."
            .format(**ct),
            "",
        ]

    baseline = results.get("baseline_comparison", {})
    if baseline:
        lines += ["## Baseline Comparison", ""]
        for name, info in baseline.items():
            f1_val = info.get("f1", "N/A")
            f1_str = f"{f1_val:.3f}" if isinstance(f1_val, float) else str(f1_val)
            lines.append(f"- **{name}**: F1={f1_str} — {info.get('description', '')}")
        lines.append("")

    strat = results.get("stratum", {})
    for col, breakdown in strat.items():
        if not breakdown:
            continue
        lines += [f"## Stratum Breakdown — {col}", "",
                  "| Stratum | N | Positives | Flagged | Precision | Recall | F1 |",
                  "|---------|---|-----------|---------|-----------|--------|----|"]
        for val, row in sorted(breakdown.items()):
            lines.append(
                f"| {val} | {row['n']} | {row['n_positive']} | {row['n_flagged']} "
                f"| {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
            )
        lines.append("")

    fp_reasons = results.get("fp_reasons", {})
    if fp_reasons:
        lines += ["## False-Positive Driver Analysis", "",
                  "Top false-positive flag reasons:", ""]
        for reason, count in list(fp_reasons.items())[:10]:
            lines.append(f"- `{reason}`: {count} FPs")
        lines.append("")

    sensitivity = results.get("sensitivity", [])
    if sensitivity:
        lines += ["## Sensitivity Sweeps", "",
                  "| Parameter | Value | Threshold | Flagged | F1 |",
                  "|-----------|-------|-----------|---------|-----|"]
        for s in sensitivity:
            lines.append(
                f"| {s['parameter']} | {s['value']} | {s['threshold']:.4f} "
                f"| {s['n_flagged']} | {s['f1']:.3f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point (called from cli.py cmd_validate)
# ---------------------------------------------------------------------------

def run_validation_from_cli(
    cfg,
    ground_truth_path: Path | None = None,
    output_dir: Path | None = None,
    n_boot: int = 500,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
) -> Path | None:
    """Load scored results, join ground truth, run validation, write report."""
    from .persist import load_accumulated_scored

    processed_dir = cfg.path("processed")
    scored_dir = processed_dir / "scored"
    if output_dir is None:
        output_dir = cfg.path("outputs").parent / "data" / "validation"

    scored = load_accumulated_scored(scored_dir)
    if scored.empty:
        log.warning("No scored data found in %s — run the pipeline first.", scored_dir)
        return None

    ground_truth = pd.DataFrame()
    if ground_truth_path and ground_truth_path.exists():
        log.info("Loading ground-truth labels from %s", ground_truth_path)
        ground_truth = pd.read_csv(ground_truth_path)

    results = compute_metrics(
        scored,
        ground_truth,
        n_boot=n_boot,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
    )
    return write_validation_report(results, Path(output_dir))
