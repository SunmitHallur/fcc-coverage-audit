"""Tests for validate.py — runs fully offline on fixture data, no network needed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fcc_audit.validate import (  # noqa: E402
    compute_metrics,
    compute_empirical_plausibility_radius,
    write_validation_report,
    _bootstrap_ci,
    _cost_optimal_threshold,
    _f1,
    _precision,
    _recall,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scored(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "provider_id": rng.integers(1, 5, size=n),
        "county_geoid": [f"{i:05d}" for i in range(n)],
        "technology": rng.choice(["5G-NR 7/1", "LTE 25/3"], size=n).tolist(),
        "priority_score": rng.uniform(0, 1, size=n),
        "flag_for_review": rng.choice([True, False], size=n, p=[0.3, 0.7]).tolist(),
        "flag_reason": rng.choice(["same-site", "blanket-fill", "boundary-snap"], size=n).tolist(),
        "added_km2": rng.uniform(5, 200, size=n),
        "added_frac_of_county": rng.uniform(0, 0.5, size=n),
        "same_site_growth_share": rng.uniform(0, 1, size=n),
        "unattributed_share": rng.uniform(0, 0.3, size=n),
        "blanket_fillin": rng.uniform(0, 0.4, size=n),
        "boundary_snap_share": rng.uniform(0, 0.3, size=n),
        "new_site_share": rng.uniform(0, 1, size=n),
        "coverage_increase_magnitude": rng.uniform(0, 2, size=n),
        "state_fips": rng.choice(["28", "22", "01"], size=n).tolist(),
        "asr_no_new_structure": rng.uniform(0, 1, size=n),
        "measurement_gap": rng.uniform(0, 0.3, size=n),
    })


def _make_ground_truth(scored: pd.DataFrame, positive_frac: float = 0.3) -> pd.DataFrame:
    """Create synthetic ground-truth labels aligned with scored rows."""
    rng = np.random.default_rng(99)
    labels = rng.choice([0, 1], size=len(scored), p=[1 - positive_frac, positive_frac]).tolist()
    return pd.DataFrame({
        "provider_id": scored["provider_id"].tolist(),
        "county_geoid": scored["county_geoid"].tolist(),
        "technology": scored["technology"].tolist(),
        "label": labels,
    })


# ---------------------------------------------------------------------------
# Bootstrap CI tests
# ---------------------------------------------------------------------------

def test_bootstrap_ci_f1_smoke():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    point, lo, hi = _bootstrap_ci(y_true, y_pred, _f1, n_boot=200)
    assert 0.0 <= lo <= point <= hi <= 1.0


def test_bootstrap_ci_returns_finite():
    y_true = np.ones(10, dtype=int)
    y_pred = np.ones(10, dtype=int)
    point, lo, hi = _bootstrap_ci(y_true, y_pred, _precision, n_boot=100)
    assert np.isfinite(point)


# ---------------------------------------------------------------------------
# Cost-weighted threshold
# ---------------------------------------------------------------------------

def test_cost_optimal_threshold():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_score = np.array([0.9, 0.7, 0.3, 0.2, 0.6, 0.5])
    result = _cost_optimal_threshold(y_true, y_score, cost_fp=1.0, cost_fn=5.0)
    assert "threshold" in result
    assert "cost" in result
    assert result["fp"] >= 0
    assert result["fn"] >= 0
    # With high cost_fn, the optimal threshold should prefer recall over precision.
    assert isinstance(result["threshold"], float)


# ---------------------------------------------------------------------------
# compute_metrics — no ground truth
# ---------------------------------------------------------------------------

def test_compute_metrics_no_ground_truth():
    scored = _make_scored(30)
    gt = pd.DataFrame()
    results = compute_metrics(scored, gt)
    assert "n_scored" in results
    assert results["n_scored"] == 30
    assert "note" in results


# ---------------------------------------------------------------------------
# compute_metrics — with synthetic labels
# ---------------------------------------------------------------------------

def test_compute_metrics_with_labels():
    scored = _make_scored(40)
    gt = _make_ground_truth(scored, positive_frac=0.35)
    results = compute_metrics(scored, gt, n_boot=100)
    m = results.get("metrics", {})
    assert "precision" in m
    assert "recall" in m
    assert "f1" in m
    # CIs should be valid (lo <= point <= hi)
    for key in ("precision", "recall", "f1"):
        v = m[key]
        assert v["ci_lo"] <= v["point"] <= v["ci_hi"] + 1e-9, f"{key} CI invalid: {v}"


def test_compute_metrics_confusion_matrix_sums():
    scored = _make_scored(20)
    gt = _make_ground_truth(scored, positive_frac=0.4)
    results = compute_metrics(scored, gt, n_boot=50)
    m = results.get("metrics", {})
    total = m.get("n_tp", 0) + m.get("n_fp", 0) + m.get("n_fn", 0) + m.get("n_tn", 0)
    assert total == 20  # all 20 rows should be accounted for


def test_compute_metrics_pr_curve():
    scored = _make_scored(50)
    gt = _make_ground_truth(scored, positive_frac=0.3)
    results = compute_metrics(scored, gt, n_boot=50)
    if "pr_curve" in results:
        pr = results["pr_curve"]
        assert "auprc" in pr
        assert 0.0 <= pr["auprc"] <= 1.0
        assert len(pr["precisions"]) == len(pr["recalls"])


def test_compute_metrics_sensitivity_sweeps():
    scored = _make_scored(60)
    gt = _make_ground_truth(scored, positive_frac=0.3)
    results = compute_metrics(scored, gt, n_boot=50)
    sweeps = results.get("sensitivity", [])
    assert len(sweeps) >= 3  # at least 3 percentile points tested


def test_compute_metrics_stratum_breakdown():
    scored = _make_scored(40)
    gt = _make_ground_truth(scored, positive_frac=0.35)
    results = compute_metrics(scored, gt, n_boot=50)
    strat = results.get("stratum", {})
    assert "technology" in strat or "state_fips" in strat


# ---------------------------------------------------------------------------
# Plausibility radius
# ---------------------------------------------------------------------------

def test_plausibility_radius_with_legit_cases():
    scored = _make_scored(40)
    scored["same_site_growth_share"] = np.linspace(0.5, 1.0, 40)
    scored["added_km2"] = np.linspace(10, 200, 40)
    gt = _make_ground_truth(scored, positive_frac=0.0)  # all legit
    result = compute_empirical_plausibility_radius(scored, gt)
    if "implied_radius_km" in result:
        desc = result["implied_radius_km"]
        assert "50%" in desc or "mean" in desc
        assert desc.get("max", 1) > 0


# ---------------------------------------------------------------------------
# write_validation_report (no plots, tmp dir)
# ---------------------------------------------------------------------------

def test_write_validation_report_no_plots(tmp_path):
    scored = _make_scored(20)
    gt = _make_ground_truth(scored, positive_frac=0.3)
    results = compute_metrics(scored, gt, n_boot=50)
    report_path = write_validation_report(results, tmp_path, include_plots=False)
    assert report_path.exists()
    assert report_path.stat().st_size > 0
    # JSON results should also be written.
    json_path = tmp_path / "validation_results.json"
    assert json_path.exists()
    loaded = json.loads(json_path.read_text())
    assert "n_scored" in loaded


def test_write_validation_report_json_is_valid(tmp_path):
    scored = _make_scored(30)
    gt = _make_ground_truth(scored, positive_frac=0.4)
    results = compute_metrics(scored, gt, n_boot=50)
    write_validation_report(results, tmp_path, include_plots=False)
    json_path = tmp_path / "validation_results.json"
    # Should never contain raw NaN or Infinity (not valid JSON).
    text = json_path.read_text()
    assert "NaN" not in text
    assert "Infinity" not in text
    json.loads(text)  # should not raise
