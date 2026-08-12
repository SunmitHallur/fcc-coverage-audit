from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fcc_audit.config import load_config
from fcc_audit.score import _FEATURE_OPERATING_RANGE, build_features, score


def _feature_row(case: str = "base") -> dict:
    return {
        "case": case,
        "provider_id": 1,
        "provider_name": "Provider",
        "technology": "5G",
        "county_geoid": "01001",
        "county_name": "County",
        "added_km2": 50.0,
        "added_frac_of_county": 0.2,
        "coverage_increase_magnitude": 0.2,
        "blanket_fillin": 0.2,
        "same_site_growth_share": 0.2,
        "unattributed_share": 0.2,
        "boundary_snap_share": 0.2,
        "new_site_share": 0.2,
        "asr_no_new_structure": 0.2,
        "measurement_gap": 0.2,
        "new_towers": 0,
        "new_towers_cross_border": 0,
    }


def _score_for(rows: list[dict], case: str) -> float:
    scored = score(pd.DataFrame(rows), load_config())
    return float(scored.loc[scored["case"] == case, "priority_score"].iloc[0])


def test_priority_score_is_strictly_monotone_in_each_configured_feature():
    cfg = load_config()
    ranges = {
        **_FEATURE_OPERATING_RANGE,
        **(cfg.scoring.get("feature_operating_ranges") or {}),
    }
    for feature, weight in cfg.scoring["feature_weights"].items():
        op = float(ranges.get(feature, 1.0))
        low = _feature_row("low")
        high = _feature_row("high")
        # Stay inside the calibrated operating range so neither value saturates.
        low[feature] = 0.2 * op
        high[feature] = 0.9 * op

        low_score = _score_for([low, high], "low")
        high_score = _score_for([low, high], "high")
        if weight > 0:
            assert high_score > low_score, feature
        elif weight < 0:
            assert high_score < low_score, feature


def test_no_single_feature_can_move_score_more_than_quarter_point():
    cfg = load_config()
    ranges = {
        **_FEATURE_OPERATING_RANGE,
        **(cfg.scoring.get("feature_operating_ranges") or {}),
    }
    for feature in cfg.scoring["feature_weights"]:
        op = float(ranges.get(feature, 1.0))
        low = _feature_row("low")
        high = _feature_row("high")
        low[feature] = 0.0
        high[feature] = op  # full calibrated strength

        swing = abs(_score_for([low, high], "high") - _score_for([low, high], "low"))
        assert swing <= 0.25 + 1e-12, (feature, swing)


def test_priority_score_is_invariant_to_other_counties_in_cohort():
    target = _feature_row("target")
    alone = _score_for([target], "target")
    cohort = _score_for(
        [target, {**_feature_row("low"), "added_frac_of_county": 0.0},
         {**_feature_row("high"), "added_frac_of_county": 1.0}],
        "target",
    )
    assert cohort == pytest.approx(alone)


def test_weight_above_quarter_is_rejected():
    cfg = load_config()
    cfg.scoring["feature_weights"]["added_frac_of_county"] = 0.2501
    with pytest.raises(ValueError, match="0.25 maximum influence"):
        score(pd.DataFrame([_feature_row()]), cfg)


def test_relative_change_transform_is_monotone_and_bounded():
    county_change = pd.DataFrame(
        {
            "county_geoid": ["01001", "01003", "01005"],
            "pct_increase": [0.5, 5.0, float("inf")],
            "added_frac_of_county": [0.05, 0.05, 0.05],
            "prior_cov_frac": [0.1, 0.1, 0.0],
            "current_cov_frac": [0.15, 0.6, 0.05],
        }
    )
    attribution = pd.DataFrame(
        {
            "county_geoid": county_change["county_geoid"],
            "added_km2_new_site": 0.0,
            "added_km2_expanded_site": 1.0,
            "added_km2_unattributed": 0.0,
        }
    )

    built = build_features(county_change, attribution)
    relative = built["coverage_increase_magnitude"].tolist()
    assert 0.0 < relative[0] < relative[1] < relative[2] <= 1.0
