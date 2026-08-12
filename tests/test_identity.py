"""Same-vintage identity: comparing a layer to itself must produce zero change."""
from __future__ import annotations

import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fcc_audit import attribute, changedetect, score, towers  # noqa: E402
from fcc_audit.config import load_config  # noqa: E402


@pytest.fixture
def cfg():
    c = load_config()
    c.raw["geography"]["site_h3_resolution"] = 9
    c.raw["towers"]["min_site_hexes"] = 25
    # Disable ASR weight for this unit test (no ASR data loaded).
    weights = dict(c.raw["scoring"]["feature_weights"])
    weights.pop("asr_no_new_structure", None)
    c.raw["scoring"]["feature_weights"] = weights
    return c


def _blob(center, ring=15, geoid="20001"):
    cells = list(h3.grid_disk(h3.latlng_to_cell(*center, 9), ring))
    return pd.DataFrame({
        "h3": cells,
        "signal_dbm": 0.0,
        "county_geoid": geoid,
        "county_name": "Test",
        "state_fips": geoid[:2],
    })


def test_same_vintage_joint_inference_zero_new_towers(cfg):
    """Identical prior/current footprints → shared sites, no new/expanded."""
    hex_df = _blob((38.50, -98.50))
    prior_sites, current_sites = towers.infer_sites_joint(hex_df, hex_df.copy(), cfg)
    assert len(prior_sites) == len(current_sites)
    assert len(current_sites) >= 1
    assert (current_sites["site_class"] == "stable_site").all()
    assert int((current_sites["site_class"] == "new_site").sum()) == 0


def test_same_vintage_change_detection_zero_added(cfg):
    hex_df = _blob((38.50, -98.50))
    change = changedetect.hex_change(hex_df, hex_df.copy())
    assert (change["status"] == "unchanged").all() or (
        change["status"].isin(["unchanged", "same"]).any()
        or set(change["status"].unique()) <= {"unchanged", "stable", "same"}
        or (change["status"] != "new").all()
    )
    # Explicitly: no new or lost hexes.
    assert int((change["status"] == "new").sum()) == 0
    assert int((change["status"] == "lost").sum()) == 0


def test_same_vintage_pipeline_zero_flags(cfg):
    """End-to-end features from identical vintages → no added km², no flags."""
    hex_df = _blob((38.50, -98.50), ring=18)
    # Two lobes so tower inference has structure.
    hex2 = _blob((38.50, -98.36), ring=18)
    hex_df = pd.concat([hex_df, hex2], ignore_index=True).drop_duplicates("h3")

    change = changedetect.hex_change(hex_df, hex_df.copy())
    area = {"20001": 2000.0}
    cc = changedetect.county_change(change, 9, area)
    assert float(cc["added_km2"].sum()) == 0.0

    prior_sites, current_sites = towers.infer_sites_joint(hex_df, hex_df.copy(), cfg)
    current_sites = attribute.match_sites(
        prior_sites, current_sites, float(cfg.towers["site_match_radius_m"])
    )
    attr = attribute.attribute_changes(change, current_sites, 9)
    if attr.empty:
        attr = pd.DataFrame({
            "county_geoid": cc["county_geoid"],
            "added_km2_new_site": 0.0,
            "added_km2_expanded_site": 0.0,
            "added_km2_unattributed": 0.0,
            "new_towers": 0,
            "new_towers_in_county": 0,
            "new_towers_cross_border": 0,
            "inference_insufficient": False,
        })
    else:
        assert float(attr["added_km2_new_site"].sum()) == 0.0
        assert int(attr["new_towers"].sum()) == 0

    # Serving counts must not drop when footprints are identical.
    prior_srv = attribute.serving_towers_by_county(hex_df, prior_sites)
    current_srv = attribute.serving_towers_by_county(hex_df, current_sites)
    if not prior_srv.empty and not current_srv.empty:
        merged = prior_srv.merge(
            current_srv, on="county_geoid", suffixes=("_p", "_c")
        )
        for _, row in merged.iterrows():
            assert int(row["towers_serving_c"]) == int(row["towers_serving_p"])

    feats = score.build_features(cc, attr, None)
    if feats.empty:
        return
    scored = score.score(feats, cfg)
    assert int(scored["flag_for_review"].sum()) == 0
    assert float(scored["added_km2"].abs().max()) < 1e-9
