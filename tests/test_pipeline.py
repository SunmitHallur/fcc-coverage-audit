"""End-to-end test of the audit pipeline on synthetic fixtures.

Verifies the core hypothesis the pipeline exists to prove:
  * a same-tower coverage inflation (the gaming pattern) is FLAGGED and ranks at
    the top, and
  * a genuine new-tower buildout is NOT flagged.

Runs fully offline using the fixture backend - no network, no FCC access.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fcc_audit import fixtures, normalize, score  # noqa: E402
from fcc_audit.acquire import get_source  # noqa: E402
from fcc_audit.cli import _resolve_providers, process_provider  # noqa: E402
from fcc_audit.config import load_config  # noqa: E402

ATT = 130077       # AT&T - builds a genuine new tower
TMOBILE = 130403   # T-Mobile - inflates an existing tower (gaming)
CHARLIE = "90003"  # county where both the new build and the inflation land


@pytest.fixture(scope="module")
def scored() -> pd.DataFrame:
    cfg = load_config()
    cfg.raw["source"]["backend"] = "fixture"
    fixtures.make_fixtures(cfg)
    source = get_source(cfg)
    current, prior = fixtures.fixture_vintages()
    counties = normalize.load_counties(cfg)
    areas = normalize.county_areas_km2(counties, cfg.geography["equal_area_crs"])

    all_feats = []
    for provider in _resolve_providers(cfg, source, current):
        feats, _ = process_provider(cfg, source, provider, current, prior, counties, areas)
        if not feats.empty:
            all_feats.append(feats)

    features = pd.concat(all_feats, ignore_index=True)
    return score.score(features, cfg)


def _row(df, provider_id, county):
    m = df[(df["provider_id"] == provider_id) & (df["county_geoid"] == county)]
    assert not m.empty, f"no row for provider {provider_id} county {county}"
    return m.iloc[0]


def test_gaming_case_is_top_flagged(scored):
    top = scored.iloc[0]
    assert top["provider_id"] == TMOBILE
    assert top["county_geoid"] == CHARLIE
    assert bool(top["flag_for_review"]) is True


def test_inflated_site_attributed_to_existing(scored):
    tmo = _row(scored, TMOBILE, CHARLIE)
    # Nearly all growth comes from the SAME (existing) site -> gaming signal.
    assert tmo["same_site_growth_share"] >= 0.9
    assert tmo["new_site_share"] <= 0.1


def test_new_tower_buildout_not_flagged(scored):
    att = _row(scored, ATT, CHARLIE)
    # AT&T's growth in Charlie is a brand-new site -> legitimate, low risk.
    assert att["new_site_share"] >= 0.5
    assert bool(att["flag_for_review"]) is False
    assert att["priority_score"] < _row(scored, TMOBILE, CHARLIE)["priority_score"]


def test_dashboard_json_is_valid(scored):
    import json
    from fcc_audit.report import build_dashboard_payload

    cfg = load_config()
    counties = normalize.load_counties(cfg)
    payload = build_dashboard_payload(scored, pd.DataFrame(), counties)
    # allow_nan=False raises if any inf/nan leaked through.
    json.dumps(payload, allow_nan=False)
