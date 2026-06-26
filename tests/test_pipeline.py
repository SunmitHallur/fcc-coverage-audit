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

ATT = 130077         # AT&T - builds a genuine new tower
TMOBILE = 130403     # T-Mobile - inflates an existing tower (gaming)
USCELLULAR = 130235  # UScellular - single tower, smooth organic expansion (must NOT flag)
CHARLIE = "90003"    # county where both the new build and the inflation land
ALPHA = "90001"      # county where UScellular's single-tower smooth expansion lands


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
    all_sites = []
    all_coverage = []
    for provider in _resolve_providers(cfg, source, current):
        feats, sites, coverage = process_provider(cfg, source, provider, current, prior, counties, areas)
        if not feats.empty:
            all_feats.append(feats)
        if not sites.empty:
            all_sites.append(sites)
        if not coverage.empty:
            all_coverage.append(coverage)

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
    # After the lobe-reach fix, all growth is correctly attributed to the same
    # existing tower (no fringe hexes mis-labelled as unattributed).
    assert tmo["same_site_growth_share"] >= 0.80
    assert tmo["new_site_share"] <= 0.1


def test_smooth_single_tower_expansion_not_flagged(scored):
    """A single existing tower that modestly grows its lobe must NOT be flagged.

    UScellular in Alpha County has exactly one tower before and after, with a
    ~47% area increase (7.0 -> 8.5 km radius). This represents normal organic
    growth (antenna upgrade, carrier add) and should never be flagged. This
    test guards the Phase 0 attribution + flag-logic fix against regression.
    """
    usc = _row(scored, USCELLULAR, ALPHA)
    # Single matched tower -> ~100% same_site after lobe-reach fix.
    assert usc["same_site_growth_share"] >= 0.80, (
        f"Expected same_site_growth_share >= 0.80 but got {usc['same_site_growth_share']:.2f}; "
        "lobe-reach attribution may have regressed"
    )
    # Small area increase relative to county -> must not trigger implausibility gate.
    assert usc.get("added_frac_of_county", 0.0) < 0.10, (
        "UScellular smooth expansion added too large a fraction of county area"
    )
    # Must not be flagged — this is the core regression guard.
    assert bool(usc["flag_for_review"]) is False, (
        "Single-tower smooth organic expansion was incorrectly flagged"
    )


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


def test_plain_language_explanation(scored):
    from fcc_audit.explain import explain_row

    row = scored.iloc[0]
    expl = explain_row(row)
    assert "headline" in expl and len(expl["headline"]) > 10
    assert isinstance(expl["bullets"], list) and len(expl["bullets"]) >= 1
    assert "recommendation" in expl
    assert expl["severity"] in ("Critical", "High", "Moderate", "Low")


def test_web_bundle_build(scored):
    import json
    from fcc_audit.report import build_web_records, build_web_meta

    records = build_web_records(scored)
    assert records  # provider -> service -> geoid
    meta = build_web_meta(scored, {"current": "a", "prior": "b"})
    assert meta["total_records"] == len(scored)
    json.dumps(records, allow_nan=False)
    assert "prior_towers" in scored.columns
    assert "current_towers" in scored.columns


def test_tower_counts_present(scored):
    row = scored.iloc[0]
    assert int(row["current_towers"]) >= int(row["prior_towers"])
