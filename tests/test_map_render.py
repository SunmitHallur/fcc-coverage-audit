"""Tests for static compare-map PNG rendering."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fcc_audit.map_render import render_county_compare_maps, render_coverage_map  # noqa: E402


@pytest.fixture
def county_feature():
    return {
        "type": "Feature",
        "properties": {"geoid": "90003", "name": "Charlie County"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-99.0, 39.0], [-98.5, 39.0], [-98.5, 39.5], [-99.0, 39.5], [-99.0, 39.0],
            ]],
        },
    }


def test_render_coverage_map_writes_png(tmp_path, county_feature):
    out = tmp_path / "prior.png"
    path = render_coverage_map(
        hexes=[["8826e64247fffff", -92.0], ["8826e64335fffff", -85.0]],
        sites=[{"lat": 39.25, "lng": -98.75, "site_class": "prior_site", "in_county": True}],
        county_feature=county_feature,
        title="Prior",
        output_path=out,
    )
    assert path == out
    assert out.exists()
    assert out.stat().st_size > 5000


def test_render_county_compare_maps(tmp_path, county_feature):
    detail = {
        "county_boundary": county_feature,
        "prior_vintage": "2025-06-30",
        "current_vintage": "2025-12-31",
        "prior_hexes": [["8826e64247fffff", -95.0]],
        "current_hexes": [["8826e64335fffff", -82.0]],
        "sites_prior": [],
        "sites_current": [{"lat": 39.25, "lng": -98.75, "site_class": "new_site", "in_county": True}],
    }
    refs = render_county_compare_maps(detail, tmp_path)
    assert refs["prior_map"] == "prior.png"
    assert refs["current_map"] == "current.png"
    assert (tmp_path / "prior.png").exists()
    assert (tmp_path / "current.png").exists()
