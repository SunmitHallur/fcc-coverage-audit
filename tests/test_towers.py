"""Tower-site inference: binary-source splitting + provider-adaptive core."""
from __future__ import annotations

import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fcc_audit.config import load_config  # noqa: E402
from fcc_audit.towers import infer_sites, _core_hexes  # noqa: E402


@pytest.fixture
def cfg():
    c = load_config()
    c.raw["geography"]["site_h3_resolution"] = 9
    c.raw["towers"]["min_site_hexes"] = 25
    return c


def _km_to(lat, lng, target):
    return float(np.hypot(
        (lat - target[0]) * 110.57,
        (lng - target[1]) * 111.32 * np.cos(np.radians(target[0])),
    ))


def test_binary_two_tower_blob_splits(cfg):
    """Flat-signal (Redshift 0/1) coverage: a merged two-lobe blob must yield
    two sites near the true tower positions, not one at the midpoint."""
    t1, t2 = (38.50, -98.50), (38.50, -98.36)  # ~12 km apart
    cells = set()
    for lat, lng in (t1, t2):
        cells |= set(h3.grid_disk(h3.latlng_to_cell(lat, lng, 9), 18))
    df = pd.DataFrame({"h3": sorted(cells), "signal_dbm": 0.0, "county_geoid": "20001"})

    sites = infer_sites(df, cfg, "T")
    assert len(sites) == 2
    for s in sites.to_dict("records"):
        nearest = min(_km_to(s["lat"], s["lng"], t) for t in (t1, t2))
        assert nearest < 2.0, f"site {s['site_id']} is {nearest:.2f} km from any true tower"


def test_binary_single_lobe_yields_one_site(cfg):
    center = (38.50, -98.50)
    cells = h3.grid_disk(h3.latlng_to_cell(*center, 9), 15)
    df = pd.DataFrame({"h3": list(cells), "signal_dbm": 0.0, "county_geoid": "20001"})
    sites = infer_sites(df, cfg, "T")
    assert len(sites) == 1
    assert _km_to(sites.iloc[0]["lat"], sites.iloc[0]["lng"], center) < 1.0


def test_adaptive_core_for_strong_bands_only_provider():
    """A provider filing only strong bands (-80/-85) must not have its entire
    footprint treated as 'core' — the threshold tightens to its own top band."""
    center = h3.latlng_to_cell(38.5, -98.5, 9)
    rows = [
        {"h3": c, "signal_dbm": -80.0 if h3.grid_distance(center, c) <= 5 else -85.0,
         "county_geoid": "20001"}
        for c in h3.grid_disk(center, 12)
    ]
    df = pd.DataFrame(rows)
    core, flat = _core_hexes(df, -95.0)
    assert not flat
    assert len(core) < len(df)
    assert core["signal_dbm"].min() == -80.0


def test_fine_banded_provider_uses_config_cutoff():
    """A provider filing fine-grained bands keeps the configured -95 cutoff."""
    center = h3.latlng_to_cell(38.5, -98.5, 9)
    rows = [
        {"h3": c, "signal_dbm": -85.0 - 5 * min(h3.grid_distance(center, c) // 3, 8),
         "county_geoid": "20001"}
        for c in h3.grid_disk(center, 12)
    ]
    df = pd.DataFrame(rows)
    core, flat = _core_hexes(df, -95.0)
    assert not flat
    assert len(core) == int((df["signal_dbm"] >= -95.0).sum())


def test_flat_signal_detection():
    df = pd.DataFrame({
        "h3": ["8926e64240fffff", "8926e642407ffff"],
        "signal_dbm": [0.0, 0.0],
        "county_geoid": ["20001", "20001"],
    })
    core, flat = _core_hexes(df, -95.0)
    assert flat
    assert len(core) == 2
