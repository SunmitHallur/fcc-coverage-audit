from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from fcc_audit.acquire import RedshiftSource
from fcc_audit.cli import _context_states, _run_key, _save_batch_results, _target_rows
from fcc_audit.config import load_config


def test_redshift_empty_query_is_cached_as_valid_layer(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.project_root = tmp_path
    cfg.raw["source"]["backend"] = "redshift"
    cfg.set_states(["20"])
    source = RedshiftSource(cfg)
    monkeypatch.setattr(source, "_query_df", lambda *_args, **_kwargs: pd.DataFrame(columns=["h3index"]))

    layer = source.fetch(130077, "5G-NR (7/1 Mbps)", "277")

    cached = pd.read_parquet(layer.local_path)
    assert cached.empty
    assert list(cached.columns) == ["h3", "signal_dbm"]
    assert layer.is_hex is True


def test_target_rows_remove_context_state_outputs():
    rows = pd.DataFrame(
        [
            {"county_geoid": "20001", "value": 1},
            {"county_geoid": "29001", "value": 2},
        ]
    )
    assert _target_rows(rows, ["20"])["county_geoid"].tolist() == ["20001"]


def test_context_states_include_nearby_but_not_distant_states():
    counties = gpd.GeoDataFrame(
        {
            "state_fips": ["20", "29", "06"],
            "county_geoid": ["20001", "29001", "06001"],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(20, 0, 21, 1)],
        crs="EPSG:4326",
    )
    assert _context_states(counties, ["20"]) == ["20", "29"]


def test_batch_persistence_isolated_and_replaces_empty_state_partition(tmp_path):
    cfg = load_config()
    cfg.project_root = tmp_path
    cfg.raw["source"]["backend"] = "redshift"
    meta = {
        "current": "277",
        "prior": "279",
        "analysis_units": [{"provider_id": 1, "technology": "5G"}],
    }
    scored = pd.DataFrame(
        [{"provider_id": 1, "technology": "5G", "county_geoid": "20001"}]
    )
    coverage = pd.DataFrame(
        [{
            "provider_id": 1,
            "technology": "5G",
            "county_geoid": "20001",
            "vintage": "current",
            "h3": "abc",
        }]
    )

    _save_batch_results(cfg, scored, pd.DataFrame(), meta, coverage, states=["20"])
    run_dir = tmp_path / "data" / "processed" / _run_key(cfg, "277", "279")
    assert (run_dir / "coverage" / "coverage_20.parquet").exists()

    _save_batch_results(cfg, scored, pd.DataFrame(), meta, pd.DataFrame(), states=["20"])
    assert not (run_dir / "coverage" / "coverage_20.parquet").exists()
    manifest = json.loads((run_dir / "manifests" / "batch_20.json").read_text())
    assert manifest["states"] == ["20"]
    assert manifest["analysis_units"] == meta["analysis_units"]
