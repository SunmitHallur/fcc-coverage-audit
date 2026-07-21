from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fcc_audit import cli
from fcc_audit.acquire import CoverageFile, RedshiftSource
from fcc_audit.cli import (
    _context_states, _run_key, _save_batch_results, _target_rows, process_provider,
)
from fcc_audit.config import Provider, load_config

NATIONAL_STATE_FIPS = {
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36",
    "37", "38", "39", "40", "41", "42", "44", "45", "46", "47", "48",
    "49", "50", "51", "53", "54", "55", "56",
}


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


def test_redshift_query_error_propagates_without_cache(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.project_root = tmp_path
    cfg.raw["source"]["backend"] = "redshift"
    source = RedshiftSource(cfg)

    def denied(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(source, "_query_df", denied)
    with pytest.raises(PermissionError, match="denied"):
        source.fetch(130077, "5G-NR (7/1 Mbps)", "277")
    assert not list((tmp_path / "data" / "raw").rglob("*.parquet"))
    assert not list((tmp_path / "data" / "raw").rglob("*.part"))


def test_redshift_provider_discovery_error_propagates(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.project_root = tmp_path
    cfg.raw["source"]["backend"] = "redshift"
    source = RedshiftSource(cfg)
    monkeypatch.setattr(
        source, "_query_df",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )

    with pytest.raises(PermissionError, match="denied"):
        source.list_providers("277")


def test_successful_all_empty_units_are_completed(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.project_root = tmp_path
    provider = Provider(id=1, name="Provider")

    class EmptySource:
        def fetch(self, provider_id, technology, vintage):
            return CoverageFile(provider_id, technology, vintage, tmp_path / "empty.parquet")

    monkeypatch.setattr(
        cli, "_analyze_unit",
        lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
    )
    feats, sites, coverage, completed = process_provider(
        cfg, EmptySource(), provider, "277", "279", gpd.GeoDataFrame(), {},
    )

    assert feats.empty and sites.empty and coverage.empty
    assert completed == {
        (provider.id, str(service["label"])) for service in cfg.services
    }


def test_successful_all_empty_batch_writes_complete_manifest(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.project_root = tmp_path
    cfg.raw["source"]["backend"] = "redshift"

    class EmptySource:
        def resolve_vintages(self, current, prior):
            return str(current), str(prior)

    monkeypatch.setattr(cli, "get_source", lambda _cfg: EmptySource())
    monkeypatch.setattr(cli.normalize, "load_counties", lambda _cfg: gpd.GeoDataFrame())
    monkeypatch.setattr(cli.normalize, "county_areas_km2", lambda *_args: {})

    def completed_empty(_cfg, _source, provider, *_args, **_kwargs):
        completed = {
            (int(provider.id), str(service["label"])) for service in cfg.services
        }
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), completed

    monkeypatch.setattr(cli, "process_provider", completed_empty)
    args = SimpleNamespace(
        states=None, current="277", prior="279", cleanup_raw=False,
        workers=1, build_web=False, verbose=False, config=None,
    )

    assert cli.cmd_run(cfg, args) == 0
    run_dir = tmp_path / "data" / "processed" / _run_key(cfg, "277", "279")
    manifest = json.loads((run_dir / "manifests" / "batch_all.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["missing_analysis_units"] == []
    as_pairs = lambda units: {
        (unit["provider_id"], unit["technology"]) for unit in units
    }
    assert as_pairs(manifest["completed_analysis_units"]) == as_pairs(manifest["analysis_units"])


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
        "completed_analysis_units": [{"provider_id": 1, "technology": "5G"}],
        "missing_analysis_units": [],
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


def test_windows_launcher_runs_exact_national_batches_and_final_build():
    root = Path(__file__).resolve().parents[1]
    powershell = (root / "run_overnight.ps1").read_text(encoding="utf-8")
    batch_block = powershell.split("$batches = @(", 1)[1].split(")", 1)[0]
    batch_states = re.findall(r"\b\d{2}\b", batch_block)

    assert set(batch_states) == NATIONAL_STATE_FIPS
    assert len(batch_states) == len(set(batch_states))
    assert "--cleanup-raw" in powershell
    assert "-m fcc_audit.cli build-web" in powershell

    launcher = (root / "run.bat").read_text(encoding="utf-8")
    assert "run_overnight.ps1" in launcher
