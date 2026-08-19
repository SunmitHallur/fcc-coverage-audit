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
    _context_states, _coverage_geoids_to_persist, _filter_coverage_for_persist,
    _run_key, _save_batch_results, _target_rows, process_provider,
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
    assert source.is_mrgd
    monkeypatch.setattr(
        source, "_query_df",
        lambda *_args, **_kwargs: pd.DataFrame(columns=["h3index", "minsignal"]),
    )

    layer = source.fetch(130077, "5G-NR (7/1 Mbps)", "292")

    cached = pd.read_parquet(layer.local_path)
    assert cached.empty
    assert list(cached.columns) == ["h3", "signal_dbm"]
    assert layer.is_hex is True


def test_write_hex_cache_round_trips_signal_dbm(tmp_path):
    cfg = load_config()
    cfg.project_root = tmp_path
    source = RedshiftSource(cfg)
    dest = tmp_path / "cache.parquet"
    # Duplicate hex keeps max minsignal.
    source._write_hex_cache(
        dest,
        ["8928308280fffff", "8928308281fffff", "8928308280fffff"],
        [-110.0, -90.0, -100.0],
    )
    cached = pd.read_parquet(dest)
    assert set(cached.columns) == {"h3", "signal_dbm"}
    by_h3 = dict(zip(cached["h3"], cached["signal_dbm"]))
    assert by_h3["8928308280fffff"] == -100.0
    assert by_h3["8928308281fffff"] == -90.0


def test_redshift_shared_scan_fans_out_all_providers(tmp_path, monkeypatch):
    """One warehouse scan should populate every configured provider's state cache."""
    cfg = load_config()
    cfg.project_root = tmp_path
    cfg.raw["source"]["backend"] = "redshift"
    cfg.set_states(["20"])
    # Keep the service list small for the test.
    cfg.raw["analysis"]["services"] = [
        {"label": "5G-NR 7/1", "desc": "5G-NR (7/1 Mbps)"},
        {"label": "4G LTE", "desc": "4G LTE"},
    ]
    cfg.raw["analysis"]["providers"] = [
        {"id": 130077, "name": "AT&T"},
        {"id": 130403, "name": "T-Mobile"},
    ]
    source = RedshiftSource(cfg)
    assert source.is_mrgd
    calls: list[str] = []

    def fake_query(sql, params=()):
        calls.append(sql)
        assert "LIKE" not in sql.upper()
        assert "minsignal" in sql.lower()
        assert "providerid" in sql.lower()
        assert "technology" in sql.lower()
        assert "mindown" in sql.lower()
        assert "environmnt" in sql.lower()
        # Two hexes for 5G (tech 500 / mindown 7); one for LTE (400 / 5).
        # Duplicate AT&T 5G hex with weaker signal → keep max.
        return pd.DataFrame({
            "h3index": [
                "8928308280fffff", "8928308281fffff", "8928308283fffff",
                "8928308281fffff", "8928308280fffff",
            ],
            "providerid": [130077, 130077, 130403, 130403, 130077],
            "technology": [500, 500, 500, 500, 400],
            "mindown": [7, 7, 7, 7, 5],
            "minsignal": [-110.0, -95.0, -100.0, -90.0, -105.0],
        })

    monkeypatch.setattr(source, "_query_df", fake_query)
    source.prefetch(["292"], ["20"], [130077, 130403])

    assert len(calls) == 1
    from fcc_audit.acquire import safe_service_name
    att_5g = pd.read_parquet(
        tmp_path / "data/raw/292/130077"
        / f"{safe_service_name('5G-NR (7/1 Mbps)')}_redshift_st20_hex9.parquet"
    )
    tm_5g = pd.read_parquet(
        tmp_path / "data/raw/292/130403"
        / f"{safe_service_name('5G-NR (7/1 Mbps)')}_redshift_st20_hex9.parquet"
    )
    att_4g = pd.read_parquet(
        tmp_path / "data/raw/292/130077"
        / f"{safe_service_name('4G LTE')}_redshift_st20_hex9.parquet"
    )
    assert set(att_5g["h3"]) == {"8928308280fffff", "8928308281fffff"}
    assert set(tm_5g["h3"]) == {"8928308281fffff", "8928308283fffff"}
    assert set(att_4g["h3"]) == {"8928308280fffff"}
    att_sig = dict(zip(att_5g["h3"], att_5g["signal_dbm"]))
    assert att_sig["8928308280fffff"] == -110.0
    assert att_sig["8928308281fffff"] == -95.0
    tm_sig = dict(zip(tm_5g["h3"], tm_5g["signal_dbm"]))
    assert tm_sig["8928308281fffff"] == -90.0

    # Second prefetch must not hit Redshift again.
    source.prefetch(["292"], ["20"], [130077, 130403])
    assert len(calls) == 1

    # fetch() after warm caches must compose locally — no new warehouse scans.
    layer = source.fetch(130077, "5G-NR (7/1 Mbps)", "292")
    assert len(calls) == 1
    cached = pd.read_parquet(layer.local_path)
    assert set(cached["h3"]) == {"8928308280fffff", "8928308281fffff"}
    assert set(cached["signal_dbm"]) == {-110.0, -95.0}


def test_redshift_single_provider_uses_direct_queries(tmp_path, monkeypatch):
    """1 provider → DIRECT slices (not a multi-provider shared union scan)."""
    cfg = load_config()
    cfg.project_root = tmp_path
    cfg.raw["source"]["backend"] = "redshift"
    cfg.set_states(["20"])
    cfg.raw["analysis"]["services"] = [
        {"label": "5G-NR 7/1", "desc": "5G-NR (7/1 Mbps)"},
    ]
    cfg.raw["analysis"]["providers"] = [{"id": 130077, "name": "AT&T"}]
    source = RedshiftSource(cfg)
    calls: list[str] = []

    def fake_query(sql, params=()):
        calls.append(sql)
        assert "LIKE" not in sql.upper()
        assert "minsignal" in sql.lower()
        assert "providerid = %s" in sql.lower() or "providerid=%s" in sql.replace(" ", "").lower()
        assert "technology" in sql.lower() and "mindown" in sql.lower()
        return pd.DataFrame({
            "h3index": ["8928308280fffff", "8928308281fffff"],
            "minsignal": [-100.0, -90.0],
        })

    monkeypatch.setattr(source, "_query_df", fake_query)
    source.prefetch(["292", "291"], ["20"], [130077])
    # 2 vintages × 1 state × 1 provider × 1 service = 2 direct queries
    assert len(calls) == 2

    layer = source.fetch(130077, "5G-NR (7/1 Mbps)", "292")
    assert len(calls) == 2  # warm
    cached = pd.read_parquet(layer.local_path)
    assert len(cached) == 2
    assert set(cached["signal_dbm"]) == {-100.0, -90.0}


def test_redshift_query_error_propagates_without_cache(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.project_root = tmp_path
    cfg.raw["source"]["backend"] = "redshift"
    source = RedshiftSource(cfg)

    def denied(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(source, "_query_df", denied)
    with pytest.raises(PermissionError, match="denied"):
        source.fetch(130077, "5G-NR (7/1 Mbps)", "292")
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
        source.list_providers("292")


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
    feats, sites, coverage, completed, skipped = process_provider(
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
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), completed, set()

    monkeypatch.setattr(cli, "process_provider", completed_empty)
    args = SimpleNamespace(
        states=None, current="277", prior="279", cleanup_raw=False,
        workers=1, build_web=False, verbose=False, config=None,
        providers=None, services=None, no_context=False, ack_fcc_national=False,
        top_n=250,
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


def test_write_batch_timing_schema(tmp_path):
    run_dir = tmp_path / "processed" / "run"
    path = cli._write_batch_timing(
        run_dir,
        "20",
        {
            "states": ["20"],
            "backend": "redshift",
            "prefetch_s": 1.5,
            "analyze_s": 10.0,
            "total_s": 12.0,
            "status": "complete",
        },
    )
    blob = json.loads(path.read_text(encoding="utf-8"))
    assert blob["prefetch_s"] == 1.5
    assert blob["analyze_s"] == 10.0
    assert blob["total_s"] == 12.0
    assert path.name == "batch_timing_20.json"


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


def test_redshift_prefetch_all_expands_to_national_fips(tmp_path, monkeypatch):
    """SHARED prefetch with states='all' must warm per-state caches (not one national)."""
    cfg = load_config()
    cfg.project_root = tmp_path
    cfg.raw["source"]["backend"] = "redshift"
    cfg.raw["analysis"]["services"] = [
        {"label": "5G-NR 7/1", "desc": "5G-NR (7/1 Mbps)"},
    ]
    cfg.raw["analysis"]["providers"] = [
        {"id": 130077, "name": "AT&T"},
        {"id": 130403, "name": "T-Mobile"},
    ]
    source = RedshiftSource(cfg)
    seen_states: list[str | None] = []

    def fake_query(sql, params=()):
        # mrgd: state_fips is first param, then environmnt
        state = params[0] if params else None
        seen_states.append(state)
        assert "minsignal" in sql.lower()
        return pd.DataFrame({
            "h3index": ["8928308280fffff"],
            "providerid": [130077],
            "technology": [500],
            "mindown": [7],
            "minsignal": [-100.0],
        })

    monkeypatch.setattr(source, "_query_df", fake_query)
    source.prefetch(["292"], "all", [130077, 130403], max_workers=1)

    assert set(seen_states) == NATIONAL_STATE_FIPS
    assert source.caches_ready(["292"], "all", [130077, 130403])


def test_redshift_prefetch_parallel_shared_scans(tmp_path, monkeypatch):
    cfg = load_config()
    cfg.project_root = tmp_path
    cfg.raw["source"]["backend"] = "redshift"
    cfg.raw["source"]["redshift"]["prefetch_workers"] = 2
    cfg.raw["analysis"]["services"] = [
        {"label": "5G-NR 7/1", "desc": "5G-NR (7/1 Mbps)"},
    ]
    cfg.raw["analysis"]["providers"] = [
        {"id": 130077, "name": "AT&T"},
        {"id": 130403, "name": "T-Mobile"},
    ]
    source = RedshiftSource(cfg)
    calls: list[tuple] = []

    def fake_query(sql, params=()):
        calls.append(params)
        return pd.DataFrame({
            "h3index": ["8928308280fffff"],
            "providerid": [130077],
            "technology": [500],
            "mindown": [7],
            "minsignal": [-100.0],
        })

    # Each parallel worker creates its own RedshiftSource; patch the class method.
    monkeypatch.setattr(RedshiftSource, "_query_df", lambda self, sql, params=(): fake_query(sql, params))
    source.prefetch(["292"], ["20", "31"], [130077, 130403], max_workers=2)
    assert len(calls) == 2
    assert {c[0] for c in calls} == {"20", "31"}


def test_coverage_persist_keeps_flagged_and_top_n():
    scored = pd.DataFrame([
        {"provider_id": 1, "technology": "5G", "county_geoid": "20001",
         "priority_score": 0.9, "flag_for_review": True},
        {"provider_id": 1, "technology": "5G", "county_geoid": "20003",
         "priority_score": 0.5, "flag_for_review": False},
        {"provider_id": 1, "technology": "5G", "county_geoid": "20005",
         "priority_score": 0.8, "flag_for_review": False},
        {"provider_id": 1, "technology": "5G", "county_geoid": "20007",
         "priority_score": 0.1, "flag_for_review": False},
    ])
    keep = _coverage_geoids_to_persist(scored, top_n=2)
    # flagged 20001 + top-2 by score (20001, 20005) → {20001, 20005}
    assert keep == {"20001", "20005"}

    coverage = pd.DataFrame([
        {"county_geoid": "20001", "h3": "a"},
        {"county_geoid": "20003", "h3": "b"},
        {"county_geoid": "20005", "h3": "c"},
        {"county_geoid": "20007", "h3": "d"},
    ])
    trimmed = _filter_coverage_for_persist(coverage, scored, top_n=2)
    assert set(trimmed["county_geoid"]) == {"20001", "20005"}


def test_windows_launcher_runs_exact_national_batches_and_final_build():
    root = Path(__file__).resolve().parents[1]
    powershell = (root / "run_overnight.ps1").read_text(encoding="utf-8")
    batch_block = powershell.split("$batches = @(", 1)[1].split(")", 1)[0]
    batch_states = re.findall(r"\b\d{2}\b", batch_block)

    assert set(batch_states) == NATIONAL_STATE_FIPS
    assert len(batch_states) == len(set(batch_states))
    # Keep Redshift/interim caches across overlapping neighbor states.
    # Overnight injects --backend via $backendArgs before the subcommand.
    run_lines = [
        line for line in powershell.splitlines()
        if "fcc_audit.cli" in line and " run " in f" {line} "
        and not line.lstrip().startswith("#")
    ]
    assert run_lines, "expected a cli run invocation in run_overnight.ps1"
    assert all("--cleanup-raw" not in line for line in run_lines)
    assert any("--workers 6" in line for line in run_lines)
    assert any("--skip-prefetch" in line for line in run_lines)
    assert "@backendArgs" in powershell or "--backend" in powershell
    assert "download" in powershell and "fcc_audit.cli" in powershell
    assert "build-web" in powershell and "fcc_audit.cli" in powershell
    assert 'Backend = $(if ($env:FCC_AUDIT_BACKEND)' in powershell or 'redshift' in powershell
    # Default overnight must not auto-push; only -Publish does.
    assert "param(" in powershell and "$Publish" in powershell
    assert "git push" in powershell
    push_block = powershell.split("if ($Publish)", 1)[1]
    assert "git push" in push_block

    launcher = (root / "run.bat").read_text(encoding="utf-8")
    assert "run_overnight.ps1" in launcher
    assert "fcc_audit.cli serve" in launcher or "web\\" in launcher


def test_unix_launchers_match_national_contract():
    root = Path(__file__).resolve().parents[1]
    overnight = (root / "run_overnight.sh").read_text(encoding="utf-8")
    run_sh = (root / "run.sh").read_text(encoding="utf-8")
    process = (root / "process_batch.sh").read_text(encoding="utf-8")
    process_bat = (root / "process_batch.bat").read_text(encoding="utf-8")

    assert "run_overnight.sh" in run_sh
    assert "fcc_audit.cli serve" in run_sh or "http.server 8000" in run_sh
    assert "dashboard/index.html" not in run_sh

    assert "--workers 6" in overnight
    assert "--skip-prefetch" in overnight
    # Overnight must select Redshift explicitly (config defaults to files).
    assert 'BACKEND="${FCC_AUDIT_BACKEND:-redshift}"' in overnight
    assert 'BACKEND_ARGS=(--backend "$BACKEND")' in overnight
    assert "download" in overnight and "fcc_audit.cli" in overnight
    assert '"${BACKEND_ARGS[@]}" download' in overnight
    assert '"${BACKEND_ARGS[@]}" run' in overnight
    assert "build-web" in overnight
    assert "--publish" in overnight
    # git push only inside the PUBLISH branch
    assert 'PUBLISH=1' in overnight or 'PUBLISH=0' in overnight
    assert 'if [[ "$PUBLISH" -eq 1 ]]' in overnight

    assert "--workers 6" in process
    # process_batch must not invoke --build-web (would wipe national site).
    assert re.search(r"fcc_audit\.cli run[^\n]*--build-web", process) is None
    assert re.search(r"fcc_audit\.cli run[^\n]*--build-web", process_bat) is None
    assert "fcc_audit.cli run" in process
    assert "--workers 6" in process
