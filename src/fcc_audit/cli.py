"""Command-line orchestration for the coverage-change audit pipeline.

Usage:
    python -m fcc_audit.cli make-fixtures        # generate offline synthetic data
    python -m fcc_audit.cli list-vintages        # list available FCC vintages
    python -m fcc_audit.cli run                   # full pipeline (auto vintages)
    python -m fcc_audit.cli run --states 01,02 --cleanup-raw
    python -m fcc_audit.cli build-web             # assemble static web bundle
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from . import attribute, changedetect, explain, normalize, report, score, towers
from .acquire import DataSource, get_source
from .config import Config, Provider, load_config

_NATIONAL_STATE_FIPS = {
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36",
    "37", "38", "39", "40", "41", "42", "44", "45", "46", "47", "48",
    "49", "50", "51", "53", "54", "55", "56",
}


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _states_list(cfg: Config) -> list[str]:
    s = cfg.states
    return [] if s == "all" else list(s)


def _run_key(cfg: Config, current: str, prior: str) -> str:
    """Stable directory key isolating one backend/vintage comparison."""
    def safe(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")

    return f"{safe(cfg.backend)}_{safe(current)}_vs_{safe(prior)}"


def _target_rows(df: pd.DataFrame, states: list[str]) -> pd.DataFrame:
    """Keep requested-state outputs after analysis with neighboring-state context."""
    if df.empty or not states:
        return df
    wanted = set(states)
    if "state_fips" in df.columns:
        state = df["state_fips"].astype(str).str.zfill(2)
    elif "county_geoid" in df.columns:
        state = df["county_geoid"].astype(str).str[:2]
    else:
        return df
    return df.loc[state.isin(wanted)].reset_index(drop=True)


def _sites_relevant_to_coverage(
    sites: pd.DataFrame,
    coverage: pd.DataFrame,
    target_states: list[str],
) -> pd.DataFrame:
    """Keep target-home sites and cross-border sites that serve target coverage."""
    if sites.empty or not target_states:
        return sites
    wanted = set(target_states)
    keep: set[int] = set(
        sites.index[
            sites["county_geoid"].astype(str).str[:2].isin(wanted)
        ].tolist()
    )
    keys = ["provider_id", "technology", "vintage"]
    if coverage.empty or any(key not in sites or key not in coverage for key in keys):
        return sites.loc[sorted(keep)].reset_index(drop=True)
    for values, site_group in sites.groupby(keys, sort=False):
        mask = pd.Series(True, index=coverage.index)
        for key, value in zip(keys, values):
            mask &= coverage[key] == value
        cov_group = coverage.loc[mask]
        if cov_group.empty:
            continue
        local_indices, _, _ = attribute.attribute_hexes_to_sites(
            cov_group, site_group.reset_index(drop=True),
        )
        valid = sorted(set(int(i) for i in local_indices if i >= 0))
        keep.update(site_group.index[valid].tolist())
    return sites.loc[sorted(keep)].reset_index(drop=True)


def _context_states(counties, target_states: list[str], buffer_m: float = 50_000) -> list[str]:
    """Add states within *buffer_m* of target states for cross-border tower context."""
    if not target_states:
        return []
    state_col = counties["state_fips"].astype(str).str.zfill(2)
    target = counties.loc[state_col.isin(target_states)]
    if target.empty:
        return sorted(set(target_states))
    projected = counties.to_crs("EPSG:5070")
    projected_states = projected["state_fips"].astype(str).str.zfill(2)
    target_geom = projected.loc[projected_states.isin(target_states)].geometry.union_all()
    nearby = projected.loc[projected.geometry.intersects(target_geom.buffer(buffer_m))]
    return sorted(set(target_states) | set(nearby["state_fips"].astype(str).str.zfill(2)))


def _states_processed_label(scored: pd.DataFrame) -> str:
    """Human-readable state scope from batch metadata or county GEOIDs."""
    if "batch_states" in scored.columns:
        tokens: set[str] = set()
        for raw in scored["batch_states"].dropna().unique():
            text = str(raw).strip()
            if text.lower() == "all":
                return "all"
            tokens.update(t.strip().zfill(2) for t in text.split(",") if t.strip())
        if tokens:
            return ",".join(sorted(tokens))
    if "county_geoid" in scored.columns:
        prefs = sorted({str(g)[:2] for g in scored["county_geoid"].astype(str) if str(g)[:2].isdigit()})
        if prefs:
            return ",".join(prefs)
    return "all"


def _drop_fixture_geographies(df: pd.DataFrame) -> pd.DataFrame:
    """Remove synthetic state-90 rows before assembling a real web bundle."""
    if df.empty:
        return df
    mask = pd.Series(False, index=df.index)
    if "county_geoid" in df.columns:
        mask |= df["county_geoid"].astype(str).str.startswith("90")
    if "state_fips" in df.columns:
        mask |= df["state_fips"].astype(str).str.zfill(2).eq("90")
    return df.loc[~mask].reset_index(drop=True)


def _demo_web_defaults(cfg: Config, scored: pd.DataFrame) -> dict[str, Any]:
    """Landing-page defaults for the fixture / synthetic county demo."""
    if scored.empty:
        return {}
    geoids = scored["county_geoid"].astype(str)
    if cfg.backend != "fixture" and not geoids.str.startswith("90").any():
        return {}
    return {
        "default_provider_id": 130077,
        "default_county_geoid": "90003",
    }


def _analyze_unit(
    cfg, provider, service_label, cur_file, pri_file, counties, county_area_km2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run change/site/attribute/scoring for one (provider, service) unit."""
    log = logging.getLogger(__name__)
    county_res = int(cfg.geography["county_h3_resolution"])
    site_res = int(cfg.geography["site_h3_resolution"])

    cur8, cur9 = normalize.normalize_layers(
        cfg, cur_file, counties, county_res, site_res, service_label
    )
    pri8, pri9 = normalize.normalize_layers(
        cfg, pri_file, counties, county_res, site_res, service_label
    )
    if cur8.empty and pri8.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    change = changedetect.hex_change(pri8, cur8)
    cc = changedetect.county_change(change, county_res, county_area_km2)

    prior_sites = towers.infer_sites(pri9, cfg, label_prefix="P")
    current_sites = towers.infer_sites(cur9, cfg, label_prefix="C")
    current_sites = attribute.match_sites(
        prior_sites, current_sites, float(cfg.towers["site_match_radius_m"])
    )
    # Compute empirical lobe reach from full-band coverage so fringe hexes
    # (beyond the strong-signal core) are attributed to their tower rather
    # than being mis-classified as 'unattributed'. A single matched tower
    # will capture ~100% of its gained hexes after this step.
    current_sites = towers.compute_lobe_reach(cur9, current_sites)
    prior_sites = towers.compute_lobe_reach(pri9, prior_sites)

    attr = attribute.attribute_changes(change, current_sites, county_res)
    bsnap = normalize.boundary_snap_share(
        change, counties,
        float(cfg.reconcile.get("boundary_snap_threshold_m", 1500.0)),
        cfg.geography["equal_area_crs"],
    )
    feats = score.build_features(cc, attr, bsnap)
    tag = {
        "provider_id": provider.id, "provider_name": provider.name,
        "technology": service_label,
    }
    if not feats.empty:
        feats["county_geoid"] = feats["county_geoid"].astype(str)
        prior_srv = attribute.serving_towers_by_county(pri8, prior_sites)
        current_srv = attribute.serving_towers_by_county(cur8, current_sites)
        if not prior_srv.empty:
            prior_srv["county_geoid"] = prior_srv["county_geoid"].astype(str)
            feats = feats.merge(
                prior_srv.rename(columns={
                    "towers_serving": "prior_towers",
                    "towers_in_county": "prior_towers_in_county",
                    "towers_cross_border": "prior_towers_cross_border",
                }),
                on="county_geoid",
                how="left",
            )
        if not current_srv.empty:
            current_srv["county_geoid"] = current_srv["county_geoid"].astype(str)
            feats = feats.merge(
                current_srv.rename(columns={
                    "towers_serving": "current_towers",
                    "towers_in_county": "current_towers_in_county",
                    "towers_cross_border": "current_towers_cross_border",
                }),
                on="county_geoid",
                how="left",
            )
        for col in [
            "prior_towers", "current_towers",
            "prior_towers_in_county", "current_towers_in_county",
            "prior_towers_cross_border", "current_towers_cross_border",
        ]:
            if col in feats:
                feats[col] = feats[col].fillna(0).astype(int)
        for k, v in tag.items():
            feats[k] = v

    prior_out = prior_sites.copy()
    if not prior_out.empty:
        prior_out["vintage"] = "prior"
        prior_out["site_class"] = "prior_site"
        for k, v in tag.items():
            prior_out[k] = v
    current_out = current_sites.copy()
    if not current_out.empty:
        current_out["vintage"] = "current"
        for k, v in tag.items():
            current_out[k] = v
    sites = pd.concat([prior_out, current_out], ignore_index=True)

    cov_cols = ["h3", "signal_dbm", "county_geoid"]
    cov_parts = []
    if not pri8.empty:
        p = pri8[cov_cols].copy()
        p["vintage"] = "prior"
        cov_parts.append(p)
    if not cur8.empty:
        c = cur8[cov_cols].copy()
        c["vintage"] = "current"
        cov_parts.append(c)
    coverage = pd.concat(cov_parts, ignore_index=True) if cov_parts else pd.DataFrame()
    if not coverage.empty:
        for k, v in tag.items():
            coverage[k] = v

    log.info(
        "  %s %s: %d counties changed, %d prior / %d current sites",
        provider.name, service_label,
        0 if feats.empty else len(feats), len(prior_sites), len(current_sites),
    )
    return feats, sites, coverage


def process_provider(
    cfg: Config,
    source: DataSource,
    provider: Provider,
    current: str,
    prior: str,
    counties,
    county_area_km2: dict | None = None,
    cleanup_raw: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[tuple[int, str]]]:
    log = logging.getLogger(__name__)
    log.info("=== %s (id=%s) ===", provider.name, provider.id)

    feats_parts, sites_parts, coverage_parts = [], [], []
    completed_units: set[tuple[int, str]] = set()
    for svc in cfg.services:
        label, desc = svc["label"], svc["desc"]
        try:
            cur_file = source.fetch(provider.id, desc, current)
            pri_file = source.fetch(provider.id, desc, prior)
        except FileNotFoundError as exc:
            # FCC/fixture backends may legitimately have no filing for a
            # provider/service. Redshift represents no coverage as an empty
            # layer; RuntimeError is therefore a real query/permission failure
            # and must abort the batch rather than silently publishing partial data.
            log.warning("no source layer for %s %s: %s", provider.name, label, exc)
            continue

        feats, sites, coverage = _analyze_unit(
            cfg, provider, label, cur_file, pri_file, counties, county_area_km2,
        )
        completed_units.add((int(provider.id), str(label)))
        if not feats.empty:
            feats_parts.append(feats)
        if not sites.empty:
            sites_parts.append(sites)
        if not coverage.empty:
            coverage_parts.append(coverage)

        if cleanup_raw:
            import os
            for f in (cur_file, pri_file):
                try:
                    os.remove(f.local_path)
                except OSError:
                    pass

    feats = pd.concat(feats_parts, ignore_index=True) if feats_parts else pd.DataFrame()
    sites = pd.concat(sites_parts, ignore_index=True) if sites_parts else pd.DataFrame()
    coverage = pd.concat(coverage_parts, ignore_index=True) if coverage_parts else pd.DataFrame()
    return feats, sites, coverage, completed_units


def _provider_worker(
    payload: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[tuple[int, str]]]:
    """Process one provider in a separate process (for ``run --workers N``).

    Each worker rebuilds its own config/source/counties so nothing large needs
    to be pickled across the process boundary; results are identical to the
    serial path. Intended for use after raw files are already downloaded, so
    workers do CPU-bound H3 indexing in parallel without multiplying the FCC
    request rate.
    """
    setup_logging(payload.get("verbose", False))
    cfg = load_config(payload["config"])
    if payload.get("backend"):
        cfg.raw["source"]["backend"] = payload["backend"]
    cfg.set_states(payload["states"])
    source = get_source(cfg)
    counties = normalize.load_counties(cfg)
    area = normalize.county_areas_km2(counties, cfg.geography["equal_area_crs"])
    provider = Provider(**payload["provider"])
    return process_provider(
        cfg, source, provider,
        payload["current"], payload["prior"],
        counties, area, payload["cleanup_raw"],
    )


def _resolve_providers(cfg: Config, source: DataSource, vintage: str) -> list[Provider]:
    """Explicit provider list, or auto-discovered from the catalog when 'all'."""
    if cfg.providers_all:
        providers = source.list_providers(vintage)
        logging.getLogger(__name__).info("discovered %d providers for %s", len(providers), vintage)
        return providers
    return cfg.providers


def _save_batch_results(
    cfg: Config,
    scored: pd.DataFrame,
    sites: pd.DataFrame,
    meta: dict[str, Any],
    coverage: pd.DataFrame | None = None,
    *,
    states: list[str] | None = None,
) -> None:
    """Persist scored rows (and optional sites/coverage) for incremental web builds."""
    states = list(states or [])
    run_dir = cfg.path("processed") / _run_key(cfg, meta["current"], meta["prior"])
    scored_dir = run_dir / "scored"
    sites_dir = run_dir / "sites"
    coverage_dir = run_dir / "coverage"
    states_key = "-".join(sorted(states)) if states else "all"

    # A rerun replaces the whole target batch, including units/states that are
    # now legitimately empty. Remove old partitions before writing new results.
    if scored_dir.exists():
        for stale in scored_dir.glob(f"scored_*_{states_key}.parquet"):
            stale.unlink()
    (sites_dir / f"sites_{states_key}.parquet").unlink(missing_ok=True)
    if coverage_dir.exists():
        stale_coverage = (
            [coverage_dir / f"coverage_{state}.parquet" for state in states]
            if states else list(coverage_dir.glob("coverage_*.parquet"))
        )
        for stale in stale_coverage:
            stale.unlink(missing_ok=True)

    if not scored.empty and "technology" in scored.columns:
        for svc in scored["technology"].unique():
            svc_rows = scored[scored["technology"] == svc]
            report.save_batch_scored(svc_rows, scored_dir, service_label=str(svc), states=states, meta=meta)

    batch_ts = datetime.now(timezone.utc).isoformat()

    if not sites.empty:
        sites_dir.mkdir(parents=True, exist_ok=True)
        sites_path = sites_dir / f"sites_{states_key}.parquet"
        batch_sites = sites.copy()
        batch_sites["batch_ts"] = batch_ts
        batch_sites.to_parquet(sites_path, index=False)

    if coverage is not None and not coverage.empty:
        coverage_dir.mkdir(parents=True, exist_ok=True)
        batch_cov = coverage.copy()
        batch_cov["batch_ts"] = batch_ts
        # State partitions bound peak RAM during the final web build.
        state_key = batch_cov["county_geoid"].astype(str).str[:2]
        for state, state_cov in batch_cov.groupby(state_key, sort=True):
            state_cov.to_parquet(coverage_dir / f"coverage_{state}.parquet", index=False)

    manifests_dir = run_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "backend": cfg.backend,
        "current": meta["current"],
        "prior": meta["prior"],
        "states": states or ["all"],
        "query_states": meta.get("query_states", states or ["all"]),
        "analysis_units": meta.get("analysis_units", []),
        "completed_analysis_units": meta.get("completed_analysis_units", []),
        "missing_analysis_units": meta.get("missing_analysis_units", []),
        "status": "complete" if not meta.get("missing_analysis_units") else "incomplete",
        "completed_at": batch_ts,
    }
    manifest_path = manifests_dir / f"batch_{states_key}.json"
    manifest_tmp = manifest_path.with_suffix(".json.part")
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_tmp.replace(manifest_path)


def cmd_run(cfg: Config, args) -> int:
    log = logging.getLogger(__name__)
    if getattr(args, "states", None):
        cfg.set_states(args.states)
    target_states = _states_list(cfg)

    source = get_source(cfg)
    current, prior = source.resolve_vintages(
        args.current or cfg.vintage_current, args.prior or cfg.vintage_prior
    )
    log.info("comparing current=%s vs prior=%s (states=%s)", current, prior, cfg.states)
    target_key = "-".join(sorted(target_states)) if target_states else "all"
    run_dir = cfg.path("processed") / _run_key(cfg, current, prior)
    (run_dir / "manifests" / f"batch_{target_key}.json").unlink(missing_ok=True)
    for stale in (run_dir / "scored").glob(f"scored_*_{target_key}.parquet"):
        stale.unlink()
    (run_dir / "sites" / f"sites_{target_key}.parquet").unlink(missing_ok=True)
    coverage_dir = run_dir / "coverage"
    stale_coverage = (
        [coverage_dir / f"coverage_{state}.parquet" for state in target_states]
        if target_states else list(coverage_dir.glob("coverage_*.parquet"))
    )
    for stale in stale_coverage:
        stale.unlink(missing_ok=True)

    counties = normalize.load_counties(cfg)
    county_area_km2 = normalize.county_areas_km2(counties, cfg.geography["equal_area_crs"])
    if cfg.backend == "redshift" and target_states:
        context_states = _context_states(counties, target_states)
        if context_states != sorted(target_states):
            log.info(
                "adding cross-border tower context: target=%s query=%s",
                target_states, context_states,
            )
            cfg.set_states(context_states)

    providers = _resolve_providers(cfg, source, current)
    expected_units = {
        (int(provider.id), str(service["label"]))
        for provider in providers
        for service in cfg.services
    }
    cleanup_raw = bool(getattr(args, "cleanup_raw", False)) and cfg.raw["source"]["backend"] != "fixture"
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    all_feats, all_sites, all_coverage = [], [], []
    completed_units: set[tuple[int, str]] = set()

    def _collect(result):
        feats, sites, coverage, completed = result
        completed_units.update(completed)
        feats = _target_rows(feats, target_states)
        coverage = _target_rows(coverage, target_states)
        if not feats.empty:
            all_feats.append(feats)
        if not sites.empty:
            all_sites.append(sites)
        if not coverage.empty:
            all_coverage.append(coverage)

    if workers > 1 and len(providers) > 1:
        from concurrent.futures import ProcessPoolExecutor

        n = min(workers, len(providers))
        log.info("processing %d providers across %d worker processes", len(providers), n)
        payloads = [
            {
                "config": getattr(args, "config", None),
                "backend": cfg.backend,
                "states": _states_list(cfg) or "all",
                "current": current,
                "prior": prior,
                "provider": {"id": p.id, "name": p.name},
                "cleanup_raw": cleanup_raw,
                "verbose": getattr(args, "verbose", False),
            }
            for p in providers
        ]
        with ProcessPoolExecutor(max_workers=n) as ex:
            for result in ex.map(_provider_worker, payloads):
                _collect(result)
    else:
        for provider in providers:
            _collect(process_provider(
                cfg, source, provider, current, prior, counties, county_area_km2, cleanup_raw
            ))

    missing_units = sorted(expected_units - completed_units)
    if missing_units:
        log.error(
            "incomplete batch: %d provider/service units did not complete: %s",
            len(missing_units),
            "; ".join(f"{provider_id}/{service}" for provider_id, service in missing_units),
        )
        return 2

    sites = pd.concat(all_sites, ignore_index=True) if all_sites else pd.DataFrame()
    coverage = pd.concat(all_coverage, ignore_index=True) if all_coverage else pd.DataFrame()
    states_label = ",".join(target_states) if target_states else "all"
    meta = {
        "current": current,
        "prior": prior,
        "providers": ", ".join(p.name for p in providers),
        "technologies": ", ".join(s["label"] for s in cfg.services),
        "states_processed": states_label,
        "analysis_units": [
            {"provider_id": int(provider.id), "technology": str(service["label"])}
            for provider in providers
            for service in cfg.services
        ],
        "completed_analysis_units": [
            {"provider_id": provider_id, "technology": service}
            for provider_id, service in sorted(completed_units)
        ],
        "missing_analysis_units": [],
        "query_states": _states_list(cfg) or ["all"],
    }

    sites = _sites_relevant_to_coverage(sites, coverage, target_states)
    if not all_feats:
        # A successful all-empty query is complete, not a failed batch. Persist
        # its success manifest so national completeness can distinguish zero
        # coverage from a skipped/erroring provider-service unit.
        _save_batch_results(
            cfg, pd.DataFrame(), sites, meta, coverage, states=target_states,
        )
        log.info("batch complete: all provider/service layers were valid but produced no rows")
        return 0

    features = pd.concat(all_feats, ignore_index=True)
    scored = score.score(features, cfg)
    scored = explain.add_explanations(scored)
    _save_batch_results(cfg, scored, sites, meta, coverage, states=target_states)

    dashboard_dir = cfg.project_root / "dashboard"
    dashboard_dir.mkdir(exist_ok=True)
    paths = report.write_outputs(
        scored, sites, counties, cfg.path("outputs"), dashboard_dir, meta
    )

    if getattr(args, "build_web", False):
        web_dir = cfg.project_root / "web"
        web_meta = dict(meta)
        web_meta.update(_demo_web_defaults(cfg, scored))
        web_paths = report.write_web_bundle(
            scored, sites, counties, web_dir, web_meta, coverage=coverage,
            top_n=getattr(args, "top_n", 250),
        )
        paths.update({f"web_{k}": v for k, v in web_paths.items()})

    flagged = int(scored["flag_for_review"].sum())
    log.info("DONE: %d provider-county rows, %d flagged", len(scored), flagged)
    print("\nOutputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print(f"\nFlagged for review: {flagged}/{len(scored)}")
    print(f"Open the dashboard: {cfg.project_root / 'web' / 'index.html'}")
    return 0


def cmd_build_web(cfg: Config, args) -> int:
    """Assemble the static web bundle from accumulated batch parquet files."""
    log = logging.getLogger(__name__)
    current = str(cfg.vintage_current)
    prior = str(cfg.vintage_prior)
    run_dir = cfg.path("processed") / _run_key(cfg, current, prior)
    scored_dir = run_dir / "scored"
    sites_dir = run_dir / "sites"
    coverage_dir = run_dir / "coverage"
    manifests_dir = run_dir / "manifests"

    manifests = []
    if manifests_dir.exists():
        manifests = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(manifests_dir.glob("batch_*.json"))
        ]
    if (
        cfg.backend != "fixture"
        and cfg.states == "all"
        and not getattr(args, "allow_incomplete", False)
    ):
        completed_states: set[str] = set()
        unit_sets: set[frozenset[tuple[int, str]]] = set()
        for manifest in manifests:
            if manifest.get("backend") != cfg.backend:
                continue
            if str(manifest.get("current")) != current or str(manifest.get("prior")) != prior:
                continue
            if manifest.get("status", "complete") != "complete":
                continue
            states = manifest.get("states") or []
            units = frozenset(
                (int(unit["provider_id"]), str(unit["technology"]))
                for unit in manifest.get("analysis_units", [])
            )
            completed = frozenset(
                (int(unit["provider_id"]), str(unit["technology"]))
                for unit in manifest.get("completed_analysis_units", manifest.get("analysis_units", []))
            )
            if completed != units or manifest.get("missing_analysis_units"):
                continue
            if units:
                unit_sets.add(units)
            if "all" in states:
                completed_states = set(_NATIONAL_STATE_FIPS)
                break
            completed_states.update(str(state).zfill(2) for state in states)
        missing_states = sorted(_NATIONAL_STATE_FIPS - completed_states)
        if missing_states:
            log.error(
                "national build incomplete for %s: missing successful batch manifests for %s "
                "(use --allow-incomplete only for a deliberate preview)",
                run_dir.name, ",".join(missing_states),
            )
            return 1
        if len(unit_sets) > 1:
            log.error(
                "national build has inconsistent provider/service units across batches in %s",
                run_dir,
            )
            return 1
        if not cfg.providers_all:
            expected_units = frozenset(
                (int(provider.id), str(service["label"]))
                for provider in cfg.providers
                for service in cfg.services
            )
            if unit_sets != {expected_units}:
                log.error(
                    "national build manifests do not match configured provider/service units"
                )
                return 1

    scored = report.load_accumulated_scored(scored_dir)

    if scored.empty:
        log.error("no accumulated scored data in %s — run batches first", scored_dir)
        return 1

    current_values = set(scored.get("batch_current", pd.Series(dtype=str)).dropna().astype(str))
    prior_values = set(scored.get("batch_prior", pd.Series(dtype=str)).dropna().astype(str))
    if current_values - {current} or prior_values - {prior}:
        log.error(
            "refusing mixed-vintage build: current=%s prior=%s in %s",
            sorted(current_values), sorted(prior_values), run_dir,
        )
        return 1

    # Batch scores are relative to each batch. Recompute over the accumulated
    # national feature frame so priority ranks and percentile flags are comparable.
    scored = score.score(scored, cfg)
    scored = explain.add_explanations(scored)
    fixture_bundle = (
        "county_geoid" in scored.columns
        and scored["county_geoid"].astype(str).str.startswith("90").all()
    )

    # Merge all site batches (dedupe on lat/lng/provider/service).
    sites = pd.DataFrame()
    if sites_dir.exists():
        site_parts = [pd.read_parquet(p) for p in sorted(sites_dir.glob("sites_*.parquet"))]
        if site_parts:
            sites = pd.concat(site_parts, ignore_index=True)
            if not fixture_bundle:
                sites = _drop_fixture_geographies(sites)
            dedup_cols = [
                c for c in ["lat", "lng", "provider_id", "technology", "vintage"]
                if c in sites.columns
            ]
            if dedup_cols:
                sites = sites.drop_duplicates(subset=dedup_cols, keep="last")

    meta = {
        "current": current,
        "prior": prior,
        "providers": ", ".join(
            scored.drop_duplicates("provider_id")["provider_name"].astype(str).tolist()
        ),
        "technologies": ", ".join(sorted(scored["technology"].unique())),
        "states_processed": _states_processed_label(scored),
    }

    counties = normalize.load_counties(cfg)
    dashboard_dir = cfg.project_root / "dashboard"
    dashboard_dir.mkdir(exist_ok=True)
    report.write_outputs(
        scored, sites, counties, cfg.path("outputs"), dashboard_dir, meta,
    )
    web_dir = cfg.project_root / "web"
    web_meta = dict(meta)
    web_meta.update(_demo_web_defaults(cfg, scored))
    render_pngs = getattr(args, "render_pngs", False)
    top_n = getattr(args, "top_n", 250)
    coverage_paths = sorted(coverage_dir.glob("coverage_*.parquet"))
    paths = report.write_web_bundle(
        scored, sites, counties, web_dir, web_meta, coverage_paths=coverage_paths,
        render_pngs=render_pngs,
        top_n=top_n,
    )
    flagged = int(scored["flag_for_review"].sum()) if "flag_for_review" in scored.columns else 0
    log.info("web bundle ready: %d records, %d flagged", len(scored), flagged)
    print("\nWeb bundle:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print(f"\nDeploy: push to git -> Vercel auto-deploys web/")
    print(f"Preview locally: cd web && python -m http.server 8000")
    return 0


def cmd_list_vintages(cfg: Config, args) -> int:
    source = get_source(cfg)
    for v in source.list_vintages():
        print(v)
    return 0


def cmd_download(cfg: Config, args) -> int:
    """Pre-stage raw coverage files from the FCC API without running analysis."""
    log = logging.getLogger(__name__)
    if getattr(args, "states", None):
        cfg.set_states(args.states)

    source = get_source(cfg)
    current, prior = source.resolve_vintages(
        args.current or cfg.vintage_current, args.prior or cfg.vintage_prior
    )
    providers = _resolve_providers(cfg, source, current)
    services = cfg.services
    total_bytes = 0
    n_files = n_skipped = 0
    log.info("downloading %d providers x %d services x 2 vintages (states=%s)",
             len(providers), len(services), cfg.states)
    for provider in providers:
        for vintage in (current, prior):
            for svc in services:
                try:
                    cov = source.fetch(provider.id, svc["desc"], vintage)
                except FileNotFoundError as exc:
                    log.warning("skip %s %s @ %s: %s", provider.name, svc["label"], vintage, exc)
                    n_skipped += 1
                    continue
                size = cov.local_path.stat().st_size if cov.local_path.exists() else 0
                total_bytes += size
                n_files += 1
                log.info("ok %s %s @ %s (%.1f MB)", provider.name, svc["label"], vintage, size / 1e6)
    print(f"\nDownloaded/cached {n_files} files ({total_bytes/1e9:.2f} GB), "
          f"{n_skipped} unavailable. Raw data under {cfg.path('raw')}")
    print("Now run offline:  python -m fcc_audit.cli run")
    return 0


def cmd_make_fixtures(cfg: Config, args) -> int:
    from . import fixtures

    fixtures.make_fixtures(cfg)
    print("Fixtures written. Set source.backend: fixture in config to use them.")
    return 0


def _cmd_case_files(cfg: Config, args) -> int:
    """Generate per-county case files from scored pipeline output."""
    from .casefile import cmd_case_files
    return cmd_case_files(cfg, args)


def cmd_validate(cfg: Config, args) -> int:
    """Backtest the pipeline against ground-truth labels; write validation report."""
    from pathlib import Path
    from .validate import run_validation_from_cli

    gt_path = Path(args.ground_truth) if getattr(args, "ground_truth", None) else None
    out_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else None
    report_path = run_validation_from_cli(
        cfg,
        ground_truth_path=gt_path,
        output_dir=out_dir,
        n_boot=int(getattr(args, "n_boot", 500)),
        cost_fp=float(getattr(args, "cost_fp", 1.0)),
        cost_fn=float(getattr(args, "cost_fn", 5.0)),
    )
    if report_path:
        log.info("Validation report: %s", report_path)
        return 0
    log.warning("Validation produced no report. Run the pipeline first.")
    return 1


def cmd_benchmark(cfg: Config, args) -> int:
    """Check the pipeline reproduces the FCC's selected/not-selected examples."""
    log = logging.getLogger(__name__)
    bench = cfg.raw.get("benchmark")
    if not bench:
        log.error("no benchmark section in config")
        return 1
    args.current = bench["vintages"]["current"]
    args.prior = bench["vintages"]["prior"]

    source = get_source(cfg)
    current, prior = source.resolve_vintages(args.current, args.prior)
    counties = normalize.load_counties(cfg)
    county_area_km2 = normalize.county_areas_km2(counties, cfg.geography["equal_area_crs"])
    all_feats = []
    for provider in _resolve_providers(cfg, source, current):
        feats, _, _, _ = process_provider(
            cfg, source, provider, current, prior, counties, county_area_km2
        )
        if not feats.empty:
            all_feats.append(feats)
    if not all_feats:
        log.error("no features; benchmark cannot run (likely no data downloaded)")
        return 1
    scored = score.score(pd.concat(all_feats, ignore_index=True), cfg)

    scored = scored.copy()
    scored["county_geoid"] = scored["county_geoid"].astype(str)
    bench_service = bench.get("service_label", "5G-NR 7/1")
    if "technology" in scored and bench_service:
        scored = scored[scored["technology"] == bench_service]
    flagged = scored[scored["flag_for_review"]]
    flagged_any = set(flagged["county_geoid"])
    flagged_by_provider = set(zip(flagged["county_geoid"], flagged["provider_id"]))

    print(f"\nBenchmark: D25 vs J25 ({current} vs {prior})\n")
    print(f"{'County':<26}{'provider':<10}{'expected':<10}{'pipeline':<10}result")
    tp = fp = tn = fn = 0
    for c in bench["counties"]:
        geoid = str(c["geoid"])
        pid = c.get("provider_id")
        if pid:
            got = (geoid, pid) in flagged_by_provider
            prov = (cfg.provider_by_id(pid).name if cfg.provider_by_id(pid) else str(pid))
        else:
            got = geoid in flagged_any
            prov = "any"
        exp = bool(c["expected_selected"])
        ok = got == exp
        tp += got and exp
        fp += got and not exp
        tn += (not got) and (not exp)
        fn += (not got) and exp
        print(f"{c['name']:<26}{prov:<10}{('select' if exp else 'skip'):<10}"
              f"{('select' if got else 'skip'):<10}{'PASS' if ok else 'FAIL'}")

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"\n{tp + tn}/{total} match  |  precision={precision:.2f}  recall={recall:.2f}  "
          f"(TP={tp} FP={fp} TN={tn} FN={fn})")
    return 0 if (fp + fn) == 0 else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fcc_audit", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    parser.add_argument(
        "--backend", default=None, choices=["fcc", "redshift", "fixture"],
        help="override source.backend from config",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline")
    p_run.add_argument("--current", default=None)
    p_run.add_argument("--prior", default=None)
    p_run.add_argument(
        "--states", default=None, nargs="+",
        help='state FIPS to scope this batch: "01,02,48" and/or 01 02 48 (or "all")',
    )
    p_run.add_argument(
        "--cleanup-raw", action="store_true",
        help="delete each provider's raw download after processing (bounds disk use)",
    )
    p_run.add_argument(
        "--build-web", action="store_true",
        help="also rebuild the static web bundle after this batch",
    )
    p_run.add_argument(
        "--top-n", type=int, default=250,
        help="max counties per provider×service with detail JSON and tier coloring (default 250)",
    )
    p_run.add_argument(
        "--workers", type=int, default=1,
        help="parallel worker processes for provider analysis (CPU-bound). "
             "Best used after `download` has cached the raw files, so workers "
             "don't multiply the FCC request rate. Default 1 (serial).",
    )
    p_run.set_defaults(func=cmd_run)

    sub.add_parser("list-vintages", help="list available vintages").set_defaults(
        func=cmd_list_vintages
    )

    p_dl = sub.add_parser(
        "download", help="pre-fetch all raw coverage files from the FCC API (no analysis)"
    )
    p_dl.add_argument("--current", default=None)
    p_dl.add_argument("--prior", default=None)
    p_dl.add_argument(
        "--states", default=None, nargs="+",
        help='state FIPS: "01,02,48" and/or 01 02 48 (or "all")',
    )
    p_dl.set_defaults(func=cmd_download)

    p_bw = sub.add_parser("build-web", help="assemble static web bundle from accumulated batches")
    p_bw.add_argument(
        "--render-pngs", dest="render_pngs", action="store_true",
        help="also render server-side prior/current PNG maps (large; client-side hex rendering is the default)",
    )
    p_bw.add_argument(
        "--top-n", type=int, default=250,
        help="max counties per provider×service with detail JSON and tier coloring (default 250)",
    )
    p_bw.add_argument(
        "--allow-incomplete", action="store_true",
        help="build a deliberate partial preview without all 51 state batch manifests",
    )
    p_bw.set_defaults(func=cmd_build_web)
    sub.add_parser("make-fixtures", help="generate synthetic offline data").set_defaults(
        func=cmd_make_fixtures
    )
    sub.add_parser(
        "benchmark", help="check against FCC's labeled selected/not-selected counties"
    ).set_defaults(func=cmd_benchmark)

    p_val = sub.add_parser(
        "validate",
        help="backtest against ground-truth labels; emit precision/recall/F1 + plots",
    )
    p_val.add_argument(
        "--ground-truth", default=None,
        help="CSV path with columns matching pipeline join keys + a 'label' column (1=gaming)",
    )
    p_val.add_argument("--output-dir", default=None, help="output directory (default: data/validation/)")
    p_val.add_argument("--n-boot", type=int, default=500, help="bootstrap resamples for CIs")
    p_val.add_argument("--cost-fp", type=float, default=1.0, help="cost of a false positive (wasted drive-test)")
    p_val.add_argument("--cost-fn", type=float, default=5.0, help="cost of a false negative (missed gaming)")
    p_val.set_defaults(func=lambda cfg, args: cmd_validate(cfg, args))

    p_cf = sub.add_parser(
        "case-files",
        help="auto-generate per-county case files (Markdown/PDF) from scored data",
    )
    p_cf.add_argument("--out-dir", dest="out_dir", default=None,
                      help="output directory for case files (default: <outputs>/case_files/)")
    p_cf.add_argument("--all", action="store_true",
                      help="write case files for all counties, not just flagged ones")
    p_cf.add_argument("--geoid", default=None,
                      help="generate a single county case file by GEOID")
    p_cf.add_argument("--pdf", action="store_true",
                      help="also write PDF files (requires weasyprint + mistune)")
    p_cf.add_argument("--llm", default="none", choices=["none", "local", "gemini"],
                      help="LLM backend for narrative drafting (default: none)")
    p_cf.add_argument("--llm-url", dest="llm_url", default="http://localhost:11434",
                      help="Ollama/llama.cpp server URL (for --llm local)")
    p_cf.add_argument("--llm-model", dest="llm_model", default="llama3",
                      help="model name on the local server (for --llm local)")
    p_cf.add_argument("--gemini-api-key", dest="gemini_api_key", default=None,
                      help="Gemini API key (for --llm gemini; or set GEMINI_API_KEY env var)")
    p_cf.set_defaults(func=lambda cfg, args: _cmd_case_files(cfg, args))

    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = load_config(args.config)
    if args.backend:
        cfg.raw["source"]["backend"] = args.backend
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
