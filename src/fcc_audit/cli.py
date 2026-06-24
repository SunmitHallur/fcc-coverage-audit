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
import logging
import sys
from typing import Any

import pandas as pd

from . import attribute, changedetect, explain, normalize, report, score, towers
from .acquire import DataSource, get_source
from .config import Config, Provider, load_config


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _states_list(cfg: Config) -> list[str]:
    s = cfg.states
    return [] if s == "all" else list(s)


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
        prior_counts = attribute.tower_counts_by_county(prior_sites)
        current_counts = attribute.tower_counts_by_county(current_sites)
        feats["prior_towers"] = (
            feats["county_geoid"].map(prior_counts).fillna(0).astype(int)
        )
        feats["current_towers"] = (
            feats["county_geoid"].map(current_counts).fillna(0).astype(int)
        )
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log = logging.getLogger(__name__)
    log.info("=== %s (id=%s) ===", provider.name, provider.id)

    feats_parts, sites_parts, coverage_parts = [], [], []
    for svc in cfg.services:
        label, desc = svc["label"], svc["desc"]
        try:
            cur_file = source.fetch(provider.id, desc, current)
            pri_file = source.fetch(provider.id, desc, prior)
        except (FileNotFoundError, RuntimeError) as exc:
            log.warning("skip %s %s: %s", provider.name, label, exc)
            continue

        feats, sites, coverage = _analyze_unit(
            cfg, provider, label, cur_file, pri_file, counties, county_area_km2,
        )
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
    return feats, sites, coverage


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
) -> None:
    """Persist scored rows (and optional sites/coverage) for incremental web builds."""
    states = _states_list(cfg)
    scored_dir = cfg.path("processed") / "scored"
    sites_dir = cfg.path("processed") / "sites"
    coverage_dir = cfg.path("processed") / "coverage"

    if not scored.empty and "technology" in scored.columns:
        for svc in scored["technology"].unique():
            svc_rows = scored[scored["technology"] == svc]
            report.save_batch_scored(svc_rows, scored_dir, service_label=str(svc), states=states, meta=meta)

    states_key = "-".join(sorted(states)) if states else "all"
    batch_ts = meta.get("generated_at", "")

    if not sites.empty:
        sites_dir.mkdir(parents=True, exist_ok=True)
        sites_path = sites_dir / f"sites_{states_key}.parquet"
        batch_sites = sites.copy()
        batch_sites["batch_ts"] = batch_ts
        batch_sites.to_parquet(sites_path, index=False)

    if coverage is not None and not coverage.empty:
        coverage_dir.mkdir(parents=True, exist_ok=True)
        cov_path = coverage_dir / f"coverage_{states_key}.parquet"
        batch_cov = coverage.copy()
        batch_cov["batch_ts"] = batch_ts
        batch_cov.to_parquet(cov_path, index=False)


def cmd_run(cfg: Config, args) -> int:
    log = logging.getLogger(__name__)
    if getattr(args, "states", None):
        cfg.set_states(args.states)

    source = get_source(cfg)
    current, prior = source.resolve_vintages(
        args.current or cfg.vintage_current, args.prior or cfg.vintage_prior
    )
    log.info("comparing current=%s vs prior=%s (states=%s)", current, prior, cfg.states)

    counties = normalize.load_counties(cfg)
    county_area_km2 = normalize.county_areas_km2(counties, cfg.geography["equal_area_crs"])

    providers = _resolve_providers(cfg, source, current)
    cleanup_raw = bool(getattr(args, "cleanup_raw", False)) and cfg.raw["source"]["backend"] != "fixture"
    all_feats, all_sites, all_coverage = [], [], []
    for provider in providers:
        feats, sites, coverage = process_provider(
            cfg, source, provider, current, prior, counties, county_area_km2, cleanup_raw
        )
        if not feats.empty:
            all_feats.append(feats)
        if not sites.empty:
            all_sites.append(sites)
        if not coverage.empty:
            all_coverage.append(coverage)

    if not all_feats:
        log.error("no features produced; nothing to score")
        return 1

    features = pd.concat(all_feats, ignore_index=True)
    scored = score.score(features, cfg)
    scored = explain.add_explanations(scored)
    sites = pd.concat(all_sites, ignore_index=True) if all_sites else pd.DataFrame()
    coverage = pd.concat(all_coverage, ignore_index=True) if all_coverage else pd.DataFrame()

    states_label = ",".join(_states_list(cfg)) if _states_list(cfg) else "all"
    meta = {
        "current": current,
        "prior": prior,
        "providers": ", ".join(p.name for p in providers),
        "technologies": ", ".join(s["label"] for s in cfg.services),
        "states_processed": states_label,
    }
    _save_batch_results(cfg, scored, sites, meta, coverage)

    dashboard_dir = cfg.project_root / "dashboard"
    dashboard_dir.mkdir(exist_ok=True)
    paths = report.write_outputs(
        scored, sites, counties, cfg.path("outputs"), dashboard_dir, meta
    )

    if getattr(args, "build_web", False):
        web_dir = cfg.project_root / "web"
        web_paths = report.write_web_bundle(
            scored, sites, counties, web_dir, meta, coverage=coverage,
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
    scored_dir = cfg.path("processed") / "scored"
    sites_dir = cfg.path("processed") / "sites"
    coverage_dir = cfg.path("processed") / "coverage"
    scored = report.load_accumulated_scored(scored_dir)

    if scored.empty:
        log.error("no accumulated scored data in %s — run batches first", scored_dir)
        return 1

    scored = explain.add_explanations(scored)

    # Merge all site batches (dedupe on lat/lng/provider/service).
    sites = pd.DataFrame()
    if sites_dir.exists():
        site_parts = [pd.read_parquet(p) for p in sorted(sites_dir.glob("sites_*.parquet"))]
        if site_parts:
            sites = pd.concat(site_parts, ignore_index=True)
            dedup_cols = [c for c in ["lat", "lng", "provider_id", "technology"] if c in sites.columns]
            if dedup_cols:
                sites = sites.drop_duplicates(subset=dedup_cols, keep="last")

    # Infer vintages from batch metadata or config.
    current = scored["batch_current"].dropna().iloc[-1] if "batch_current" in scored.columns else cfg.vintage_current
    prior = scored["batch_prior"].dropna().iloc[-1] if "batch_prior" in scored.columns else cfg.vintage_prior
    states_processed = "all"
    if "batch_states" in scored.columns:
        states_processed = ",".join(sorted(set(scored["batch_states"].dropna().unique())))

    meta = {
        "current": current,
        "prior": prior,
        "providers": ", ".join(
            scored.drop_duplicates("provider_id")["provider_name"].astype(str).tolist()
        ),
        "technologies": ", ".join(sorted(scored["technology"].unique())),
        "states_processed": states_processed,
    }

    coverage = report.load_accumulated_coverage(coverage_dir)
    counties = normalize.load_counties(cfg)
    web_dir = cfg.project_root / "web"
    paths = report.write_web_bundle(
        scored, sites, counties, web_dir, meta, coverage=coverage,
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
                except (FileNotFoundError, RuntimeError) as exc:
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
        feats, _ = process_provider(
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
        "--states", default=None,
        help='comma-separated state FIPS codes to scope this batch (e.g. "01,02,48")',
    )
    p_run.add_argument(
        "--cleanup-raw", action="store_true",
        help="delete each provider's raw download after processing (bounds disk use)",
    )
    p_run.add_argument(
        "--build-web", action="store_true",
        help="also rebuild the static web bundle after this batch",
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
    p_dl.add_argument("--states", default=None, help="comma-separated state FIPS codes")
    p_dl.set_defaults(func=cmd_download)

    sub.add_parser("build-web", help="assemble static web bundle from accumulated batches").set_defaults(
        func=cmd_build_web
    )
    sub.add_parser("make-fixtures", help="generate synthetic offline data").set_defaults(
        func=cmd_make_fixtures
    )
    sub.add_parser(
        "benchmark", help="check against FCC's labeled selected/not-selected counties"
    ).set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    cfg = load_config(args.config)
    if args.backend:
        cfg.raw["source"]["backend"] = args.backend
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
