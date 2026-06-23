"""Command-line orchestration for the coverage-change audit pipeline.

Usage:
    python -m fcc_audit.cli make-fixtures        # generate offline synthetic data
    python -m fcc_audit.cli list-vintages        # list available FCC vintages
    python -m fcc_audit.cli run                   # full pipeline (auto vintages)
    python -m fcc_audit.cli run --current 2025-12-31 --prior 2025-06-30
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from . import attribute, changedetect, normalize, report, score, towers
from .acquire import DataSource, get_source
from .config import Config, Provider, load_config


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _fetch_tech_files(source: DataSource, cfg: Config, provider: Provider, vintage: str) -> dict:
    """Fetch one coverage file per technology for a provider+vintage."""
    log = logging.getLogger(__name__)
    files: dict[str, object] = {}
    for tech in cfg.technologies:
        try:
            files[tech] = source.fetch(provider.id, tech, vintage)
        except (FileNotFoundError, RuntimeError) as exc:
            log.warning("skip %s %s @ %s: %s", provider.name, tech, vintage, exc)
    return files


def _analyze_unit(
    cfg, provider, tech, tier_label, tier_spec, env_label, env_codes,
    cur_file, pri_file, counties, county_area_km2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run change/site/attribute/scoring features for one (provider, tech, tier,
    env-group) analysis unit. Returns (features, current_sites)."""
    log = logging.getLogger(__name__)
    county_res = int(cfg.geography["county_h3_resolution"])
    site_res = int(cfg.geography["site_h3_resolution"])

    cur8 = normalize.normalize_layer(cfg, cur_file, counties, county_res, tier_label, tier_spec, env_label, env_codes)
    pri8 = normalize.normalize_layer(cfg, pri_file, counties, county_res, tier_label, tier_spec, env_label, env_codes)
    if cur8.empty and pri8.empty:
        return pd.DataFrame(), pd.DataFrame()
    change = changedetect.hex_change(pri8, cur8)
    cc = changedetect.county_change(change, county_res, county_area_km2)

    cur9 = normalize.normalize_layer(cfg, cur_file, counties, site_res, tier_label, tier_spec, env_label, env_codes)
    pri9 = normalize.normalize_layer(cfg, pri_file, counties, site_res, tier_label, tier_spec, env_label, env_codes)
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
        "technology": tech, "speed_tier": tier_label, "environment": env_label,
    }
    if not feats.empty:
        for k, v in tag.items():
            feats[k] = v
    if not current_sites.empty:
        current_sites = current_sites.copy()
        for k, v in tag.items():
            current_sites[k] = v
    log.info(
        "  %s %s/%s/%s: %d counties changed, %d current sites",
        provider.name, tech, tier_label, env_label,
        0 if feats.empty else len(feats), len(current_sites),
    )
    return feats, current_sites


def process_provider(
    cfg: Config,
    source: DataSource,
    provider: Provider,
    current: str,
    prior: str,
    counties,
    county_area_km2: dict | None = None,
    cleanup_raw: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = logging.getLogger(__name__)
    log.info("=== %s (id=%s) ===", provider.name, provider.id)

    cur_files = _fetch_tech_files(source, cfg, provider, current)
    pri_files = _fetch_tech_files(source, cfg, provider, prior)

    feats_parts, sites_parts = [], []
    env_groups = cfg.environment_groups()
    for tech, tier_label, tier_spec in cfg.analysis_units():
        cur_file = cur_files.get(tech)
        pri_file = pri_files.get(tech)
        if cur_file is None or pri_file is None:
            continue
        for env_label, env_codes in env_groups:
            feats, sites = _analyze_unit(
                cfg, provider, tech, tier_label, tier_spec, env_label, env_codes,
                cur_file, pri_file, counties, county_area_km2,
            )
            if not feats.empty:
                feats_parts.append(feats)
            if not sites.empty:
                sites_parts.append(sites)

    if cleanup_raw:
        # Free disk between providers: the compact interim hex parquet is kept,
        # so only the large raw source files are removed.
        import os
        for f in (*cur_files.values(), *pri_files.values()):
            try:
                os.remove(f.local_path)
            except OSError:
                pass

    feats = pd.concat(feats_parts, ignore_index=True) if feats_parts else pd.DataFrame()
    sites = pd.concat(sites_parts, ignore_index=True) if sites_parts else pd.DataFrame()
    return feats, sites


def _resolve_providers(cfg: Config, source: DataSource, vintage: str) -> list[Provider]:
    """Explicit provider list, or auto-discovered from the catalog when 'all'."""
    if cfg.providers_all:
        providers = source.list_providers(vintage)
        logging.getLogger(__name__).info("discovered %d providers for %s", len(providers), vintage)
        return providers
    return cfg.providers


def cmd_run(cfg: Config, args) -> int:
    log = logging.getLogger(__name__)
    source = get_source(cfg)
    current, prior = source.resolve_vintages(
        args.current or cfg.vintage_current, args.prior or cfg.vintage_prior
    )
    log.info("comparing current=%s vs prior=%s", current, prior)

    counties = normalize.load_counties(cfg)
    county_area_km2 = normalize.county_areas_km2(counties, cfg.geography["equal_area_crs"])

    providers = _resolve_providers(cfg, source, current)
    # Never delete fixture source files (they are inputs, not throwaway downloads).
    cleanup_raw = bool(getattr(args, "cleanup_raw", False)) and cfg.raw["source"]["backend"] != "fixture"
    all_feats, all_sites = [], []
    for provider in providers:
        feats, sites = process_provider(
            cfg, source, provider, current, prior, counties, county_area_km2, cleanup_raw
        )
        if not feats.empty:
            all_feats.append(feats)
        if not sites.empty:
            all_sites.append(sites)

    if not all_feats:
        log.error("no features produced; nothing to score")
        return 1

    features = pd.concat(all_feats, ignore_index=True)
    scored = score.score(features, cfg)
    sites = pd.concat(all_sites, ignore_index=True) if all_sites else pd.DataFrame()

    dashboard_dir = cfg.project_root / "dashboard"
    dashboard_dir.mkdir(exist_ok=True)
    meta = {
        "current": current,
        "prior": prior,
        "providers": ", ".join(p.name for p in providers),
        "technologies": ", ".join(f"{t}/{tier}" for t, tier, _ in cfg.analysis_units()),
    }
    paths = report.write_outputs(
        scored, sites, counties, cfg.path("outputs"), dashboard_dir, meta
    )
    flagged = int(scored["flag_for_review"].sum())
    log.info("DONE: %d provider-county rows, %d flagged", len(scored), flagged)
    print("\nOutputs:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print(f"\nFlagged for review: {flagged}/{len(scored)}")
    print(f"Open the dashboard: {dashboard_dir / 'index.html'}")
    return 0


def cmd_list_vintages(cfg: Config, args) -> int:
    source = get_source(cfg)
    for v in source.list_vintages():
        print(v)
    return 0


def cmd_download(cfg: Config, args) -> int:
    """Pre-stage raw coverage files from the FCC API without running analysis.

    Useful to pull everything in one pass (e.g. onto an external drive) before a
    later offline `run`. Already-downloaded files are skipped, so it's resumable.
    """
    log = logging.getLogger(__name__)
    source = get_source(cfg)
    current, prior = source.resolve_vintages(
        args.current or cfg.vintage_current, args.prior or cfg.vintage_prior
    )
    providers = _resolve_providers(cfg, source, current)
    techs = list(cfg.technologies.keys())
    total_bytes = 0
    n_files = n_skipped = 0
    log.info("downloading %d providers x %d techs x 2 vintages", len(providers), len(techs))
    for provider in providers:
        for vintage in (current, prior):
            for tech in techs:
                try:
                    cov = source.fetch(provider.id, tech, vintage)
                except (FileNotFoundError, RuntimeError) as exc:
                    log.warning("skip %s %s @ %s: %s", provider.name, tech, vintage, exc)
                    n_skipped += 1
                    continue
                size = cov.local_path.stat().st_size if cov.local_path.exists() else 0
                total_bytes += size
                n_files += 1
                log.info("ok %s %s @ %s (%.1f MB)", provider.name, tech, vintage, size / 1e6)
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
    """Check the pipeline reproduces the FCC's selected/not-selected examples.

    Runs the comparison for the benchmark vintages, then for each labeled county
    compares whether ANY provider was flagged there against the FCC's decision.
    """
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
    # Scope the comparison to the benchmark's service (e.g. 5G-NR 7/1), so flags
    # from other technologies/tiers don't count as matches.
    bench_tech = bench.get("technology", "5G-NR")
    bench_tier = bench.get("speed_tier")
    if "technology" in scored and bench_tech:
        scored = scored[scored["technology"] == bench_tech]
    if "speed_tier" in scored and bench_tier:
        scored = scored[scored["speed_tier"] == bench_tier]
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
        "--cleanup-raw", action="store_true",
        help="delete each provider's raw download after processing (bounds disk use)",
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
    p_dl.set_defaults(func=cmd_download)
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
