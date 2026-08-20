"""On-demand county detail HTTP server (the "cook").

Overnight ``run`` writes ingredients: per-state coverage parquet, batch sites
parquet, scored parquet, and the TIGER county GeoPackage. A county click is
``GET /api/county?...`` — Python slices those files and returns the JSON the
map already knows. Towers are never re-inferred on a request.

Static files under ``web/`` are served as usual. This is not a live Redshift
query.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import geopandas as gpd
import pandas as pd

from .config import Config, load_config
from .normalize import load_counties
from .webbundle import build_county_detail, apply_scored_tower_counts

log = logging.getLogger(__name__)


def _run_key(cfg: Config, current: str, prior: str) -> str:
    def safe(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")

    return f"{safe(cfg.backend)}_{safe(current)}_vs_{safe(prior)}"


def run_dir_for(cfg: Config) -> Path:
    return cfg.path("processed") / _run_key(
        cfg, str(cfg.vintage_current), str(cfg.vintage_prior)
    )


def batch_paths_for_state(directory: Path, glob_prefix: str, state: str) -> list[Path]:
    """Parquet whose overnight batch key includes *state* (e.g. ``sites_20-31.parquet``)."""
    if not directory.exists():
        return []
    state = str(state).zfill(2)
    hits: list[Path] = []
    for path in sorted(directory.glob(f"{glob_prefix}*.parquet")):
        key = path.stem.rsplit("_", 1)[-1]
        parts = key.split("-")
        if key == "all" or state in parts:
            hits.append(path)
    return hits


def _read_parquet_filtered(path: Path, filters: list[tuple]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path, filters=filters)
    except Exception:  # noqa: BLE001 — pyarrow filter pushdown is optional
        df = pd.read_parquet(path)
        for col, op, val in filters:
            if col not in df.columns or op != "==":
                continue
            if col in ("provider_id",):
                df = df[df[col].astype(int) == int(val)]
            else:
                df = df[df[col].astype(str) == str(val)]
        return df


def load_county_coverage(
    run_dir: Path,
    geoid: str,
    provider_id: int,
    service: str,
) -> pd.DataFrame:
    """Load hex rows for one county from ``coverage/coverage_<state>.parquet``."""
    state = str(geoid).zfill(5)[:2]
    path = run_dir / "coverage" / f"coverage_{state}.parquet"
    filters = [
        ("county_geoid", "==", str(geoid)),
        ("provider_id", "==", int(provider_id)),
        ("technology", "==", str(service)),
    ]
    df = _read_parquet_filtered(path, filters)
    if df.empty:
        return df
    if "county_geoid" in df.columns:
        df = df[df["county_geoid"].astype(str) == str(geoid)]
    if "provider_id" in df.columns:
        df = df[df["provider_id"].astype(int) == int(provider_id)]
    if "technology" in df.columns:
        df = df[df["technology"].astype(str) == str(service)]
    return df.reset_index(drop=True)


def load_county_sites(
    run_dir: Path,
    geoid: str,
    provider_id: int,
    service: str,
) -> pd.DataFrame:
    sites_dir = run_dir / "sites"
    state = str(geoid).zfill(5)[:2]
    paths = batch_paths_for_state(sites_dir, "sites_", state)
    if not paths:
        return pd.DataFrame()
    parts = []
    for path in paths:
        df = _read_parquet_filtered(
            path,
            [
                ("provider_id", "==", int(provider_id)),
                ("technology", "==", str(service)),
            ],
        )
        if df.empty:
            continue
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    sites = pd.concat(parts, ignore_index=True)
    if "provider_id" in sites.columns:
        sites = sites[sites["provider_id"].astype(int) == int(provider_id)]
    if "technology" in sites.columns:
        sites = sites[sites["technology"].astype(str) == str(service)]
    return sites.reset_index(drop=True)


def load_scored_row(
    run_dir: Path,
    geoid: str,
    provider_id: int,
    service: str,
) -> dict[str, Any] | None:
    scored_dir = run_dir / "scored"
    state = str(geoid).zfill(5)[:2]
    paths = batch_paths_for_state(scored_dir, "scored_", state)
    if not paths:
        return None
    for path in paths:
        df = _read_parquet_filtered(
            path,
            [
                ("county_geoid", "==", str(geoid)),
                ("provider_id", "==", int(provider_id)),
                ("technology", "==", str(service)),
            ],
        )
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        return row
    return None


def county_slice(
    cfg: Config,
    geoid: str,
    provider_id: int,
    service: str,
    *,
    counties: gpd.GeoDataFrame | None = None,
) -> dict[str, Any]:
    """Extract one county's detail payload (GeoPackage boundary + parquet hexes)."""
    geoid = str(geoid).zfill(5)
    run_dir = run_dir_for(cfg)
    if counties is None:
        counties = load_counties(cfg)
    one = counties[counties["county_geoid"].astype(str) == geoid]
    coverage = load_county_coverage(run_dir, geoid, provider_id, service)
    sites = load_county_sites(run_dir, geoid, provider_id, service)
    meta = {
        "current": cfg.vintage_current,
        "prior": cfg.vintage_prior,
    }
    detail = build_county_detail(geoid, coverage, sites, meta, counties=one)
    row = load_scored_row(run_dir, geoid, provider_id, service)
    apply_scored_tower_counts(detail, row)
    detail["source"] = "api"
    return detail


def make_handler(cfg: Config, web_dir: Path):
    """Build a GET handler bound to *cfg* and the static ``web/`` directory."""

    @lru_cache(maxsize=1)
    def _counties() -> gpd.GeoDataFrame:
        return load_counties(cfg)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web_dir), **kwargs)

        def log_message(self, fmt: str, *args: Any) -> None:
            log.info("%s - " + fmt, self.address_string(), *args)

        def do_GET(self) -> None:  # noqa: N802 — stdlib handler name
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path == "/api/health":
                run_dir = run_dir_for(cfg)
                self._json(200, {
                    "ok": True,
                    "run_dir": str(run_dir),
                    "ingredients": run_dir.exists(),
                })
                return
            if path == "/api/county":
                self._serve_county(parse_qs(parsed.query))
                return
            super().do_GET()

        def _serve_county(self, qs: dict[str, list[str]]) -> None:
            geoid = (qs.get("geoid") or [""])[0].strip()
            provider = (qs.get("provider") or [""])[0].strip()
            service = unquote((qs.get("service") or [""])[0].strip())
            if not (geoid and provider and service):
                self._json(400, {"error": "geoid, provider, and service are required"})
                return
            run_dir = run_dir_for(cfg)
            if not run_dir.exists():
                self._json(404, {"error": f"processed ingredients missing: {run_dir}"})
                return
            try:
                pid = int(provider)
            except ValueError:
                self._json(400, {"error": "provider must be an integer id"})
                return
            try:
                payload = county_slice(cfg, geoid, pid, service, counties=_counties())
            except Exception as exc:  # noqa: BLE001
                log.exception("county extract failed")
                self._json(500, {"error": str(exc)})
                return
            self._json(200, payload)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "private, max-age=300")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8000) -> None:
    web_dir = cfg.project_root / "web"
    handler = make_handler(cfg, web_dir)
    httpd = ThreadingHTTPServer((host, port), handler)
    log.info("serving %s at http://%s:%s  (county API: GET /api/county)", web_dir, host, port)
    print(f"Open http://{host}:{port}/")
    print("County detail: GET /api/county?geoid=20155&provider=130077&service=5G-NR%207/1")
    httpd.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve web/ and cook one-county JSON from processed parquet + GeoPackage",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    serve(load_config(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
