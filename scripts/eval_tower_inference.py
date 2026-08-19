"""County-level tower-inference accuracy against FCC ASR + cross-provider colocation.

Downloads are expected under data/raw/<vintage>/<provider_id>/ already
(see FccDownloadSource). Hex caches live in data/eval/hex/.
"""
from __future__ import annotations

import h3
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fcc_audit.acquire import FccDownloadSource, safe_service_name  # noqa: E402
from fcc_audit.config import load_config  # noqa: E402
from fcc_audit.groundtruth_asr import load_asr_structures  # noqa: E402
from fcc_audit.normalize import coverage_to_hex, detect_signal_column, load_counties  # noqa: E402
from fcc_audit.towers import infer_sites, _n_sector_signature, _FWD  # noqa: E402

log = logging.getLogger("eval_towers")

VINTAGE = "December 31, 2025"
SERVICE = "5G-NR (35/3 Mbps)"
PROVIDERS = [(130403, "T-Mobile"), (130077, "AT&T"), (131425, "Verizon")]
# Different regions, each expected to have >10 cell sites.
COUNTIES = [
    ("34013", "Essex, NJ"),
    ("34037", "Sussex, NJ"),
    ("34039", "Union, NJ"),
    ("10003", "New Castle, DE"),
    ("22073", "Ouachita, LA"),
    ("22111", "Union, LA"),
    ("20161", "Riley, KS"),
    ("49005", "Cache, UT"),
]
ASR_MATCH_M = 2000.0
SHARED_MATCH_M = 400.0
FOOTPRINT_M = 2500.0
CLIP_BUFFER_DEG = 0.08  # ~8 km, so border towers still appear
SITE_IN_COUNTY_DEG = 0.08
_INV = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)


def _xy(lats, lngs) -> np.ndarray:
    x, y = _FWD.transform(np.asarray(lngs, dtype=float), np.asarray(lats, dtype=float))
    return np.column_stack([x, y])


def _match(src: np.ndarray, dst: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    if len(src) == 0 or len(dst) == 0:
        inf = np.full(len(src), np.inf)
        return inf, np.zeros(len(src), dtype=bool)
    tree = cKDTree(dst)
    dist, _ = tree.query(src, k=1)
    dist = np.asarray(dist, dtype=float)
    return dist, dist <= radius_m


def _hex_county(
    *,
    cfg,
    zip_path: Path,
    county_geom,
    geoid: str,
    provider_id: int,
    cache_dir: Path,
) -> pd.DataFrame:
    cache = cache_dir / f"{geoid}_{provider_id}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    gdf = FccDownloadSource._read_coverage_zip(zip_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    clip_geom = county_geom.buffer(CLIP_BUFFER_DEG)
    gdf = gdf[gdf.intersects(clip_geom)].copy()
    if gdf.empty:
        out = pd.DataFrame(columns=["h3", "signal_dbm", "county_geoid"])
        out.to_parquet(cache, index=False)
        return out
    import geopandas as gpd

    gdf = gpd.clip(gdf, clip_geom)
    signal_col = detect_signal_column(gdf)
    hex_df = coverage_to_hex(gdf, 9, signal_col)
    hex_df["county_geoid"] = geoid
    hex_df.to_parquet(cache, index=False)
    return hex_df


def _lobe_counts(sites: pd.DataFrame, hex_df: pd.DataFrame) -> dict[str, int]:
    counts = {"1": 0, "2": 0, "3": 0, "other": 0}
    if sites.empty or hex_df.empty:
        return counts
    ll = np.array([h3.cell_to_latlng(c) for c in hex_df["h3"].astype(str)])
    xs, ys = _FWD.transform(ll[:, 1], ll[:, 0])
    for rec in sites.itertuples(index=False):
        sx, sy = _FWD.transform(rec.lng, rec.lat)
        reach = float(getattr(rec, "reach_m", 8000) or 8000)
        d2 = (xs - sx) ** 2 + (ys - sy) ** 2
        mask = d2 <= (reach * 1.15) ** 2
        if int(mask.sum()) < 24:
            counts["1"] += 1
            continue
        r = max(reach, 2500.0)
        if _n_sector_signature(float(sx), float(sy), xs[mask], ys[mask], r * 0.15, r * 1.20, 3):
            counts["3"] += 1
        elif _n_sector_signature(float(sx), float(sy), xs[mask], ys[mask], r * 0.12, r * 1.40, 2):
            counts["2"] += 1
        else:
            counts["1"] += 1
    return counts


def _shared_clusters(sites_by_pid: dict[int, pd.DataFrame]) -> dict:
    rows = []
    for pid, sdf in sites_by_pid.items():
        if sdf.empty:
            continue
        xy = _xy(sdf["lat"], sdf["lng"])
        for i, rec in enumerate(sdf.itertuples(index=False)):
            rows.append((pid, rec.lat, rec.lng, xy[i, 0], xy[i, 1]))
    if len(rows) < 2:
        return {"n_clusters_2plus": 0, "n_sites_in_shared": 0, "median_offset_m": None}
    pts = np.array([(r[3], r[4]) for r in rows])
    pids = np.array([r[0] for r in rows])
    tree = cKDTree(pts)
    used = np.zeros(len(rows), dtype=bool)
    n_shared = 0
    n_sites = 0
    offsets = []
    for i in range(len(rows)):
        if used[i]:
            continue
        members = tree.query_ball_point(pts[i], SHARED_MATCH_M)
        used[members] = True
        uniq = set(int(pids[j]) for j in members)
        if len(uniq) >= 2:
            n_shared += 1
            n_sites += len(members)
            sub = pts[members]
            c = sub.mean(axis=0)
            offsets.extend(np.hypot(sub[:, 0] - c[0], sub[:, 1] - c[1]).tolist())
    return {
        "n_clusters_2plus": n_shared,
        "n_sites_in_shared": n_sites,
        "median_offset_m": (float(np.median(offsets)) if offsets else None),
    }


def evaluate_county(
    *,
    geoid: str,
    name: str,
    county_geom,
    asr_xy: np.ndarray,
    asr_in_county: pd.DataFrame,
    cfg,
    zip_by_pid: dict[int, Path],
    cache_dir: Path,
    flatten: bool,
) -> dict:
    sites_by_pid: dict[int, pd.DataFrame] = {}
    provider_rows = []
    hex_xy_all = []
    for pid, pname in PROVIDERS:
        hex_df = _hex_county(
            cfg=cfg,
            zip_path=zip_by_pid[pid],
            county_geom=county_geom,
            geoid=geoid,
            provider_id=pid,
            cache_dir=cache_dir,
        )
        work = hex_df.copy()
        if flatten and not work.empty:
            work["signal_dbm"] = 0.0
        if not work.empty:
            ll = np.array([h3.cell_to_latlng(c) for c in work["h3"].astype(str)])
            hex_xy_all.append(_xy(ll[:, 0], ll[:, 1]))
        sites = infer_sites(work, cfg, label_prefix=pname[:1]) if not work.empty else work
        # Keep sites inside the county (plus 2 km) so border-buffer extras drop out.
        if not sites.empty:
            sxy = _xy(sites["lat"], sites["lng"])
            keep = []
            for lat, lng in zip(sites["lat"], sites["lng"]):
                keep.append(county_geom.buffer(SITE_IN_COUNTY_DEG).contains(shapely.Point(lng, lat)))
            sites = sites.loc[np.array(keep)].reset_index(drop=True)
            sxy = _xy(sites["lat"], sites["lng"]) if not sites.empty else np.empty((0, 2))
        else:
            sxy = np.empty((0, 2))
        sites_by_pid[pid] = sites

        if len(asr_xy) and len(sxy):
            dist, hit = _match(sxy, asr_xy, ASR_MATCH_M)
            prec = float(hit.mean()) if len(hit) else None
            d2, rec = _match(asr_xy, sxy, ASR_MATCH_M)
            rec_v = float(rec.mean()) if len(rec) else None
            med = float(np.median(dist[hit])) if hit.any() else None
        else:
            prec = rec_v = med = None
            hit = np.array([], dtype=bool)
        lobes = _lobe_counts(sites, work)
        provider_rows.append({
            "provider": pname,
            "provider_id": pid,
            "n_hexes": int(len(work)),
            "n_sites": int(len(sites)),
            "asr_precision": prec,
            "asr_recall": rec_v,
            "median_match_m": med,
            "n_matched": int(hit.sum()) if len(hit) else 0,
            "lobes": lobes,
        })

    hex_pts = np.vstack(hex_xy_all) if hex_xy_all else np.empty((0, 2))
    if len(asr_xy) and len(hex_pts):
        _, in_fp = _match(asr_xy, hex_pts, FOOTPRINT_M)
    else:
        in_fp = np.zeros(len(asr_xy), dtype=bool)

    # Footprint-conditioned ASR recall (union of provider sites vs ASR in coverage).
    all_sites = pd.concat(
        [s.assign(provider_id=pid) for pid, s in sites_by_pid.items() if not s.empty],
        ignore_index=True,
    ) if any(not s.empty for s in sites_by_pid.values()) else pd.DataFrame()
    if not all_sites.empty and in_fp.any():
        sxy = _xy(all_sites["lat"], all_sites["lng"])
        _, rec_fp = _match(asr_xy[in_fp], sxy, ASR_MATCH_M)
        footprint_recall = float(rec_fp.mean())
        n_asr_fp = int(in_fp.sum())
    else:
        footprint_recall = None
        n_asr_fp = int(in_fp.sum())

    shared = _shared_clusters(sites_by_pid)
    n_sites_total = int(sum(r["n_sites"] for r in provider_rows))
    return {
        "geoid": geoid,
        "county": name,
        "flatten_signal": flatten,
        "n_asr_county": int(len(asr_in_county)),
        "n_asr_in_footprint": n_asr_fp,
        "n_sites_all_providers": n_sites_total,
        "asr_recall_in_footprint": footprint_recall,
        "shared": shared,
        "providers": provider_rows,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    cfg.raw["geography"]["site_h3_resolution"] = 9
    raw_root = cfg.path("raw") / VINTAGE
    eval_dir = ROOT / "data" / "eval"
    hex_dir = eval_dir / "hex"
    hex_dir.mkdir(parents=True, exist_ok=True)

    counties = load_counties(cfg)
    asr = load_asr_structures(ROOT / "data" / "groundtruth" / "asr")
    asr_xy_all = _xy(asr["lat"], asr["lng"]) if not asr.empty else np.empty((0, 2))

    safe = safe_service_name(SERVICE)
    zip_by_pid: dict[int, dict[str, Path]] = {}
    for pid, _name in PROVIDERS:
        zip_by_pid[pid] = {}
        for geoid, _label in COUNTIES:
            st = geoid[:2]
            z = raw_root / str(pid) / f"{safe}_{st}.zip"
            if not z.exists():
                raise FileNotFoundError(z)
            zip_by_pid[pid][st] = z

    reports = []
    for flatten in (True, False):
        mode = "binary" if flatten else "signal"
        log.info("=== mode %s ===", mode)
        for geoid, name in COUNTIES:
            row = counties[counties["county_geoid"].astype(str) == geoid]
            if row.empty:
                log.warning("missing county geom %s", geoid)
                continue
            geom = row.iloc[0].geometry
            asr_c = asr[asr["county_geoid"].astype(str) == geoid]
            asr_xy = _xy(asr_c["lat"], asr_c["lng"]) if not asr_c.empty else np.empty((0, 2))
            zips = {pid: zip_by_pid[pid][geoid[:2]] for pid, _ in PROVIDERS}
            log.info("eval %s %s (%s ASR)", geoid, name, len(asr_c))
            rec = evaluate_county(
                geoid=geoid,
                name=name,
                county_geom=geom,
                asr_xy=asr_xy,
                asr_in_county=asr_c,
                cfg=cfg,
                zip_by_pid=zips,
                cache_dir=hex_dir,
                flatten=flatten,
            )
            rec["mode"] = mode
            reports.append(rec)
            log.info(
                "  sites=%s  ASR-fp recall=%s  shared clusters=%s",
                rec["n_sites_all_providers"],
                rec["asr_recall_in_footprint"],
                rec["shared"]["n_clusters_2plus"],
            )

    out = eval_dir / "tower_inference_report.json"
    out.write_text(json.dumps(reports, indent=2, default=str))
    print(json.dumps(reports, indent=2, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
