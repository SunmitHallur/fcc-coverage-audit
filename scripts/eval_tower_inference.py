"""County-level tower-inference accuracy against FCC ASR + cross-provider colocation.

Downloads are expected under data/raw/<vintage>/<provider_id>/ already
(see FccDownloadSource). Hex caches live in data/eval/hex/.
"""
from __future__ import annotations

import argparse
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
# Locations used by the synthetic gate (must not be only the unit-test point).
SYNTH_LOCS = [
    ("unit_ks", 38.50, -98.50),
    ("atlanta", 33.75, -84.39),
    ("seattle", 47.61, -122.33),
    ("manhattan_ks", 39.18, -96.57),
    ("newark", 40.74, -74.17),
    ("monroe_la", 32.51, -92.12),
    ("logan_ut", 41.74, -111.83),
    ("wichita", 37.69, -97.34),
]
# Regression floors for 8-county signal-mode (cell-like ASR). These are
# "do not get worse than the post-T1–T4 report", not a 25% pin-accuracy claim.
GATE_MEDIAN_PREC_250 = 0.05
GATE_MEDIAN_PREC_500 = 0.10
GATE_MAX_CLOSE_PAIR_COLLAPSE = 0.65
ASR_MATCH_M = 2000.0
ASR_RADII_M = (250.0, 500.0, 1000.0, 2000.0)
CELLISH_TYPES = {"TOWER", "MTOWER", "LTOWER", "POLE", "MAST", "GTOWER", "BANT"}
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


def _close_pair_collapse(
    asr_xy: np.ndarray,
    site_xy: np.ndarray,
    pair_radius_m: float = ASR_MATCH_M,
) -> dict:
    """Share of ASR pairs closer than ``pair_radius_m`` that share a nearest site."""
    n = len(asr_xy)
    if n < 2 or len(site_xy) == 0:
        return {"n_pairs": 0, "n_collapsed": 0, "rate": None}
    tree_asr = cKDTree(asr_xy)
    pairs = tree_asr.query_pairs(pair_radius_m)
    if not pairs:
        return {"n_pairs": 0, "n_collapsed": 0, "rate": None}
    tree_s = cKDTree(site_xy)
    _, nn = tree_s.query(asr_xy, k=1)
    nn = np.asarray(nn, dtype=int)
    collapsed = sum(1 for i, j in pairs if nn[i] == nn[j])
    n_pairs = len(pairs)
    return {
        "n_pairs": int(n_pairs),
        "n_collapsed": int(collapsed),
        "rate": float(collapsed / n_pairs),
    }


def _match_at_radii(
    src: np.ndarray, dst: np.ndarray, radii: tuple[float, ...] = ASR_RADII_M
) -> dict:
    """Precision-style hit rates for each radius (fraction of src within radius of dst)."""
    if len(src) == 0:
        return {
            str(int(r)): {"n": 0, "hit": 0, "rate": None, "median_hit_m": None}
            for r in radii
        }
    if len(dst) == 0:
        return {
            str(int(r)): {"n": int(len(src)), "hit": 0, "rate": 0.0, "median_hit_m": None}
            for r in radii
        }
    tree = cKDTree(dst)
    dist, _ = tree.query(src, k=1)
    dist = np.asarray(dist, dtype=float)
    out = {}
    for r in radii:
        hit = dist <= r
        out[str(int(r))] = {
            "n": int(len(src)),
            "hit": int(hit.sum()),
            "rate": float(hit.mean()),
            "median_hit_m": float(np.median(dist[hit])) if hit.any() else None,
        }
    return out


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
    asr_cell_xy: np.ndarray,
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
            keep = []
            for lat, lng in zip(sites["lat"], sites["lng"]):
                keep.append(county_geom.buffer(SITE_IN_COUNTY_DEG).contains(shapely.Point(lng, lat)))
            sites = sites.loc[np.array(keep)].reset_index(drop=True)
            sxy = _xy(sites["lat"], sites["lng"]) if not sites.empty else np.empty((0, 2))
        else:
            sxy = np.empty((0, 2))
        sites_by_pid[pid] = sites

        # Backward-compatible 2 km metrics against all constructed ASR.
        if len(asr_xy) and len(sxy):
            dist, hit = _match(sxy, asr_xy, ASR_MATCH_M)
            prec = float(hit.mean()) if len(hit) else None
            d2, rec = _match(asr_xy, sxy, ASR_MATCH_M)
            rec_v = float(rec.mean()) if len(rec) else None
            med = float(np.median(dist[hit])) if hit.any() else None
        else:
            prec = rec_v = med = None
            hit = np.array([], dtype=bool)

        prec_all = _match_at_radii(sxy, asr_xy)
        rec_all = _match_at_radii(asr_xy, sxy)
        prec_cell = _match_at_radii(sxy, asr_cell_xy)
        rec_cell = _match_at_radii(asr_cell_xy, sxy)
        collapse = _close_pair_collapse(asr_cell_xy, sxy)

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
            "precision_by_radius_m": {k: v["rate"] for k, v in prec_all.items()},
            "recall_by_radius_m": {k: v["rate"] for k, v in rec_all.items()},
            "cellish_precision_by_radius_m": {k: v["rate"] for k, v in prec_cell.items()},
            "cellish_recall_by_radius_m": {k: v["rate"] for k, v in rec_cell.items()},
            "close_pair_collapse": collapse,
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

    if len(asr_cell_xy) and len(hex_pts):
        _, cell_in_fp = _match(asr_cell_xy, hex_pts, FOOTPRINT_M)
    else:
        cell_in_fp = np.zeros(len(asr_cell_xy), dtype=bool)
    if not all_sites.empty and cell_in_fp.any():
        sxy = _xy(all_sites["lat"], all_sites["lng"])
        _, rec_cell_fp = _match(asr_cell_xy[cell_in_fp], sxy, ASR_MATCH_M)
        cellish_footprint_recall = float(rec_cell_fp.mean())
        n_asr_cell_fp = int(cell_in_fp.sum())
    else:
        cellish_footprint_recall = None
        n_asr_cell_fp = int(cell_in_fp.sum())

    shared = _shared_clusters(sites_by_pid)
    n_sites_total = int(sum(r["n_sites"] for r in provider_rows))
    if not all_sites.empty:
        all_xy = _xy(all_sites["lat"], all_sites["lng"])
        county_collapse = _close_pair_collapse(asr_cell_xy, all_xy)
    else:
        county_collapse = {"n_pairs": 0, "n_collapsed": 0, "rate": None}
    return {
        "geoid": geoid,
        "county": name,
        "flatten_signal": flatten,
        "n_asr_county": int(len(asr_in_county)),
        "n_asr_cellish": int(len(asr_cell_xy)),
        "n_asr_in_footprint": n_asr_fp,
        "n_asr_cellish_in_footprint": n_asr_cell_fp,
        "n_sites_all_providers": n_sites_total,
        "asr_recall_in_footprint": footprint_recall,
        "cellish_recall_in_footprint": cellish_footprint_recall,
        "close_pair_collapse": county_collapse,
        "shared": shared,
        "providers": provider_rows,
    }


def _petal_cloverleaf_hex(lat: float, lng: float) -> pd.DataFrame:
    hub = h3.latlng_to_cell(lat, lng, 9)
    cells = set(h3.grid_disk(hub, 4))
    origins = []
    for ang in (0.0, 120.0, 240.0):
        dlat = (4.0 / 110.57) * np.cos(np.radians(ang))
        dlng = (4.0 / (111.32 * np.cos(np.radians(lat)))) * np.sin(np.radians(ang))
        origin = h3.latlng_to_cell(lat + dlat, lng + dlng, 9)
        origins.append(origin)
        cells |= set(h3.grid_disk(origin, 10))
    rows = []
    for c in cells:
        d_pet = min(h3.grid_distance(o, c) for o in origins)
        rows.append({"h3": c, "signal_dbm": -78.0 - 2.5 * d_pet, "county_geoid": "20001"})
    return pd.DataFrame(rows)


def _signal_pair_hex(lat: float, lng: float, sep_km: float, ring: int = 10) -> pd.DataFrame:
    t1 = (lat, lng)
    t2 = (lat, lng + sep_km / (111.32 * np.cos(np.radians(lat))))
    rows = []
    for la, lo in (t1, t2):
        origin = h3.latlng_to_cell(la, lo, 9)
        for c in h3.grid_disk(origin, ring):
            d = h3.grid_distance(origin, c)
            rows.append({"h3": c, "signal_dbm": -80.0 - 2.0 * d, "county_geoid": "20001"})
    return pd.DataFrame(rows).sort_values("signal_dbm", ascending=False).drop_duplicates("h3")


def run_synthetic_gates(cfg) -> dict:
    """Location-general T2/T3 checks that do not need coverage zips."""
    cfg.raw["geography"]["site_h3_resolution"] = 9
    t2 = []
    for name, lat, lng in SYNTH_LOCS:
        n = int(len(infer_sites(_petal_cloverleaf_hex(lat, lng), cfg, "T")))
        t2.append({"location": name, "n_sites": n, "ok": n == 1})
    t3 = []
    for sep in (0.4, 0.8, 1.6):
        expect = 1 if sep < 0.6 else 2
        for name, lat, lng in SYNTH_LOCS:
            n = int(len(infer_sites(_signal_pair_hex(lat, lng, sep), cfg, "T")))
            t3.append({
                "location": name, "sep_km": sep, "n_sites": n,
                "ok": n == expect,
            })
    t2_pass = sum(1 for r in t2 if r["ok"])
    t3_08 = sum(1 for r in t3 if r["sep_km"] == 0.8 and r["ok"])
    t3_16 = sum(1 for r in t3 if r["sep_km"] == 1.6 and r["ok"])
    t3_04 = sum(1 for r in t3 if r["sep_km"] == 0.4 and r["ok"])
    failures = []
    if t2_pass < len(SYNTH_LOCS):
        failures.append(f"T2 cloverleaf {t2_pass}/{len(SYNTH_LOCS)} merge to 1 site")
    if t3_08 < len(SYNTH_LOCS):
        failures.append(f"T3 0.8 km {t3_08}/{len(SYNTH_LOCS)} split to 2 sites")
    if t3_16 < len(SYNTH_LOCS):
        failures.append(f"T3 1.6 km {t3_16}/{len(SYNTH_LOCS)} split to 2 sites")
    return {
        "t2": t2,
        "t3": t3,
        "t2_pass": t2_pass,
        "t3_0p8_pass": t3_08,
        "t3_1p6_pass": t3_16,
        "t3_0p4_stay_one": t3_04,
        "n_locs": len(SYNTH_LOCS),
        "failures": failures,
        "ok": not failures,
    }


def _gate_county_reports(reports: list[dict]) -> list[str]:
    """Fail on 250/500 m cellish precision and close-pair collapse, not 2 km recall."""
    sig = [r for r in reports if r.get("mode") == "signal"]
    if not sig:
        return ["no signal-mode county rows to gate"]
    prec250, prec500, collapse = [], [], []
    for r in sig:
        if r.get("close_pair_collapse") and r["close_pair_collapse"].get("rate") is not None:
            collapse.append(r["close_pair_collapse"]["rate"])
        for p in r.get("providers") or []:
            c250 = (p.get("cellish_precision_by_radius_m") or {}).get("250")
            c500 = (p.get("cellish_precision_by_radius_m") or {}).get("500")
            if c250 is not None:
                prec250.append(c250)
            if c500 is not None:
                prec500.append(c500)
    fails = []
    if prec250:
        med = float(np.median(prec250))
        if med < GATE_MEDIAN_PREC_250:
            fails.append(
                f"median cellish 250 m precision {med:.3f} < {GATE_MEDIAN_PREC_250}"
            )
    else:
        fails.append("no 250 m cellish precision values")
    if prec500:
        med = float(np.median(prec500))
        if med < GATE_MEDIAN_PREC_500:
            fails.append(
                f"median cellish 500 m precision {med:.3f} < {GATE_MEDIAN_PREC_500}"
            )
    if collapse:
        med = float(np.median(collapse))
        if med > GATE_MAX_CLOSE_PAIR_COLLAPSE:
            fails.append(
                f"median close-pair collapse {med:.3f} > {GATE_MAX_CLOSE_PAIR_COLLAPSE}"
            )
    return fails


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Tower-inference eval + gate")
    parser.add_argument(
        "--synthetics-only", action="store_true",
        help="Run T2/T3 location synthetics only (no coverage zips).",
    )
    parser.add_argument(
        "--gate", action="store_true",
        help="Exit 1 if synthetic or 250/500 m / collapse thresholds fail.",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    cfg.raw["geography"]["site_h3_resolution"] = 9
    eval_dir = ROOT / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    synth = run_synthetic_gates(cfg)
    log.info(
        "synthetics T2=%s/%s T3_0.8=%s/%s T3_1.6=%s/%s",
        synth["t2_pass"], synth["n_locs"],
        synth["t3_0p8_pass"], synth["n_locs"],
        synth["t3_1p6_pass"], synth["n_locs"],
    )
    (eval_dir / "tower_inference_synthetics.json").write_text(
        json.dumps(synth, indent=2)
    )

    reports: list[dict] = []
    county_fails: list[str] = []
    if not args.synthetics_only:
        raw_root = cfg.path("raw") / VINTAGE
        hex_dir = eval_dir / "hex"
        hex_dir.mkdir(parents=True, exist_ok=True)
        counties = load_counties(cfg)
        asr = load_asr_structures(ROOT / "data" / "groundtruth" / "asr")
        safe = safe_service_name(SERVICE)
        zip_by_pid: dict[int, dict[str, Path]] = {}
        missing_zips = []
        for pid, _name in PROVIDERS:
            zip_by_pid[pid] = {}
            for geoid, _label in COUNTIES:
                st = geoid[:2]
                z = raw_root / str(pid) / f"{safe}_{st}.zip"
                if not z.exists():
                    missing_zips.append(str(z))
                else:
                    zip_by_pid[pid][st] = z
        if missing_zips:
            log.warning("skipping county eval, missing zips: %s", missing_zips[:3])
        else:
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
                    if not asr_c.empty and "structure_type" in asr_c.columns:
                        asr_cell = asr_c[asr_c["structure_type"].isin(CELLISH_TYPES)]
                    else:
                        asr_cell = asr_c
                    asr_cell_xy = (
                        _xy(asr_cell["lat"], asr_cell["lng"]) if not asr_cell.empty else np.empty((0, 2))
                    )
                    zips = {pid: zip_by_pid[pid][geoid[:2]] for pid, _ in PROVIDERS}
                    log.info("eval %s %s (%s ASR, %s cellish)", geoid, name, len(asr_c), len(asr_cell))
                    rec = evaluate_county(
                        geoid=geoid,
                        name=name,
                        county_geom=geom,
                        asr_xy=asr_xy,
                        asr_cell_xy=asr_cell_xy,
                        asr_in_county=asr_c,
                        cfg=cfg,
                        zip_by_pid=zips,
                        cache_dir=hex_dir,
                        flatten=flatten,
                    )
                    rec["mode"] = mode
                    reports.append(rec)
                    log.info(
                        "  sites=%s  ASR-fp recall=%s  collapse=%s",
                        rec["n_sites_all_providers"],
                        rec["asr_recall_in_footprint"],
                        rec.get("close_pair_collapse"),
                    )
            out = eval_dir / "tower_inference_report.json"
            out.write_text(json.dumps(reports, indent=2, default=str))
            print(json.dumps(reports, indent=2, default=str))
            print(f"wrote {out}")
            county_fails = _gate_county_reports(reports)
    # County 250/500 m + collapse gate only when we actually scored counties.

    print(json.dumps({"synthetics": {
        k: synth[k] for k in (
            "t2_pass", "t3_0p8_pass", "t3_1p6_pass", "t3_0p4_stay_one",
            "n_locs", "failures", "ok",
        )
    }, "county_gate_failures": county_fails}, indent=2))

    if args.gate:
        fails = list(synth["failures"]) + county_fails
        if fails:
            log.error("GATE FAIL: %s", "; ".join(fails))
            return 1
        log.info("GATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
