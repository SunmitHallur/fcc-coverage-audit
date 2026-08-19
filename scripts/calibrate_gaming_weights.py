#!/usr/bin/env python3
"""Calibrate June→December gaming weights on live Verizon 7/1 filings.

Builds features for every county in the states we have cached, then searches
score weights / implausibility gates. Slide labels are a check, not a freeze:
new-tower buildout is never a same-site gaming target even if an old slide
selected the county.

Prefers zero false flags on skip-like physics over catching every selected slide.
"""
from __future__ import annotations

import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fcc_audit.acquire import FccDownloadSource, safe_service_name
from fcc_audit.attribute import anchor_sites_to_asr, attribute_changes
from fcc_audit.changedetect import county_change, hex_change
from fcc_audit.config import load_config
from fcc_audit.groundtruth_asr import load_asr_structures
from fcc_audit.normalize import (
    assign_counties,
    county_areas_km2,
    coverage_to_hex,
    detect_signal_column,
    load_counties,
)
from fcc_audit.score import build_features, score
from fcc_audit.towers import infer_sites_joint

log = logging.getLogger("calibrate")
PID = 131425
SERVICE = "5G-NR (7/1 Mbps)"
PRIOR = "June 30, 2025"
CUR = "December 31, 2025"
HEX_DIR = ROOT / "data/eval/hex_states"
OUT_DIR = ROOT / "data/eval"

# Ranking cohort: hex the whole state so percentile ranking is real (N>>50).
FULL_STATES = ["25", "28", "46", "16", "08", "20"]
# Large zips: only clip the labeled counties instead of polyfilling the state.
LABELED_ONLY_STATES = ["48", "36", "40"]


def _zip_path(vintage: str, st: str, safe: str) -> Path:
    return ROOT / "data/raw" / vintage / str(PID) / f"{safe}_{st}.zip"


def hex_state(vintage: str, st: str, safe: str, counties, cfg) -> pd.DataFrame:
    HEX_DIR.mkdir(parents=True, exist_ok=True)
    cache = HEX_DIR / f"{st}_{vintage.replace(' ', '_')}.parquet"
    if cache.exists() and cache.stat().st_size > 1000:
        log.info("hex cache hit %s", cache.name)
        return pd.read_parquet(cache)
    z = _zip_path(vintage, st, safe)
    if not z.exists() or z.stat().st_size < 10_000:
        raise FileNotFoundError(z)
    log.info("hex state %s %s from %s (%.0f MB)", st, vintage, z.name, z.stat().st_size / 1e6)
    gdf = FccDownloadSource._read_coverage_zip(z)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    df = coverage_to_hex(gdf, 9, detect_signal_column(gdf))
    df = assign_counties(
        df, counties, clip_to_states=[st],
        buffer_m=20_000.0,
        equal_area_crs=cfg.geography["equal_area_crs"],
    )
    df.to_parquet(cache, index=False)
    log.info("  wrote %s rows=%s", cache.name, f"{len(df):,}")
    return df


def features_for_state(st: str, safe: str, counties, area, asr, cfg) -> pd.DataFrame:
    feat_cache = HEX_DIR / f"{st}_features.parquet"
    if feat_cache.exists() and feat_cache.stat().st_size > 500:
        log.info("feature cache hit %s", feat_cache.name)
        return pd.read_parquet(feat_cache)
    prior = hex_state(PRIOR, st, safe, counties, cfg)
    current = hex_state(CUR, st, safe, counties, cfg)
    log.info("infer sites state=%s prior=%s current=%s", st, f"{len(prior):,}", f"{len(current):,}")
    _ps, cs = infer_sites_joint(prior, current, cfg)
    if not cs.empty:
        sub = counties[counties["state_fips"].astype(str).str.zfill(2) == st]
        if not sub.empty:
            minx, miny, maxx, maxy = sub.total_bounds
            pad = 0.25
            asr_near = asr[
                (asr["lat"] >= miny - pad) & (asr["lat"] <= maxy + pad)
                & (asr["lng"] >= minx - pad) & (asr["lng"] <= maxx + pad)
            ]
        else:
            asr_near = asr
        gt = (cfg.raw.get("groundtruth") or {}).get("asr") or {}
        snap = float(gt.get("site_snap_radius_m", 750))
        match = float(gt.get("site_match_radius_m", 2000))
        cs = anchor_sites_to_asr(cs, asr_near if not asr_near.empty else asr, radius_m=match, snap_radius_m=snap)
    ch = hex_change(prior, current)
    cc = county_change(ch, 9, area)
    attr = attribute_changes(ch, cs, 9)
    feats = build_features(cc, attr)
    if feats.empty:
        return feats
    feats["state_fips"] = st
    if not cs.empty and "asr_snapped" in cs.columns and "county_geoid" in cs.columns:
        snap_n = cs.groupby(cs["county_geoid"].astype(str))["asr_snapped"].sum()
        n_sites = cs.groupby(cs["county_geoid"].astype(str)).size()
        feats["n_current_sites"] = feats["county_geoid"].astype(str).map(n_sites).fillna(0).astype(int)
        feats["n_snapped"] = feats["county_geoid"].astype(str).map(snap_n).fillna(0).astype(int)
    else:
        feats["n_current_sites"] = 0
        feats["n_snapped"] = 0
    log.info("  state %s counties=%s sites=%s snapped=%s", st, len(feats), len(cs), int(cs["asr_snapped"].sum()) if not cs.empty and "asr_snapped" in cs.columns else 0)
    feats.to_parquet(feat_cache, index=False)
    return feats


def hex_labeled_counties(vintage: str, geoids: list[str], safe: str, counties) -> pd.DataFrame:
    """Clip labeled counties from one state zip. Reuse per-county parquet when present."""
    county_cache = ROOT / "data/eval/hex_slide_counties"
    county_cache.mkdir(parents=True, exist_ok=True)
    frames = []
    missing = []
    for geoid in geoids:
        cache = county_cache / f"{geoid}_{vintage.replace(' ', '_')}.parquet"
        if cache.exists() and cache.stat().st_size > 500:
            frames.append(pd.read_parquet(cache))
        else:
            missing.append(geoid)
    if not missing:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    st = missing[0][:2]
    z = _zip_path(vintage, st, safe)
    log.info("clip labeled %s %s missing=%s from %s", st, vintage, missing, z.name)
    gdf = FccDownloadSource._read_coverage_zip(z)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    for geoid in missing:
        row = counties[counties["county_geoid"].astype(str) == geoid]
        if row.empty:
            continue
        geom = row.iloc[0].geometry
        clip = geom.buffer(0.08)
        sub = gdf[gdf.intersects(clip)].copy()
        cache = county_cache / f"{geoid}_{vintage.replace(' ', '_')}.parquet"
        if sub.empty:
            df = pd.DataFrame(columns=["h3", "signal_dbm", "county_geoid", "county_name", "state_fips"])
        else:
            import geopandas as gpd
            sub = gpd.clip(sub, clip)
            df = coverage_to_hex(sub, 9, detect_signal_column(sub))
            df["county_geoid"] = geoid
            df["county_name"] = str(row.iloc[0].get("county_name", ""))
            df["state_fips"] = geoid[:2]
        df.to_parquet(cache, index=False)
        frames.append(df)
        log.info("  cached %s rows=%s", cache.name, len(df))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def features_for_labeled(st: str, geoids: list[str], safe: str, counties, area, asr, cfg) -> pd.DataFrame:
    """Score each labeled county on its own hex clip (buffers must not share h3)."""
    parts = []
    for geoid in geoids:
        prior = hex_labeled_counties(PRIOR, [geoid], safe, counties)
        current = hex_labeled_counties(CUR, [geoid], safe, counties)
        if prior.empty and current.empty:
            continue
        for df in (prior, current):
            if not df.empty and df["h3"].duplicated().any():
                df.drop_duplicates("h3", inplace=True)
        log.info("infer labeled %s prior=%s current=%s", geoid, f"{len(prior):,}", f"{len(current):,}")
        _ps, cs = infer_sites_joint(prior, current, cfg)
        if not cs.empty:
            row = counties[counties["county_geoid"].astype(str) == geoid]
            if not row.empty:
                minx, miny, maxx, maxy = row.total_bounds
                pad = 0.20
                asr_near = asr[
                    (asr["lat"] >= miny - pad) & (asr["lat"] <= maxy + pad)
                    & (asr["lng"] >= minx - pad) & (asr["lng"] <= maxx + pad)
                ]
            else:
                asr_near = asr
            gt = (cfg.raw.get("groundtruth") or {}).get("asr") or {}
            snap = float(gt.get("site_snap_radius_m", 750))
            match = float(gt.get("site_match_radius_m", 2000))
            cs = anchor_sites_to_asr(
                cs, asr_near if not asr_near.empty else asr,
                radius_m=match, snap_radius_m=snap,
            )
        ch = hex_change(prior, current)
        cc = county_change(ch, 9, area)
        attr = attribute_changes(ch, cs, 9)
        feats = build_features(cc, attr)
        if feats.empty:
            continue
        feats["state_fips"] = st
        feats["n_current_sites"] = int(len(cs))
        feats["n_snapped"] = int(cs["asr_snapped"].sum()) if (not cs.empty and "asr_snapped" in cs.columns) else 0
        parts.append(feats)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def live_should_flag(row) -> bool | None:
    """Physics target. None = unlabeled ranking-only row."""
    added = float(row.get("added_km2") or 0)
    same = float(row.get("same_site_growth_share") or 0)
    new_share = float(row.get("new_site_share") or 0)
    new_towers = int(row.get("new_towers") or 0)
    frac = float(row.get("added_frac_of_county") or 0)
    blanket = float(row.get("blanket_fillin") or 0)
    if "slide_selected" not in row or pd.isna(row.get("slide_selected")):
        return None
    slide = bool(row["slide_selected"])
    if added < 10:
        return False
    if new_share >= 0.50 and new_towers >= 1:
        return False
    if slide and same >= 0.50 and (frac >= 0.05 or blanket >= 0.20):
        return True
    if (not slide) and same < 0.50:
        return False
    if (not slide) and added < 10:
        return False
    # Skip-labeled but current filing looks like same-site blanket: still not a
    # forced positive (company may have a real pattern the slide didn't show).
    # Treat as negative unless the jump is Pearl-River-scale.
    if (not slide) and same >= 0.50 and frac >= 0.15:
        return True
    if not slide:
        return False
    return False


def apply_labels(feat: pd.DataFrame, bench: list[dict]) -> pd.DataFrame:
    by = {str(s["geoid"]).zfill(5): s for s in bench}
    feat = feat.copy()
    feat["county_geoid"] = feat["county_geoid"].astype(str).str.zfill(5)
    feat["name"] = feat["county_geoid"].astype(str).map(lambda g: by.get(g, {}).get("name"))
    feat["slide_selected"] = feat["county_geoid"].astype(str).map(
        lambda g: by[g]["expected_selected"] if g in by else None
    )
    feat["live_selected"] = [live_should_flag(r) for r in feat.to_dict("records")]
    return feat


def evaluate(feat: pd.DataFrame, cfg) -> dict:
    scored = score(feat.copy(), cfg)
    scored["county_geoid"] = scored["county_geoid"].astype(str)
    base = feat.copy()
    base["county_geoid"] = base["county_geoid"].astype(str)
    # score() re-sorts by priority; join back so labels stay on the right row.
    keep = ["county_geoid", "flag_for_review", "priority_score"]
    merged = base.merge(scored[keep], on="county_geoid", how="left", suffixes=("", "_scored"))
    if "flag_for_review_scored" in merged.columns:
        merged["flag_for_review"] = merged["flag_for_review_scored"]
    labeled = merged["live_selected"].notna()
    tp = fp = tn = fn = 0
    details = []
    for rec in merged.loc[labeled].to_dict("records"):
        exp = bool(rec["live_selected"])
        got = bool(rec.get("flag_for_review"))
        tp += got and exp
        fp += got and not exp
        tn += (not got) and (not exp)
        fn += (not got) and exp
        details.append({
            "name": rec.get("name"),
            "geoid": rec["county_geoid"],
            "slide": rec.get("slide_selected"),
            "live": exp,
            "flag": got,
            "ok": got == exp,
            "added_km2": rec.get("added_km2"),
            "frac": rec.get("added_frac_of_county"),
            "same": rec.get("same_site_growth_share"),
            "new": rec.get("new_site_share"),
            "blanket": rec.get("blanket_fillin"),
            "score": rec.get("priority_score"),
        })
    n_flag = int(scored["flag_for_review"].fillna(False).sum())
    n_elig = int((scored["added_km2"].fillna(0) >= 10).sum())
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) else 1.0,
        "recall": tp / (tp + fn) if (tp + fn) else 1.0,
        "n_counties": int(len(scored)),
        "n_flagged": n_flag,
        "n_eligible": n_elig,
        "flag_rate_eligible": n_flag / n_elig if n_elig else 0.0,
        "details": details,
        "scored": scored,
    }


def search(feat: pd.DataFrame, base_cfg) -> dict:
    best_key = None
    best = None
    # Prefer 0 FP, then max TP, then low eligible flag-rate (don't spam KS), then high precision.
    same_cuts = (0.50, 0.60)
    frac_cuts = (0.05, 0.08, 0.10, 0.12, 0.15)
    blanket_cuts = (0.20, 0.25)
    added_w = (0.20, 0.25)
    same_w = (0.16, 0.22, 0.25)
    new_w = (-0.10, -0.16, -0.22)
    blanket_w = (0.10, 0.14, 0.18)
    unattr_w = (0.0, 0.08)
    pctiles = (0.90, 0.95, 0.97)
    n = 0
    for sc in same_cuts:
        for fc in frac_cuts:
            for bc in blanket_cuts:
                for aw in added_w:
                    for sw in same_w:
                        for nw in new_w:
                            for bw in blanket_w:
                                for uw in unattr_w:
                                    for pct in pctiles:
                                        n += 1
                                        cfg = deepcopy(base_cfg)
                                        cfg.scoring["suspicious_same_site_growth"] = sc
                                        cfg.scoring["suspicious_same_site_min_county_frac"] = fc
                                        cfg.scoring["suspicious_same_site_min_blanket"] = bc
                                        cfg.scoring["flag_percentile"] = pct
                                        w = dict(cfg.scoring["feature_weights"])
                                        w["added_frac_of_county"] = aw
                                        w["same_site_growth_share"] = sw
                                        w["new_site_share"] = nw
                                        w["blanket_fillin"] = bw
                                        w["unattributed_share"] = uw
                                        w["asr_no_new_structure"] = 0.0
                                        cfg.scoring["feature_weights"] = w
                                        ev = evaluate(feat, cfg)
                                        # Penalize flooding reviewers: >25% of eligible is too many FPs.
                                        flood = ev["flag_rate_eligible"] > 0.25
                                        key = (
                                            ev["fp"],
                                            1 if flood else 0,
                                            -ev["tp"],
                                            ev["flag_rate_eligible"],
                                            -ev["precision"],
                                        )
                                        if best_key is None or key < best_key:
                                            best_key = key
                                            best = {
                                                "same_cut": sc, "frac_cut": fc, "blanket_cut": bc,
                                                "flag_percentile": pct, "weights": w,
                                                **{k: ev[k] for k in (
                                                    "tp", "fp", "tn", "fn", "precision", "recall",
                                                    "n_counties", "n_flagged", "n_eligible",
                                                    "flag_rate_eligible", "details",
                                                )},
                                            }
    log.info("searched %s grids", n)
    return best


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    cfg.raw["geography"]["site_h3_resolution"] = 9
    cfg.raw["source"]["backend"] = "fcc"
    safe = safe_service_name(SERVICE)
    counties = load_counties(cfg)
    area = county_areas_km2(counties, cfg.geography["equal_area_crs"])
    asr = load_asr_structures(ROOT / "data/groundtruth/asr")
    bench = cfg.raw["benchmark"]["counties"]
    labeled_by_st: dict[str, list[str]] = {}
    for spec in bench:
        geoid = str(spec["geoid"])
        labeled_by_st.setdefault(geoid[:2], []).append(geoid)
    parts = []
    for st in FULL_STATES:
        z0, z1 = _zip_path(PRIOR, st, safe), _zip_path(CUR, st, safe)
        if not (z0.exists() and z1.exists() and z0.stat().st_size > 10_000 and z1.stat().st_size > 10_000):
            log.warning("skip full state %s (missing zip)", st)
            continue
        try:
            parts.append(features_for_state(st, safe, counties, area, asr, cfg))
        except Exception:
            log.exception("state %s failed", st)
    for st in LABELED_ONLY_STATES:
        z0, z1 = _zip_path(PRIOR, st, safe), _zip_path(CUR, st, safe)
        geoids = labeled_by_st.get(st, [])
        if not geoids:
            continue
        if not (z0.exists() and z1.exists() and z0.stat().st_size > 10_000 and z1.stat().st_size > 10_000):
            log.warning("skip labeled state %s (missing zip)", st)
            continue
        try:
            parts.append(features_for_labeled(st, geoids, safe, counties, area, asr, cfg))
        except Exception:
            log.exception("labeled state %s failed", st)
    if not parts:
        log.error("no features")
        return 1
    feat = pd.concat(parts, ignore_index=True)
    feat["county_geoid"] = feat["county_geoid"].astype(str).str.zfill(5)
    feat = feat.drop_duplicates("county_geoid", keep="first")
    feat = apply_labels(feat, cfg.raw["benchmark"]["counties"])
    feat_path = OUT_DIR / "benchmark_live_features.json"
    # Drop huge internals; keep scoring columns.
    keep = [c for c in feat.columns if not str(c).startswith("score_")]
    feat[keep].to_json(feat_path, orient="records", indent=2, default_handler=str)
    log.info("wrote %s n=%s labeled=%s", feat_path, len(feat), int(feat["live_selected"].notna().sum()))
    baseline = evaluate(feat, cfg)
    base_out = {k: baseline[k] for k in baseline if k != "scored"}
    (OUT_DIR / "weight_search_baseline.json").write_text(json.dumps(base_out, indent=2, default=str))
    print("BASELINE", json.dumps(base_out, indent=2, default=str))
    best = search(feat, cfg)
    (OUT_DIR / "weight_search.json").write_text(json.dumps(best, indent=2, default=str))
    print("BEST", json.dumps(best, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
