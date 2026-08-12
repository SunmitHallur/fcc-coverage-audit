"""Attribute coverage increases to NEW sites vs EXPANDED existing sites.

1. Match current-vintage inferred sites to prior-vintage sites within a radius.
   - no prior site nearby  -> NEW site
   - prior site nearby, materially more coverage now -> EXPANDED site
   - otherwise -> STABLE
2. Attribute each newly-covered / upgraded hex to its nearest current site, so
   per-county added area can be split into "from new towers" vs "from expanded
   existing towers". A large share of growth coming from EXPANDED (same) sites is
   the key gaming signal: a provider claiming big coverage jumps without building.

Sites are matched to hexes by propagation reach, regardless of county boundaries:
a tower in an adjacent county that covers hexes in this county counts toward
this county's serving-tower totals and growth attribution.
"""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

_FWD = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

# A matched site counts as "expanded" if its covered-hex count grew by >= this.
_EXPANSION_GROWTH = 0.20
# Coverage lobes (incl. weak bands) extend beyond the strong core used to infer
# a site, so a hex is attributed to its nearest site if within reach*margin.
REACH_MARGIN = 1.6
# Floor on reach so small/new sites still capture their immediate lobe.
_MIN_REACH_M = 3000.0


def tower_counts_by_county(sites: pd.DataFrame) -> dict[str, int]:
    """Count inferred sites whose strong-signal core sits inside each county."""
    if sites.empty or "county_geoid" not in sites.columns:
        return {}
    return (
        sites.dropna(subset=["county_geoid"])
        .groupby("county_geoid")
        .size()
        .astype(int)
        .to_dict()
    )


def _site_reach_m(sites: pd.DataFrame) -> np.ndarray:
    """Per-site propagation reach for hex attribution.

    Prefers ``lobe_reach_m`` (empirical, from ``towers.compute_lobe_reach``)
    over the core-based ``reach_m * REACH_MARGIN`` heuristic. The lobe reach
    is derived from ALL covered hexes so it captures how far a tower's signal
    actually propagates — not just to the edge of the strong-signal core. This
    ensures a single matched tower captures ~100% of its gained hexes without
    mis-attributing fringe hexes as 'unattributed'.
    """
    if "lobe_reach_m" in sites.columns:
        lobe = sites["lobe_reach_m"].to_numpy(dtype=float)
        core = sites.get("reach_m", pd.Series(0.0, index=sites.index)).to_numpy(dtype=float)
        fallback = np.maximum(core * REACH_MARGIN, _MIN_REACH_M)
        valid = np.isfinite(lobe) & (lobe > 0)
        return np.where(valid, np.maximum(lobe, _MIN_REACH_M), fallback)
    reach = sites.get("reach_m", pd.Series(0.0, index=sites.index)).to_numpy(dtype=float)
    return np.maximum(reach * REACH_MARGIN, _MIN_REACH_M)


def _hex_xy_m(hex_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    centers = [h3.cell_to_latlng(c) for c in hex_ids]
    lats = np.array([lat for lat, _ in centers])
    lngs = np.array([lng for _, lng in centers])
    return _FWD.transform(lngs, lats)


def _prepare_sites(sites: pd.DataFrame) -> pd.DataFrame:
    """Ensure projected coordinates and reach exist for spatial attribution."""
    if sites.empty:
        return sites
    out = sites.copy()
    if "x_m" not in out.columns or "y_m" not in out.columns:
        xs, ys = _FWD.transform(out["lng"].to_numpy(), out["lat"].to_numpy())
        out["x_m"] = xs
        out["y_m"] = ys
    if "reach_m" not in out.columns:
        out["reach_m"] = _MIN_REACH_M
    return out


def attribute_hexes_to_sites(
    hex_df: pd.DataFrame,
    sites: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each hex row to nearest site index, distance (m), and site class label.

    Returns parallel arrays (site_idx, dist_m, attribution) with site_idx=-1 when
    no site is within propagation reach.
    """
    n = len(hex_df)
    if n == 0 or sites.empty:
        return (
            np.full(n, -1, dtype=int),
            np.full(n, np.nan),
            np.array(["unattributed"] * n, dtype=object),
        )

    sites = _prepare_sites(sites.reset_index(drop=True))
    xs, ys = _hex_xy_m(hex_df["h3"].astype(str).tolist())
    tree = cKDTree(sites[["x_m", "y_m"]].to_numpy())
    dist, idx = tree.query(np.column_stack([xs, ys]), k=1)
    reach = _site_reach_m(sites)
    within = dist <= reach[idx]
    site_idx = np.where(within, idx.astype(int), -1)
    site_class = sites["site_class"].to_numpy() if "site_class" in sites.columns else None
    if site_class is None:
        attribution = np.where(within, "site", "unattributed")
    else:
        attribution = np.where(within, site_class[idx], "unattributed")
    return site_idx, dist, attribution


def serving_towers_by_county(
    hex_df: pd.DataFrame,
    sites: pd.DataFrame,
) -> pd.DataFrame:
    """Per county, count sites whose lobes cover hexes there (incl. cross-border)."""
    cols = [
        "county_geoid", "towers_serving", "towers_in_county", "towers_cross_border",
    ]
    if hex_df.empty or sites.empty:
        return pd.DataFrame(columns=cols)

    sub = hex_df.dropna(subset=["county_geoid", "h3"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=cols)

    site_idx, _, _ = attribute_hexes_to_sites(sub, sites)
    sub["_site_idx"] = site_idx
    attributed = sub.loc[sub["_site_idx"] >= 0, ["county_geoid", "_site_idx"]]
    if attributed.empty:
        geoids = sub["county_geoid"].astype(str).unique()
        return pd.DataFrame({
            "county_geoid": geoids,
            "towers_serving": 0,
            "towers_in_county": 0,
            "towers_cross_border": 0,
        })

    attributed = attributed.drop_duplicates()
    attributed["county_geoid"] = attributed["county_geoid"].astype(str)
    home = sites["county_geoid"].astype(str).to_numpy()
    idx = attributed["_site_idx"].to_numpy(dtype=int)
    attributed["in_county"] = home[idx] == attributed["county_geoid"].to_numpy()

    serving = attributed.groupby("county_geoid", sort=False)["_site_idx"].nunique()
    in_county = attributed.loc[attributed["in_county"]].groupby(
        "county_geoid", sort=False
    )["_site_idx"].nunique()
    out = pd.DataFrame({"county_geoid": serving.index.astype(str), "towers_serving": serving.to_numpy()})
    out["towers_in_county"] = out["county_geoid"].map(in_county).fillna(0).astype(int)
    out["towers_cross_border"] = (out["towers_serving"] - out["towers_in_county"]).astype(int)
    # Include counties that had hexes but zero attributed towers.
    all_geoids = set(sub["county_geoid"].astype(str))
    missing = all_geoids - set(out["county_geoid"])
    if missing:
        out = pd.concat([
            out,
            pd.DataFrame({
                "county_geoid": sorted(missing),
                "towers_serving": 0,
                "towers_in_county": 0,
                "towers_cross_border": 0,
            }),
        ], ignore_index=True)
    return out[cols]


def site_indices_serving_county(
    hex_df: pd.DataFrame,
    sites: pd.DataFrame,
    geoid: str,
) -> np.ndarray:
    """Unique site indices that cover any hex in ``geoid`` (cross-border included)."""
    if hex_df.empty or sites.empty:
        return np.array([], dtype=int)
    sub = hex_df[hex_df["county_geoid"].astype(str) == str(geoid)]
    if sub.empty:
        return np.array([], dtype=int)
    sub_sites = _prepare_sites(sites.reset_index(drop=True))
    site_idx, _, _ = attribute_hexes_to_sites(sub, sub_sites)
    valid = site_idx[site_idx >= 0]
    return np.unique(valid.astype(int))


def match_sites(
    prior_sites: pd.DataFrame, current_sites: pd.DataFrame, radius_m: float
) -> pd.DataFrame:
    """Label each current site as new_site / expanded_site / stable_site.

    Uses a radius-gated Hungarian assignment (``linear_sum_assignment``) so
    matching is one-to-one: two current sites cannot claim the same prior site.
    Prefer :func:`towers.infer_sites_joint` when both vintages are available —
    that path shares site identity by construction and skips this fallback.
    """
    cur = current_sites.copy()
    if cur.empty:
        cur["matched_prior_id"] = []
        cur["match_dist_m"] = []
        cur["site_class"] = []
        return cur

    # Already classified by joint inference — leave alone.
    if "site_class" in cur.columns and cur["site_class"].notna().all():
        if "matched_prior_id" in cur.columns and "match_dist_m" in cur.columns:
            return cur

    if prior_sites.empty:
        cur["matched_prior_id"] = None
        cur["match_dist_m"] = np.nan
        cur["site_class"] = "new_site"
        return cur

    from scipy.optimize import linear_sum_assignment

    prior_xy = prior_sites[["x_m", "y_m"]].to_numpy(dtype=float)
    cur_xy = cur[["x_m", "y_m"]].to_numpy(dtype=float)
    # Pairwise distances; values beyond radius become a large sentinel so the
    # assignment prefers in-radius pairs, then we reject the rest.
    n_c, n_p = len(cur), len(prior_sites)
    cost = np.full((n_c, n_p), radius_m * 10.0, dtype=float)
    tree = cKDTree(prior_xy)
    # Query a generous k so candidates within radius are available for assignment.
    k = min(n_p, 8)
    dists, idxs = tree.query(cur_xy, k=k)
    if k == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)
    for ci in range(n_c):
        for d, pi in zip(dists[ci], idxs[ci]):
            if np.isfinite(d) and d <= radius_m:
                cost[ci, int(pi)] = float(d)

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_prior_for_cur: dict[int, tuple[int, float]] = {}
    for ci, pi in zip(row_ind, col_ind):
        d = float(cost[ci, pi])
        if d <= radius_m:
            matched_prior_for_cur[int(ci)] = (int(pi), d)

    matched_id = []
    site_class = []
    match_dist = []
    n_hexes = cur["n_hexes"].to_numpy()
    for ci in range(n_c):
        hit = matched_prior_for_cur.get(ci)
        if hit is None:
            matched_id.append(None)
            site_class.append("new_site")
            match_dist.append(np.nan)
            continue
        pi, d = hit
        prior_row = prior_sites.iloc[pi]
        n_prior = max(int(prior_row["n_hexes"]), 1)
        growth = (int(n_hexes[ci]) - n_prior) / n_prior
        matched_id.append(prior_row["site_id"])
        match_dist.append(d)
        site_class.append("expanded_site" if growth >= _EXPANSION_GROWTH else "stable_site")
    cur["matched_prior_id"] = matched_id
    cur["match_dist_m"] = match_dist
    cur["site_class"] = site_class
    return cur


def anchor_sites_to_asr(
    sites: pd.DataFrame,
    asr_structures: pd.DataFrame,
    radius_m: float = 2000.0,
) -> pd.DataFrame:
    """Spatially join inferred sites to ASR registered structures.

    Adds ``asr_matched`` (bool) and ``asr_distance_m`` (float). ASR rows need
    ``lat`` / ``lng`` columns (from :func:`groundtruth_asr._load_or_build_asr_df`).
    """
    out = sites.copy()
    if out.empty:
        out["asr_matched"] = []
        out["asr_distance_m"] = []
        return out
    out["asr_matched"] = False
    out["asr_distance_m"] = np.nan
    if asr_structures is None or asr_structures.empty:
        return out
    if "lat" not in asr_structures.columns or "lng" not in asr_structures.columns:
        return out
    asr = asr_structures.dropna(subset=["lat", "lng"]).copy()
    if asr.empty:
        return out
    xs_a, ys_a = _FWD.transform(asr["lng"].to_numpy(dtype=float), asr["lat"].to_numpy(dtype=float))
    tree = cKDTree(np.column_stack([xs_a, ys_a]))
    if "x_m" not in out.columns or "y_m" not in out.columns:
        xs, ys = _FWD.transform(out["lng"].to_numpy(dtype=float), out["lat"].to_numpy(dtype=float))
        out["x_m"] = xs
        out["y_m"] = ys
    dist, _ = tree.query(out[["x_m", "y_m"]].to_numpy(dtype=float), k=1)
    matched = dist <= float(radius_m)
    out["asr_matched"] = matched
    out["asr_distance_m"] = np.where(matched, dist, np.nan)
    return out


def attribute_changes(
    change_df: pd.DataFrame, current_sites: pd.DataFrame, resolution: int
) -> pd.DataFrame:
    """Split per-county added area into new-site vs expanded-site contributions."""
    hex_km2 = h3.average_hexagon_area(resolution, unit="km^2")
    gained = change_df[change_df["status"].isin(["new", "upgraded"])].dropna(
        subset=["county_geoid"]
    ).copy()
    if gained.empty:
        return pd.DataFrame(
            columns=[
                "county_geoid", "added_km2_new_site",
                "added_km2_expanded_site", "added_km2_unattributed",
                "new_towers", "new_towers_in_county", "new_towers_cross_border",
                "inference_insufficient",
            ]
        )
    if current_sites.empty:
        # A thin/small full-gain layer may not meet tower-inference thresholds.
        # Those gained cells did not disappear analytically: with no inferred
        # site, all of their area is explicitly unattributed — but scoring must
        # treat this as inference_insufficient (not gaming).
        counts = gained.groupby("county_geoid").size()
        return pd.DataFrame({
            "county_geoid": counts.index.astype(str),
            "added_km2_new_site": 0.0,
            "added_km2_expanded_site": 0.0,
            "added_km2_unattributed": counts.to_numpy(dtype=float) * hex_km2,
            "new_towers": 0,
            "new_towers_in_county": 0,
            "new_towers_cross_border": 0,
            "inference_insufficient": True,
        })

    site_idx, _, attribution = attribute_hexes_to_sites(gained, current_sites)
    gained["attributed_site_idx"] = site_idx
    gained["attribution"] = attribution

    rows = []
    for county, grp in gained.groupby("county_geoid"):
        geoid = str(county)
        counts = grp["attribution"].value_counts()
        new_idxs = grp.loc[
            grp["attribution"] == "new_site", "attributed_site_idx"
        ]
        new_idxs = new_idxs[new_idxs >= 0].astype(int).unique()
        new_towers = int(len(new_idxs))
        new_in = new_cross = 0
        if new_towers:
            new_sites = current_sites.iloc[new_idxs]
            home = new_sites["county_geoid"].astype(str)
            new_in = int((home == geoid).sum())
            new_cross = new_towers - new_in

        rows.append(
            {
                "county_geoid": geoid,
                "added_km2_new_site": float(counts.get("new_site", 0) * hex_km2),
                # Stable matched-site hexes are not "expanded growth" — counting
                # them as expanded inflated same_site_growth_share for modest upgrades.
                "added_km2_expanded_site": float(counts.get("expanded_site", 0) * hex_km2),
                "added_km2_unattributed": float(counts.get("unattributed", 0) * hex_km2),
                "new_towers": new_towers,
                "new_towers_in_county": new_in,
                "new_towers_cross_border": new_cross,
                "inference_insufficient": False,
            }
        )
    return pd.DataFrame(rows)
