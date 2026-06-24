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

    rows = []
    for county, grp in sub.groupby("county_geoid"):
        geoid = str(county)
        sidx = grp.loc[grp["_site_idx"] >= 0, "_site_idx"].astype(int).unique()
        if len(sidx) == 0:
            rows.append({
                "county_geoid": geoid,
                "towers_serving": 0,
                "towers_in_county": 0,
                "towers_cross_border": 0,
            })
            continue
        serving = sites.iloc[sidx]
        in_county = int((serving["county_geoid"].astype(str) == geoid).sum())
        rows.append({
            "county_geoid": geoid,
            "towers_serving": int(len(serving)),
            "towers_in_county": in_county,
            "towers_cross_border": int(len(serving) - in_county),
        })
    return pd.DataFrame(rows)


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
    """Label each current site as new_site / expanded_site / stable_site."""
    cur = current_sites.copy()
    if cur.empty:
        cur["matched_prior_id"] = []
        cur["match_dist_m"] = []
        cur["site_class"] = []
        return cur

    if prior_sites.empty:
        cur["matched_prior_id"] = None
        cur["match_dist_m"] = np.nan
        cur["site_class"] = "new_site"
        return cur

    tree = cKDTree(prior_sites[["x_m", "y_m"]].to_numpy())
    dist, idx = tree.query(cur[["x_m", "y_m"]].to_numpy(), k=1)
    matched_id = []
    site_class = []
    match_dist = []
    for d, i, n_now in zip(dist, idx, cur["n_hexes"].to_numpy()):
        if d > radius_m:
            matched_id.append(None)
            site_class.append("new_site")
            match_dist.append(np.nan)
            continue
        prior_row = prior_sites.iloc[int(i)]
        n_prior = max(int(prior_row["n_hexes"]), 1)
        growth = (n_now - n_prior) / n_prior
        matched_id.append(prior_row["site_id"])
        match_dist.append(float(d))
        site_class.append("expanded_site" if growth >= _EXPANSION_GROWTH else "stable_site")
    cur["matched_prior_id"] = matched_id
    cur["match_dist_m"] = match_dist
    cur["site_class"] = site_class
    return cur


def attribute_changes(
    change_df: pd.DataFrame, current_sites: pd.DataFrame, resolution: int
) -> pd.DataFrame:
    """Split per-county added area into new-site vs expanded-site contributions."""
    hex_km2 = h3.average_hexagon_area(resolution, unit="km^2")
    gained = change_df[change_df["status"].isin(["new", "upgraded"])].dropna(
        subset=["county_geoid"]
    ).copy()
    if gained.empty or current_sites.empty:
        return pd.DataFrame(
            columns=[
                "county_geoid", "added_km2_new_site",
                "added_km2_expanded_site", "added_km2_unattributed",
                "new_towers", "new_towers_in_county", "new_towers_cross_border",
            ]
        )

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
                "added_km2_expanded_site": float(counts.get("expanded_site", 0) * hex_km2)
                + float(counts.get("stable_site", 0) * hex_km2),
                "added_km2_unattributed": float(counts.get("unattributed", 0) * hex_km2),
                "new_towers": new_towers,
                "new_towers_in_county": new_in,
                "new_towers_cross_border": new_cross,
            }
        )
    return pd.DataFrame(rows)
