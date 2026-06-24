"""Attribute coverage increases to NEW sites vs EXPANDED existing sites.

1. Match current-vintage inferred sites to prior-vintage sites within a radius.
   - no prior site nearby  -> NEW site
   - prior site nearby, materially more coverage now -> EXPANDED site
   - otherwise -> STABLE
2. Attribute each newly-covered / upgraded hex to its nearest current site, so
   per-county added area can be split into "from new towers" vs "from expanded
   existing towers". A large share of growth coming from EXPANDED (same) sites is
   the key gaming signal: a provider claiming big coverage jumps without building.
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
_REACH_MARGIN = 1.6
# Floor on reach so small/new sites still capture their immediate lobe.
_MIN_REACH_M = 3000.0


def tower_counts_by_county(sites: pd.DataFrame) -> dict[str, int]:
    """Count inferred sites per county_geoid."""
    if sites.empty or "county_geoid" not in sites.columns:
        return {}
    return (
        sites.dropna(subset=["county_geoid"])
        .groupby("county_geoid")
        .size()
        .astype(int)
        .to_dict()
    )


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
            ]
        )

    centers = [h3.cell_to_latlng(c) for c in gained["h3"]]
    lats = np.array([lat for lat, _ in centers])
    lngs = np.array([lng for _, lng in centers])
    xs, ys = _FWD.transform(lngs, lats)

    tree = cKDTree(current_sites[["x_m", "y_m"]].to_numpy())
    dist, idx = tree.query(np.column_stack([xs, ys]), k=1)
    site_class = current_sites["site_class"].to_numpy()
    reach = np.maximum(
        current_sites.get("reach_m", pd.Series(0.0, index=current_sites.index)).to_numpy()
        * _REACH_MARGIN,
        _MIN_REACH_M,
    )
    within = dist <= reach[idx]
    assigned = np.where(within, site_class[idx], "unattributed")
    gained["attribution"] = assigned

    # Count genuinely-new towers per county (used as a legitimacy signal).
    new_tower_counts = {}
    if "site_class" in current_sites and "county_geoid" in current_sites:
        new_sites = current_sites[current_sites["site_class"] == "new_site"]
        new_tower_counts = (
            new_sites.dropna(subset=["county_geoid"]).groupby("county_geoid").size().to_dict()
        )

    rows = []
    for county, grp in gained.groupby("county_geoid"):
        counts = grp["attribution"].value_counts()
        rows.append(
            {
                "county_geoid": county,
                "added_km2_new_site": float(counts.get("new_site", 0) * hex_km2),
                "added_km2_expanded_site": float(counts.get("expanded_site", 0) * hex_km2)
                + float(counts.get("stable_site", 0) * hex_km2),
                "added_km2_unattributed": float(counts.get("unattributed", 0) * hex_km2),
                "new_towers": int(new_tower_counts.get(county, 0)),
            }
        )
    return pd.DataFrame(rows)
