"""Approximate cell-site inference from coverage structure.

Cell sites are not published in the FCC data. Coverage from a site forms a lobe
that radiates from a point, with the strongest modeled signal concentrated near
the site. We therefore:

1. keep hexes at/above a high signal band (the "core" of each lobe), and
2. group them into CONTIGUOUS blobs using H3 grid adjacency (connected
   components). Each blob is treated as one inferred site, located at the
   signal-weighted centroid of the blob.

Connected components (rather than density clustering) make this robust to lobe
size: a single tower with a huge footprint still yields a single site, which is
essential for correctly attributing coverage growth to new vs. expanded sites.

This is intentionally approximate - the output is "where a site probably is",
used to attribute coverage changes and prioritize manual review, not to pinpoint
hardware. Dense urban areas may merge nearby towers into one blob; that is an
acceptable trade-off for prioritization.
"""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from .config import Config

_FWD = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
_INV = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)

# Floor on reach so small/new sites still capture their immediate lobe.
# Kept in sync with attribute._MIN_REACH_M.
_MIN_REACH_M = 3000.0
# For compute_lobe_reach: use the 95th-percentile of all assigned-hex distances.
_LOBE_REACH_PERCENTILE = 95.0
# Minimum hexes assigned to a site before computing a stable percentile.
_LOBE_REACH_MIN_HEXES = 3
# Fallback multiplier on core reach_m when no full-coverage hex data is available.
_LOBE_REACH_FALLBACK_MARGIN = 2.5

SITE_COLUMNS = [
    "site_id", "lat", "lng", "x_m", "y_m", "reach_m",
    "n_hexes", "max_signal_dbm", "mean_signal_dbm", "county_geoid",
]


def _connected_components(cells: set[str]) -> list[list[str]]:
    """Group H3 cells into contiguous blobs via 1-ring grid adjacency."""
    seen: set[str] = set()
    components: list[list[str]] = []
    for start in cells:
        if start in seen:
            continue
        stack = [start]
        comp: list[str] = []
        while stack:
            cell = stack.pop()
            if cell in seen:
                continue
            seen.add(cell)
            comp.append(cell)
            for neighbor in h3.grid_disk(cell, 1):
                if neighbor != cell and neighbor in cells and neighbor not in seen:
                    stack.append(neighbor)
        components.append(comp)
    return components


def infer_sites(hex_df: pd.DataFrame, cfg: Config, label_prefix: str = "S") -> pd.DataFrame:
    """Infer approximate site locations from a provider+vintage hex table."""
    tcfg = cfg.towers
    if hex_df.empty:
        return pd.DataFrame(columns=SITE_COLUMNS)

    strong = hex_df[hex_df["signal_dbm"] >= float(tcfg["min_signal_band_dbm"])].copy()
    # Auto-scale min_site_hexes to keep the minimum physical blob area consistent
    # across H3 resolutions. Config value is authoritative for the configured
    # site_h3_resolution; infer actual resolution from the data and scale.
    base_hexes = int(tcfg["min_site_hexes"])
    if not strong.empty:
        try:
            actual_res = h3.get_resolution(strong["h3"].iloc[0])
            cfg_res = int(cfg.geography.get("site_h3_resolution", actual_res))
            if actual_res != cfg_res:
                # Scale by inverse hex area ratio: each step in H3 resolution
                # is ~7x finer in area, so keep the total blob area constant.
                area_ratio = h3.average_hexagon_area(cfg_res, unit="km^2") / max(
                    h3.average_hexagon_area(actual_res, unit="km^2"), 1e-9
                )
                base_hexes = max(3, round(base_hexes * area_ratio))
        except Exception:
            pass
    min_hexes = base_hexes
    if len(strong) < min_hexes:
        return pd.DataFrame(columns=SITE_COLUMNS)

    signal_by_cell = dict(zip(strong["h3"], strong["signal_dbm"]))
    county_by_cell = dict(zip(strong["h3"], strong["county_geoid"]))

    sites = []
    for i, comp in enumerate(_connected_components(set(strong["h3"]))):
        if len(comp) < min_hexes:
            continue
        centers = np.array([h3.cell_to_latlng(c) for c in comp])  # (lat, lng)
        lats, lngs = centers[:, 0], centers[:, 1]
        xs, ys = _FWD.transform(lngs, lats)
        sigs = np.array([signal_by_cell[c] for c in comp])
        # Signal-weighted centroid (shift dBm to positive weights).
        w = np.clip(sigs + 130.0, 1.0, None)
        cx = float(np.average(xs, weights=w))
        cy = float(np.average(ys, weights=w))
        # Effective reach: distance from centroid to the farthest core hex. The
        # full coverage lobe (incl. weaker bands) extends beyond the strong core,
        # so attribution scales this up by a margin.
        reach = float(np.max(np.hypot(xs - cx, ys - cy)))
        # Back to lat/lng for display.
        lng, lat = _INV.transform(cx, cy)
        counties = pd.Series([county_by_cell.get(c) for c in comp]).dropna()
        county = counties.mode().iloc[0] if not counties.empty else None
        sites.append(
            {
                "site_id": f"{label_prefix}{i}",
                "lat": float(lat),
                "lng": float(lng),
                "x_m": cx,
                "y_m": cy,
                "reach_m": reach,
                "n_hexes": int(len(comp)),
                "max_signal_dbm": float(sigs.max()),
                "mean_signal_dbm": float(sigs.mean()),
                "county_geoid": county,
            }
        )
    return pd.DataFrame(sites, columns=SITE_COLUMNS)


def compute_lobe_reach(
    hex_df: pd.DataFrame,
    sites: pd.DataFrame,
    percentile: float = _LOBE_REACH_PERCENTILE,
) -> pd.DataFrame:
    """Augment inferred sites with an empirical full-lobe propagation reach.

    ``infer_sites`` derives ``reach_m`` from the strong-signal core only
    (hexes at/above ``min_signal_band_dbm``). Real coverage lobes extend well
    beyond that core at weaker signal bands, which means gained fringe hexes
    fall outside ``reach_m * REACH_MARGIN`` and get mis-attributed as
    ``unattributed`` — a false gaming signal.

    This function assigns every covered hex in ``hex_df`` (all signal bands,
    site-resolution) to its nearest site via KD-tree, then sets
    ``lobe_reach_m`` = <percentile>th percentile of those per-site distances.
    Attribution in ``attribute.py`` uses ``lobe_reach_m`` when present, so a
    single matched tower captures ~100% of its gained hexes and
    ``unattributed_share`` is reserved for coverage genuinely orphaned from
    every inferred tower.

    Returns a copy of ``sites`` with ``lobe_reach_m`` added.
    """
    s = sites.copy()
    if s.empty:
        return s

    core_reach = s.get("reach_m", pd.Series(0.0, index=s.index)).to_numpy(dtype=float)
    fallback = np.maximum(core_reach * _LOBE_REACH_FALLBACK_MARGIN, _MIN_REACH_M)

    if hex_df.empty:
        s["lobe_reach_m"] = fallback
        return s

    xs_s = s["x_m"].to_numpy(dtype=float)
    ys_s = s["y_m"].to_numpy(dtype=float)

    hex_ids = hex_df["h3"].astype(str).tolist()
    centers = np.array([h3.cell_to_latlng(c) for c in hex_ids])
    lats, lngs = centers[:, 0], centers[:, 1]
    xs_h, ys_h = _FWD.transform(lngs, lats)

    tree = cKDTree(np.column_stack([xs_s, ys_s]))
    dist, idx = tree.query(np.column_stack([xs_h, ys_h]), k=1)

    lobe_reach = fallback.copy()
    for i in range(len(s)):
        mask = idx == i
        n = int(mask.sum())
        if n >= _LOBE_REACH_MIN_HEXES:
            emp = float(np.percentile(dist[mask], percentile))
            # Always at least as large as the fallback so we never shrink reach.
            lobe_reach[i] = max(emp, fallback[i])

    s["lobe_reach_m"] = np.maximum(lobe_reach, _MIN_REACH_M)
    return s
