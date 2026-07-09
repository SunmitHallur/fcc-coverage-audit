"""Approximate cell-site inference from coverage structure.

Cell sites are not published in the FCC data. Coverage from a site forms a lobe
that radiates from a point, with the strongest modeled signal concentrated near
the site. We therefore:

1. keep hexes at/above a high signal band (the "core" of each lobe), and
2. group them into CONTIGUOUS blobs using H3 grid adjacency (connected
   components). Each blob is treated as one inferred site, located at the
   signal-weighted centroid of the blob.

Provider heterogeneity: providers report signal in different band schemes
(some file fine-grained RSRP bands down to -120 dBm, others a single coarse
band), and the Redshift hex snapshots carry only a BINARY 0/1 coverage flag
with no signal at all. Two adaptations handle this:

* **Adaptive core threshold** — if the configured dBm cutoff keeps most of a
  provider's footprint (their reported bands are all "strong"), the cutoff is
  tightened to that provider's own strongest-signal quantile so the "core"
  stays a meaningful fraction of the lobe.
* **Boundary-depth splitting** — when signal is flat (binary sources), lobe
  structure is recovered from coverage *shape*: each blob's cells are scored
  by grid distance to the blob edge (a discrete distance transform), local
  depth maxima are treated as probable tower positions, and multi-lobe blobs
  are split by nearest-peak assignment. Centroids are depth-weighted so a
  site lands at its lobe's interior core, not the footprint's geometric mean.

Connected components (rather than density clustering) make this robust to lobe
size: a single tower with a huge footprint still yields a single site, which is
essential for correctly attributing coverage growth to new vs. expanded sites.

This is intentionally approximate - the output is "where a site probably is",
used to attribute coverage changes and prioritize manual review, not to pinpoint
hardware. Interior towers of a fully-covered region are not recoverable from
binary coverage; only lobe/edge structure is.
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
# Adaptive threshold: if the configured dBm cutoff keeps more than this share of
# a provider's hexes, tighten to the provider's own strong-signal quantile.
_MAX_CORE_FRACTION = 0.60
# ... and in that case keep the top-N fraction strongest hexes as the core.
_ADAPTIVE_CORE_QUANTILE = 0.35
# Depth transform saturation (rings). At res 9 (~0.35 km/ring) this is ~14 km —
# deeper interiors of huge binary blobs carry no tower information anyway.
_DEPTH_CAP = 40
# Minimum depth for a cell to qualify as a peak (avoids splitting thin strips).
_MIN_PEAK_DEPTH = 2
# Minimum separation between accepted peaks, in meters. Peaks closer than this
# are indistinguishable given the site_match_radius used downstream.
_MIN_PEAK_SEPARATION_M = 3000.0

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


def _boundary_depth(cells: set[str], cap: int = _DEPTH_CAP) -> dict[str, int]:
    """Grid distance (rings) from each cell to the nearest cell outside the set.

    Multi-source BFS from the blob edge — a discrete distance transform on the
    H3 grid. Depth 1 = edge cell; interior cells grow deeper. Saturates at
    ``cap`` (deep interiors of huge binary blobs carry no tower information).
    """
    depth: dict[str, int] = {}
    frontier: list[str] = []
    for c in cells:
        for n in h3.grid_disk(c, 1):
            if n != c and n not in cells:
                depth[c] = 1
                frontier.append(c)
                break
    if not depth:
        return {c: 1 for c in cells}
    d = 1
    while frontier and d < cap:
        nxt: list[str] = []
        for c in frontier:
            for n in h3.grid_disk(c, 1):
                if n in cells and n not in depth:
                    depth[n] = d + 1
                    nxt.append(n)
        frontier = nxt
        d += 1
    for c in cells:
        if c not in depth:
            depth[c] = cap
    return depth


def _find_depth_peaks(
    comp: list[str],
    depth: dict[str, int],
    xy_by_cell: dict[str, tuple[float, float]],
    min_separation_m: float = _MIN_PEAK_SEPARATION_M,
) -> list[str]:
    """Locate probable tower positions as local maxima of the boundary depth.

    A tower sits deep inside its own lobe, so depth local-maxima are the best
    binary-data estimate of tower positions. Adjacent equal-depth candidates
    (plateaus) are merged into one representative so a flat interior doesn't
    spawn a grid of fake towers, then peaks are accepted greedily (deepest
    first) with a minimum separation.
    """
    comp_set = set(comp)
    candidates: set[str] = set()
    for c in comp:
        dc = depth[c]
        if dc < _MIN_PEAK_DEPTH:
            continue
        if all(depth.get(n, 0) <= dc for n in h3.grid_disk(c, 1) if n != c and n in comp_set):
            candidates.add(c)
    if not candidates:
        # Thin strip / tiny blob: treat the whole component as one lobe.
        return [max(comp, key=lambda c: depth[c])]

    # Merge adjacent candidates (plateaus) into single representatives.
    reps: list[str] = []
    for plateau in _connected_components(candidates):
        xs = np.array([xy_by_cell[c][0] for c in plateau])
        ys = np.array([xy_by_cell[c][1] for c in plateau])
        cx, cy = xs.mean(), ys.mean()
        rep = plateau[int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))]
        reps.append(rep)

    reps.sort(key=lambda c: depth[c], reverse=True)
    accepted: list[str] = []
    for c in reps:
        x, y = xy_by_cell[c]
        if all(
            np.hypot(x - xy_by_cell[a][0], y - xy_by_cell[a][1]) >= min_separation_m
            for a in accepted
        ):
            accepted.append(c)
    return accepted


def _split_component_by_peaks(
    comp: list[str],
    peaks: list[str],
    xy_by_cell: dict[str, tuple[float, float]],
) -> list[list[str]]:
    """Partition a blob's cells by nearest depth-peak (Voronoi in meters)."""
    if len(peaks) <= 1:
        return [comp]
    peak_xy = np.array([xy_by_cell[p] for p in peaks])
    tree = cKDTree(peak_xy)
    cell_xy = np.array([xy_by_cell[c] for c in comp])
    _, idx = tree.query(cell_xy, k=1)
    parts: list[list[str]] = [[] for _ in peaks]
    for cell, i in zip(comp, idx):
        parts[int(i)].append(cell)
    return [p for p in parts if p]


def _core_hexes(hex_df: pd.DataFrame, threshold_dbm: float) -> tuple[pd.DataFrame, bool]:
    """Select the strong-signal "core" hexes, adapting to the provider's bands.

    Returns ``(core_df, signal_is_flat)``. Providers report signal in different
    band schemes; if the configured cutoff keeps most of the footprint (the
    provider only files strong bands), tighten to that provider's own signal
    quantile so the core stays discriminative. Binary sources (constant
    signal, e.g. the Redshift 0/1 hex snapshots) carry no signal information:
    the full footprint is returned and flagged flat so the caller relies on
    coverage-shape (boundary depth) instead.
    """
    sig = hex_df["signal_dbm"]
    flat = int(sig.nunique()) <= 1
    if flat:
        return hex_df.copy(), True
    strong = hex_df[sig >= threshold_dbm]
    if len(strong) > _MAX_CORE_FRACTION * len(hex_df):
        # Provider files only strong bands, so the configured cutoff keeps
        # nearly everything. Walk the provider's own signal bands from
        # strongest down and stop at the weakest band that still keeps the
        # core under the cap; if even the top band alone exceeds the cap,
        # use just the top band (best discrimination available).
        counts = sig.value_counts().sort_index(ascending=False)
        total = len(hex_df)
        threshold = float(counts.index[0])
        cum = 0
        for value, n in counts.items():
            cum += int(n)
            if cum > _MAX_CORE_FRACTION * total:
                break
            threshold = float(value)
        strong = hex_df[sig >= threshold]
    return strong.copy(), False


def infer_sites(hex_df: pd.DataFrame, cfg: Config, label_prefix: str = "S") -> pd.DataFrame:
    """Infer approximate site locations from a provider+vintage hex table."""
    tcfg = cfg.towers
    if hex_df.empty:
        return pd.DataFrame(columns=SITE_COLUMNS)

    strong, signal_flat = _core_hexes(hex_df, float(tcfg["min_signal_band_dbm"]))
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

    all_cells = set(strong["h3"])
    # Project every core cell once (used for depth peaks, splitting, centroids).
    cell_list = list(all_cells)
    centers = np.array([h3.cell_to_latlng(c) for c in cell_list])  # (lat, lng)
    xs_all, ys_all = _FWD.transform(centers[:, 1], centers[:, 0])
    xy_by_cell = {c: (float(x), float(y)) for c, x, y in zip(cell_list, xs_all, ys_all)}

    sites = []
    site_idx = 0
    for comp in _connected_components(all_cells):
        if len(comp) < min_hexes:
            continue

        # Boundary-depth transform: recovers lobe structure from coverage
        # SHAPE. Only needed for binary sources (flat signal) — when real
        # signal is present the strong-signal core already isolates each
        # tower's lobe, and splitting there would fragment legitimately
        # inflated single-tower lobes and break same-site attribution.
        if signal_flat:
            depth = _boundary_depth(set(comp))
            peaks = _find_depth_peaks(comp, depth, xy_by_cell)
            lobes = _split_component_by_peaks(comp, peaks, xy_by_cell)
        else:
            depth = None
            lobes = [comp]

        for lobe in lobes:
            if len(lobe) < max(3, min_hexes // max(1, len(lobes))):
                continue
            xs = np.array([xy_by_cell[c][0] for c in lobe])
            ys = np.array([xy_by_cell[c][1] for c in lobe])
            sigs = np.array([signal_by_cell[c] for c in lobe])
            # Centroid weighting: with real signal, weight by signal strength
            # (strongest bands sit nearest the tower). With flat (binary)
            # signal, weight by interior depth so the estimate anchors at the
            # lobe core instead of the footprint's geometric mean.
            if signal_flat:
                w = np.array([depth[c] for c in lobe], dtype=float)
            else:
                w = np.clip(sigs + 130.0, 1.0, None)
            w = np.clip(w, 1e-9, None)
            cx = float(np.average(xs, weights=w))
            cy = float(np.average(ys, weights=w))
            # Effective reach: distance from centroid to the farthest core hex.
            # The full coverage lobe (incl. weaker bands) extends beyond the
            # strong core, so attribution scales this up by a margin.
            reach = float(np.max(np.hypot(xs - cx, ys - cy)))
            lng, lat = _INV.transform(cx, cy)
            counties = pd.Series([county_by_cell.get(c) for c in lobe]).dropna()
            county = counties.mode().iloc[0] if not counties.empty else None
            sites.append(
                {
                    "site_id": f"{label_prefix}{site_idx}",
                    "lat": float(lat),
                    "lng": float(lng),
                    "x_m": cx,
                    "y_m": cy,
                    "reach_m": reach,
                    "n_hexes": int(len(lobe)),
                    "max_signal_dbm": float(sigs.max()),
                    "mean_signal_dbm": float(sigs.mean()),
                    "county_geoid": county,
                }
            )
            site_idx += 1
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
