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
band, and some systematically file 15–30 dB hotter or colder than peers).
The Redshift hex snapshots can also carry only a BINARY 0/1 coverage flag.
Adaptations:

* **Relative core (per provider×vintage)** — the "strong" hexes that seed a
  site are the top ~35% of *that layer's own* signal distribution, not hexes
  above a global dBm cutoff. A provider that files -50 dBm cores and one that
  files -90 dBm cores of the same shape therefore produce the same blobs.
* **Boundary-depth splitting** — when signal is flat (binary sources), lobe
  structure is recovered from coverage *shape*: each blob's cells are scored
  by grid distance to the blob edge (a discrete distance transform) and local
  depth maxima are candidate towers. A 1-lobe disk has one maximum. 2-sector
  bowties and 3-sector cloverleafs produce *petal* maxima, not a hub maximum —
  those pairs/triples are merged back to the junction when the angular gaps
  say they are cones of one site (not two nearby omnis).
  Sites are placed on the peak/junction cell, not a mass centroid.
* **Signal-peak splitting** — when real ``minsignal`` is present, overlapping
  towers are separated at local signal maxima. A single inflated lobe still has
  one maximum, so same-site growth is not fragmented. Peak *locations* use
  rank within the layer, so a hotter filing does not move the mast.

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
# Relative core: strongest-first share of *this* provider's hexes.
_ADAPTIVE_CORE_QUANTILE = 0.35
_MAX_CORE_FRACTION = 0.60
_MIN_CORE_FRACTION = 0.18
# Depth transform saturation (rings). At res 9 (~0.35 km/ring) this is ~14 km —
# deeper interiors of huge binary blobs carry no tower information anyway.
_DEPTH_CAP = 40
# Minimum depth for a cell to qualify as a peak (avoids splitting thin strips).
_MIN_PEAK_DEPTH = 2
# Cloverleaf merge: absolute petal-to-petal span (not scaled by peak NMS).
# ~4 km petals at 120° are ~6.9 km apart; keep headroom to ~10 km.
_CLOVER_MIN_SIDE_M = 1500.0
_CLOVER_MAX_SIDE_M = 10_000.0
# Signal field is "discriminative" enough to split on local maxima.
_SIGNAL_SPLIT_RANGE_DB = 5.0
# Default minimum separation between accepted peaks (meters). Independent of
# ``site_match_radius_m`` (cross-vintage identity). 500 m is ~1.4 H3-9 cells
# so 0.8 km urban macros can split after hex snapping; 800 m collapsed them
# at half of US test locations. Large *depth* plateaus use a wider gate.
_MIN_PEAK_SEPARATION_M = 500.0
# Depth-field NMS for huge connected blobs (county-wide fill). 500 m tiling
# of a saturated interior plateau minted urban fake sites (Sedgwick T-Mobile
# 24 → 167). Signal maxima still use _MIN_PEAK_SEPARATION_M.
_LARGE_BLOB_DEPTH_SEP_M = 2000.0
_LARGE_BLOB_HEXES = 4000
# Signal-field saddle prominence (dB) for dropping overlap shoulders.
_MIN_SIGNAL_PROMINENCE_DB = 3.0
# Binary Redshift footprints can be multi-million-hex mega-blobs; full
# boundary-depth lobe splitting on those dominates overnight runtime. Above this
# many *flat* core cells, infer sites on a coarser parent grid and scale reach
# back. Real minsignal layers stay on res 9 — parent centroids were landing
# on county lines and 1-cell parent lobes were dropped as "too small".
_FLAT_COARSE_HEX_THRESHOLD = 25_000
_FLAT_INFER_PARENT_STEPS = 2  # res 9 -> res 7 (~49x fewer cells)
# Joint-inference: a site with prior hexes below this fraction of current (or
# absolute floor) is classified as new_site rather than expanded/stable.
_JOINT_NEW_PRIOR_MAX_HEXES = 2
_JOINT_EXPANSION_GROWTH = 0.20

SITE_COLUMNS = [
    "site_id", "lat", "lng", "x_m", "y_m", "reach_m",
    "n_hexes", "max_signal_dbm", "mean_signal_dbm", "county_geoid",
]


def _rollup_flat_for_inference(hex_df: pd.DataFrame, parent_steps: int) -> pd.DataFrame:
    """Collapse hexes to coarser parents for tower inference only.

    Keeps the strongest ``signal_dbm`` child in each parent and remembers that
    child as ``_seed_h3`` so the pin can snap back to res-9 instead of the
    parent centroid (which is often in the next county).
    """
    if parent_steps <= 0 or hex_df.empty:
        return hex_df
    src_res = h3.get_resolution(str(hex_df["h3"].iloc[0]))
    parent_res = max(0, src_res - parent_steps)
    orig = hex_df["h3"].astype(str).tolist()
    parents = [h3.cell_to_parent(c, parent_res) for c in orig]
    out = hex_df.assign(_parent=parents, _seed_h3=orig)
    if "signal_dbm" in out.columns:
        out = out.sort_values("signal_dbm", ascending=False)
    out = out.drop_duplicates(subset=["_parent"], keep="first").copy()
    out["h3"] = out["_parent"].astype(str)
    return out.drop(columns=["_parent"])


def _flat_parent_steps(n_hexes: int) -> int:
    """Choose parent rollup depth so inference stays near ~60k cells."""
    if n_hexes < _FLAT_COARSE_HEX_THRESHOLD:
        return 0
    steps = _FLAT_INFER_PARENT_STEPS
    while n_hexes / (7 ** steps) > 60_000 and steps < 4:
        steps += 1
    return steps


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


def _find_score_peaks(
    comp: list[str],
    score: dict[str, float],
    xy_by_cell: dict[str, tuple[float, float]],
    min_separation_m: float,
    min_score: float,
) -> tuple[list[str], float]:
    """Local maxima of *score*, plateau-merged, greedily separated.

    Returns ``(peaks, largest_plateau_fraction)`` so callers can detect a
    non-discriminative field (one giant plateau) and fall back to shape.
    """
    comp_set = set(comp)
    candidates: set[str] = set()
    for c in comp:
        sc = score[c]
        if sc < min_score:
            continue
        if all(score.get(n, sc - 1) <= sc for n in h3.grid_disk(c, 1) if n != c and n in comp_set):
            candidates.add(c)
    if not candidates:
        return [max(comp, key=lambda c: score[c])], 0.0

    plateaus = _connected_components(candidates)
    largest_frac = max(len(p) for p in plateaus) / max(len(comp), 1)
    reps: list[str] = []
    for plateau in plateaus:
        xs = np.array([xy_by_cell[c][0] for c in plateau])
        ys = np.array([xy_by_cell[c][1] for c in plateau])
        cx, cy = xs.mean(), ys.mean()
        rep = plateau[int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))]
        reps.append(rep)

    # Deterministic NMS: score descending, then cell id, so tied bands do not
    # flip which peak survives across runs.
    reps.sort(key=lambda c: (-score[c], c))
    accepted: list[str] = []
    for c in reps:
        x, y = xy_by_cell[c]
        if all(
            np.hypot(x - xy_by_cell[a][0], y - xy_by_cell[a][1]) >= min_separation_m
            for a in accepted
        ):
            accepted.append(c)
    return accepted, float(largest_frac)


_MIN_PEAK_PROMINENCE = 3


def _saddle_score(
    a: str,
    b: str,
    score: dict[str, float],
    xy_by_cell: dict[str, tuple[float, float]],
    tree: cKDTree,
    cells: list[str],
    n_samples: int = 24,
) -> float:
    """Minimum score on the straight line between two cells."""
    xa, ya = xy_by_cell[a]
    xb, yb = xy_by_cell[b]
    ts = np.linspace(0.05, 0.95, n_samples)
    xs = xa + ts * (xb - xa)
    ys = ya + ts * (yb - ya)
    _, idx = tree.query(np.column_stack([xs, ys]))
    return float(min(score[cells[int(i)]] for i in idx))


def _filter_low_prominence_peaks(
    peaks: list[str],
    score: dict[str, float],
    xy_by_cell: dict[str, tuple[float, float]],
    min_prominence: float = float(_MIN_PEAK_PROMINENCE),
) -> list[str]:
    """Drop shoulder maxima that sit on the slope of a stronger nearby peak.

    Two overlapping circular towers grow extra depth/signal bumps in the
    overlap; those have almost no drop toward the true center. Petals of a
    2-/3-sector site stay: the path between petals dips at the hub saddle.
    Works for boundary-depth (integer rings) or signal (dBm) scores.

    A peak is dropped when any strictly stronger peak has a path saddle so
    shallow that ``score[peak] - saddle < min_prominence`` — even if the
    stronger peak is only 1–2 rings/dB hotter (common for overlap shoulders).
    """
    if len(peaks) <= 1:
        return peaks
    cells = list(xy_by_cell)
    tree = cKDTree(np.array([xy_by_cell[c] for c in cells], dtype=float))
    scores = [float(score[p]) for p in peaks]
    keep = [True] * len(peaks)
    for i, peak in enumerate(peaks):
        higher = [j for j in range(len(peaks)) if scores[j] > scores[i]]
        if not higher:
            continue
        px, py = xy_by_cell[peak]
        j = min(
            higher,
            key=lambda k: (xy_by_cell[peaks[k]][0] - px) ** 2
            + (xy_by_cell[peaks[k]][1] - py) ** 2,
        )
        saddle = _saddle_score(peak, peaks[j], score, xy_by_cell, tree, cells)
        if scores[i] - saddle < min_prominence:
            keep[i] = False
    return [p for p, ok in zip(peaks, keep) if ok]


def _find_depth_peaks(
    comp: list[str],
    depth: dict[str, int],
    xy_by_cell: dict[str, tuple[float, float]],
    min_separation_m: float = _MIN_PEAK_SEPARATION_M,
) -> list[str]:
    """Locate probable tower positions as local maxima of the boundary depth."""
    score = {c: float(depth[c]) for c in comp}
    peaks, _ = _find_score_peaks(
        comp, score, xy_by_cell, min_separation_m, float(_MIN_PEAK_DEPTH),
    )
    local_xy = {c: xy_by_cell[c] for c in comp}
    return _filter_low_prominence_peaks(
        peaks, score, local_xy, min_prominence=float(_MIN_PEAK_PROMINENCE),
    )


def _n_sector_signature(
    cx: float,
    cy: float,
    xs: np.ndarray,
    ys: np.ndarray,
    ring_inner_m: float,
    ring_outer_m: float,
    n_sectors: int,
    n_bins: int = 18,
) -> bool:
    """True when a ring around (cx, cy) has ``n_sectors`` coverage lobes with gaps.

    3-sector cloverleafs and 2-sector "bowtie" sites both leave angular notches.
    Nearby circular footprints look filled from their overlap centroid (no gaps).
    """
    dx = xs - cx
    dy = ys - cy
    r2 = dx * dx + dy * dy
    mask = (r2 >= ring_inner_m * ring_inner_m) & (r2 <= ring_outer_m * ring_outer_m)
    min_pts = 18 if n_sectors == 2 else 24
    if int(mask.sum()) < min_pts:
        return False
    ang = np.arctan2(dy[mask], dx[mask])
    bins = np.bincount(
        (((ang + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins),
        minlength=n_bins,
    )
    occ = bins >= max(3.0, 0.40 * float(bins.max()))
    starts_occ = int((occ & ~np.roll(occ, 1)).sum())
    starts_gap = int((~occ & np.roll(occ, 1)).sum())
    frac = float(occ.mean())
    if n_sectors == 2:
        return starts_occ == 2 and starts_gap == 2 and 0.18 <= frac <= 0.80
    # 0.28 (not 0.35): Albers-warped petals at some longitudes occupy ~6/18
    # bins (Logan UT core frac 0.33) while still having three clear gaps.
    return starts_occ == 3 and starts_gap == 3 and 0.28 <= frac <= 0.90


def _three_sector_signature(
    cx: float,
    cy: float,
    xs: np.ndarray,
    ys: np.ndarray,
    ring_inner_m: float,
    ring_outer_m: float,
    n_bins: int = 18,
) -> bool:
    """True when a ring around (cx, cy) has three coverage sectors with gaps."""
    return _n_sector_signature(
        cx, cy, xs, ys, ring_inner_m, ring_outer_m, n_sectors=3, n_bins=n_bins,
    )


def _xy_near(
    cx: float,
    cy: float,
    xs: np.ndarray,
    ys: np.ndarray,
    radius: float,
    tree: cKDTree | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates inside ``radius`` of (cx, cy). KDTree when the field is large."""
    if len(xs) == 0 or radius <= 0:
        return xs[:0], ys[:0]
    if tree is not None and len(xs) > 4_000:
        idx = tree.query_ball_point([cx, cy], float(radius))
        if not idx:
            return xs[:0], ys[:0]
        return xs[idx], ys[idx]
    r2 = float(radius) * float(radius)
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r2
    return xs[mask], ys[mask]


def _two_peak_hub_xy(
    p1: np.ndarray,
    p2: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    tree: cKDTree | None = None,
) -> tuple[float, float]:
    """Hub of a 2-sector site along the perpendicular bisector of the petal peaks.

    Opposed (180°) petals look like two sectors from a wide range of points
    centered on the chord midpoint; a cloverleaf missing one sector is only
    two-sectored from points shifted into the remaining wedge. Take the median
    of the points that still have a 2-sector ring signature.
    """
    mx = float(p1[0] + p2[0]) / 2.0
    my = float(p1[1] + p2[1]) / 2.0
    vx, vy = float(p2[0] - p1[0]), float(p2[1] - p1[1])
    dist = float(np.hypot(vx, vy))
    if dist < 50.0:
        return mx, my
    loc_x, loc_y = _xy_near(mx, my, xs, ys, dist * 1.70, tree)
    if len(loc_x) == 0:
        loc_x, loc_y = xs, ys
    px, py = -vy / dist, vx / dist
    radius = dist / 2.0
    good: list[float] = []
    for t in np.linspace(-0.70 * dist, 0.70 * dist, 29):
        hx = mx + float(t) * px
        hy = my + float(t) * py
        if _n_sector_signature(hx, hy, loc_x, loc_y, radius * 0.12, radius * 1.55, 2):
            good.append(float(t))
    if not good:
        return mx, my
    t = float(np.median(np.array(good)))
    return mx + t * px, my + t * py


def _low_far_side_mass(
    p1: np.ndarray,
    p2: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    radius: float,
    tree: cKDTree | None = None,
) -> bool:
    """True when little coverage sits beyond each peak away from the hub.

    Petals of a 2-sector bowtie end near the peak (far/hub mass ratio ~0.8);
    two overlapping circular towers keep nearly symmetric far/hub halves
    (~0.97). Reject the bowtie merge when either peak looks omni-symmetric.
    """
    hx = float(p1[0] + p2[0]) / 2.0
    hy = float(p1[1] + p2[1]) / 2.0
    loc_x, loc_y = _xy_near(hx, hy, xs, ys, radius * 3.5, tree)
    if len(loc_x) == 0:
        loc_x, loc_y = xs, ys
    far_limit2 = (radius * 1.25) ** 2
    for px, py in (p1, p2):
        vx, vy = float(px - hx), float(py - hy)
        vlen = float(np.hypot(vx, vy)) or 1.0
        ux, uy = vx / vlen, vy / vlen
        dx = loc_x - px
        dy = loc_y - py
        near = (dx * dx + dy * dy) <= far_limit2
        proj = dx * ux + dy * uy
        far_n = int((near & (proj > 0.0)).sum())
        hub_n = int((near & (proj < 0.0)).sum())
        if far_n >= 80 and far_n >= 0.90 * max(hub_n, 1):
            return False
    return True


def _merge_cloverleaf_peaks(
    peaks: list[str],
    xy_by_cell: dict[str, tuple[float, float]],
    cell_ids: list[str],
    xs: np.ndarray,
    ys: np.ndarray,
    min_separation_m: float,
    *,
    core_xs: np.ndarray | None = None,
    core_ys: np.ndarray | None = None,
) -> list[str]:
    """Replace petal-peaks of a 2- or 3-sector site with one site at the hub.

    Depth/signal maxima of a multi-cone footprint sit in the petals; the real
    tower is the junction (a saddle). Equilateral peak triples with a
    three-sector angular signature, and opposite pairs with a two-sector
    signature, collapse to the junction cell. Two nearby circular towers fail
    the sector test and stay split.

    Cloverleaf side lengths are absolute meters (not scaled by peak NMS) so a
    tighter urban peak gate does not shrink the merge window below real petal
    spans (~7 km for 4 km petals at 120°).

    ``xs``/``ys`` are the full footprint (2-sector + far-side test). Weak
    between-petal hexes fill angular gaps, so 3-sector uses ``core_xs``/
    ``core_ys`` when given.
    """
    n = len(peaks)
    if n < 2:
        return peaks
    pxy = np.array([xy_by_cell[p] for p in peaks], dtype=float)
    tree = cKDTree(pxy)
    max_side = _CLOVER_MAX_SIDE_M
    min_side = _CLOVER_MIN_SIDE_M
    tri_xs = core_xs if core_xs is not None else xs
    tri_ys = core_ys if core_ys is not None else ys
    tri_tree = cKDTree(np.column_stack([tri_xs, tri_ys])) if len(tri_xs) else None
    full_tree = cKDTree(np.column_stack([xs, ys])) if len(xs) else None
    triples: list[tuple[float, int, int, int, float, float]] = []
    for i in range(n):
        near = [j for j in tree.query_ball_point(pxy[i], max_side) if j > i]
        for a, j in enumerate(near):
            for k in near[a + 1 :]:
                sides = sorted([
                    float(np.hypot(*(pxy[i] - pxy[j]))),
                    float(np.hypot(*(pxy[i] - pxy[k]))),
                    float(np.hypot(*(pxy[j] - pxy[k]))),
                ])
                if sides[0] < min_side or sides[2] > max_side:
                    continue
                if sides[2] / max(sides[0], 1.0) > 1.85:
                    continue
                cx, cy = pxy[[i, j, k]].mean(axis=0)
                angs = np.sort([
                    np.arctan2(pxy[t][1] - cy, pxy[t][0] - cx) for t in (i, j, k)
                ])
                gaps = np.diff(angs, append=angs[0] + 2 * np.pi)
                if float(gaps.min()) < np.deg2rad(70) or float(gaps.max()) > np.deg2rad(165):
                    continue
                radius = float(np.mean([
                    np.hypot(pxy[t][0] - cx, pxy[t][1] - cy) for t in (i, j, k)
                ]))
                loc_x, loc_y = _xy_near(
                    float(cx), float(cy), tri_xs, tri_ys, radius * 1.40, tri_tree,
                )
                if not _n_sector_signature(
                    float(cx), float(cy), loc_x, loc_y, radius * 0.15, radius * 1.40, 3,
                ):
                    continue
                triples.append((radius, i, j, k, float(cx), float(cy)))
    triples.sort(key=lambda t: -t[0])
    used = np.zeros(n, dtype=bool)
    out: list[str] = []
    for radius, i, j, k, cx, cy in triples:
        if used[i] or used[j] or used[k]:
            continue
        dist = np.hypot(pxy[:, 0] - cx, pxy[:, 1] - cy)
        nearby = np.flatnonzero(dist <= radius * 1.25)
        used[nearby] = True
        jidx = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        out.append(cell_ids[jidx])

    # 2-sector bowtie / missing-third-sector: two petal peaks, hub at the saddle.
    pairs: list[tuple[float, int, int, float, float]] = []
    unused = [i for i in range(n) if not used[i]]
    for a, i in enumerate(unused):
        for j in unused[a + 1 :]:
            side = float(np.hypot(*(pxy[i] - pxy[j])))
            if side < min_side or side > max_side:
                continue
            cx, cy = pxy[[i, j]].mean(axis=0)
            radius = side / 2.0
            loc_x, loc_y = _xy_near(
                float(cx), float(cy), xs, ys, radius * 1.55, full_tree,
            )
            if not _n_sector_signature(
                float(cx), float(cy), loc_x, loc_y, radius * 0.12, radius * 1.55, 2,
            ):
                continue
            if not _low_far_side_mass(pxy[i], pxy[j], xs, ys, radius, full_tree):
                continue
            pairs.append((radius, i, j, float(cx), float(cy)))
    pairs.sort(key=lambda t: -t[0])
    for radius, i, j, cx, cy in pairs:
        if used[i] or used[j]:
            continue
        hx, hy = _two_peak_hub_xy(pxy[i], pxy[j], xs, ys, full_tree)
        dist = np.hypot(pxy[:, 0] - hx, pxy[:, 1] - hy)
        nearby = np.flatnonzero(dist <= max(radius * 1.20, min_separation_m))
        used[nearby] = True
        jidx = int(np.argmin((xs - hx) ** 2 + (ys - hy) ** 2))
        out.append(cell_ids[jidx])

    for i, peak in enumerate(peaks):
        if not used[i]:
            out.append(peak)
    dedup: list[str] = []
    for c in out:
        x, y = xy_by_cell[c]
        if all(
            np.hypot(x - xy_by_cell[a][0], y - xy_by_cell[a][1]) >= min_separation_m
            for a in dedup
        ):
            dedup.append(c)
    return dedup


def _split_component_by_peaks(
    comp: list[str],
    peaks: list[str],
    xy_by_cell: dict[str, tuple[float, float]],
) -> list[tuple[str, list[str]]]:
    """Partition a blob's cells by nearest peak. Returns ``(peak, cells)``."""
    if not peaks:
        return [(comp[0], comp)]
    if len(peaks) == 1:
        return [(peaks[0], comp)]
    peak_xy = np.array([xy_by_cell[p] for p in peaks])
    tree = cKDTree(peak_xy)
    cell_xy = np.array([xy_by_cell[c] for c in comp])
    _, idx = tree.query(cell_xy, k=1)
    parts: list[list[str]] = [[] for _ in peaks]
    for cell, i in zip(comp, idx):
        parts[int(i)].append(cell)
    return [(peaks[i], parts[i]) for i in range(len(peaks)) if parts[i]]


def _peaks_for_component(
    comp: list[str],
    *,
    signal_flat: bool,
    signal_by_cell: dict,
    xy_by_cell: dict[str, tuple[float, float]],
    peak_sep: float,
) -> tuple[list[str], dict[str, int] | None]:
    """Choose tower peaks for one connected blob (signal or depth maxima).

    Cloverleaf merge is applied later over the full footprint so sector
    signatures see below-core hub cells and do not mistake two overlapping
    omnis (core-only rings look like a bowtie) for a 2-sector site.
    """
    sigs = np.array([float(signal_by_cell[c]) for c in comp], dtype=float)
    sig_range = float(sigs.max() - sigs.min()) if len(sigs) else 0.0
    use_depth = signal_flat or sig_range < _SIGNAL_SPLIT_RANGE_DB
    if not use_depth:
        score = {c: float(signal_by_cell[c]) for c in comp}
        # Ignore fringe local maxes: only peaks within 10 dB of the hottest
        # cell. Two macros both peaked at -80 still both qualify.
        min_score = float(sigs.max() - 10.0)
        peaks, plat_frac = _find_score_peaks(
            comp, score, xy_by_cell, peak_sep, min_score=min_score,
        )
        if not (plat_frac >= 0.45 and len(peaks) <= 1):
            local_xy = {c: xy_by_cell[c] for c in comp}
            peaks = _filter_low_prominence_peaks(
                peaks, score, local_xy, min_prominence=_MIN_SIGNAL_PROMINENCE_DB,
            )
            return peaks, None
    depth_sep = (
        max(peak_sep, _LARGE_BLOB_DEPTH_SEP_M)
        if len(comp) >= _LARGE_BLOB_HEXES
        else peak_sep
    )
    depth = _boundary_depth(set(comp))
    peaks = _find_depth_peaks(comp, depth, xy_by_cell, min_separation_m=depth_sep)
    return peaks, depth


def _relative_core_threshold(sig: pd.Series) -> float:
    """Strongest-band cutoff that keeps ~35% of this layer, not a global dBm.

    Walks the provider's own bands from hottest to weakest and stops near
    ``_ADAPTIVE_CORE_QUANTILE``, clamped to ``[_MIN_CORE_FRACTION, _MAX_CORE_FRACTION]``.
    A T-Mobile filing peaked at -50 dBm and an AT&T filing peaked at -90 dBm
    with the same spatial ranks therefore share one core geometry.
    """
    counts = sig.value_counts().sort_index(ascending=False)
    total = max(int(len(sig)), 1)
    target = _ADAPTIVE_CORE_QUANTILE * total
    lo = _MIN_CORE_FRACTION * total
    hi = _MAX_CORE_FRACTION * total
    threshold = float(counts.index[0])
    cum = 0
    for value, n in counts.items():
        nxt = cum + int(n)
        if cum > 0 and nxt > hi and cum >= lo:
            break
        cum = nxt
        threshold = float(value)
        if cum >= target:
            break
    return threshold


def _core_hexes(hex_df: pd.DataFrame, threshold_dbm: float) -> tuple[pd.DataFrame, bool]:
    """Select the strong-signal core as a *relative* slice of this layer.

    ``threshold_dbm`` is accepted for call-site compatibility; membership is
    decided from this provider×vintage's own band histogram so hotter/colder
    filings of the same footprint do not change which hexes seed a site.
    Binary sources (one signal value) return the full footprint and
    ``signal_is_flat=True``.
    """
    sig = hex_df["signal_dbm"]
    flat = int(sig.nunique()) <= 1
    if flat:
        return hex_df.copy(), True
    _ = threshold_dbm
    cutoff = _relative_core_threshold(sig)
    return hex_df[sig >= cutoff].copy(), False


def infer_sites(
    hex_df: pd.DataFrame,
    cfg: Config,
    label_prefix: str = "S",
    *,
    parent_steps_override: int | None = None,
    min_peak_separation_m: float | None = None,
) -> pd.DataFrame:
    """Infer approximate site locations from a provider+vintage hex table.

    ``parent_steps_override`` forces a fixed coarse-grid depth (used by joint
    cross-vintage inference so both vintages share one H3 resolution).
    ``min_peak_separation_m`` defaults to ``peak_separation_m`` from config
    (fallback ``_MIN_PEAK_SEPARATION_M``), independent of cross-vintage
    ``site_match_radius_m``.
    """
    tcfg = cfg.towers
    if hex_df.empty:
        return pd.DataFrame(columns=SITE_COLUMNS)

    strong, signal_flat = _core_hexes(hex_df, float(tcfg["min_signal_band_dbm"]))
    # Auto-scale min_site_hexes to keep the minimum physical blob area consistent
    # across H3 resolutions. Config value is authoritative for the configured
    # site_h3_resolution; infer actual resolution from the data and scale.
    base_hexes_orig = int(tcfg["min_site_hexes"])
    base_hexes = base_hexes_orig
    infer_df = strong
    if parent_steps_override is not None:
        parent_steps = int(parent_steps_override)
    else:
        parent_steps = 0
        if signal_flat and len(strong) >= _FLAT_COARSE_HEX_THRESHOLD:
            parent_steps = _flat_parent_steps(len(strong))
    if parent_steps > 0:
        infer_df = _rollup_flat_for_inference(strong, parent_steps)
        # Scale min hexes by ~7^steps so the physical area threshold is preserved.
        base_hexes = max(3, round(base_hexes / (7 ** parent_steps)))
    if not infer_df.empty:
        try:
            actual_res = h3.get_resolution(infer_df["h3"].iloc[0])
            cfg_res = int(cfg.geography.get("site_h3_resolution", actual_res))
            if actual_res != cfg_res and parent_steps == 0:
                # Scale by inverse hex area ratio: each step in H3 resolution
                # is ~7x finer in area, so keep the total blob area constant.
                area_ratio = h3.average_hexagon_area(cfg_res, unit="km^2") / max(
                    h3.average_hexagon_area(actual_res, unit="km^2"), 1e-9
                )
                base_hexes = max(3, round(base_hexes * area_ratio))
        except Exception:
            pass
    min_hexes = base_hexes
    child_per = (7 ** parent_steps) if parent_steps else 1
    min_total_hexes = min_hexes if parent_steps == 0 else base_hexes_orig
    if len(infer_df) * child_per < min_total_hexes:
        return pd.DataFrame(columns=SITE_COLUMNS)

    peak_sep = float(
        min_peak_separation_m
        if min_peak_separation_m is not None
        else tcfg.get("peak_separation_m", _MIN_PEAK_SEPARATION_M)
    )

    signal_by_cell = dict(zip(infer_df["h3"], infer_df["signal_dbm"]))
    county_by_cell = dict(zip(infer_df["h3"], infer_df["county_geoid"]))
    # Hottest res-9 child inside each rolled-up parent (pin snap, not centroid).
    seed_by_parent: dict[str, str] = {}
    if "_seed_h3" in infer_df.columns:
        seed_by_parent = {
            str(p): str(s)
            for p, s in zip(infer_df["h3"].astype(str), infer_df["_seed_h3"].astype(str))
        }
    # Full footprint counties (for hub snap cells below the relative core cut).
    for h, g in zip(hex_df["h3"].astype(str), hex_df["county_geoid"]):
        county_by_cell.setdefault(h, g)

    all_cells = set(infer_df["h3"].astype(str))
    # Project every core cell once (used for depth peaks, splitting, centroids).
    cell_list = list(all_cells)
    centers = np.array([h3.cell_to_latlng(c) for c in cell_list])  # (lat, lng)
    xs_all, ys_all = _FWD.transform(centers[:, 1], centers[:, 0])
    xy_by_cell = {c: (float(x), float(y)) for c, x, y in zip(cell_list, xs_all, ys_all)}

    sites = []
    site_idx = 0
    # Collect peaks across components first. Relative core can disconnect the
    # petals of a multi-sector site (hub is a signal saddle below the core cut);
    # a second, global cloverleaf merge over the full footprint re-joins them.
    pending: list[tuple[list[str], list[str], dict[str, int] | None]] = []
    all_peaks: list[str] = []
    # After rollup, one parent cell is ~49 res-9 hexes. Compare *child-hex*
    # area to the configured minimum so a rural macro that collapsed to 1–2
    # parents is not discarded as "too small".
    min_comp_hexes = min_hexes if parent_steps == 0 else base_hexes_orig
    for comp in _connected_components(all_cells):
        if len(comp) * child_per < min_comp_hexes:
            continue
        peaks, depth = _peaks_for_component(
            comp,
            signal_flat=signal_flat,
            signal_by_cell=signal_by_cell,
            xy_by_cell=xy_by_cell,
            peak_sep=peak_sep,
        )
        pending.append((comp, peaks, depth))
        all_peaks.extend(peaks)

    # Full-footprint coordinates for sector signature + hub snap (includes
    # below-core hub cells that relative core dropped).
    full_cells = list(dict.fromkeys(
        str(c) for c in hex_df["h3"].astype(str).tolist()
    ))
    if parent_steps > 0:
        # Inference ran on rolled-up parents; snap within that grid.
        full_cells = cell_list
    full_missing = [c for c in full_cells if c not in xy_by_cell]
    if full_missing:
        centers_f = np.array([h3.cell_to_latlng(c) for c in full_missing])
        xs_f, ys_f = _FWD.transform(centers_f[:, 1], centers_f[:, 0])
        for c, x, y in zip(full_missing, xs_f, ys_f):
            xy_by_cell[c] = (float(x), float(y))
    full_xs = np.array([xy_by_cell[c][0] for c in full_cells], dtype=float)
    full_ys = np.array([xy_by_cell[c][1] for c in full_cells], dtype=float)

    if len(all_peaks) >= 2:
        core_xs = np.array([xy_by_cell[c][0] for c in cell_list], dtype=float)
        core_ys = np.array([xy_by_cell[c][1] for c in cell_list], dtype=float)
        merged_peaks = _merge_cloverleaf_peaks(
            list(dict.fromkeys(all_peaks)),
            xy_by_cell,
            full_cells,
            full_xs,
            full_ys,
            peak_sep,
            core_xs=core_xs,
            core_ys=core_ys,
        )
    else:
        merged_peaks = list(dict.fromkeys(all_peaks))

    # Partition every core cell by nearest merged peak (cross-component).
    core_cells = [c for comp, _, _ in pending for c in comp]
    # Prefer depth from the component that owned each cell when present.
    depth_by_cell: dict[str, int] = {}
    for comp, _, depth in pending:
        if depth is not None:
            depth_by_cell.update(depth)

    if not merged_peaks:
        return pd.DataFrame(columns=SITE_COLUMNS)

    lobes = _split_component_by_peaks(core_cells, merged_peaks, xy_by_cell)
    min_lobe_hexes = min_hexes if parent_steps == 0 else base_hexes_orig
    for peak, lobe in lobes:
        if len(lobe) * child_per < max(3, min_lobe_hexes // max(1, len(lobes))):
            continue
        xs = np.array([xy_by_cell[c][0] for c in lobe])
        ys = np.array([xy_by_cell[c][1] for c in lobe])
        sigs = np.array([signal_by_cell[c] for c in lobe])
        # Anchor on the peak/junction cell. A depth-weighted centroid of a
        # 3-sector cloverleaf drifts into the fattest petal and then looks
        # like a different tower in the next vintage.
        if peak in xy_by_cell:
            cx, cy = xy_by_cell[peak]
            nearest_cell = peak
        else:
            w = np.clip(sigs - float(np.min(sigs)) + 1.0, 1.0, None)
            if depth_by_cell:
                w = np.array([float(depth_by_cell.get(c, 1)) for c in lobe], dtype=float)
            w = np.clip(w, 1e-9, None)
            cx = float(np.average(xs, weights=w))
            cy = float(np.average(ys, weights=w))
            nearest_cell = lobe[int(np.argmin(np.hypot(xs - cx, ys - cy)))]
        seed = seed_by_parent.get(str(nearest_cell))
        if seed:
            if seed not in xy_by_cell:
                lat_s, lng_s = h3.cell_to_latlng(seed)
                xs_s, ys_s = _FWD.transform(lng_s, lat_s)
                xy_by_cell[seed] = (float(xs_s), float(ys_s))
            cx, cy = xy_by_cell[seed]
            nearest_cell = seed
        reach = float(np.max(np.hypot(xs - cx, ys - cy)))
        if parent_steps:
            reach *= (7 ** 0.5) ** parent_steps
        # A one-cell lobe has reach 0; attribution then counts 0 serving towers.
        reach = max(reach, _MIN_REACH_M)
        lng, lat = _INV.transform(cx, cy)
        county = county_by_cell.get(nearest_cell)
        sites.append(
            {
                "site_id": f"{label_prefix}{site_idx}",
                "lat": float(lat),
                "lng": float(lng),
                "x_m": cx,
                "y_m": cy,
                "reach_m": reach,
                "n_hexes": int(len(lobe) * (7 ** parent_steps if parent_steps else 1)),
                "max_signal_dbm": float(sigs.max()),
                "mean_signal_dbm": float(sigs.mean()),
                "county_geoid": county,
            }
        )
        site_idx += 1
    return pd.DataFrame(sites, columns=SITE_COLUMNS)


def _count_hexes_per_site(hex_df: pd.DataFrame, sites: pd.DataFrame) -> np.ndarray:
    """Count hexes of ``hex_df`` attributed to each site (within lobe reach)."""
    n = len(sites)
    counts = np.zeros(n, dtype=int)
    if hex_df.empty or sites.empty:
        return counts
    from scipy.spatial import cKDTree

    xs_s = sites["x_m"].to_numpy(dtype=float)
    ys_s = sites["y_m"].to_numpy(dtype=float)
    reach = np.maximum(
        sites.get("lobe_reach_m", sites["reach_m"]).to_numpy(dtype=float),
        _MIN_REACH_M,
    )
    # Prefer lobe_reach when present; else core reach * fallback margin.
    if "lobe_reach_m" not in sites.columns:
        reach = np.maximum(sites["reach_m"].to_numpy(dtype=float) * _LOBE_REACH_FALLBACK_MARGIN, _MIN_REACH_M)

    hex_ids = hex_df["h3"].astype(str).tolist()
    centers = np.array([h3.cell_to_latlng(c) for c in hex_ids])
    xs_h, ys_h = _FWD.transform(centers[:, 1], centers[:, 0])
    tree = cKDTree(np.column_stack([xs_s, ys_s]))
    dist, idx = tree.query(np.column_stack([xs_h, ys_h]), k=1)
    within = dist <= reach[idx]
    for i in idx[within]:
        counts[int(i)] += 1
    return counts


def infer_sites_joint(
    prior_hex: pd.DataFrame,
    current_hex: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Infer sites once on the union footprint; classify by per-vintage counts.

    Site geometry and identity are shared across vintages, so centroid jitter
    cannot manufacture a "new" tower. Returns ``(prior_sites, current_sites)``
    where ``current_sites`` already carry ``site_class`` /
    ``matched_prior_id`` / ``match_dist_m``.
    """
    empty = pd.DataFrame(columns=SITE_COLUMNS)
    if prior_hex.empty and current_hex.empty:
        return empty.copy(), empty.copy()

    # Union footprint for geometry; prefer current county tags, fill from prior.
    parts = []
    if not prior_hex.empty:
        p = prior_hex[["h3", "signal_dbm", "county_geoid"]].copy()
        p["_v"] = "prior"
        parts.append(p)
    if not current_hex.empty:
        c = current_hex[["h3", "signal_dbm", "county_geoid"]].copy()
        c["_v"] = "current"
        parts.append(c)
    combined = pd.concat(parts, ignore_index=True)
    # Prefer current row for county/signal when a hex appears in both.
    # Explicit vintage rank — do NOT sort by the string label ("current" <
    # "prior" alphabetically, which used to keep prior when keep="last").
    combined["_rank"] = combined["_v"].map({"prior": 0, "current": 1}).fillna(0)
    combined = combined.sort_values("_rank", ascending=True)
    union = combined.drop_duplicates(subset=["h3"], keep="last").drop(
        columns=["_v", "_rank"]
    )

    # Deterministic coarse grid from the UNION size so vintages never straddle
    # the 25k threshold independently.
    strong, signal_flat = _core_hexes(union, float(cfg.towers["min_signal_band_dbm"]))
    parent_steps = 0
    if signal_flat and len(strong) >= _FLAT_COARSE_HEX_THRESHOLD:
        parent_steps = _flat_parent_steps(len(strong))

    sites = infer_sites(
        union, cfg, label_prefix="J",
        parent_steps_override=parent_steps,
        min_peak_separation_m=float(
            cfg.towers.get("peak_separation_m", _MIN_PEAK_SEPARATION_M)
        ),
    )
    if sites.empty:
        return empty.copy(), empty.copy()

    sites = compute_lobe_reach(union, sites)
    n_prior = _count_hexes_per_site(prior_hex, sites)
    n_current = _count_hexes_per_site(current_hex, sites)
    sites = sites.copy()
    sites["n_hexes_prior"] = n_prior
    sites["n_hexes_current"] = n_current
    # n_hexes for downstream = current count (fallback to prior for prior-only).
    sites["n_hexes"] = np.where(n_current > 0, n_current, n_prior).astype(int)

    site_class: list[str] = []
    matched_prior: list[str | None] = []
    match_dist: list[float] = []
    for np_i, nc_i, sid in zip(n_prior, n_current, sites["site_id"].tolist()):
        if nc_i <= 0:
            # Prior-only: not present in current_sites.
            site_class.append("prior_only")
            matched_prior.append(None)
            match_dist.append(np.nan)
            continue
        if np_i <= _JOINT_NEW_PRIOR_MAX_HEXES:
            site_class.append("new_site")
            matched_prior.append(None)
            match_dist.append(np.nan)
        else:
            growth = (nc_i - np_i) / max(np_i, 1)
            matched_prior.append(sid)  # same identity
            match_dist.append(0.0)
            site_class.append(
                "expanded_site" if growth >= _JOINT_EXPANSION_GROWTH else "stable_site"
            )
    sites["site_class"] = site_class
    sites["matched_prior_id"] = matched_prior
    sites["match_dist_m"] = match_dist

    prior_mask = n_prior > 0
    current_mask = n_current > 0
    prior_sites = sites.loc[prior_mask].copy()
    if not prior_sites.empty:
        prior_sites["n_hexes"] = prior_sites["n_hexes_prior"].astype(int)
        prior_sites["site_class"] = "prior_site"
    current_sites = sites.loc[current_mask].copy()
    if not current_sites.empty:
        current_sites["n_hexes"] = current_sites["n_hexes_current"].astype(int)
    return prior_sites.reset_index(drop=True), current_sites.reset_index(drop=True)


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

    # Flat / binary Redshift layers have no weaker fringe bands — the core
    # footprint IS the lobe. Projecting every hex again is pure cost.
    if hex_df.empty or (
        "signal_dbm" in hex_df.columns and int(hex_df["signal_dbm"].nunique(dropna=True)) <= 1
    ):
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
    order = np.argsort(idx, kind="mergesort")
    idx_sorted = idx[order]
    dist_sorted = dist[order]
    splits = np.flatnonzero(np.diff(idx_sorted)) + 1
    groups = np.split(dist_sorted, splits)
    site_ids = idx_sorted[np.concatenate([[0], splits])]
    for site_i, dists in zip(site_ids, groups):
        if len(dists) >= _LOBE_REACH_MIN_HEXES:
            emp = float(np.percentile(dists, percentile))
            lobe_reach[int(site_i)] = max(emp, fallback[int(site_i)])

    s["lobe_reach_m"] = np.maximum(lobe_reach, _MIN_REACH_M)
    return s
