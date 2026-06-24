"""Static FCC-style coverage comparison maps (PNG) for the web detail panel."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from shapely.geometry import Polygon

log = logging.getLogger(__name__)

# Strong fringe (red) → orange → green → yellow-green, matching the web legend.
_CMAP = LinearSegmentedColormap.from_list(
    "fcc_signal", ["#dc2626", "#f97316", "#22c55e", "#a3e635"], N=256
)

_TOWER_IN = {
    "new_site": "#15803d",
    "expanded_site": "#c2410c",
    "prior_site": "#475569",
}
_TOWER_OUT = {
    "new_site": "#2563eb",
    "expanded_site": "#7c3aed",
    "prior_site": "#64748b",
}


def _hex_polygon(cell: str) -> Polygon | None:
    try:
        ring = h3.cell_to_boundary(cell, geo_json=True)
        if not ring:
            return None
        return Polygon(ring)
    except Exception:
        return None


def _hexes_gdf(hexes: list, clip_geom) -> gpd.GeoDataFrame:
    rows = []
    for item in hexes or []:
        if not item or len(item) < 2:
            continue
        cell, dbm = str(item[0]), float(item[1])
        poly = _hex_polygon(cell)
        if poly is None or poly.is_empty:
            continue
        rows.append({"h3": cell, "signal_dbm": dbm, "geometry": poly})
    if not rows:
        return gpd.GeoDataFrame(columns=["h3", "signal_dbm", "geometry"], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    if clip_geom is not None and not clip_geom.is_empty:
        gdf = gpd.clip(gdf, clip_geom)
    return gdf


def render_coverage_map(
    *,
    hexes: list,
    sites: list[dict[str, Any]],
    county_feature: dict | None,
    title: str,
    output_path: Path,
    dpi: int = 160,
    figsize: tuple[float, float] = (6.8, 5.2),
) -> Path | None:
    """Render one vintage panel to PNG. Returns path when written."""
    county_gdf = (
        gpd.GeoDataFrame.from_features([county_feature], crs="EPSG:4326")
        if county_feature
        else None
    )
    if county_gdf is None or county_gdf.empty:
        return None

    county_geom = county_gdf.geometry.union_all()
    gdf = _hexes_gdf(hexes, county_geom)

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="#e8e4dc")
    ax.set_facecolor("#ebe6dc")

    county_gdf.plot(ax=ax, facecolor="#f5f0e6", edgecolor="none", zorder=1)

    if not gdf.empty:
        gdf.plot(
            ax=ax,
            column="signal_dbm",
            cmap=_CMAP,
            vmin=-120,
            vmax=-70,
            alpha=0.9,
            edgecolor="none",
            zorder=2,
        )

    for s in sites or []:
        lat, lng = float(s["lat"]), float(s["lng"])
        in_county = s.get("in_county", True)
        site_class = str(s.get("site_class", "site"))
        palette = _TOWER_IN if in_county else _TOWER_OUT
        color = palette.get(site_class, "#334155" if not in_county else "#b91c1c")
        ax.plot(
            lng,
            lat,
            marker="o",
            markersize=7,
            markeredgecolor="#111827",
            markeredgewidth=1.2,
            color=color,
            zorder=4,
        )

    county_gdf.boundary.plot(ax=ax, color="#111827", linewidth=2.8, zorder=5)

    has_cross = any(not s.get("in_county", True) for s in (sites or []))
    has_in = any(s.get("in_county", True) for s in (sites or []))
    if has_cross or has_in:
        from matplotlib.lines import Line2D

        legend_items = []
        if has_in:
            legend_items.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#475569",
                       markeredgecolor="#111827", markersize=7, label="In-county tower")
            )
        if has_cross:
            legend_items.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#64748b",
                       markeredgecolor="#111827", markersize=7, label="Neighboring tower")
            )
        ax.legend(handles=legend_items, loc="lower right", fontsize=8, framealpha=0.92)

    ax.set_title(title, fontsize=11, fontweight="600", pad=8, color="#111827")
    ax.set_axis_off()

    minx, miny, maxx, maxy = county_gdf.total_bounds
    for s in sites or []:
        lng, lat = float(s["lng"]), float(s["lat"])
        minx = min(minx, lng)
        miny = min(miny, lat)
        maxx = max(maxx, lng)
        maxy = max(maxy, lat)
    pad_x = (maxx - minx) * 0.08 or 0.04
    pad_y = (maxy - miny) * 0.08 or 0.04
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return output_path


def render_county_compare_maps(detail: dict[str, Any], out_dir: Path) -> dict[str, str]:
    """Write prior.png / current.png beside the detail JSON; return relative URLs."""
    county = detail.get("county_boundary")
    if not county:
        return {}

    prior_v = detail.get("prior_vintage") or "Prior"
    current_v = detail.get("current_vintage") or "Current"
    refs: dict[str, str] = {}

    panels = [
        ("prior_hexes", "sites_prior", "prior_map", "prior.png", f"Prior — {prior_v}"),
        ("current_hexes", "sites_current", "current_map", "current.png", f"Current — {current_v}"),
    ]
    for hex_key, site_key, json_key, filename, title in panels:
        path = out_dir / filename
        if render_coverage_map(
            hexes=detail.get(hex_key, []),
            sites=detail.get(site_key, []),
            county_feature=county,
            title=title,
            output_path=path,
        ):
            refs[json_key] = filename
    return refs
