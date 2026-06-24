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
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from shapely.geometry import Polygon, box, shape

log = logging.getLogger(__name__)

# FCC-style: weak (red) → strong (green/teal), dBm -120 .. -50.
_CMAP = LinearSegmentedColormap.from_list(
    "fcc_signal",
    ["#8b0000", "#dc2626", "#f97316", "#eab308", "#84cc16", "#22c55e", "#14b8a6"],
    N=256,
)
_NORM = Normalize(vmin=-120, vmax=-50)

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

_BG_FALLBACK = "#ebe6dc"
_COUNTY_OUTLINE = "#111827"
_DPI = 180
_FIGSIZE = (7.2, 5.6)
_MARGIN_FRAC = 0.18


def _hex_polygon(cell: str) -> Polygon | None:
    try:
        ring = h3.cell_to_boundary(cell)
        if not ring:
            return None
        coords = [(lng, lat) for lat, lng in ring]
        return Polygon(coords)
    except Exception:
        return None


def _hexes_gdf(hexes: list, *, clip_geom=None) -> gpd.GeoDataFrame:
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


def _county_gdf(county_feature: dict | None) -> gpd.GeoDataFrame | None:
    if not county_feature:
        return None
    gdf = gpd.GeoDataFrame.from_features([county_feature], crs="EPSG:4326")
    return None if gdf.empty else gdf


def compute_render_extent(
    county_feature: dict | None,
    sites: list[dict[str, Any]] | None = None,
    *,
    margin_frac: float = _MARGIN_FRAC,
    aspect: float | None = None,
) -> tuple[float, float, float, float] | None:
    """Shared WGS84 extent (minx, miny, maxx, maxy) for prior/current panels."""
    county_gdf = _county_gdf(county_feature)
    if county_gdf is None:
        return None
    minx, miny, maxx, maxy = county_gdf.total_bounds
    for s in sites or []:
        lng, lat = float(s["lng"]), float(s["lat"])
        minx = min(minx, lng)
        miny = min(miny, lat)
        maxx = max(maxx, lng)
        maxy = max(maxy, lat)
    pad_x = (maxx - minx) * margin_frac or 0.04
    pad_y = (maxy - miny) * margin_frac or 0.04
    minx -= pad_x
    maxx += pad_x
    miny -= pad_y
    maxy += pad_y
    if aspect is None:
        aspect = _FIGSIZE[0] / _FIGSIZE[1]
    width = maxx - minx
    height = maxy - miny
    if width / height < aspect:
        extra = (height * aspect - width) / 2.0
        minx -= extra
        maxx += extra
    else:
        extra = (width / aspect - height) / 2.0
        miny -= extra
        maxy += extra
    return (minx, miny, maxx, maxy)


def _add_basemap(ax, extent_3857: tuple[float, float, float, float]) -> bool:
    """Try OpenTopoMap tiles; return False on offline/network failure."""
    try:
        import contextily as ctx

        ax.set_xlim(extent_3857[0], extent_3857[2])
        ax.set_ylim(extent_3857[1], extent_3857[3])
        ctx.add_basemap(
            ax,
            source=ctx.providers.OpenTopoMap,
            crs="EPSG:3857",
            zoom="auto",
            attribution=False,
        )
        return True
    except Exception as exc:
        log.debug("basemap unavailable: %s", exc)
        ax.set_facecolor(_BG_FALLBACK)
        return False


def _add_scale_bar(ax, extent_3857: tuple[float, float, float, float]) -> None:
    """Simple km scale bar bottom-left."""
    minx, miny, maxx, maxy = extent_3857
    width_m = maxx - minx
    if width_m <= 0:
        return
    # Nice round bar length ~15% of map width.
    target_m = width_m * 0.15
    candidates = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
    bar_m = min(candidates, key=lambda c: abs(c - target_m))
    x0 = minx + width_m * 0.04
    y0 = miny + (maxy - miny) * 0.06
    x1 = x0 + bar_m
    ax.plot([x0, x1], [y0, y0], color="#111827", linewidth=3, solid_capstyle="butt", zorder=8)
    ax.plot([x0, x0], [y0 - 200, y0 + 200], color="#111827", linewidth=2, zorder=8)
    ax.plot([x1, x1], [y0 - 200, y0 + 200], color="#111827", linewidth=2, zorder=8)
    label = f"{bar_m // 1000} km" if bar_m >= 1000 else f"{bar_m} m"
    ax.text(
        (x0 + x1) / 2,
        y0 + (maxy - miny) * 0.025,
        label,
        ha="center",
        va="bottom",
        fontsize=7,
        color="#111827",
        zorder=8,
    )


def render_coverage_map(
    *,
    hexes: list,
    sites: list[dict[str, Any]],
    county_feature: dict | None,
    title: str,
    output_path: Path,
    extent_wgs84: tuple[float, float, float, float] | None = None,
    dpi: int = _DPI,
    figsize: tuple[float, float] = _FIGSIZE,
    show_colorbar: bool = True,
) -> Path | None:
    """Render one vintage panel to PNG. Returns path when written."""
    county_gdf = _county_gdf(county_feature)
    if county_gdf is None:
        return None

    all_sites = sites or []
    extent = extent_wgs84 or compute_render_extent(
        county_feature, all_sites, aspect=figsize[0] / figsize[1]
    )
    if extent is None:
        return None

    gdf = _hexes_gdf(hexes)
    clip_box = box(*extent)
    if not gdf.empty:
        gdf = gpd.clip(gdf, clip_box)

    county_3857 = county_gdf.to_crs(epsg=3857)
    if not gdf.empty:
        gdf = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor=_BG_FALLBACK)
    extent_3857 = (
        gpd.GeoSeries([clip_box], crs="EPSG:4326").to_crs(epsg=3857).total_bounds
    )
    ax.set_xlim(extent_3857[0], extent_3857[2])
    ax.set_ylim(extent_3857[1], extent_3857[3])

    _add_basemap(ax, extent_3857)

    if not gdf.empty:
        gdf.plot(
            ax=ax,
            column="signal_dbm",
            cmap=_CMAP,
            norm=_NORM,
            alpha=0.68,
            edgecolor="none",
            zorder=3,
        )

    for s in all_sites:
        lat, lng = float(s["lat"]), float(s["lng"])
        in_county = s.get("in_county", True)
        site_class = str(s.get("site_class", "site"))
        palette = _TOWER_IN if in_county else _TOWER_OUT
        color = palette.get(site_class, "#334155" if not in_county else "#b91c1c")
        pt = gpd.GeoSeries.from_xy([lng], [lat], crs="EPSG:4326").to_crs(epsg=3857)
        ax.plot(
            pt.x.iloc[0],
            pt.y.iloc[0],
            marker="o",
            markersize=6,
            markeredgecolor="#111827",
            markeredgewidth=1.1,
            color=color,
            zorder=6,
        )

    county_3857.boundary.plot(ax=ax, color=_COUNTY_OUTLINE, linewidth=3.0, zorder=7)

    has_cross = any(not s.get("in_county", True) for s in all_sites)
    has_in = any(s.get("in_county", True) for s in all_sites)
    legend_items = []
    if has_in:
        legend_items.append(
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="#475569",
                markeredgecolor="#111827", markersize=6, label="In-county tower",
            )
        )
    if has_cross:
        legend_items.append(
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="#2563eb",
                markeredgecolor="#111827", markersize=6, label="Neighboring tower",
            )
        )
    if legend_items:
        ax.legend(handles=legend_items, loc="lower left", fontsize=7, framealpha=0.92)

    _add_scale_bar(ax, extent_3857)

    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=_NORM)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, aspect=28)
        cbar.set_label("dBm", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    ax.set_title(title, fontsize=10, fontweight="600", pad=6, color="#111827")
    ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    return output_path


def render_county_compare_maps(
    detail: dict[str, Any],
    out_dir: Path,
    *,
    context: dict[str, list] | None = None,
) -> dict[str, str]:
    """Write prior.png / current.png beside the detail JSON; return relative URLs."""
    county = detail.get("county_boundary")
    if not county:
        return {}

    ctx = context or {}
    prior_hexes = ctx.get("prior_hexes") or detail.get("prior_hexes", [])
    current_hexes = ctx.get("current_hexes") or detail.get("current_hexes", [])

    all_sites = (detail.get("sites_prior") or []) + (detail.get("sites_current") or [])
    extent = compute_render_extent(county, all_sites, aspect=_FIGSIZE[0] / _FIGSIZE[1])
    if extent is None:
        return {}

    prior_v = detail.get("prior_vintage") or "Prior"
    current_v = detail.get("current_vintage") or "Current"
    refs: dict[str, str] = {}

    panels = [
        ("prior_hexes", "sites_prior", "prior_map", "prior.png", f"Prior — {prior_v}"),
        ("current_hexes", "sites_current", "current_map", "current.png", f"Current — {current_v}"),
    ]
    hex_by_key = {"prior_hexes": prior_hexes, "current_hexes": current_hexes}
    for hex_key, site_key, json_key, filename, title in panels:
        path = out_dir / filename
        if render_coverage_map(
            hexes=hex_by_key[hex_key],
            sites=detail.get(site_key, []),
            county_feature=county,
            title=title,
            output_path=path,
            extent_wgs84=extent,
            show_colorbar=(json_key == "current_map"),
        ):
            refs[json_key] = filename
    return refs
