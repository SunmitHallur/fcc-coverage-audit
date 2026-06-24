"""Synthetic offline fixtures for development / CI / demos.

Generates two vintages of 5G coverage for the Big-4 in a small synthetic area
with its own synthetic county grid (so no network is needed). The scenario is
designed to exercise the flagging logic:

* AT&T       - builds a genuine NEW tower (legitimate growth, low risk).
* T-Mobile   - massively inflates coverage from an EXISTING tower with no new
               build (the gaming pattern -> should be flagged).
* Verizon    - modest organic growth.
* UScellular - unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path

import zipfile

import geopandas as gpd
import requests
from shapely.geometry import Point, box
from shapely.ops import transform
from pyproj import Transformer

from .acquire import safe_service_name
from .config import Config

log = logging.getLogger(__name__)

_FWD = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
_INV = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)

# Synthetic 2x2 county grid centered near Kansas (avoids real-county overlap).
_COUNTY_GRID = [
    ("90001", "Alpha County", 38.5, 39.0, -99.0, -98.5),
    ("90002", "Bravo County", 38.5, 39.0, -98.5, -98.0),
    ("90003", "Charlie County", 39.0, 39.5, -99.0, -98.5),
    ("90004", "Delta County", 39.0, 39.5, -98.5, -98.0),
]

# Tower layout per provider per vintage: (lat, lng, outer_radius_km).
# AT&T Charlie (90003) is the cross-border demo: towers in Alpha/Delta cover
# Osborne before a new in-county build appears in the current vintage.
_LAYOUT = {
    "prior": {
        130077: [
            (39.10, -98.82, 12.0),   # Alpha (Russell) — north edge, covers Charlie
            (39.12, -98.58, 11.0),   # Alpha — covers SE Charlie
            (39.30, -98.28, 12.0),   # Delta (Mitchell) — covers W Charlie
        ],
        130403: [(39.25, -98.75, 8.0)],                          # T-Mobile: Charlie
        131425: [(39.25, -98.25, 6.0)],                          # Verizon: Delta
        130235: [(38.70, -98.70, 7.0)],                          # UScellular: Alpha
    },
    "current": {
        130077: [
            (39.10, -98.82, 12.0),
            (39.12, -98.58, 11.0),
            (39.30, -98.28, 12.0),
            (39.35, -98.77, 9.0),    # NEW tower in Charlie (Osborne)
        ],
        130403: [(39.25, -98.75, 22.0)],                         # SAME tower, huge inflation
        131425: [(39.25, -98.25, 7.5)],                          # modest growth
        130235: [(38.70, -98.70, 7.0)],                          # unchanged
    },
}

# Concentric signal bands as fractions of the outer radius -> minsignal (dBm).
# More rings → smoother heat map when rasterized for compare panels.
_BANDS = [
    (0.20, -78.0),
    (0.35, -85.0),
    (0.50, -92.0),
    (0.65, -98.0),
    (0.80, -105.0),
    (0.92, -112.0),
    (1.0, -118.0),
]


def _circle(lat: float, lng: float, radius_m: float):
    cx, cy = _FWD.transform(lng, lat)
    from shapely.geometry import Point

    buf = Point(cx, cy).buffer(radius_m)
    return transform(lambda x, y, z=None: _INV.transform(x, y), buf)


# The single synthetic service the fixtures emulate (5G 7/1), matching a real
# FCC per-(provider, service) coverage file.
_FIX_SERVICE = "5G-NR (7/1 Mbps)"


def _tower_rings(lat: float, lng: float, outer_km: float):
    feats = []
    prev = None
    for frac, dbm in _BANDS:
        ring = _circle(lat, lng, outer_km * 1000.0 * frac)
        geom = ring if prev is None else ring.difference(prev)
        feats.append({"geometry": geom, "minsignal": dbm})
        prev = ring
    return feats


def _try_load_tiger(cfg: Config) -> gpd.GeoDataFrame | None:
    """Load TIGER county boundaries when available (offline CI falls back to boxes)."""
    raw = cfg.path("raw") / "tl_us_county.zip"
    if not raw.exists():
        url = cfg.geography.get("counties_url")
        if not url:
            return None
        try:
            log.info("downloading TIGER counties for fixture shapes: %s", url)
            resp = requests.get(url, timeout=300, headers={"user-agent": "fcc-coverage-audit/0.1"})
            resp.raise_for_status()
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(resp.content)
        except Exception as exc:
            log.warning("could not download TIGER counties for fixtures: %s", exc)
            return None
    if not zipfile.is_zipfile(raw):
        return None
    try:
        gdf = gpd.read_file(f"zip://{raw}")[["GEOID", "NAME", "STATEFP", "geometry"]]
        gdf = gdf.rename(columns={"GEOID": "county_geoid", "NAME": "county_name", "STATEFP": "state_fips"})
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        return gdf
    except Exception as exc:
        log.warning("could not read TIGER counties: %s", exc)
        return None


def _fixture_county_geometries(cfg: Config) -> gpd.GeoDataFrame:
    """Synthetic GEOIDs with real TIGER shapes when a county zip is on disk."""
    tiger = _try_load_tiger(cfg)
    rows = []
    for geoid, name, lat0, lat1, lng0, lng1 in _COUNTY_GRID:
        cell = box(lng0, lat0, lng1, lat1)
        cx, cy = (lng0 + lng1) / 2.0, (lat0 + lat1) / 2.0
        geom = cell
        display_name = name
        if tiger is not None and not tiger.empty:
            pt = Point(cx, cy)
            hit = tiger[tiger.contains(pt)]
            if hit.empty:
                hit = tiger[tiger.intersects(cell)]
            if not hit.empty:
                row = hit.iloc[0]
                geom = row.geometry
                display_name = f"{name} ({row['county_name']})"
        rows.append({
            "county_geoid": geoid,
            "county_name": display_name,
            "state_fips": geoid[:2],
            "geometry": geom,
        })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def make_fixtures(cfg: Config) -> None:
    """Write synthetic counties + per-provider/vintage coverage GeoJSON."""
    # 1) Synthetic county grid -> interim cache so normalize.load_counties uses it.
    counties = _fixture_county_geometries(cfg)
    cache = cfg.path("interim") / "tl_us_county.gpkg"
    counties.to_file(cache, driver="GPKG")
    log.info("wrote synthetic counties: %s", cache)

    # 2) Coverage layers.
    fixture_dir = cfg.project_root / cfg.fixture["dir"]
    for vintage_key, vintage_date in [("prior", "2025-06-30"), ("current", "2025-12-31")]:
        vdir = fixture_dir / vintage_date
        vdir.mkdir(parents=True, exist_ok=True)
        for provider_id, towers in _LAYOUT[vintage_key].items():
            feats = []
            for (lat, lng, r_km) in towers:
                feats.extend(_tower_rings(lat, lng, r_km))
            gdf = gpd.GeoDataFrame(feats, crs="EPSG:4326")
            # One file per (provider, service), matching the real FCC layout.
            out = vdir / f"{provider_id}_{safe_service_name(_FIX_SERVICE)}.geojson"
            gdf.to_file(out, driver="GeoJSON")
        log.info("wrote fixture vintage %s (%s)", vintage_key, vintage_date)


def fixture_vintages() -> tuple[str, str]:
    return "2025-12-31", "2025-06-30"
