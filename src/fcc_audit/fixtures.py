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

import geopandas as gpd
from shapely.geometry import box
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
_LAYOUT = {
    "prior": {
        130077: [(38.75, -98.75, 8.0), (38.75, -98.25, 8.0)],   # AT&T: Alpha, Bravo
        130403: [(39.25, -98.75, 8.0)],                          # T-Mobile: Charlie
        131425: [(39.25, -98.25, 6.0)],                          # Verizon: Delta
        130235: [(38.70, -98.70, 7.0)],                          # UScellular: Alpha
    },
    "current": {
        130077: [(38.75, -98.75, 8.0), (38.75, -98.25, 8.0), (39.25, -98.70, 8.0)],  # +NEW Charlie
        130403: [(39.25, -98.75, 22.0)],                         # SAME tower, huge inflation
        131425: [(39.25, -98.25, 7.5)],                          # modest growth
        130235: [(38.70, -98.70, 7.0)],                          # unchanged
    },
}

# Concentric signal bands as fractions of the outer radius -> minsignal (dBm).
_BANDS = [(0.4, -85.0), (0.7, -95.0), (1.0, -105.0)]


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


def make_fixtures(cfg: Config) -> None:
    """Write synthetic counties + per-provider/vintage coverage GeoJSON."""
    # 1) Synthetic county grid -> interim cache so normalize.load_counties uses it.
    counties = gpd.GeoDataFrame(
        [
            {
                "county_geoid": geoid,
                "county_name": name,
                "state_fips": geoid[:2],
                "geometry": box(lng0, lat0, lng1, lat1),
            }
            for geoid, name, lat0, lat1, lng0, lng1 in _COUNTY_GRID
        ],
        crs="EPSG:4326",
    )
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
