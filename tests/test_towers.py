"""Tower-site inference: binary-source splitting + provider-adaptive core."""
from __future__ import annotations

import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fcc_audit.config import load_config  # noqa: E402
from fcc_audit.towers import infer_sites, _core_hexes  # noqa: E402


@pytest.fixture
def cfg():
    c = load_config()
    c.raw["geography"]["site_h3_resolution"] = 9
    c.raw["towers"]["min_site_hexes"] = 25
    return c


def _km_to(lat, lng, target):
    return float(np.hypot(
        (lat - target[0]) * 110.57,
        (lng - target[1]) * 111.32 * np.cos(np.radians(target[0])),
    ))


def test_binary_two_tower_blob_splits(cfg):
    """Flat-signal (Redshift 0/1) coverage: a merged two-lobe blob must yield
    two sites near the true tower positions, not one at the midpoint."""
    t1, t2 = (38.50, -98.50), (38.50, -98.36)  # ~12 km apart
    cells = set()
    for lat, lng in (t1, t2):
        cells |= set(h3.grid_disk(h3.latlng_to_cell(lat, lng, 9), 18))
    df = pd.DataFrame({"h3": sorted(cells), "signal_dbm": 0.0, "county_geoid": "20001"})

    sites = infer_sites(df, cfg, "T")
    assert len(sites) == 2
    for s in sites.to_dict("records"):
        nearest = min(_km_to(s["lat"], s["lng"], t) for t in (t1, t2))
        assert nearest < 2.0, f"site {s['site_id']} is {nearest:.2f} km from any true tower"


def test_binary_single_lobe_yields_one_site(cfg):
    center = (38.50, -98.50)
    cells = h3.grid_disk(h3.latlng_to_cell(*center, 9), 15)
    df = pd.DataFrame({"h3": list(cells), "signal_dbm": 0.0, "county_geoid": "20001"})
    sites = infer_sites(df, cfg, "T")
    assert len(sites) == 1
    assert _km_to(sites.iloc[0]["lat"], sites.iloc[0]["lng"], center) < 1.0


def test_site_home_county_comes_from_centroid_not_lobe_majority(cfg):
    center = h3.latlng_to_cell(38.50, -98.50, 9)
    cells = list(h3.grid_disk(center, 15))
    df = pd.DataFrame({
        "h3": cells,
        "signal_dbm": 0.0,
        # Simulate a border-crossing lobe: almost all coverage cells are across
        # the border, but the inferred tower centroid is in county 20001.
        "county_geoid": ["20001" if cell == center else "29001" for cell in cells],
    })

    sites = infer_sites(df, cfg, "T")

    assert len(sites) == 1
    assert sites.iloc[0]["county_geoid"] == "20001"


def test_adaptive_core_for_strong_bands_only_provider():
    """A provider filing only strong bands (-80/-85) must not have its entire
    footprint treated as 'core' — the cutoff is the hottest band that still
    stays under the 60% cap."""
    center = h3.latlng_to_cell(38.5, -98.5, 9)
    rows = [
        {"h3": c, "signal_dbm": -80.0 if h3.grid_distance(center, c) <= 5 else -85.0,
         "county_geoid": "20001"}
        for c in h3.grid_disk(center, 12)
    ]
    df = pd.DataFrame(rows)
    core, flat = _core_hexes(df, -95.0)
    assert not flat
    assert len(core) < len(df)
    assert core["signal_dbm"].min() == -80.0


def test_weak_rural_core_is_loosened():
    """A filing whose strongest bands are a tiny fraction still keeps a connected core."""
    center = h3.latlng_to_cell(38.5, -98.5, 9)
    rows = []
    for c in h3.grid_disk(center, 18):
        d = h3.grid_distance(center, c)
        if d <= 2:
            dbm = -80.0
        elif d <= 5:
            dbm = -90.0
        else:
            dbm = -110.0
        rows.append({"h3": c, "signal_dbm": dbm, "county_geoid": "20001"})
    df = pd.DataFrame(rows)
    core, flat = _core_hexes(df, -95.0)
    assert not flat
    assert len(core) >= 0.18 * len(df)
    assert core["signal_dbm"].min() <= -100.0


def test_core_ignores_absolute_dbm_scale():
    """The same footprint filed 20 dB colder must keep the same core hexes."""
    center = h3.latlng_to_cell(38.5, -98.5, 9)
    rows_hot, rows_cold = [], []
    for c in h3.grid_disk(center, 14):
        d = h3.grid_distance(center, c)
        dbm = -70.0 - 5.0 * min(d // 2, 8)
        rows_hot.append({"h3": c, "signal_dbm": dbm, "county_geoid": "20001"})
        rows_cold.append({"h3": c, "signal_dbm": dbm - 25.0, "county_geoid": "20001"})
    hot = pd.DataFrame(rows_hot)
    cold = pd.DataFrame(rows_cold)
    core_hot, _ = _core_hexes(hot, -95.0)
    core_cold, _ = _core_hexes(cold, -95.0)
    assert set(core_hot["h3"]) == set(core_cold["h3"])
    assert 0.18 * len(hot) <= len(core_hot) <= 0.60 * len(hot)


def test_hotter_filing_does_not_move_inferred_sites(cfg):
    """Two overlapping towers filed 25 dB colder must land on the same hexes."""
    t1, t2 = (38.50, -98.50), (38.50, -98.40)

    def _layer(offset: float) -> pd.DataFrame:
        rows = []
        for lat, lng in (t1, t2):
            origin = h3.latlng_to_cell(lat, lng, 9)
            for c in h3.grid_disk(origin, 16):
                d = h3.grid_distance(origin, c)
                rows.append({
                    "h3": c,
                    "signal_dbm": -80.0 - 2.0 * d + offset,
                    "county_geoid": "20001",
                })
        return (
            pd.DataFrame(rows)
            .sort_values("signal_dbm", ascending=False)
            .drop_duplicates("h3")
        )

    hot = infer_sites(_layer(0.0), cfg, "H")
    cold = infer_sites(_layer(-25.0), cfg, "C")
    assert len(hot) == len(cold) == 2
    hot_xy = list(zip(hot["lat"], hot["lng"]))
    for rec in cold.to_dict("records"):
        nearest = min(_km_to(rec["lat"], rec["lng"], t) for t in hot_xy)
        assert nearest < 0.35, f"cold site moved {nearest:.2f} km vs hot filing"


def test_flat_signal_detection():
    df = pd.DataFrame({
        "h3": ["8926e64240fffff", "8926e642407ffff"],
        "signal_dbm": [0.0, 0.0],
        "county_geoid": ["20001", "20001"],
    })
    core, flat = _core_hexes(df, -95.0)
    assert flat
    assert len(core) == 2


def test_flat_coarse_rollup_still_finds_site(cfg, monkeypatch):
    """Large flat footprints roll up to parents; a single lobe still yields one site."""
    import fcc_audit.towers as towers_mod

    monkeypatch.setattr(towers_mod, "_FLAT_COARSE_HEX_THRESHOLD", 50)
    center = (38.50, -98.50)
    cells = list(h3.grid_disk(h3.latlng_to_cell(*center, 9), 12))
    assert len(cells) >= 50
    df = pd.DataFrame({"h3": cells, "signal_dbm": 0.0, "county_geoid": "20001"})
    sites = infer_sites(df, cfg, "T")
    assert len(sites) == 1
    # Pin snaps to the hottest child, not the parent centroid (~1–2 km error).
    assert _km_to(sites.iloc[0]["lat"], sites.iloc[0]["lng"], center) < 1.5
    assert float(sites.iloc[0]["reach_m"]) >= 3000.0


def test_flat_rollup_keeps_single_parent_lobes(cfg, monkeypatch):
    """Two small rural disks must survive rollup even if each is 1–2 parent cells."""
    import fcc_audit.towers as towers_mod

    monkeypatch.setattr(towers_mod, "_FLAT_COARSE_HEX_THRESHOLD", 40)
    t1, t2 = (38.50, -98.50), (38.80, -98.50)  # ~33 km
    cells = []
    for lat, lng in (t1, t2):
        cells.extend(h3.grid_disk(h3.latlng_to_cell(lat, lng, 9), 4))
    df = pd.DataFrame({"h3": cells, "signal_dbm": 0.0, "county_geoid": "20001"})
    assert len(df) >= 40
    sites = infer_sites(df, cfg, "T")
    assert len(sites) == 2, f"rolled-up rural disks collapsed to {len(sites)} site(s)"
    for s in sites.to_dict("records"):
        nearest = min(_km_to(s["lat"], s["lng"], t) for t in (t1, t2))
        assert nearest < 1.5, f"site is {nearest:.2f} km from any true tower"
        assert float(s["reach_m"]) >= 3000.0


def test_minsignal_large_core_does_not_rollup(cfg, monkeypatch):
    """Real minsignal stays on res 9 — 0.8 km peaks must not collapse into one parent."""
    import fcc_audit.towers as towers_mod

    monkeypatch.setattr(towers_mod, "_FLAT_COARSE_HEX_THRESHOLD", 50)
    t1, t2 = (38.50, -98.50), (38.5072, -98.50)  # ~0.8 km
    rows = []
    for lat, lng in (t1, t2):
        origin = h3.latlng_to_cell(lat, lng, 9)
        for c in h3.grid_disk(origin, 10):
            d = h3.grid_distance(origin, c)
            rows.append({
                "h3": c,
                "signal_dbm": -80.0 - 2.0 * d,
                "county_geoid": "20001",
            })
    df = pd.DataFrame(rows).sort_values("signal_dbm", ascending=False).drop_duplicates("h3")
    assert len(df) >= 50
    sites = infer_sites(df, cfg, "T")
    assert len(sites) == 2, f"minsignal pair collapsed to {len(sites)} site(s) (rollup?)"
    for s in sites.to_dict("records"):
        nearest = min(_km_to(s["lat"], s["lng"], t) for t in (t1, t2))
        assert nearest < 0.6, f"site is {nearest:.2f} km from true peak"


def _cloverleaf_cells(
    lat: float,
    lng: float,
    *,
    hub: int = 4,
    lobe_r: int = 10,
    lobe_off_km: float = 4.0,
    angles: tuple[float, ...] = (0.0, 120.0, 240.0),
):
    cells = set(h3.grid_disk(h3.latlng_to_cell(lat, lng, 9), hub))
    for ang in angles:
        dlat = (lobe_off_km / 110.57) * np.cos(np.radians(ang))
        dlng = (lobe_off_km / (111.32 * np.cos(np.radians(lat)))) * np.sin(np.radians(ang))
        cells |= set(h3.grid_disk(h3.latlng_to_cell(lat + dlat, lng + dlng, 9), lobe_r))
    return cells


def test_binary_cloverleaf_is_one_site_at_hub(cfg):
    """Three cones from one point are one 3-sector site, not three petal towers."""
    center = (38.50, -98.50)
    cells = _cloverleaf_cells(*center)
    df = pd.DataFrame({"h3": list(cells), "signal_dbm": 0.0, "county_geoid": "20001"})
    sites = infer_sites(df, cfg, "T")
    assert len(sites) == 1, f"cloverleaf split into {len(sites)} sites"
    assert _km_to(sites.iloc[0]["lat"], sites.iloc[0]["lng"], center) < 1.5


def test_binary_cloverleaf_petal_growth_is_not_a_new_tower(cfg):
    """Growing one sector of an existing cloverleaf must not mint a new site."""
    from fcc_audit.towers import infer_sites_joint

    center = (38.50, -98.50)
    prior = pd.DataFrame({
        "h3": list(_cloverleaf_cells(*center)),
        "signal_dbm": 0.0,
        "county_geoid": "20001",
    })
    cur_cells = set(prior["h3"])
    dlat = (6.0 / 110.57)
    cur_cells |= set(h3.grid_disk(h3.latlng_to_cell(center[0] + dlat, center[1], 9), 14))
    current = pd.DataFrame({
        "h3": list(cur_cells), "signal_dbm": 0.0, "county_geoid": "20001",
    })
    _prior_sites, current_sites = infer_sites_joint(prior, current, cfg)
    assert len(current_sites) == 1
    assert current_sites.iloc[0]["site_class"] in {"stable_site", "expanded_site"}
    assert current_sites.iloc[0]["site_class"] != "new_site"


def test_binary_two_lobe_is_one_site_at_hub(cfg):
    """Two cones (missing a third sector) are one site, not two petal towers."""
    center = (38.50, -98.50)
    cells = _cloverleaf_cells(*center, angles=(0.0, 180.0))
    df = pd.DataFrame({"h3": list(cells), "signal_dbm": 0.0, "county_geoid": "20001"})
    sites = infer_sites(df, cfg, "T")
    assert len(sites) == 1, f"2-lobe split into {len(sites)} sites"
    assert _km_to(sites.iloc[0]["lat"], sites.iloc[0]["lng"], center) < 1.5


def test_binary_two_lobe_120deg_is_one_site(cfg):
    """A cloverleaf missing one sector still collapses to the hub."""
    center = (38.50, -98.50)
    cells = _cloverleaf_cells(*center, angles=(0.0, 120.0))
    df = pd.DataFrame({"h3": list(cells), "signal_dbm": 0.0, "county_geoid": "20001"})
    sites = infer_sites(df, cfg, "T")
    assert len(sites) == 1, f"2-lobe 120° split into {len(sites)} sites"
    assert _km_to(sites.iloc[0]["lat"], sites.iloc[0]["lng"], center) < 1.5


def test_two_nearby_circular_towers_stay_split(cfg):
    """Two omnis ~8 km apart must not merge as a 2-sector bowtie."""
    t1, t2 = (38.50, -98.50), (38.50, -98.41)  # ~7.9 km
    cells = set()
    for lat, lng in (t1, t2):
        cells |= set(h3.grid_disk(h3.latlng_to_cell(lat, lng, 9), 16))
    df = pd.DataFrame({"h3": sorted(cells), "signal_dbm": 0.0, "county_geoid": "20001"})
    sites = infer_sites(df, cfg, "T")
    assert len(sites) == 2, f"nearby omnis collapsed to {len(sites)} site(s)"
    for s in sites.to_dict("records"):
        nearest = min(_km_to(s["lat"], s["lng"], t) for t in (t1, t2))
        assert nearest < 2.0, f"site {s['site_id']} is {nearest:.2f} km from any true tower"


def test_three_separate_towers_are_not_merged_as_cloverleaf(cfg):
    """Three real circular sites in a triangle stay three, not one hub."""
    cells: set[str] = set()
    for ang in (0.0, 120.0, 240.0):
        dlat = (12.0 / 110.57) * np.cos(np.radians(ang))
        dlng = (12.0 / (111.32 * np.cos(np.radians(38.50)))) * np.sin(np.radians(ang))
        cells |= set(h3.grid_disk(h3.latlng_to_cell(38.50 + dlat, -98.50 + dlng, 9), 12))
    df = pd.DataFrame({"h3": list(cells), "signal_dbm": 0.0, "county_geoid": "20001"})
    sites = infer_sites(df, cfg, "T")
    assert len(sites) == 3


def test_signal_peaks_split_two_overlapping_towers(cfg):
    """Real minsignal: two local maxima in one connected core stay two sites."""
    t1, t2 = (38.50, -98.50), (38.50, -98.40)  # ~8.7 km
    rows = []
    for lat, lng in (t1, t2):
        origin = h3.latlng_to_cell(lat, lng, 9)
        for c in h3.grid_disk(origin, 16):
            d = h3.grid_distance(origin, c)
            rows.append({
                "h3": c,
                "signal_dbm": -80.0 - 2.0 * d,
                "county_geoid": "20001",
            })
    df = pd.DataFrame(rows).sort_values("signal_dbm", ascending=False).drop_duplicates("h3")
    sites = infer_sites(df, cfg, "T")
    assert len(sites) == 2
    for s in sites.to_dict("records"):
        nearest = min(_km_to(s["lat"], s["lng"], t) for t in (t1, t2))
        assert nearest < 2.0


def test_signal_inflated_single_tower_stays_one_site(cfg):
    """A large gradient lobe from one tower must not fragment into new sites."""
    center = (38.50, -98.50)
    origin = h3.latlng_to_cell(*center, 9)
    rows = [{
        "h3": c,
        "signal_dbm": -78.0 - 1.5 * h3.grid_distance(origin, c),
        "county_geoid": "20001",
    } for c in h3.grid_disk(origin, 22)]
    sites = infer_sites(pd.DataFrame(rows), cfg, "T")
    assert len(sites) == 1
    assert _km_to(sites.iloc[0]["lat"], sites.iloc[0]["lng"], center) < 1.0


def test_cloverleaf_hexes_are_attributed(cfg):
    """The hub site's reach must cover the cones, not leave them unattributed."""
    from fcc_audit.attribute import attribute_hexes_to_sites

    center = (38.50, -98.50)
    df = pd.DataFrame({
        "h3": list(_cloverleaf_cells(*center)),
        "signal_dbm": 0.0,
        "county_geoid": "20001",
    })
    sites = infer_sites(df, cfg, "T")
    idx, _, attr = attribute_hexes_to_sites(df, sites)
    assert float((idx >= 0).mean()) >= 0.95
    assert float((attr == "unattributed").mean()) <= 0.05


def test_lobe_reach_skips_projection_for_flat_signal():
    from fcc_audit.towers import compute_lobe_reach

    hex_df = pd.DataFrame({
        "h3": ["8926e64240fffff"],
        "signal_dbm": [0.0],
        "county_geoid": ["20001"],
    })
    sites = pd.DataFrame([{
        "site_id": "C0", "lat": 38.5, "lng": -98.5,
        "x_m": 0.0, "y_m": 0.0, "reach_m": 4000.0,
        "n_hexes": 10, "max_signal_dbm": 0.0, "mean_signal_dbm": 0.0,
        "county_geoid": "20001",
    }])
    out = compute_lobe_reach(hex_df, sites)
    assert float(out.iloc[0]["lobe_reach_m"]) >= 4000.0 * 2.5

SYNTH_LOCS = [
    ("unit_ks", 38.50, -98.50),
    ("atlanta", 33.75, -84.39),
    ("seattle", 47.61, -122.33),
    ("manhattan_ks", 39.18, -96.57),
    ("newark", 40.74, -74.17),
    ("monroe_la", 32.51, -92.12),
    ("logan_ut", 41.74, -111.83),
    ("wichita", 37.69, -97.34),
]


def _petal_hex(lat, lng):
    hub = h3.latlng_to_cell(lat, lng, 9)
    cells = set(h3.grid_disk(hub, 4))
    origins = []
    for ang in (0.0, 120.0, 240.0):
        dlat = (4.0 / 110.57) * np.cos(np.radians(ang))
        dlng = (4.0 / (111.32 * np.cos(np.radians(lat)))) * np.sin(np.radians(ang))
        origin = h3.latlng_to_cell(lat + dlat, lng + dlng, 9)
        origins.append(origin)
        cells |= set(h3.grid_disk(origin, 10))
    rows = []
    for c in cells:
        d_pet = min(h3.grid_distance(o, c) for o in origins)
        rows.append({"h3": c, "signal_dbm": -78.0 - 2.5 * d_pet, "county_geoid": "20001"})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("name,lat,lng", SYNTH_LOCS)
def test_signal_petal_peaked_cloverleaf_is_one_site_everywhere(cfg, name, lat, lng):
    """T2: petal-peaked 3-sector must merge at more than the unit-test lat/lng."""
    sites = infer_sites(_petal_hex(lat, lng), cfg, "T")
    assert len(sites) == 1, f"{name}: petal cloverleaf split into {len(sites)} sites"
    assert _km_to(sites.iloc[0]["lat"], sites.iloc[0]["lng"], (lat, lng)) < 2.0


@pytest.mark.parametrize("sep_km,expect", [(0.8, 2), (1.6, 2)])
@pytest.mark.parametrize("name,lat,lng", SYNTH_LOCS)
def test_signal_close_macros_split_everywhere(cfg, name, lat, lng, sep_km, expect):
    """T3: 0.8/1.6 km signal-peaked macros split at every US test location."""
    t2_lng = lng + sep_km / (111.32 * np.cos(np.radians(lat)))
    rows = []
    for la, lo in ((lat, lng), (lat, t2_lng)):
        origin = h3.latlng_to_cell(la, lo, 9)
        for c in h3.grid_disk(origin, 10):
            d = h3.grid_distance(origin, c)
            rows.append({"h3": c, "signal_dbm": -80.0 - 2.0 * d, "county_geoid": "20001"})
    df = pd.DataFrame(rows).sort_values("signal_dbm", ascending=False).drop_duplicates("h3")
    sites = infer_sites(df, cfg, "T")
    assert len(sites) == expect, (
        f"{name} {sep_km} km: got {len(sites)} sites, expected {expect}"
    )


@pytest.mark.parametrize("name,lat,lng", SYNTH_LOCS)
def test_signal_sub_cell_pair_does_not_fragment(cfg, name, lat, lng):
    """0.4 km is inside one/two H3-9 cells; must not mint a third site."""
    t2_lng = lng + 0.4 / (111.32 * np.cos(np.radians(lat)))
    rows = []
    for la, lo in ((lat, lng), (lat, t2_lng)):
        origin = h3.latlng_to_cell(la, lo, 9)
        for c in h3.grid_disk(origin, 10):
            d = h3.grid_distance(origin, c)
            rows.append({"h3": c, "signal_dbm": -80.0 - 2.0 * d, "county_geoid": "20001"})
    df = pd.DataFrame(rows).sort_values("signal_dbm", ascending=False).drop_duplicates("h3")
    sites = infer_sites(df, cfg, "T")
    assert 1 <= len(sites) <= 2, f"{name} 0.4 km: got {len(sites)} sites"


def test_joint_union_prefers_current_signal(cfg):
    """Overlapping hexes: current signal must win (not alphabetical prior)."""
    from fcc_audit.towers import infer_sites_joint

    cells = list(h3.grid_disk(h3.latlng_to_cell(38.5, -98.5, 9), 10))
    prior = pd.DataFrame({"h3": cells, "signal_dbm": -50.0, "county_geoid": "20001"})
    current = pd.DataFrame({"h3": cells, "signal_dbm": -110.0, "county_geoid": "20001"})
    p = prior.assign(_v="prior")
    c = current.assign(_v="current")
    combined = pd.concat([p, c], ignore_index=True)
    combined["_rank"] = combined["_v"].map({"prior": 0, "current": 1})
    union = (
        combined.sort_values("_rank", ascending=True)
        .drop_duplicates(subset=["h3"], keep="last")
    )
    assert float(union["signal_dbm"].iloc[0]) == -110.0

    prior_sites, current_sites = infer_sites_joint(prior, current, cfg)
    assert len(current_sites) >= 1
    assert float(current_sites["max_signal_dbm"].max()) <= -100.0


def test_peak_nms_is_deterministic(cfg):
    """Tied signal-band peaks must not flip which site survives across runs."""
    t1, t2 = (38.50, -98.50), (38.50, -98.42)
    rows = []
    for lat, lng in (t1, t2):
        origin = h3.latlng_to_cell(lat, lng, 9)
        for c in h3.grid_disk(origin, 14):
            d = h3.grid_distance(origin, c)
            band = -80.0 - 5.0 * (d // 3)
            rows.append({"h3": c, "signal_dbm": band, "county_geoid": "20001"})
    df = pd.DataFrame(rows).sort_values("signal_dbm", ascending=False).drop_duplicates("h3")
    a = infer_sites(df, cfg, "A")
    b = infer_sites(df.sample(frac=1.0, random_state=1).reset_index(drop=True), cfg, "B")
    assert len(a) == len(b)
    a_xy = sorted(zip(a["lat"].round(5), a["lng"].round(5)))
    b_xy = sorted(zip(b["lat"].round(5), b["lng"].round(5)))
    assert a_xy == b_xy
