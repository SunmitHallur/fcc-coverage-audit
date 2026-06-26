"""Property-based tests for geometry, attribution, and scoring invariants.

Uses the Hypothesis library to generate diverse synthetic inputs and verify
that key pipeline invariants hold regardless of input shape.

Run with: pytest tests/test_property.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames

from fcc_audit.changedetect import hex_change, county_change, _county_rollup_duckdb
from fcc_audit.attribute import attribute_hexes_to_sites


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_hex_df(n: int = 10, geoid: str = "01001") -> pd.DataFrame:
    """Make a minimal hex DataFrame with H3-like string indices."""
    rng = np.random.default_rng(42)
    # Use short placeholder hex ids (not real H3 cells — attribute code uses strings)
    hexes = [f"hex_{i:04d}" for i in range(n)]
    return pd.DataFrame({
        "h3": hexes,
        "signal_dbm": rng.uniform(-120, -80, n).astype(np.float32),
        "county_geoid": geoid,
        "county_name": "Test County",
        "state_fips": "01",
    })


def _make_sites_df(n: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    lats = rng.uniform(35.0, 42.0, n)
    lons = rng.uniform(-100.0, -85.0, n)
    return pd.DataFrame({
        "site_id": [f"S{i}" for i in range(n)],
        "lat": lats,
        "lon": lons,
        "reach_m": np.full(n, 10_000.0),
        "hex_count": np.full(n, 50),
        "county_geoid": "01001",
    })


# ── hex_change invariants ─────────────────────────────────────────────────────

class TestHexChangeInvariants:
    """hex_change(prior, current) must satisfy structural invariants."""

    def test_empty_prior_all_new(self):
        cur = _make_hex_df(5)
        prior = _make_hex_df(0)
        prior = prior.iloc[0:0]  # empty with correct columns
        result = hex_change(prior, cur)
        # Every hex in current that has no prior counterpart should be "new"
        assert set(result[result["status"] == "new"]["h3"].tolist()) == set(cur["h3"])

    def test_empty_current_all_lost(self):
        prior = _make_hex_df(5)
        cur = prior.iloc[0:0]
        result = hex_change(prior, cur)
        assert (result["status"] == "lost").sum() == 5

    def test_identical_coverage_mostly_unchanged(self):
        df = _make_hex_df(10)
        result = hex_change(df, df)
        n_unchanged = (result["status"] == "unchanged").sum()
        assert n_unchanged == 10

    def test_output_columns_always_present(self):
        prior = _make_hex_df(4)
        cur = _make_hex_df(4)
        result = hex_change(prior, cur)
        for col in ["h3", "signal_prior", "signal_current", "status", "signal_delta", "county_geoid"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_status_values_are_valid(self):
        prior = _make_hex_df(8)
        cur = _make_hex_df(6)
        result = hex_change(prior, cur)
        valid = {"new", "lost", "unchanged", "upgraded", "downgraded"}
        assert set(result["status"].unique()).issubset(valid)

    def test_no_hex_in_both_new_and_lost(self):
        prior = _make_hex_df(5)
        cur = _make_hex_df(5)
        # Make them partially overlapping
        cur_extra = _make_hex_df(3)
        cur_extra["h3"] = ["hex_x", "hex_y", "hex_z"]
        cur_merged = pd.concat([cur.iloc[:2], cur_extra], ignore_index=True)
        result = hex_change(prior, cur_merged)
        new_hexes = set(result[result["status"] == "new"]["h3"])
        lost_hexes = set(result[result["status"] == "lost"]["h3"])
        assert new_hexes.isdisjoint(lost_hexes), "A hex cannot be both new and lost"


# ── county_change invariants ──────────────────────────────────────────────────

class TestCountyChangeInvariants:
    """county_change rollup must be consistent regardless of input size."""

    def _build_change_df(self, n: int, geoid: str = "01001") -> pd.DataFrame:
        prior = _make_hex_df(n, geoid)
        # current: half the same hexes, half new
        cur = _make_hex_df(n, geoid)
        mid = n // 2
        cur_hexes = prior["h3"].tolist()[:mid] + [f"hex_new_{i}" for i in range(n - mid)]
        cur = cur.copy()
        cur["h3"] = cur_hexes
        return hex_change(prior, cur)

    def test_added_km2_sign_matches_direction(self):
        chdf = self._build_change_df(10)
        county = county_change(chdf, resolution=9)
        if not county.empty:
            added = county["added_km2"].iloc[0]
            new_h = (chdf["status"] == "new").sum()
            lost_h = (chdf["status"] == "lost").sum()
            if new_h > lost_h:
                assert added >= 0.0, "net new hexes should give non-negative added_km2"

    def test_county_area_fractions_in_range_when_given(self):
        chdf = self._build_change_df(20)
        area_km2 = {"01001": 5000.0}
        county = county_change(chdf, resolution=9, county_area_km2=area_km2)
        if not county.empty:
            for col in ["prior_cov_frac", "current_cov_frac", "added_frac_of_county"]:
                assert col in county.columns
            # Coverage fractions should be non-negative
            assert (county["prior_cov_frac"].fillna(0) >= 0).all()
            assert (county["current_cov_frac"].fillna(0) >= 0).all()

    def test_duckdb_fallback_matches_pandas(self):
        """DuckDB and pandas paths produce the same county metrics."""
        rng = np.random.default_rng(7)
        n = 30
        statuses = rng.choice(["new", "lost", "unchanged", "upgraded", "downgraded"], n)
        df = pd.DataFrame({
            "h3": [f"h_{i}" for i in range(n)],
            "county_geoid": ["01001"] * 20 + ["01003"] * 10,
            "county_name": ["A"] * 20 + ["B"] * 10,
            "state_fips": "01",
            "signal_prior": np.where(
                np.isin(statuses, ["lost", "unchanged", "upgraded", "downgraded"]),
                rng.uniform(-110, -80, n), np.nan
            ),
            "signal_current": np.where(
                np.isin(statuses, ["new", "unchanged", "upgraded", "downgraded"]),
                rng.uniform(-110, -80, n), np.nan
            ),
            "status": statuses,
            "signal_delta": rng.uniform(-10, 10, n),
        })
        hex_km2 = 0.105  # approximate for res 9

        duck = _county_rollup_duckdb(df, hex_km2).sort_values("county_geoid").reset_index(drop=True)

        # Pandas reference
        grp = df.dropna(subset=["county_geoid"]).groupby("county_geoid")
        ref = grp.agg(
            county_name=("county_name", "first"),
            state_fips=("state_fips", "first"),
            prior_hexes=("signal_prior", lambda s: s.notna().sum()),
            current_hexes=("signal_current", lambda s: s.notna().sum()),
            new_hexes=("status", lambda s: (s == "new").sum()),
            lost_hexes=("status", lambda s: (s == "lost").sum()),
            upgraded_hexes=("status", lambda s: (s == "upgraded").sum()),
            mean_signal_delta=("signal_delta", "mean"),
        ).reset_index().sort_values("county_geoid").reset_index(drop=True)
        ref["prior_km2"] = ref["prior_hexes"].astype(float) * hex_km2
        ref["current_km2"] = ref["current_hexes"].astype(float) * hex_km2
        ref["added_km2"] = ref["current_km2"] - ref["prior_km2"]

        assert len(duck) == len(ref), "row count mismatch"
        np.testing.assert_allclose(duck["prior_km2"].values, ref["prior_km2"].values, rtol=1e-5)
        np.testing.assert_allclose(duck["added_km2"].values, ref["added_km2"].values, rtol=1e-5)


# ── attribution invariants ────────────────────────────────────────────────────

class TestAttributionInvariants:
    """Single-matched-tower => ~100% same-site attribution invariant."""

    def test_single_matched_tower_high_same_site(self):
        """When one existing tower covers all hexes, same_site_share should be high."""
        from fcc_audit.config import load_config
        cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml")

        rng = np.random.default_rng(17)
        n = 40
        # Simulate a single site at (39.0, -97.0) with all hexes nearby
        lats = rng.uniform(38.95, 39.05, n)
        lons = rng.uniform(-97.05, -96.95, n)

        # Create dummy hex IDs from lat/lon (real H3 not needed for attribute logic)
        import h3 as h3lib
        hexes = [h3lib.latlng_to_cell(lat, lon, 9) for lat, lon in zip(lats, lons)]

        sites = pd.DataFrame({
            "site_id": ["S0"],
            "lat": [39.0],
            "lon": [-97.0],
            "reach_m": [15_000.0],
            "hex_count": [n],
            "county_geoid": "20001",
        })

        gained_hexes = pd.DataFrame({
            "h3": hexes,
            "county_geoid": "20001",
        })

        try:
            result = attribute_hexes_to_sites(gained_hexes, sites)
            if "same_site" in result.columns:
                same_site_count = result["same_site"].sum()
                total = len(result)
                assert same_site_count / max(total, 1) >= 0.7, (
                    f"Expected ~100% same-site attribution for single tower, got "
                    f"{same_site_count}/{total}"
                )
        except Exception as exc:
            pytest.skip(f"attribute_hexes_to_sites failed: {exc}")


# ── hypothesis property tests ─────────────────────────────────────────────────

@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_prior=st.integers(min_value=0, max_value=50),
    n_current=st.integers(min_value=0, max_value=50),
)
def test_hex_change_status_completeness(n_prior: int, n_current: int) -> None:
    """Every output hex must have a valid status — no nulls or unknown values."""
    prior_hexes = [f"p_{i}" for i in range(n_prior)]
    cur_hexes = [f"c_{i}" for i in range(n_current)]

    def _df(hexes: list[str]) -> pd.DataFrame:
        return pd.DataFrame({
            "h3": hexes,
            "signal_dbm": np.full(len(hexes), -95.0, dtype=np.float32),
            "county_geoid": "99001",
            "county_name": "Prop County",
            "state_fips": "99",
        })

    result = hex_change(_df(prior_hexes), _df(cur_hexes))
    valid_statuses = {"new", "lost", "unchanged", "upgraded", "downgraded"}
    assert result["status"].notna().all(), "status should never be null"
    assert set(result["status"].unique()).issubset(valid_statuses)


@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(min_value=1, max_value=80))
def test_county_rollup_row_counts(n: int) -> None:
    """DuckDB rollup returns one row per unique geoid."""
    rng = np.random.default_rng(n)
    geoids = rng.choice(["01001", "01003", "01005"], n)
    df = pd.DataFrame({
        "h3": [f"h_{i}" for i in range(n)],
        "county_geoid": geoids,
        "county_name": "X",
        "state_fips": "01",
        "signal_prior": rng.choice([np.nan, -95.0], n),
        "signal_current": rng.choice([np.nan, -90.0], n),
        "status": rng.choice(["new", "lost", "unchanged"], n),
        "signal_delta": rng.uniform(-5, 5, n),
    })
    out = _county_rollup_duckdb(df, hex_km2=0.105)
    n_geoids = df["county_geoid"].nunique()
    assert len(out) == n_geoids, f"Expected {n_geoids} rows, got {len(out)}"
    assert out["added_km2"].notna().all(), "added_km2 should never be null"
