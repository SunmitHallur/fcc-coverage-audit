"""Per-hex and per-county coverage change between two vintages.

Given a provider's normalized hex coverage for the prior and current vintages,
classify every hex as new / lost / upgraded / downgraded / unchanged, then roll
up to county level for scoring.

The county_change rollup uses DuckDB SQL when available for ~10× faster
aggregation over large parquet datasets (critical at H3 res 9/10).
"""
from __future__ import annotations

import h3
import numpy as np
import pandas as pd

# Signal increase (dBm) above which an already-covered hex counts as "upgraded".
_UPGRADE_DBM = 5.0


def combine_layers(layers: list[pd.DataFrame]) -> pd.DataFrame:
    """Union multiple (tier/environment) hex tables for one provider+vintage.

    Keeps the strongest signal per hex and the (consistent) county attributes.
    """
    non_empty = [df for df in layers if not df.empty]
    if not non_empty:
        return pd.DataFrame(
            columns=["h3", "signal_dbm", "county_geoid", "county_name", "state_fips"]
        )
    allrows = pd.concat(non_empty, ignore_index=True)
    agg = (
        allrows.sort_values("signal_dbm", ascending=False)
        .groupby("h3", as_index=False)
        .agg(
            signal_dbm=("signal_dbm", "max"),
            county_geoid=("county_geoid", "first"),
            county_name=("county_name", "first"),
            state_fips=("state_fips", "first"),
        )
    )
    return agg


def hex_change(prior: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Classify each hex's change between prior and current vintages."""
    p = prior[["h3", "signal_dbm"]].rename(columns={"signal_dbm": "signal_prior"})
    c = current[["h3", "signal_dbm", "county_geoid", "county_name", "state_fips"]].rename(
        columns={"signal_dbm": "signal_current"}
    )
    merged = p.merge(c, on="h3", how="outer")

    # Carry county info from whichever vintage has it (prefer current).
    if "county_geoid" not in merged:
        merged["county_geoid"] = None
    prior_counties = prior.set_index("h3")[["county_geoid", "county_name", "state_fips"]]
    for col in ["county_geoid", "county_name", "state_fips"]:
        fill = merged["h3"].map(prior_counties[col])
        merged[col] = merged[col].fillna(fill)

    sp = merged["signal_prior"]
    sc = merged["signal_current"]
    status = np.full(len(merged), "unchanged", dtype=object)
    status[sp.isna() & sc.notna()] = "new"
    status[sp.notna() & sc.isna()] = "lost"
    both = sp.notna() & sc.notna()
    status[both & ((sc - sp) >= _UPGRADE_DBM)] = "upgraded"
    status[both & ((sp - sc) >= _UPGRADE_DBM)] = "downgraded"
    merged["status"] = status
    merged["signal_delta"] = sc.fillna(sp) - sp.fillna(sc)
    return merged


def county_change(
    change_df: pd.DataFrame,
    resolution: int,
    county_area_km2: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Aggregate hex-level change to per-county metrics.

    If ``county_area_km2`` (geoid -> land area) is supplied, also computes the
    coverage fractions and the area-normalized increase that the FCC-verified
    selection patterns key on (in-county area increase + blanket fill-in).
    """
    hex_km2 = h3.average_hexagon_area(resolution, unit="km^2")
    area_map = county_area_km2 or {}
    df = change_df.dropna(subset=["county_geoid"]).copy()
    if df.empty:
        return pd.DataFrame()

    out = _county_rollup_duckdb(df, hex_km2)

    # Area-normalized features (the FCC-verified primary driver).
    area = out["county_geoid"].astype(str).map(area_map).astype(float)
    out["county_area_km2"] = area
    safe_area = area.where(area > 0)
    out["prior_cov_frac"] = out["prior_km2"] / safe_area
    out["current_cov_frac"] = out["current_km2"] / safe_area
    out["added_frac_of_county"] = out["added_km2"] / safe_area
    return out


def _county_rollup_duckdb(df: pd.DataFrame, hex_km2: float) -> pd.DataFrame:
    """County-level aggregation using DuckDB SQL for ~10× speedup over groupby.apply.

    Falls back to a vectorized pandas path if DuckDB is unavailable.
    """
    try:
        import duckdb  # already a required dep
        con = duckdb.connect()
        con.register("change_df", df)
        out = con.execute(f"""
            SELECT
                county_geoid,
                first(county_name)                                       AS county_name,
                first(state_fips)                                        AS state_fips,
                count(*) FILTER (WHERE signal_prior IS NOT NULL)         AS prior_hexes,
                count(*) FILTER (WHERE signal_current IS NOT NULL)       AS current_hexes,
                count(*) FILTER (WHERE status = 'new')                   AS new_hexes,
                count(*) FILTER (WHERE status = 'lost')                  AS lost_hexes,
                count(*) FILTER (WHERE status = 'upgraded')              AS upgraded_hexes,
                avg(signal_delta)                                        AS mean_signal_delta
            FROM change_df
            WHERE county_geoid IS NOT NULL
            GROUP BY county_geoid
        """).df()
        con.close()
    except Exception:
        # Vectorized pandas fallback
        grp = df.dropna(subset=["county_geoid"]).groupby("county_geoid")
        out = grp.agg(
            county_name=("county_name", "first"),
            state_fips=("state_fips", "first"),
            prior_hexes=("signal_prior", lambda s: s.notna().sum()),
            current_hexes=("signal_current", lambda s: s.notna().sum()),
            new_hexes=("status", lambda s: (s == "new").sum()),
            lost_hexes=("status", lambda s: (s == "lost").sum()),
            upgraded_hexes=("status", lambda s: (s == "upgraded").sum()),
            mean_signal_delta=("signal_delta", "mean"),
        ).reset_index()

    out["prior_km2"] = out["prior_hexes"].astype(float) * hex_km2
    out["current_km2"] = out["current_hexes"].astype(float) * hex_km2
    out["added_km2"] = out["current_km2"] - out["prior_km2"]
    out["pct_increase"] = np.where(
        out["prior_km2"] > 0,
        out["added_km2"] / out["prior_km2"],
        np.where(out["added_km2"] > 0, np.inf, 0.0),
    )
    out["mean_signal_delta"] = out["mean_signal_delta"].astype(float)
    return out.reset_index(drop=True)
