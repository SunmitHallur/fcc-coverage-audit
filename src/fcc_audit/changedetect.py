"""Per-hex and per-county coverage change between two vintages.

Given a provider's normalized hex coverage for the prior and current vintages,
classify every hex as new / lost / upgraded / downgraded / unchanged, then roll
up to county level for scoring.
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

    def _agg(g: pd.DataFrame) -> pd.Series:
        prior_hexes = int(g["signal_prior"].notna().sum())
        current_hexes = int(g["signal_current"].notna().sum())
        new_hexes = int((g["status"] == "new").sum())
        lost_hexes = int((g["status"] == "lost").sum())
        upgraded = int((g["status"] == "upgraded").sum())
        prior_km2 = prior_hexes * hex_km2
        current_km2 = current_hexes * hex_km2
        added_km2 = current_km2 - prior_km2
        pct_increase = (added_km2 / prior_km2) if prior_km2 > 0 else (np.inf if added_km2 > 0 else 0.0)
        return pd.Series(
            {
                "county_name": g["county_name"].iloc[0],
                "state_fips": g["state_fips"].iloc[0],
                "prior_km2": prior_km2,
                "current_km2": current_km2,
                "added_km2": added_km2,
                "pct_increase": pct_increase,
                "new_hexes": new_hexes,
                "lost_hexes": lost_hexes,
                "upgraded_hexes": upgraded,
                "mean_signal_delta": float(g["signal_delta"].mean(skipna=True)),
            }
        )

    out = df.groupby("county_geoid", as_index=False).apply(_agg, include_groups=False)
    out = out.reset_index(drop=True)

    # Area-normalized features (the FCC-verified primary driver). Computed after
    # the groupby because the group key is excluded inside apply().
    area = out["county_geoid"].astype(str).map(area_map).astype(float)
    out["county_area_km2"] = area
    safe_area = area.where(area > 0)
    out["prior_cov_frac"] = out["prior_km2"] / safe_area
    out["current_cov_frac"] = out["current_km2"] / safe_area
    out["added_frac_of_county"] = out["added_km2"] / safe_area
    return out
