"""Batch parquet persistence for scored rows and coverage snapshots."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def save_batch_scored(
    scored: pd.DataFrame,
    scored_dir: Path,
    *,
    service_label: str,
    states: list[str],
    meta: dict[str, Any],
) -> Path:
    """Persist one batch's scored rows for later web bundle assembly."""
    scored_dir.mkdir(parents=True, exist_ok=True)
    states_key = "-".join(sorted(states)) if states else "all"
    safe_svc = service_label.replace("/", "-").replace(" ", "")
    path = scored_dir / f"scored_{safe_svc}_{states_key}.parquet"
    df = scored.copy()
    df["batch_ts"] = datetime.now(timezone.utc).isoformat()
    df["batch_states"] = ",".join(states) if states else "all"
    df["batch_current"] = meta.get("current")
    df["batch_prior"] = meta.get("prior")
    df.to_parquet(path, index=False)
    log.info("saved batch scored: %s (%d rows)", path.name, len(df))
    return path


def load_accumulated_scored(scored_dir: Path) -> pd.DataFrame:
    """Load and merge all batch parquet files from ``data/processed/scored/``."""
    if not scored_dir.exists():
        return pd.DataFrame()
    parts = sorted(scored_dir.glob("scored_*.parquet"))
    if not parts:
        return pd.DataFrame()
    dfs = [pd.read_parquet(p) for p in parts]
    combined = pd.concat(dfs, ignore_index=True)
    # Drop synthetic fixture counties present from earlier test runs.
    combined = combined[~combined["county_geoid"].astype(str).str.startswith("900")]
    # De-duplicate on provider + service + county, keeping the latest batch.
    if "batch_ts" in combined.columns:
        combined = combined.sort_values("batch_ts").drop_duplicates(
            subset=["provider_id", "technology", "county_geoid"], keep="last"
        )
    else:
        combined = combined.drop_duplicates(
            subset=["provider_id", "technology", "county_geoid"], keep="last"
        )
    return combined.reset_index(drop=True)


def load_accumulated_coverage(coverage_dir: Path) -> pd.DataFrame:
    """Load and merge all batch coverage snapshot parquet files."""
    if not coverage_dir.exists():
        return pd.DataFrame()
    parts = sorted(coverage_dir.glob("coverage_*.parquet"))
    if not parts:
        return pd.DataFrame()
    combined = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    dedup = ["provider_id", "technology", "county_geoid", "vintage", "h3"]
    present = [c for c in dedup if c in combined.columns]
    if "batch_ts" in combined.columns:
        combined = combined.sort_values("batch_ts").drop_duplicates(subset=present, keep="last")
    elif present:
        combined = combined.drop_duplicates(subset=present, keep="last")
    return combined.reset_index(drop=True)
