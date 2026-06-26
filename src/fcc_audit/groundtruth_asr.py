"""FCC Antenna Structure Registration (ASR) ground-truth ingestion.

ASR is the FCC's public database of registered antenna structures with
geocoded locations and application/construction dates. It provides an
*independent* signal — derived from tower registration, not from coverage
maps — that answers: "was any structure actually built or registered in this
county during the window between two BDC vintages?"

This breaks the circularity in the current detection engine, which infers
tower sites from the very coverage data it is judging. When the pipeline
says "coverage grew but we see no new inferred tower", that inference is
circular; ASR says "coverage grew and there is no FCC-registered structure
constructed here" — a genuinely independent, regulator-admissible fact.

Usage
-----
  from fcc_audit.groundtruth_asr import fetch_asr_labels

  labels = fetch_asr_labels(
      prior_vintage="June 30, 2025",
      current_vintage="December 31, 2025",
      cache_dir=Path("data/groundtruth/asr"),
  )
  # labels: DataFrame with columns:
  #   county_geoid, has_new_structure, new_structure_count, min_app_date

The join to BDC carriers is intentionally county-level (not carrier-level)
because the ASR owner field does not reliably map to BDC provider IDs. A
county-level "was anything built here?" label is robust and defensible.
See docs/methodology.md for the ASR-to-carrier join caveat.
"""
from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

log = logging.getLogger(__name__)

# FCC ASR full database download (no auth, public).
# CO (construction) records with location + dates.
_ASR_CO_URL = (
    "https://wireless2.fcc.gov/UlsApp/AsrSearch/asrRegistration.jsp"
    "?fileType=CO&downloadFile=yes"
)
# Fallback: FCC bulk data download page for the CO file.
_ASR_CO_FALLBACK_URL = (
    "https://wireless2.fcc.gov/UlsApp/AsrSearch/asrRegistration.jsp"
    "?fileType=CO&downloadFile=yes"
)
# The actual bulk download that ships as a ZIP:
_ASR_BULK_URL = "https://wireless2.fcc.gov/UlsApp/AsrSearch/asrRegistration.zip"

# Column positions in the FCC ASR CO fixed-width / pipe-delimited format.
# Format reference: https://www.fcc.gov/wireless/bureau-divisions/technologies-systems-and-innovation-division/tower-construction-notification
_ASR_COLS = {
    "registration_number": 0,
    "unique_system_identifier": 1,
    "lat_degrees": 4,
    "lat_minutes": 5,
    "lat_seconds": 6,
    "lat_direction": 7,
    "lon_degrees": 8,
    "lon_minutes": 9,
    "lon_seconds": 10,
    "lon_direction": 11,
    "height_support": 14,
    "height_overall": 15,
    "construction_date": 20,
    "application_date": 26,
    "status_code": 3,
    "state_code": 17,
    "county_code": 18,
}
# Alternatively use the simpler EN (entity) file which has cleaner columns.
# The RA (registration) file has the most useful date fields.

_DATE_FMTS = ["%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"]

_REQUEST_TIMEOUT = 300


def _parse_vintage_to_date(vintage: str) -> datetime:
    """Parse an FCC vintage label ('December 31, 2025' or '2025-12-31') to a date."""
    for fmt in ("%B %d, %Y", "%Y-%m-%d", "%b %d, %Y"):
        try:
            return datetime.strptime(vintage.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse vintage date: {vintage!r}")


def _parse_date(value: str) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _to_decimal(degrees: Any, minutes: Any, seconds: Any, direction: Any) -> float | None:
    try:
        deg = float(degrees)
        mins = float(minutes)
        secs = float(seconds)
        dec = deg + mins / 60.0 + secs / 3600.0
        if str(direction).strip().upper() in ("S", "W"):
            dec = -dec
        return dec
    except (TypeError, ValueError):
        return None


def _download_asr_co(cache_dir: Path) -> Path:
    """Download the FCC ASR CO (construction) file and cache locally."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_zip = cache_dir / "asr_co_raw.zip"
    raw_txt = cache_dir / "asr_co_raw.dat"

    if raw_txt.exists():
        log.info("using cached ASR CO file: %s", raw_txt)
        return raw_txt

    log.info("downloading FCC ASR CO database (~50-100 MB) ...")
    headers = {"User-Agent": "fcc-coverage-audit/0.1 (public-records-research)"}
    resp = requests.get(_ASR_BULK_URL, timeout=_REQUEST_TIMEOUT, headers=headers, stream=True)
    resp.raise_for_status()
    raw_zip.write_bytes(resp.content)
    log.info("download complete, extracting ...")

    with zipfile.ZipFile(raw_zip) as zf:
        # Find the CO.dat or co.dat file inside.
        names = zf.namelist()
        co_names = [n for n in names if n.lower().startswith("co") and n.lower().endswith(".dat")]
        if not co_names:
            co_names = [n for n in names if n.lower().endswith(".dat")]
        if not co_names:
            raise RuntimeError(f"No .dat file found in ASR ZIP. Contents: {names}")
        log.info("extracting %s from ASR ZIP", co_names[0])
        with zf.open(co_names[0]) as src, open(raw_txt, "wb") as dst:
            dst.write(src.read())

    raw_zip.unlink(missing_ok=True)
    return raw_txt


def _parse_asr_co(raw_path: Path) -> pd.DataFrame:
    """Parse the ASR CO (construction) pipe-delimited file into a DataFrame."""
    log.info("parsing ASR CO file: %s", raw_path)
    rows = []
    with open(raw_path, "r", encoding="latin-1", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 25:
                continue
            try:
                lat = _to_decimal(
                    parts[_ASR_COLS["lat_degrees"]],
                    parts[_ASR_COLS["lat_minutes"]],
                    parts[_ASR_COLS["lat_seconds"]],
                    parts[_ASR_COLS["lat_direction"]],
                )
                lng = _to_decimal(
                    parts[_ASR_COLS["lon_degrees"]],
                    parts[_ASR_COLS["lon_minutes"]],
                    parts[_ASR_COLS["lon_seconds"]],
                    parts[_ASR_COLS["lon_direction"]],
                )
                if lat is None or lng is None:
                    continue
                state = parts[_ASR_COLS["state_code"]].strip()
                county = parts[_ASR_COLS["county_code"]].strip()
                construct_dt = _parse_date(parts[_ASR_COLS["construction_date"]])
                app_dt = _parse_date(parts[_ASR_COLS["application_date"]])
                status = parts[_ASR_COLS["status_code"]].strip()
                rows.append({
                    "lat": lat,
                    "lng": lng,
                    "state_code": state,
                    "county_code": county,
                    "construction_date": construct_dt,
                    "application_date": app_dt,
                    "status_code": status,
                })
            except (IndexError, ValueError):
                continue

    df = pd.DataFrame(rows)
    log.info("parsed %d ASR CO records", len(df))
    return df


def _build_county_geoid(state_code: str, county_code: str) -> str | None:
    """Map ASR state + county FIPS codes to a 5-digit county GEOID."""
    try:
        state_fips = str(int(state_code)).zfill(2)
        county_fips = str(int(county_code)).zfill(3)
        return f"{state_fips}{county_fips}"
    except (ValueError, TypeError):
        return None


def _load_or_build_asr_df(cache_dir: Path) -> pd.DataFrame:
    """Load cached ASR structures parquet, or download and parse from scratch."""
    parsed_cache = cache_dir / "asr_structures.parquet"
    if parsed_cache.exists():
        log.info("loading cached ASR structures: %s", parsed_cache)
        return pd.read_parquet(parsed_cache)

    raw_path = _download_asr_co(cache_dir)
    df = _parse_asr_co(raw_path)

    # Build county_geoid and filter to continental US records with usable dates.
    df["county_geoid"] = df.apply(
        lambda r: _build_county_geoid(r["state_code"], r["county_code"]), axis=1
    )
    df = df.dropna(subset=["county_geoid"])

    # Use construction_date; fall back to application_date.
    df["event_date"] = df["construction_date"].combine_first(df["application_date"])
    df = df.dropna(subset=["event_date"])

    keep = ["lat", "lng", "county_geoid", "event_date", "status_code"]
    df = df[keep].copy()
    df.to_parquet(parsed_cache, index=False)
    log.info("cached %d ASR structures -> %s", len(df), parsed_cache)
    return df


def fetch_asr_labels(
    prior_vintage: str,
    current_vintage: str,
    cache_dir: Path | str = Path("data/groundtruth/asr"),
    grace_days: int = 90,
) -> pd.DataFrame:
    """Produce per-county 'was any structure built during this vintage window' labels.

    Parameters
    ----------
    prior_vintage : str
        FCC vintage label for the start of the window (e.g. "June 30, 2025").
    current_vintage : str
        FCC vintage label for the end of the window (e.g. "December 31, 2025").
    cache_dir : Path
        Where to cache the raw ASR download and derived labels.
    grace_days : int
        Include structures registered up to this many days before the prior
        vintage to capture towers that were built just before the filing window
        started (common for coverage that appears in the current vintage).

    Returns
    -------
    DataFrame with columns:
        county_geoid        : 5-digit FIPS string
        has_new_structure   : bool — at least one structure event in window
        new_structure_count : int  — number of structure events
        min_event_date      : datetime | NaT — earliest event in window

    Notes
    -----
    The join to BDC provider is deliberately county-level: ASR records structure
    owners (tower companies, not always the carrier). A county-level "did anything
    get built?" label is conservative and regulator-admissible. See
    docs/methodology.md for the ASR-to-carrier join caveat.
    """
    cache_dir = Path(cache_dir)
    prior_dt = _parse_vintage_to_date(prior_vintage)
    current_dt = _parse_vintage_to_date(current_vintage)

    # Labels are keyed to the vintage pair; check for a cached result.
    prior_key = prior_vintage.replace(" ", "_").replace(",", "")
    current_key = current_vintage.replace(" ", "_").replace(",", "")
    label_cache = cache_dir / f"labels_{prior_key}_to_{current_key}.parquet"
    if label_cache.exists():
        log.info("loading cached ASR labels: %s", label_cache)
        return pd.read_parquet(label_cache)

    asr_df = _load_or_build_asr_df(cache_dir)

    window_start = prior_dt - timedelta(days=grace_days)
    window_end = current_dt

    in_window = asr_df[
        (asr_df["event_date"] >= window_start)
        & (asr_df["event_date"] <= window_end)
    ].copy()
    log.info(
        "ASR structures in window %s to %s: %d records",
        window_start.date(), window_end.date(), len(in_window),
    )

    if in_window.empty:
        labels = pd.DataFrame(columns=[
            "county_geoid", "has_new_structure", "new_structure_count", "min_event_date",
        ])
    else:
        agg = (
            in_window.groupby("county_geoid")
            .agg(
                new_structure_count=("event_date", "count"),
                min_event_date=("event_date", "min"),
            )
            .reset_index()
        )
        agg["has_new_structure"] = True
        labels = agg[["county_geoid", "has_new_structure", "new_structure_count", "min_event_date"]]

    labels.to_parquet(label_cache, index=False)
    log.info("wrote ASR labels: %s (%d counties with new structures)", label_cache, len(labels))
    return labels


def merge_asr_into_features(
    features: pd.DataFrame,
    asr_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Join ASR county-level labels into the scored features DataFrame.

    Adds columns:
        asr_has_new_structure : bool (False when county not in ASR labels = no build)
        asr_new_structure_count : int
        asr_min_event_date : object (datetime or NaT)
    """
    if asr_labels.empty:
        features = features.copy()
        features["asr_has_new_structure"] = False
        features["asr_new_structure_count"] = 0
        features["asr_min_event_date"] = pd.NaT
        return features

    labels = asr_labels.rename(columns={
        "has_new_structure": "asr_has_new_structure",
        "new_structure_count": "asr_new_structure_count",
        "min_event_date": "asr_min_event_date",
    })
    merged = features.merge(
        labels[["county_geoid", "asr_has_new_structure", "asr_new_structure_count", "asr_min_event_date"]],
        on="county_geoid",
        how="left",
    )
    merged["asr_has_new_structure"] = merged["asr_has_new_structure"].fillna(False).astype(bool)
    merged["asr_new_structure_count"] = merged["asr_new_structure_count"].fillna(0).astype(int)
    return merged
