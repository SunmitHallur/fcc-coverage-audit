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
# Weekly complete ASR registration bundle (CO coordinates + RA metadata).
# The old wireless2.fcc.gov asrRegistration.zip endpoint 404s.
_ASR_BULK_URL = "https://data.fcc.gov/download/pub/uls/complete/r_tower.zip"

# CO.dat (coordinates) — pipe-delimited, record type in field 0.
# FCC TOWER_PUBACC_CO: registration_number @3, unique_system_identifier @4.
_CO_COLS = {
    "registration_number": 3,
    "unique_system_identifier": 4,
    "lat_degrees": 6,
    "lat_minutes": 7,
    "lat_seconds": 8,
    "lat_direction": 9,
    "lon_degrees": 11,
    "lon_minutes": 12,
    "lon_seconds": 13,
    "lon_direction": 14,
}
# RA.dat (registration): status, county GEOID, heights, dates.
# FCC TOWER_PUBACC_RA: registration @3, unique_id @4, height_of_structure @28,
# ground_elevation @29, overall_height_agl @30, overall_amsl @31, structure_type @32.
_RA_COLS = {
    "registration_number": 3,
    "unique_system_identifier": 4,
    "status_code": 8,
    "county_geoid": 26,
    "height_support_m": 28,
    "ground_elevation_m": 29,
    "height_overall_m": 30,
    "structure_type": 32,
}

_DATE_FMTS = ["%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y"]

_REQUEST_TIMEOUT = 300


def _parse_vintage_to_date(
    vintage: str,
    vintage_dates: dict[str, str] | None = None,
) -> datetime:
    """Parse an FCC vintage label or mapped build id to a date.

    Accepts filing labels (``December 31, 2025``), ISO dates, or Redshift build
    ids when ``vintage_dates`` maps them (e.g. ``{"277": "December 31, 2025"}``).
    """
    token = vintage.strip()
    if vintage_dates and token in vintage_dates:
        token = str(vintage_dates[token]).strip()
    for fmt in ("%B %d, %Y", "%Y-%m-%d", "%b %d, %Y"):
        try:
            return datetime.strptime(token, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse vintage date: {vintage!r}. "
        f"Add it under groundtruth.asr.vintage_dates in pipeline.yaml."
    )


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


def _download_asr_bundle(cache_dir: Path) -> tuple[Path, Path]:
    """Download the FCC weekly ASR zip and extract CO.dat + RA.dat."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_zip = cache_dir / "r_tower.zip"
    co_path = cache_dir / "CO.dat"
    ra_path = cache_dir / "RA.dat"
    if co_path.exists() and ra_path.exists():
        return co_path, ra_path

    if not raw_zip.exists():
        log.info("downloading FCC ASR registrations (~36 MB) ...")
        headers = {"User-Agent": "fcc-coverage-audit/0.1 (public-records-research)"}
        resp = requests.get(_ASR_BULK_URL, timeout=_REQUEST_TIMEOUT, headers=headers, stream=True)
        resp.raise_for_status()
        tmp = raw_zip.with_suffix(".zip.part")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        tmp.replace(raw_zip)
        log.info("download complete, extracting ...")

    with zipfile.ZipFile(raw_zip) as zf:
        names = {n.lower(): n for n in zf.namelist()}
        for dest, key in ((co_path, "co.dat"), (ra_path, "ra.dat")):
            if dest.exists():
                continue
            src_name = names.get(key)
            if src_name is None:
                raise RuntimeError(f"{key} not in ASR ZIP. Contents: {zf.namelist()}")
            dest.write_bytes(zf.read(src_name))
    return co_path, ra_path


def _parse_asr_co_coords(co_path: Path) -> pd.DataFrame:
    """Parse ASR CO.dat coordinates keyed by unique system identifier."""
    rows = []
    with open(co_path, "r", encoding="latin-1", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 15 or parts[0] != "CO":
                continue
            try:
                lat = _to_decimal(
                    parts[_CO_COLS["lat_degrees"]],
                    parts[_CO_COLS["lat_minutes"]],
                    parts[_CO_COLS["lat_seconds"]],
                    parts[_CO_COLS["lat_direction"]],
                )
                lng = _to_decimal(
                    parts[_CO_COLS["lon_degrees"]],
                    parts[_CO_COLS["lon_minutes"]],
                    parts[_CO_COLS["lon_seconds"]],
                    parts[_CO_COLS["lon_direction"]],
                )
            except (IndexError, ValueError):
                continue
            if lat is None or lng is None:
                continue
            if not (15.0 <= lat <= 72.0 and -180.0 <= lng <= -60.0):
                continue
            uid = parts[_CO_COLS["unique_system_identifier"]].strip()
            reg = parts[_CO_COLS["registration_number"]].strip()
            if not uid and not reg:
                continue
            rows.append({
                "unique_id": uid or f"reg:{reg}",
                "registration_number": reg,
                "lat": lat,
                "lng": lng,
            })
    df = pd.DataFrame(rows)
    # Prefer unique_id; fall back to registration when unique_id is empty.
    if not df.empty:
        df = df.sort_values("unique_id", kind="mergesort")
        df = df.drop_duplicates("unique_id", keep="first")
    log.info("parsed %d ASR CO coordinates", len(df))
    return df


def _parse_asr_ra(ra_path: Path) -> pd.DataFrame:
    """Parse ASR RA.dat registration metadata."""
    rows = []
    with open(ra_path, "r", encoding="latin-1", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 33 or parts[0] != "RA":
                continue
            geoid = parts[_RA_COLS["county_geoid"]].strip()
            if len(geoid) != 5 or not geoid.isdigit():
                continue
            # Prefer overall AGL (appurtenances included); fall back to support AGL.
            # Do NOT use ground elevation (field 29) — that is AMSL site elev.
            height = None
            for idx in (_RA_COLS["height_overall_m"], _RA_COLS["height_support_m"]):
                try:
                    raw = parts[idx].strip()
                    if raw:
                        height = float(raw)
                        break
                except (TypeError, ValueError, IndexError):
                    continue
            stype = ""
            try:
                stype = parts[_RA_COLS["structure_type"]].strip()
            except IndexError:
                stype = ""
            event_date = None
            for idx in (12, 14, 9, 10, 11):
                if idx < len(parts):
                    event_date = _parse_date(parts[idx])
                    if event_date is not None:
                        break
            rows.append({
                "unique_id": (
                    parts[_RA_COLS["unique_system_identifier"]].strip()
                    or f"reg:{parts[_RA_COLS['registration_number']].strip()}"
                ),
                "registration_number": parts[_RA_COLS["registration_number"]].strip(),
                "status_code": parts[_RA_COLS["status_code"]].strip(),
                "county_geoid": geoid,
                "height_m": height,
                "structure_type": stype,
                "event_date": event_date,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("unique_id", kind="mergesort")
        df = df.drop_duplicates("unique_id", keep="first")
    log.info("parsed %d ASR RA registrations", len(df))
    return df


def _parse_asr_joined(co_path: Path, ra_path: Path) -> pd.DataFrame:
    """Join RA metadata to CO coordinates (unique_id, then registration fallback)."""
    coords = _parse_asr_co_coords(co_path)
    meta = _parse_asr_ra(ra_path)
    if coords.empty or meta.empty:
        return pd.DataFrame()

    by_uid = meta.merge(coords, on="unique_id", how="inner", suffixes=("", "_co"))
    if "registration_number_co" in by_uid.columns:
        by_uid = by_uid.drop(columns=["registration_number_co"])

    # Structures whose unique_id did not match: try registration_number.
    matched_uids = set(by_uid["unique_id"]) if not by_uid.empty else set()
    meta_miss = meta[~meta["unique_id"].isin(matched_uids)]
    coords_miss = coords[~coords["unique_id"].isin(matched_uids)]
    if not meta_miss.empty and not coords_miss.empty:
        by_reg = meta_miss.merge(
            coords_miss.drop(columns=["unique_id"]).drop_duplicates(
                "registration_number", keep="first"
            ),
            on="registration_number",
            how="inner",
        )
        df = pd.concat([by_uid, by_reg], ignore_index=True) if not by_uid.empty else by_reg
    else:
        df = by_uid
    log.info("joined %d ASR structures with coordinates", len(df))
    return df


def _download_asr_co(cache_dir: Path) -> Path:
    """Back-compat wrapper: extract CO.dat and return its path."""
    co_path, _ra_path = _download_asr_bundle(cache_dir)
    return co_path


def _parse_asr_co(raw_path: Path) -> pd.DataFrame:
    """Back-compat: CO coordinates only (no county). Prefer load_asr_structures."""
    df = _parse_asr_co_coords(raw_path)
    df = df.rename(columns={"unique_id": "status_code"})
    df["state_code"] = ""
    df["county_code"] = ""
    df["construction_date"] = pd.NaT
    df["application_date"] = pd.NaT
    df["status_code"] = ""
    return df[["lat", "lng", "state_code", "county_code", "construction_date", "application_date", "status_code"]]


def _build_county_geoid(state_code: str, county_code: str) -> str | None:
    """Map ASR state + county FIPS codes to a 5-digit county GEOID."""
    try:
        state_fips = str(int(state_code)).zfill(2)
        county_fips = str(int(county_code)).zfill(3)
        return f"{state_fips}{county_fips}"
    except (ValueError, TypeError):
        return None


# Bump when RA/CO column map or join key changes so a leftover parquet
# from a previous parser cannot silently keep ground-elevation heights.
_ASR_PARSE_VERSION = 2
_REQUIRED_ASR_COLS = (
    "unique_id", "lat", "lng", "county_geoid", "status_code",
    "height_m", "structure_type", "registration_number",
)


def _asr_cache_is_current(parsed_cache: Path, cache_dir: Path) -> bool:
    """False when parquet is missing, stale relative to RA/CO, or old schema."""
    if not parsed_cache.exists():
        return False
    try:
        df = pd.read_parquet(parsed_cache)
    except Exception:
        return False
    if any(c not in df.columns for c in _REQUIRED_ASR_COLS):
        return False
    marker = cache_dir / ".asr_parse_version"
    if not marker.exists() or marker.read_text().strip() != str(_ASR_PARSE_VERSION):
        return False
    cache_mtime = parsed_cache.stat().st_mtime
    for src_name in ("RA.dat", "CO.dat"):
        src = cache_dir / src_name
        if src.exists() and src.stat().st_mtime > cache_mtime + 1.0:
            return False
    return True


def load_asr_structures(
    cache_dir: Path | str = Path("data/groundtruth/asr"),
    *,
    status_codes: tuple[str, ...] = ("C",),
    min_height_m: float | None = 15.0,
) -> pd.DataFrame:
    """Return geocoded ASR structures (one row per unique system id).

    Default keeps constructed (``C``) structures at least 15 m tall — a
    conservative cell-tower-like subset. ASR is incomplete (many rooftop and
    small sites are unregistered) so this is corroboration, not a full census.
    """
    cache_dir = Path(cache_dir)
    parsed_cache = cache_dir / "asr_structures.parquet"
    if _asr_cache_is_current(parsed_cache, cache_dir):
        df = pd.read_parquet(parsed_cache)
    else:
        if parsed_cache.exists():
            log.info("rebuilding ASR cache (schema/source newer than %s)", parsed_cache)
            parsed_cache.unlink()
        co_path, ra_path = _download_asr_bundle(cache_dir)
        df = _parse_asr_joined(co_path, ra_path)
        keep = [c for c in _REQUIRED_ASR_COLS if c in df.columns]
        extra = [c for c in ("event_date",) if c in df.columns]
        df = df[keep + extra].copy()
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parsed_cache, index=False)
        (cache_dir / ".asr_parse_version").write_text(str(_ASR_PARSE_VERSION))
        log.info("cached %d ASR structures -> %s", len(df), parsed_cache)

    if status_codes:
        df = df[df["status_code"].isin(status_codes)]
    if min_height_m is not None and "height_m" in df.columns:
        df = df[df["height_m"].fillna(min_height_m + 1) >= min_height_m]
    return df.reset_index(drop=True)


def _load_or_build_asr_df(cache_dir: Path) -> pd.DataFrame:
    """Load cached ASR structures parquet, or download and parse from scratch."""
    df = load_asr_structures(cache_dir, status_codes=(), min_height_m=None)
    df = df.dropna(subset=["county_geoid"])
    if "event_date" in df.columns:
        df = df.dropna(subset=["event_date"])
    keep = [c for c in ("lat", "lng", "county_geoid", "event_date", "status_code") if c in df.columns]
    return df[keep].copy()


def fetch_asr_labels(
    prior_vintage: str,
    current_vintage: str,
    cache_dir: Path | str = Path("data/groundtruth/asr"),
    grace_days: int = 90,
    vintage_dates: dict[str, str] | None = None,
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
    prior_dt = _parse_vintage_to_date(prior_vintage, vintage_dates)
    current_dt = _parse_vintage_to_date(current_vintage, vintage_dates)

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
