"""ASR RA/CO field parsing: height is overall AGL, not ground elevation."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fcc_audit.groundtruth_asr import (  # noqa: E402
    _ASR_PARSE_VERSION,
    _RA_COLS,
    _parse_asr_co_coords,
    _parse_asr_joined,
    _parse_asr_ra,
    load_asr_structures,
)


# Magnolia AR GTOWER — ground elev 88.4 m AMSL, overall AGL 106.3 m.
_RA_FIXTURE = (
    "RA|REG|A1302667|1000135|97000|NT|MD|I|C|"
    "11/11/2024|11/11/2024|11/11/2024|09/19/1996||11/11/2024|C||"
    "Lucas||Conder||Authorized Representative|Y|"
    "921 Columbia Rd 15 (686567)|MAGNOLIA|AR|05027|71753|"
    "103.6|88.4|106.3|194.7|GTOWER|"
    "11/04/2024|2024-ASW-13076-OE|70/7460-1J|1|4, 8, 13|7||||\n"
)
_CO_FIXTURE = (
    "CO|REG|A1302667|1000135|97000|T|"
    "33|14|13.3|N|119653.3|93|16|20.9|W|335780.9||\n"
)


def test_ra_height_is_overall_agl_not_ground_elev(tmp_path: Path):
    ra = tmp_path / "RA.dat"
    ra.write_text(_RA_FIXTURE, encoding="latin-1")
    df = _parse_asr_ra(ra)
    assert len(df) == 1
    assert float(df.iloc[0]["height_m"]) == 106.3
    assert df.iloc[0]["structure_type"] == "GTOWER"
    assert df.iloc[0]["unique_id"] == "97000"
    assert df.iloc[0]["registration_number"] == "1000135"
    # Guard: field 29 (ground elev) must not be used as height.
    assert _RA_COLS["ground_elevation_m"] == 29
    assert _RA_COLS["height_overall_m"] == 30
    assert float(df.iloc[0]["height_m"]) != 88.4


def test_co_ra_join_on_unique_id(tmp_path: Path):
    ra = tmp_path / "RA.dat"
    co = tmp_path / "CO.dat"
    ra.write_text(_RA_FIXTURE, encoding="latin-1")
    co.write_text(_CO_FIXTURE, encoding="latin-1")
    coords = _parse_asr_co_coords(co)
    assert coords.iloc[0]["unique_id"] == "97000"
    joined = _parse_asr_joined(co, ra)
    assert len(joined) == 1
    assert abs(float(joined.iloc[0]["lat"]) - 33.2370) < 0.01
    assert float(joined.iloc[0]["height_m"]) == 106.3


def _write_asr_dat(tmp_path: Path) -> None:
    (tmp_path / "RA.dat").write_text(_RA_FIXTURE, encoding="latin-1")
    (tmp_path / "CO.dat").write_text(_CO_FIXTURE, encoding="latin-1")


def test_stale_asr_parquet_missing_columns_is_rebuilt(tmp_path: Path):
    """A leftover parquet without structure_type must not be reused."""
    _write_asr_dat(tmp_path)
    stale = pd.DataFrame({
        "unique_id": ["stale"],
        "lat": [0.0],
        "lng": [0.0],
        "county_geoid": ["05027"],
        "status_code": ["C"],
        "height_m": [88.4],
    })
    stale.to_parquet(tmp_path / "asr_structures.parquet", index=False)
    (tmp_path / ".asr_parse_version").write_text(str(_ASR_PARSE_VERSION))

    df = load_asr_structures(tmp_path, status_codes=(), min_height_m=None)
    assert "structure_type" in df.columns
    assert float(df.iloc[0]["height_m"]) == 106.3
    assert df.iloc[0]["unique_id"] == "97000"


def test_asr_cache_missing_version_marker_is_rebuilt(tmp_path: Path):
    _write_asr_dat(tmp_path)
    first = load_asr_structures(tmp_path, status_codes=(), min_height_m=None)
    assert float(first.iloc[0]["height_m"]) == 106.3
    (tmp_path / ".asr_parse_version").unlink()
    cached = pd.read_parquet(tmp_path / "asr_structures.parquet")
    cached["height_m"] = 88.4
    cached.to_parquet(tmp_path / "asr_structures.parquet", index=False)

    df = load_asr_structures(tmp_path, status_codes=(), min_height_m=None)
    assert float(df.iloc[0]["height_m"]) == 106.3
    assert (tmp_path / ".asr_parse_version").read_text().strip() == str(_ASR_PARSE_VERSION)


def test_current_asr_cache_is_reused(tmp_path: Path):
    parsed = tmp_path / "asr_structures.parquet"
    pd.DataFrame({
        "unique_id": ["cached"],
        "lat": [1.0],
        "lng": [2.0],
        "county_geoid": ["20001"],
        "status_code": ["C"],
        "height_m": [50.0],
        "structure_type": ["GTOWER"],
        "registration_number": ["1"],
    }).to_parquet(parsed, index=False)
    (tmp_path / ".asr_parse_version").write_text(str(_ASR_PARSE_VERSION))

    df = load_asr_structures(tmp_path, status_codes=(), min_height_m=None)
    assert df.iloc[0]["unique_id"] == "cached"
