"""ASR RA/CO field parsing: height is overall AGL, not ground elevation."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fcc_audit.groundtruth_asr import (  # noqa: E402
    _RA_COLS,
    _parse_asr_co_coords,
    _parse_asr_joined,
    _parse_asr_ra,
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
