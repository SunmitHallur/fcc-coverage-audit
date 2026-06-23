"""Data acquisition with a pluggable backend.

Three backends implement the same :class:`DataSource` interface:

* :class:`FccDownloadSource` - pulls Big-4 5G-NR vector files straight from the
  FCC National Broadband Map public API. Works today (general internet + FCC.gov
  reachable). This is the default.
* :class:`RedshiftSource` - queries the same coverage data from Amazon Redshift.
  Stubbed until AWS access is granted; enable via ``source.backend: redshift``.
* :class:`FixtureSource` - reads synthetic GeoJSON for offline development / CI.

Downstream stages depend only on the interface, so swapping backends is a
one-line config change.

FCC API contract (reverse-engineered from the public Data Download portal):

* ``GET {base}/published/filing`` -> ``{"data": [ {process_uuid, filing_type,
  filing_subtype, as_of_date, ...}, ... ]}`` - the list of published releases.
* ``GET {base}/national_map_process/nbm_get_data_download/{process_uuid}`` ->
  ``{"data": [ {file_id, file_name, file_type, data_type, data_category,
  technology_code, state_fips, provider_id, ...}, ... ]}`` - the file catalog
  for one release.
* ``GET {base}/downloads/downloadFile/{data_type}/{file_id}/{file_type}`` ->
  the binary (zipped shapefile / gpkg) for one catalog row.

The FCC silently drops requests without a non-default User-Agent, so every
request sets one from config.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

from .config import Config, Provider

log = logging.getLogger(__name__)

# FCC BDC mobile technology codes. 5G-NR is 400; included here so we can filter
# the catalog by technology regardless of how file names are spelled.
TECHNOLOGY_CODES: dict[str, int] = {
    "3G": 300,
    "4G-LTE": 83,
    "5G-NR": 400,
}


@dataclass(frozen=True)
class CoverageFile:
    """A downloaded per-(provider, technology) coverage file on local disk.

    One file holds all speed tiers and environments for that technology; the
    normalize stage filters by tier (mindown/minup) and environment (environmnt).
    """

    provider_id: int
    technology: str
    vintage: str
    local_path: Path


class DataSource(ABC):
    """Backend-agnostic coverage data interface."""

    @abstractmethod
    def list_vintages(self) -> list[str]:
        """Return available mobile-broadband vintages (as-of dates), newest first."""

    @abstractmethod
    def list_providers(self, vintage: str) -> list[Provider]:
        """Return all mobile providers available for a vintage."""

    @abstractmethod
    def fetch(self, provider_id: int, technology: str, vintage: str) -> CoverageFile:
        """Materialize one (provider, technology) coverage file and describe it."""

    def resolve_vintages(self, current: str | None, prior: str | None) -> tuple[str, str]:
        """Pick (current, prior) vintages, auto-selecting the two newest if unset."""
        if current and prior:
            return current, prior
        available = self.list_vintages()
        if len(available) < 2:
            raise RuntimeError(
                f"Need >=2 vintages to compare; backend reported {available!r}"
            )
        return current or available[0], prior or available[1]


# ---------------------------------------------------------------------------
# FCC direct-download backend (works today)
# ---------------------------------------------------------------------------
class FccDownloadSource(DataSource):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        fcc = cfg.fcc
        self.base_url: str = fcc["base_url"].rstrip("/")
        self.timeout: int = int(fcc.get("request_timeout_s", 120))
        self.max_retries: int = int(fcc.get("max_retries", 5))
        self.min_interval: float = float(fcc.get("min_seconds_between_requests", 2.0))
        self.raw_dir = cfg.path("raw")
        self._last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "user-agent": fcc.get("user_agent", "fcc-coverage-audit/0.1"),
                "accept": "application/json",
            }
        )

    # -- low level --
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()

    def _get(self, url: str, *, stream: bool = False) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=self.timeout, stream=stream)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:  # network / 5xx
                last_exc = exc
                backoff = min(60, 2 ** attempt)
                log.warning(
                    "GET %s failed (attempt %d/%d): %s - retrying in %ss",
                    url, attempt, self.max_retries, exc, backoff,
                )
                time.sleep(backoff)
        raise RuntimeError(f"GET {url} failed after {self.max_retries} attempts") from last_exc

    # -- catalog --
    def _filings(self) -> list[dict[str, Any]]:
        data = self._get(f"{self.base_url}/published/filing").json().get("data", [])
        # Mobile broadband filings only.
        return [
            f for f in data
            if "mobile" in str(f.get("filing_type", "")).lower()
            or "mobile" in str(f.get("filing_subtype", "")).lower()
        ] or data

    def list_vintages(self) -> list[str]:
        seen: list[str] = []
        for f in self._filings():
            v = f.get("as_of_date") or f.get("filing_subtype")
            if v and v not in seen:
                seen.append(str(v))
        return seen

    def _catalog_for_vintage(self, vintage: str) -> list[dict[str, Any]]:
        process_uuid = None
        for f in self._filings():
            if str(f.get("as_of_date")) == vintage or str(f.get("filing_subtype")) == vintage:
                process_uuid = f.get("process_uuid")
                break
        if not process_uuid:
            raise RuntimeError(f"No published filing found for vintage {vintage!r}")
        url = f"{self.base_url}/national_map_process/nbm_get_data_download/{process_uuid}"
        return self._get(url).json().get("data", [])

    @staticmethod
    def _is_mobile(row: dict[str, Any]) -> bool:
        return "mobile" in str(row.get("data_type", "")).lower()

    def list_providers(self, vintage: str) -> list[Provider]:
        rows = self._catalog_for_vintage(vintage)
        seen: dict[int, str] = {}
        for r in rows:
            if not self._is_mobile(r):
                continue
            pid = r.get("provider_id")
            if pid in (None, "", "null"):
                continue
            pid = int(pid)
            name = r.get("provider_name") or r.get("brand_name") or str(pid)
            seen.setdefault(pid, str(name))
        return [Provider(id=pid, name=name) for pid, name in sorted(seen.items())]

    def _match_row(
        self, rows: Iterable[dict[str, Any]], provider_id: int, technology: str
    ) -> dict[str, Any] | None:
        tech_code = TECHNOLOGY_CODES.get(technology)
        candidates = []
        for r in rows:
            if not self._is_mobile(r):
                continue
            if int(r.get("provider_id") or -1) != provider_id:
                continue
            rc = r.get("technology_code")
            if tech_code is not None and rc is not None and int(rc) != tech_code:
                # also allow a filename match in case codes differ across vintages
                if technology.replace("-", "").lower() not in str(r.get("file_name", "")).replace("-", "").lower():
                    continue
            candidates.append(r)
        # Prefer GeoPackage > Shapefile when multiple file types exist.
        candidates.sort(key=lambda r: 0 if "gpkg" in str(r.get("file_type", "")).lower() else 1)
        return candidates[0] if candidates else None

    def fetch(self, provider_id: int, technology: str, vintage: str) -> CoverageFile:
        out_dir = self.raw_dir / vintage / str(provider_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = self._catalog_for_vintage(vintage)
        row = self._match_row(rows, provider_id, technology)
        if not row:
            raise RuntimeError(
                f"No mobile {technology} catalog file for provider {provider_id} "
                f"in vintage {vintage}. Catalog had {len(rows)} rows."
            )
        file_id = row["file_id"]
        file_type = row["file_type"]
        data_type = row["data_type"]
        fname = row.get("file_name", f"{file_id}.{file_type}")
        dest = out_dir / f"{technology}_{fname}"
        if dest.exists() and dest.stat().st_size > 0:
            log.info("cached %s", dest.name)
            return CoverageFile(provider_id, technology, vintage, dest)

        url = f"{self.base_url}/downloads/downloadFile/{data_type}/{file_id}/{file_type}"
        log.info("downloading %s -> %s", url, dest.name)
        resp = self._get(url, stream=True)
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        return CoverageFile(provider_id, technology, vintage, dest)


# ---------------------------------------------------------------------------
# Redshift backend (enable once AWS access is granted)
# ---------------------------------------------------------------------------
class RedshiftSource(DataSource):
    """Queries coverage polygons from Amazon Redshift.

    Stubbed until access is granted. To enable:
      1. ``pip install redshift-connector`` (uncomment in requirements.txt).
      2. Fill ``source.redshift`` in config (host/db/user/password via env vars).
      3. Adjust ``coverage_query`` to your warehouse schema.
      4. Set ``source.backend: redshift``.
    The query must return WKT geometry in EPSG:4326; this class writes it to a
    local GeoPackage so the rest of the pipeline is identical to the FCC path.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rs = cfg.redshift
        self.raw_dir = cfg.path("raw")

    def _connect(self):  # pragma: no cover - requires live credentials
        try:
            import redshift_connector
        except ImportError as exc:
            raise RuntimeError(
                "redshift-connector not installed. Uncomment it in requirements.txt "
                "and `pip install -r requirements.txt`."
            ) from exc
        return redshift_connector.connect(
            host=self.rs["host"],
            port=int(self.rs.get("port", 5439)),
            database=self.rs["database"],
            user=self.rs["user"],
            password=self.rs["password"],
        )

    def list_vintages(self) -> list[str]:  # pragma: no cover - live only
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT as_of_date FROM bdc.mobile_coverage ORDER BY as_of_date DESC")
            return [str(r[0]) for r in cur.fetchall()]

    def list_providers(self, vintage: str) -> list[Provider]:  # pragma: no cover - live only
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT provider_id, provider_name FROM bdc.mobile_coverage "
                "WHERE as_of_date = %s ORDER BY provider_id", (vintage,)
            )
            return [Provider(id=int(r[0]), name=str(r[1] or r[0])) for r in cur.fetchall()]

    def fetch(self, provider_id, technology, vintage) -> CoverageFile:  # pragma: no cover
        import geopandas as gpd
        from shapely import wkt

        out_dir = self.raw_dir / vintage / str(provider_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{technology}.gpkg"
        query = self.rs["coverage_query"].format(
            vintage=vintage, provider_id=provider_id, tech=technology
        )
        with self._connect() as conn:
            df = conn.cursor().execute(query).fetch_dataframe()
        df["geometry"] = df["geometry_wkt"].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df.drop(columns=["geometry_wkt"]), geometry="geometry", crs="EPSG:4326")
        gdf.to_file(dest, driver="GPKG")
        return CoverageFile(provider_id, technology, vintage, dest)


# ---------------------------------------------------------------------------
# Fixture backend (offline development / CI)
# ---------------------------------------------------------------------------
class FixtureSource(DataSource):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = cfg.project_root / cfg.fixture["dir"]

    def list_vintages(self) -> list[str]:
        if not self.dir.exists():
            return []
        return sorted((p.name for p in self.dir.iterdir() if p.is_dir()), reverse=True)

    def list_providers(self, vintage: str) -> list[Provider]:
        vdir = self.dir / vintage
        if not vdir.exists():
            return []
        ids = sorted({int(p.name.split("_")[0]) for p in vdir.glob("*.geojson")})
        known = {p.id: p.name for p in self.cfg.known_providers}
        return [Provider(id=i, name=known.get(i, str(i))) for i in ids]

    def fetch(self, provider_id, technology, vintage) -> CoverageFile:
        path = self.dir / vintage / f"{provider_id}_{technology}.geojson"
        if not path.exists():
            raise FileNotFoundError(
                f"Fixture not found: {path}. Generate it with "
                f"`python -m fcc_audit.cli make-fixtures`."
            )
        return CoverageFile(provider_id, technology, vintage, path)


def get_source(cfg: Config) -> DataSource:
    backend = cfg.backend
    if backend == "fcc":
        return FccDownloadSource(cfg)
    if backend == "redshift":
        return RedshiftSource(cfg)
    if backend == "fixture":
        return FixtureSource(cfg)
    raise ValueError(f"Unknown source backend: {backend!r}")
