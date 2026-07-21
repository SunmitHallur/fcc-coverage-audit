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
  ``{"data": [ {id, file_name, file_type, data_type, data_category,
  technology_code, state_fips, provider_id, ...}, ... ]}`` - the file catalog
  for one release.
* ``GET {base}/getNBMDataDownloadFile/{file_id}/{file_type}`` -> a ZIP holding
  the shapefile (file_type=1) or GeoPackage (file_type=2) for one catalog row.
  This is the exact endpoint the website's own "Download" buttons hit; it needs
  NO API token, just the browser Referer/Origin headers below.

The FCC silently drops requests without a non-default User-Agent, so every
request sets one from config.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .config import Config, Provider

log = logging.getLogger(__name__)

# FCC BDC mobile technology codes (verified against the live catalog):
#   300 = 3G, 400 = 4G LTE, 500 = 5G-NR. 5G ships as separate speed-tier files,
# distinguished by `technology_code_desc` ("5G-NR (7/1 Mbps)" / "(35/3 Mbps)").
# We select files by `technology_code_desc`, so these are informational.
TECHNOLOGY_CODES: dict[str, int] = {
    "3G": 300,
    "4G-LTE": 400,
    "5G-NR": 500,
}


@dataclass(frozen=True)
class CoverageFile:
    """A per-(provider, technology) coverage layer materialized on local disk.

    Two shapes flow through the pipeline behind this one type:

    * **Polygon coverage** (``is_hex=False``, the FCC/fixture backends): a
      shapefile / GeoPackage / WKT table the normalize stage polyfills to H3.
      One file holds all speed tiers and environments for that technology; the
      normalize stage filters by tier (mindown/minup) and environment (environmnt).
    * **Pre-indexed H3 hexes** (``is_hex=True``, the Redshift backend): a parquet
      of columns ``h3`` (H3 cell id string) + ``signal_dbm`` for one already
      resolved (provider, service, vintage). Because the warehouse already did
      the H3 indexing, normalize SKIPS the (expensive) polygon polyfill and only
      tags counties. ``hex_resolution`` records the H3 resolution of those cells.
    """

    provider_id: int
    technology: str
    vintage: str
    local_path: Path
    is_hex: bool = False
    hex_resolution: int | None = None


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
_RAW_COVERAGE_TYPE = "Mobile Broadband Raw Coverage"


def safe_service_name(name: str) -> str:
    """Filesystem-safe token for a service label/desc (no spaces/slashes/parens)."""
    return (
        name.replace("/", "-").replace(" ", "").replace("(", "").replace(")", "")
    )


class FccDownloadSource(DataSource):
    """Pulls per-(provider, service, state) mobile coverage from the FCC NBM.

    Two-stage API (verified against the live service), both public / no token:
      * Public catalog (browser Referer/Origin headers): ``/published/filing``
        -> releases; ``/national_map_process/nbm_get_data_download/{process_uuid}``
        -> file rows.
      * Public download: ``/getNBMDataDownloadFile/{id}/{file_type}`` -> a ZIP
        with the shapefile / GeoPackage. Same endpoint the website buttons use.

    Mobile coverage ships per state x provider x service (5G tiers are separate
    files), so a national layer for one (provider, service) is the union of its
    per-state files, merged here into one local file.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        fcc = cfg.fcc
        self.base_url: str = fcc["base_url"].rstrip("/")
        self.download_tmpl: str = fcc["download_url_template"]
        self.timeout: int = int(fcc.get("request_timeout_s", 180))
        self.max_retries: int = int(fcc.get("max_retries", 5))
        self.min_interval: float = float(fcc.get("min_seconds_between_requests", 6.5))
        self.file_format: int = int(fcc.get("file_format", 2))  # 1=shp, 2=gpkg
        self.raw_dir = cfg.path("raw")
        self._last_request = 0.0
        self._catalog_cache: dict[str, list[dict[str, Any]]] = {}

        self.session = requests.Session()
        self.session.headers.update({
            "user-agent": fcc.get("user_agent", "Mozilla/5.0"),
            "accept": "application/json, text/plain, */*",
            "referer": fcc.get("referer", "https://broadbandmap.fcc.gov/data-download/nationwide-data"),
            "origin": fcc.get("origin", "https://broadbandmap.fcc.gov"),
        })

    # -- low level --
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()

    def _get(self, url: str, *, stream: bool = False, headers: dict | None = None) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=self.timeout, stream=stream, headers=headers)
                if resp.status_code in (401, 403):
                    # Permission errors won't fix themselves on retry. The FCC
                    # endpoints need no token, only the browser Referer/Origin
                    # headers (set from config) and open egress to fcc.gov.
                    raise RuntimeError(
                        f"HTTP {resp.status_code} for {url}. The FCC blocked this "
                        "request - check that fcc.user_agent / referer / origin are "
                        "set in config/pipeline.yaml and that your network allows "
                        "broadbandmap.fcc.gov."
                    )
                resp.raise_for_status()
                return resp
            except RuntimeError:
                raise
            except requests.RequestException as exc:  # network / 5xx
                last_exc = exc
                backoff = min(60, 2 ** attempt)
                log.warning("GET %s failed (attempt %d/%d): %s - retry in %ss",
                            url, attempt, self.max_retries, exc, backoff)
                time.sleep(backoff)
        raise RuntimeError(f"GET {url} failed after {self.max_retries} attempts") from last_exc

    # -- catalog --
    def _filings(self) -> list[dict[str, Any]]:
        return self._get(f"{self.base_url}/published/filing").json().get("data", [])

    def list_vintages(self) -> list[str]:
        seen: list[str] = []
        for f in self._filings():
            v = f.get("filing_subtype") or f.get("as_of_date")
            if v and str(v) not in seen:
                seen.append(str(v))
        # The API does not return filings newest-first; sort by parsed date desc
        # so resolve_vintages() picks the true current vs prior.
        from datetime import datetime

        def _key(label: str) -> tuple[int, str]:
            for fmt in ("%B %d, %Y", "%Y-%m-%d"):
                try:
                    return (int(datetime.strptime(label, fmt).timestamp()), label)
                except ValueError:
                    continue
            return (0, label)

        return sorted(seen, key=_key, reverse=True)

    def _process_uuid(self, vintage: str) -> str:
        for f in self._filings():
            if str(f.get("filing_subtype")) == vintage or str(f.get("as_of_date")) == vintage:
                return f["process_uuid"]
        raise RuntimeError(
            f"No published filing for vintage {vintage!r}. Available: {self.list_vintages()}"
        )

    def _catalog_for_vintage(self, vintage: str) -> list[dict[str, Any]]:
        if vintage not in self._catalog_cache:
            uuid = self._process_uuid(vintage)
            url = f"{self.base_url}/national_map_process/nbm_get_data_download/{uuid}"
            self._catalog_cache[vintage] = self._get(url).json().get("data", [])
        return self._catalog_cache[vintage]

    def list_providers(self, vintage: str) -> list[Provider]:
        seen: dict[int, str] = {}
        known = {p.id: p.name for p in self.cfg.known_providers}
        for r in self._catalog_for_vintage(vintage):
            if r.get("data_type") != _RAW_COVERAGE_TYPE:
                continue
            pid = r.get("provider_id")
            if pid in (None, "", "null"):
                continue
            pid = int(pid)
            seen.setdefault(pid, known.get(pid, str(pid)))
        return [Provider(id=pid, name=name) for pid, name in sorted(seen.items())]

    def _rows_for(self, vintage: str, provider_id: int, service_desc: str) -> list[dict[str, Any]]:
        states = self.cfg.states
        rows = []
        for r in self._catalog_for_vintage(vintage):
            if r.get("data_type") != _RAW_COVERAGE_TYPE:
                continue
            if int(r.get("provider_id") or -1) != provider_id:
                continue
            if str(r.get("technology_code_desc")) != service_desc:
                continue
            if str(r.get("download_available", "Yes")).lower() == "no":
                continue
            if states != "all" and str(r.get("state_fips")) not in states:
                continue
            rows.append(r)
        return rows

    def _download_one(self, row: dict[str, Any], dest: Path) -> Path:
        """Download one catalog row to ``dest`` (a .zip). No token required."""
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        url = self.download_tmpl.format(
            file_id=row["id"],
            file_type=self.file_format,
        )
        tmp = dest.with_suffix(dest.suffix + ".part")
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._get(
                    url,
                    stream=True,
                    headers={"accept": "application/zip, application/octet-stream, */*"},
                )
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
                tmp.rename(dest)
                return dest
            except (requests.RequestException, OSError) as exc:
                last_exc = exc
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
                backoff = min(60, 2 ** attempt)
                log.warning(
                    "download %s failed (attempt %d/%d): %s - retry in %ss",
                    dest.name, attempt, self.max_retries, exc, backoff,
                )
                time.sleep(backoff)
        raise RuntimeError(
            f"download {dest.name} failed after {self.max_retries} attempts"
        ) from last_exc

    @staticmethod
    def _read_coverage_zip(zip_path: Path):
        """Read the shapefile / GeoPackage held inside an FCC coverage ZIP.

        A GeoPackage is a SQLite DB and does many random seeks; reading it
        through GDAL's /vsizip/ forces repeated decompression and is
        pathologically slow for large files, so we extract it to disk first
        (on the same drive as the zip) and read the real file, cleaning up
        after. Shapefiles read sequentially, so /vsizip/ is fine for those."""
        import shutil
        import zipfile

        import geopandas as gpd

        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            gpkg = [n for n in names if n.lower().endswith(".gpkg")]
            shp = [n for n in names if n.lower().endswith(".shp")]
            if gpkg:
                workdir = zip_path.parent / (zip_path.stem + "_extract")
                workdir.mkdir(exist_ok=True)
                try:
                    extracted = Path(zf.extract(gpkg[0], workdir))
                    return gpd.read_file(extracted)
                finally:
                    shutil.rmtree(workdir, ignore_errors=True)
        if shp:
            return gpd.read_file(f"/vsizip/{zip_path.resolve()}/{shp[0]}")
        raise RuntimeError(f"No .gpkg/.shp inside {zip_path.name} (has {names[:5]})")

    def fetch(self, provider_id: int, technology: str, vintage: str) -> CoverageFile:
        """`technology` here is the service *desc* (e.g. '5G-NR (7/1 Mbps)').
        Downloads every per-state ZIP for this provider+service and merges them
        into one local GeoPackage."""
        import geopandas as gpd
        import pandas as pd

        rows = self._rows_for(vintage, provider_id, technology)
        if not rows:
            raise FileNotFoundError(
                f"No '{technology}' raw-coverage files for provider {provider_id} "
                f"in vintage {vintage} (states={self.cfg.states})."
            )
        out_dir = self.raw_dir / vintage / str(provider_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = safe_service_name(technology)
        scope = self.cfg.states_scope_key()
        merged = out_dir / f"{safe}_{self.cfg.backend}_{scope}_merged.gpkg"
        if merged.exists() and merged.stat().st_size > 0:
            return CoverageFile(provider_id, technology, vintage, merged)

        parts = []
        for r in rows:
            st = str(r.get("state_fips"))
            dest = out_dir / f"{safe}_{st}.zip"
            log.info("  download %s state %s (id=%s)", technology, st, r["id"])
            self._download_one(r, dest)
            try:
                parts.append(self._read_coverage_zip(dest))
            except Exception as exc:  # noqa: BLE001 - skip a bad/empty state file
                log.warning("  could not read %s: %s", dest.name, exc)
        if not parts:
            raise RuntimeError(f"Downloaded 0 readable files for provider {provider_id} {technology}")
        gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
        tmp = merged.with_suffix(merged.suffix + ".part")
        gdf.to_file(tmp, driver="GPKG")
        tmp.replace(merged)
        return CoverageFile(provider_id, technology, vintage, merged)


# ---------------------------------------------------------------------------
# Redshift backend (enable once AWS access is granted)
# ---------------------------------------------------------------------------
class RedshiftSource(DataSource):
    """Reads coverage from the warehouse's pre-aggregated H3 res-9 hex tables.

    The BDC data platform publishes national res-9 hex snapshots
    (``<schema>.bbmap_mobile_bb_tech_hex9s_<build>``), one row per H3 cell with:

    * ``h3index``  - the H3 res-9 cell id (string, same form the ``h3`` lib uses),
    * ``state_fips`` - the cell's state,
    * per (technology, speed tier, environment) a ``0/1`` coverage flag column
      (e.g. ``tech5g_spd1_env0``) plus a companion ``..._prov`` column holding a
      COMMA-DELIMITED list of the provider ids that cover the cell for that service.

    The table suffix ``<build>`` is a monotonic build/process id and serves as
    the **vintage token**: set ``analysis.vintages.current/prior`` to two builds.

    Because the warehouse already did the H3 indexing, this backend returns the
    covered cells for one (provider, service) directly and the pipeline SKIPS the
    expensive polygon polyfill (see ``CoverageFile.is_hex``). These hex tables
    carry only a 0/1 coverage flag (no modeled signal), so coverage is treated as
    a flat band; tower inference then works from contiguous-coverage blobs.

    To enable:
      1. ``pip install -r requirements.txt`` (redshift-connector is included).
      2. Fill ``source.redshift`` (host/db/user/password via env vars) + schema.
      3. Set two build ids in ``analysis.vintages`` and ``source.backend: redshift``.
    """

    # Map an analysis service (by its catalog `desc`) to the hex table's coverage
    # column base. The environment suffix (`_env0`/`_env1`) is appended per config.
    _DEFAULT_SERVICE_COLUMNS: dict[str, str] = {
        "5G-NR (7/1 Mbps)": "tech5g_spd1",
        "5G-NR (35/3 Mbps)": "tech5g_spd2",
        "4G LTE": "tech4g",
        "3G": "tech3g",
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rs = cfg.redshift
        self.raw_dir = cfg.path("raw")
        self.schema: str = self.rs.get("schema", "bdc_dataplatform")
        self.hex_prefix: str = self.rs.get("hex_table_prefix", "bbmap_mobile_bb_tech_hex9s_")
        self.merged_prefix: str = self.rs.get("merged_table_prefix", "bbmap_mobile_bb_merged_all_")
        self.environment: int = int(self.rs.get("environment", 0))
        self.hex_resolution: int = int(self.rs.get("hex_resolution", 9))
        self.service_columns: dict[str, str] = {
            **self._DEFAULT_SERVICE_COLUMNS,
            **(self.rs.get("service_hex_columns") or {}),
        }

    # -- connection / query helpers --
    def _connect(self):  # pragma: no cover - requires live credentials
        try:
            import redshift_connector
        except ImportError as exc:
            raise RuntimeError(
                "redshift-connector not installed. Run `pip install -r requirements.txt`."
            ) from exc
        return redshift_connector.connect(
            host=self.rs["host"],
            port=int(self.rs.get("port", 5439)),
            database=self.rs["database"],
            user=self.rs["user"],
            password=self.rs["password"],
        )

    def _query_df(self, sql: str, params: tuple = ()):  # pragma: no cover - live only
        import pandas as pd

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            try:
                df = cur.fetch_dataframe()
            except Exception:  # noqa: BLE001 - older connector lacks fetch_dataframe
                cols = [d[0] for d in cur.description]
                df = pd.DataFrame(cur.fetchall(), columns=cols)
        return df

    def _hex_table(self, vintage: str) -> str:
        return f"{self.schema}.{self.hex_prefix}{vintage}"

    def _service_column(self, service_desc: str) -> str:
        base = self.service_columns.get(service_desc)
        if base is None:
            raise RuntimeError(
                f"No Redshift hex column mapped for service {service_desc!r}. Add it "
                f"under source.redshift.service_hex_columns (known: "
                f"{sorted(self.service_columns)})."
            )
        return f"{base}_env{self.environment}"

    # -- DataSource interface --
    def list_vintages(self) -> list[str]:  # pragma: no cover - live only
        """Available hex-snapshot build ids, newest (largest) first."""
        df = self._query_df(
            "SELECT table_name FROM svv_tables "
            "WHERE table_schema = %s AND table_name LIKE %s",
            (self.schema, self.hex_prefix + "%"),
        )
        builds: set[int] = set()
        for name in df.iloc[:, 0].astype(str):
            suffix = name[len(self.hex_prefix):]
            if suffix.isdigit():
                builds.add(int(suffix))
        return [str(b) for b in sorted(builds, reverse=True)]

    def list_providers(self, vintage: str) -> list[Provider]:  # pragma: no cover - live only
        """Discover providers from the matching raw build (distinct providerid).

        The hex tables encode providers only as delimited strings, so provider
        discovery uses the companion ``bbmap_mobile_bb_merged_all_<build>`` table.
        Falls back to the configured known providers if that table is absent.
        """
        known = {p.id: p.name for p in self.cfg.known_providers}
        try:
            df = self._query_df(
                f"SELECT DISTINCT providerid, provider_name "
                f"FROM {self.schema}.{self.merged_prefix}{vintage}"
            )
        except Exception as exc:  # noqa: BLE001 - fall back to configured providers
            log.warning("provider discovery failed (%s); using known_providers", exc)
            return [Provider(id=i, name=n) for i, n in sorted(known.items())]
        out: dict[int, str] = {}
        for _, row in df.iterrows():
            pid = int(row["providerid"])
            out[pid] = known.get(pid, str(row.get("provider_name") or pid))
        return [Provider(id=i, name=n) for i, n in sorted(out.items())]

    def fetch(self, provider_id, technology, vintage) -> CoverageFile:  # pragma: no cover
        """Return the covered res-9 H3 cells for one (provider, service, build).

        Result is cached to parquet under ``data/raw`` so runs are resumable, and
        is flagged ``is_hex`` so normalize skips polyfill.
        """
        import pandas as pd

        col = self._service_column(technology)
        table = self._hex_table(vintage)
        scope = self.cfg.states_scope_key()
        out_dir = self.raw_dir / str(vintage) / str(provider_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = (
            out_dir
            / f"{safe_service_name(technology)}_{self.cfg.backend}_{scope}_hex{self.hex_resolution}.parquet"
        )
        if dest.exists() and dest.stat().st_size > 0:
            return CoverageFile(
                provider_id, technology, vintage, dest,
                is_hex=True, hex_resolution=self.hex_resolution,
            )

        # Delimiter-guarded membership so provider 130077 doesn't match 1300770 /
        # 131310 etc. Read-only: the `||` concat only shapes the value for the
        # LIKE comparison; it never modifies the table. The `_prov` list is
        # comma-separated with no spaces (verified).
        where = [
            f"{col} = 1",
            f"',' || {col}_prov || ',' LIKE %s",
        ]
        params: list = [f"%,{int(provider_id)},%"]
        states = self.cfg.states
        if states != "all":
            where.append("TRIM(state_fips) IN (" + ",".join(["%s"] * len(states)) + ")")
            params.extend(str(s).zfill(2) for s in states)
        sql = f"SELECT h3index FROM {table} WHERE " + " AND ".join(where)

        log.info("  redshift %s provider %s %s (%s)", vintage, provider_id, technology, col)
        df = self._query_df(sql, tuple(params))
        if df.empty:
            # No covered cells is a valid layer, especially for sparse carriers
            # in a state batch. Persist an empty parquet so the opposite vintage
            # can still be analyzed as a complete gain/loss and retries remain
            # resumable. Query/permission errors still propagate from _query_df.
            log.info(
                "  no covered hexes for provider %s %s in %s (states=%s)",
                provider_id, technology, table, states,
            )
            hexes = pd.Series(dtype="string")
        else:
            hexes = df["h3index"].astype(str)
        # Flat band: these hex tables carry a 0/1 coverage flag, not signal.
        pd.DataFrame({"h3": hexes, "signal_dbm": 0.0}).to_parquet(dest, index=False)
        return CoverageFile(
            provider_id, technology, vintage, dest,
            is_hex=True, hex_resolution=self.hex_resolution,
        )


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

    # Accepted local coverage formats, in priority order. CSV/TSV (WKT geometry)
    # are the easiest to export by hand from DBeaver; geojson/gpkg also work.
    _EXTS = (".geojson", ".gpkg", ".csv", ".tsv")

    def list_providers(self, vintage: str) -> list[Provider]:
        vdir = self.dir / vintage
        if not vdir.exists():
            return []
        ids: set[int] = set()
        for ext in self._EXTS:
            for p in vdir.glob(f"*{ext}"):
                try:
                    ids.add(int(p.name.split("_")[0]))
                except ValueError:
                    continue
        known = {p.id: p.name for p in self.cfg.known_providers}
        return [Provider(id=i, name=known.get(i, str(i))) for i in sorted(ids)]

    def fetch(self, provider_id, technology, vintage) -> CoverageFile:
        stem = f"{provider_id}_{safe_service_name(technology)}"
        vdir = self.dir / vintage
        for ext in self._EXTS:
            path = vdir / f"{stem}{ext}"
            if path.exists():
                return CoverageFile(provider_id, technology, vintage, path)
        raise FileNotFoundError(
            f"No local coverage file for {stem} in {vdir} "
            f"(looked for {', '.join(self._EXTS)}). Export it from DBeaver, or "
            f"generate synthetic data with `python -m fcc_audit.cli make-fixtures`."
        )


def get_source(cfg: Config) -> DataSource:
    backend = cfg.backend
    if backend == "fcc":
        return FccDownloadSource(cfg)
    if backend == "redshift":
        return RedshiftSource(cfg)
    if backend == "fixture":
        return FixtureSource(cfg)
    raise ValueError(f"Unknown source backend: {backend!r}")
