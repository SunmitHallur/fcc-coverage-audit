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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

# 50 states + DC. Used when Redshift prefetch expands ``states: all`` into
# per-state caches (overnight batches compose from these, not one national slice).
NATIONAL_STATE_FIPS: frozenset[str] = frozenset({
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36",
    "37", "38", "39", "40", "41", "42", "44", "45", "46", "47", "48",
    "49", "50", "51", "53", "54", "55", "56",
})

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
        picked_current = current or available[0]
        picked_prior = prior or available[1]
        logging.getLogger(__name__).warning(
            "auto-picked vintages current=%s prior=%s from available=%s — "
            "set analysis.vintages explicitly for production D25/J25 comparisons",
            picked_current, picked_prior, available[:6],
        )
        return picked_current, picked_prior


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

    **Performance:** each (vintage, state) is scanned **once** for all configured
    services (``SELECT h3index, …_prov … WHERE state_fips=? AND (flags)``). Provider
    membership is filtered in Python into per-provider parquet caches. That replaces
    the old per-provider ``LIKE %,<id>,%`` query pattern (~12× fewer warehouse scans
    for Big-4 × 3 services). Call :meth:`prefetch` before ``run --workers N`` so
    analyze workers never compete for Redshift connections.

    The table suffix ``<build>`` is a monotonic build/process id and serves as
    the **vintage token**: set ``analysis.vintages.current/prior`` to two builds.

    Because the warehouse already did the H3 indexing, this backend returns the
    covered cells for one (provider, service) directly and the pipeline SKIPS the
    expensive polygon polyfill (see ``CoverageFile.is_hex``). These hex tables
    carry only a 0/1 coverage flag (no modeled signal), so coverage is treated as
    a flat band; tower inference then works from contiguous-coverage blobs.
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
        self._conn = None

    def _validate_redshift_credentials(self) -> None:
        """Fail fast with a clear setup hint when .env was not loaded."""
        missing = []
        for key in ("host", "database", "user", "password"):
            val = str(self.rs.get(key) or "").strip()
            if not val or "${" in val:
                missing.append(key)
        if missing:
            raise RuntimeError(
                "Redshift credentials are missing or still contain unresolved "
                f"${{ENV}} placeholders ({', '.join(missing)}). "
                "Copy .env.example → .env and set REDSHIFT_HOST / REDSHIFT_DB / "
                "REDSHIFT_USER / REDSHIFT_PASSWORD before using --backend redshift."
            )

    # -- connection / query helpers --
    def _connect(self):  # pragma: no cover - requires live credentials
        self._validate_redshift_credentials()
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

    def _get_conn(self):  # pragma: no cover - live only
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def close(self) -> None:  # pragma: no cover - live only
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None

    def _query_df(self, sql: str, params: tuple = ()):  # pragma: no cover - live only
        import pandas as pd

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                try:
                    df = cur.fetch_dataframe()
                except Exception:  # noqa: BLE001 - older connector lacks fetch_dataframe
                    cols = [d[0] for d in cur.description]
                    df = pd.DataFrame(cur.fetchall(), columns=cols)
            return df
        except Exception:
            # Drop a broken connection so the next call reconnects.
            self.close()
            raise

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
        Discovery failures propagate: silently substituting a provider list can
        make a credential/schema failure look like a successful partial run.
        """
        known = {p.id: p.name for p in self.cfg.known_providers}
        df = self._query_df(
            f"SELECT DISTINCT providerid, provider_name "
            f"FROM {self.schema}.{self.merged_prefix}{vintage}"
        )
        if df.empty:
            raise RuntimeError(f"Redshift provider discovery returned no providers for {vintage}")
        out: dict[int, str] = {}
        for _, row in df.iterrows():
            pid = int(row["providerid"])
            out[pid] = known.get(pid, str(row.get("provider_name") or pid))
        return [Provider(id=i, name=n) for i, n in sorted(out.items())]

    def _state_cache_path(
        self, out_dir: Path, technology: str, state: str | None,
    ) -> Path:
        safe = safe_service_name(technology)
        token = "all" if state is None else f"st{str(state).zfill(2)}"
        return out_dir / f"{safe}_{self.cfg.backend}_{token}_hex{self.hex_resolution}.parquet"

    def _shared_slice_path(
        self,
        vintage: str,
        state: str | None,
        service_descs: list[str] | None = None,
    ) -> Path:
        """One Redshift scan per (vintage, state, service-set) shared across providers."""
        token = "all" if state is None else f"st{str(state).zfill(2)}"
        descs = list(service_descs) if service_descs is not None else self._configured_service_descs()
        svc_token = "-".join(sorted(safe_service_name(d) for d in descs)) or "none"
        return (
            self.raw_dir / str(vintage) / "_shared"
            / f"services_{svc_token}_{self.cfg.backend}_{token}_hex{self.hex_resolution}.parquet"
        )

    def _configured_service_descs(self) -> list[str]:
        return [str(s["desc"]) for s in self.cfg.services]

    def _configured_provider_ids(self) -> list[int]:
        if self.cfg.providers_all:
            # Discovery is expensive; fan-out only for known Big-4 when 'all'.
            return [p.id for p in self.cfg.known_providers] or [
                p.id for p in self.cfg.providers
            ]
        return [p.id for p in self.cfg.providers]

    def _write_hex_cache(self, dest: Path, hexes) -> None:
        import pandas as pd

        dest.parent.mkdir(parents=True, exist_ok=True)
        series = pd.Series(hexes, dtype="string")
        cached = pd.DataFrame({
            "h3": series,
            "signal_dbm": pd.Series(0.0, index=series.index, dtype="float64"),
        })
        tmp = dest.with_suffix(dest.suffix + ".part")
        cached.to_parquet(tmp, index=False)
        tmp.replace(dest)

    @staticmethod
    def _cache_ready(path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0

    @staticmethod
    def _provider_mask(prov_series, provider_id: int):
        """Vectorized membership test for comma-delimited provider id lists."""
        needle = f",{int(provider_id)},"
        filled = prov_series.fillna("").astype(str)
        return ("," + filled + ",").str.contains(needle, regex=False)

    def _provider_state_caches_complete(
        self,
        vintage: str,
        state: str | None,
        provider_ids: list[int],
        service_descs: list[str] | None = None,
    ) -> bool:
        descs = list(service_descs) if service_descs is not None else self._configured_service_descs()
        for desc in descs:
            for pid in provider_ids:
                dest = self._state_cache_path(
                    self.raw_dir / str(vintage) / str(pid), desc, state,
                )
                if not self._cache_ready(dest):
                    return False
        return True

    def caches_ready(
        self,
        vintages: list[str],
        states: list[str] | str,
        provider_ids: list[int] | None = None,
        service_descs: list[str] | None = None,
    ) -> bool:
        """True when every requested (vintage, state, provider, service) parquet exists."""
        if states == "all":
            state_list: list[str | None] = sorted(NATIONAL_STATE_FIPS)
        else:
            state_list = [str(s).zfill(2) for s in states]
        pids = [int(p) for p in (provider_ids or self._configured_provider_ids())]
        descs = list(service_descs or self._configured_service_descs())
        for vintage in vintages:
            for state in state_list:
                if not self._provider_state_caches_complete(vintage, state, pids, descs):
                    return False
        return True

    def _ensure_shared_slice(
        self,
        vintage: str,
        state: str | None,
        service_descs: list[str] | None = None,
    ) -> Path:
        """Pull every configured service's flags+provider lists for one state in ONE query.

        Avoids per-provider ``LIKE`` scans. Downstream fan-out writes the existing
        per-(provider, service, state) parquet caches from this shared frame.
        """
        import pandas as pd

        descs = list(service_descs) if service_descs is not None else self._configured_service_descs()
        if not descs:
            raise RuntimeError("No analysis.services configured for Redshift fetch")

        dest = self._shared_slice_path(vintage, state, descs)
        if self._cache_ready(dest):
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        # Directory lock so parallel workers cannot double-query the same slice.
        lock_dir = Path(str(dest) + ".lock")
        while True:
            if self._cache_ready(dest):
                return dest
            try:
                lock_dir.mkdir(parents=False)
                break
            except FileExistsError:
                try:
                    # Recover from a crashed holder (lock dirs are empty markers).
                    if time.time() - lock_dir.stat().st_mtime > 7200:
                        lock_dir.rmdir()
                        continue
                except OSError:
                    pass
                time.sleep(0.25)

        try:
            if self._cache_ready(dest):
                return dest

            table = self._hex_table(vintage)
            select_parts = ["h3index"]
            flag_ors: list[str] = []
            flag_aliases: list[str] = []
            prov_aliases: list[str] = []
            for desc in descs:
                col = self._service_column(desc)
                flag_aliases.append(col)
                prov_aliases.append(f"{col}_prov")
                select_parts.append(f"{col} AS {col}")
                select_parts.append(f"{col}_prov AS {col}_prov")
                flag_ors.append(f"{col} = 1")

            where = [f"({' OR '.join(flag_ors)})"]
            params: list = []
            if state is not None:
                where.insert(0, "state_fips = %s")
                params.append(str(state).zfill(2))

            sql = (
                f"SELECT {', '.join(select_parts)} FROM {table} "
                f"WHERE {' AND '.join(where)}"
            )
            log.info(
                "  redshift SHARED scan %s states=%s services=%d (no per-provider LIKE)",
                vintage, state or "all", len(descs),
            )
            df = self._query_df(sql, tuple(params))
            dest.parent.mkdir(parents=True, exist_ok=True)
            if df.empty:
                empty = pd.DataFrame({"h3": pd.Series(dtype="string")})
                for alias in flag_aliases:
                    empty[alias] = pd.Series(dtype="float64")
                for alias in prov_aliases:
                    empty[alias] = pd.Series(dtype="string")
                tmp = dest.with_suffix(dest.suffix + ".part")
                empty.to_parquet(tmp, index=False)
                tmp.replace(dest)
                return dest

            rename = {c: c.lower() for c in df.columns}
            df = df.rename(columns=rename)
            if "h3index" not in df.columns:
                raise RuntimeError(
                    f"Redshift shared scan for {table} did not return h3index"
                )
            out = pd.DataFrame({"h3": df["h3index"].astype(str)})
            for alias in flag_aliases:
                key = alias.lower()
                out[alias] = (
                    pd.to_numeric(df[key], errors="coerce").fillna(0).astype("int8")
                    if key in df.columns
                    else pd.Series(0, index=out.index, dtype="int8")
                )
            for alias in prov_aliases:
                key = alias.lower()
                out[alias] = (
                    df[key].astype("string")
                    if key in df.columns
                    else pd.Series(pd.NA, index=out.index, dtype="string")
                )
            tmp = dest.with_suffix(dest.suffix + ".part")
            out.to_parquet(tmp, index=False)
            tmp.replace(dest)
            log.info("  shared slice cached %s (%s rows)", dest.name, f"{len(out):,}")
            return dest
        finally:
            try:
                lock_dir.rmdir()
            except OSError:
                pass

    def _fetch_direct_state_slice(
        self,
        provider_id: int,
        technology: str,
        vintage: str,
        state: str | None,
        dest: Path,
    ) -> Path:
        """Minimal Redshift query: one provider × one service × one state.

        Transfers only that provider's covered hexes. This is the sub-minute
        path for a single analysis unit; multi-provider overnight uses
        :meth:`prefetch` shared scans instead.
        """
        import pandas as pd

        if self._cache_ready(dest):
            return dest

        col = self._service_column(technology)
        table = self._hex_table(vintage)
        where = [f"{col} = 1", f"',' || {col}_prov || ',' LIKE %s"]
        params: list = [f"%,{int(provider_id)},%"]
        if state is not None:
            where.insert(0, "state_fips = %s")
            params.insert(0, str(state).zfill(2))
        sql = f"SELECT h3index FROM {table} WHERE " + " AND ".join(where)
        log.info(
            "  redshift DIRECT %s provider %s %s state=%s",
            vintage, provider_id, technology, state or "all",
        )
        t0 = time.perf_counter()
        df = self._query_df(sql, tuple(params))
        elapsed = time.perf_counter() - t0
        if df.empty:
            hexes = pd.Series(dtype="string")
            n = 0
        else:
            if "h3index" not in df.columns:
                # connector may lower-case
                cols = {c.lower(): c for c in df.columns}
                if "h3index" not in cols:
                    raise RuntimeError(
                        f"Redshift direct query for {table}.{col} missing h3index"
                    )
                df = df.rename(columns={cols["h3index"]: "h3index"})
            hexes = df["h3index"].astype(str)
            n = len(hexes)
        self._write_hex_cache(dest, hexes)
        log.info(
            "  direct cache %s (%s hexes in %.1fs)", dest.name, f"{n:,}", elapsed,
        )
        return dest

    def _fanout_shared_to_provider_caches(
        self,
        vintage: str,
        state: str | None,
        provider_ids: list[int] | None = None,
        service_descs: list[str] | None = None,
    ) -> None:
        """Write per-provider hex caches from the shared (vintage, state) slice."""
        import pandas as pd

        pids = [int(p) for p in (provider_ids or self._configured_provider_ids())]
        descs = list(service_descs or self._configured_service_descs())
        if not descs:
            return
        if self._provider_state_caches_complete(vintage, state, pids, descs):
            return

        # Pass service_descs explicitly (thread-safe; do not mutate cfg.services).
        shared_path = self._ensure_shared_slice(vintage, state, descs)
        shared = pd.read_parquet(shared_path)

        for desc in descs:
            col = self._service_column(desc)
            prov_col = f"{col}_prov"
            for pid in pids:
                out_dir = self.raw_dir / str(vintage) / str(pid)
                dest = self._state_cache_path(out_dir, desc, state)
                if self._cache_ready(dest):
                    continue
                if shared.empty or prov_col not in shared.columns:
                    self._write_hex_cache(dest, pd.Series(dtype="string"))
                    continue
                covered = (
                    pd.to_numeric(shared[col], errors="coerce").fillna(0).gt(0)
                    if col in shared.columns
                    else pd.Series(True, index=shared.index)
                )
                mask = covered & self._provider_mask(shared[prov_col], int(pid))
                self._write_hex_cache(dest, shared.loc[mask, "h3"])

    def _prefetch_shared_job(
        self,
        vintage: str,
        state: str | None,
        provider_ids: list[int],
        service_descs: list[str],
    ) -> None:
        """One (vintage, state) shared scan on a dedicated warehouse connection."""
        worker = RedshiftSource(self.cfg)
        try:
            worker._fanout_shared_to_provider_caches(
                vintage, state, provider_ids, service_descs,
            )
        finally:
            worker.close()

    def prefetch(
        self,
        vintages: list[str],
        states: list[str] | str,
        provider_ids: list[int] | None = None,
        service_descs: list[str] | None = None,
        *,
        max_workers: int | None = None,
    ) -> None:
        """Materialize raw caches before parallel analyze.

        * **Multi-provider:** one shared scan per (vintage, state) then local fan-out
          (best for overnight Big-4). ``states="all"`` expands to all 51 FIPS so
          overnight batches compose from per-state caches.
        * **Single provider:** direct per-(provider, service, state) queries so a
          1×1×1 unit only transfers that provider's hexes (sub-minute target).
        * **max_workers:** parallel shared scans (default from
          ``source.redshift.prefetch_workers``, capped at 3 for WLM safety).
        """
        pids = [int(p) for p in (provider_ids or self._configured_provider_ids())]
        descs = list(service_descs or self._configured_service_descs())
        use_direct = len(pids) <= 1
        if states == "all":
            # Shared mode: per-state caches (overnight batches). Direct mode keeps
            # one national slice for a single-provider smoke.
            state_list: list[str | None] = (
                [None] if use_direct else sorted(NATIONAL_STATE_FIPS)
            )
        else:
            state_list = [str(s).zfill(2) for s in states]

        if max_workers is None:
            try:
                max_workers = int(self.rs.get("prefetch_workers", 3))
            except (TypeError, ValueError):
                max_workers = 3
        # Cap concurrency: Redshift WLM thrash hurts more than serial scans.
        max_workers = max(1, min(int(max_workers), 3))

        mode = "DIRECT" if use_direct else "SHARED"
        total = len(vintages) * len(state_list) * (len(pids) * len(descs) if use_direct else 1)
        log.info(
            "prefetch mode=%s vintages=%d states=%d providers=%d services=%d workers=%d",
            mode, len(vintages), len(state_list), len(pids), len(descs),
            1 if use_direct else max_workers,
        )

        if use_direct:
            done = 0
            for vintage in vintages:
                for state in state_list:
                    for pid in pids:
                        for desc in descs:
                            done += 1
                            dest = self._state_cache_path(
                                self.raw_dir / str(vintage) / str(pid), desc, state,
                            )
                            log.info(
                                "prefetch [%d/%d] DIRECT vintage=%s state=%s provider=%s %s",
                                done, total, vintage, state or "all", pid, desc,
                            )
                            self._fetch_direct_state_slice(
                                pid, desc, vintage, state, dest,
                            )
            return

        jobs = [(v, s) for v in vintages for s in state_list]
        if max_workers <= 1 or len(jobs) <= 1:
            for i, (vintage, state) in enumerate(jobs, 1):
                log.info(
                    "prefetch [%d/%d] SHARED vintage=%s state=%s providers=%d",
                    i, total, vintage, state or "all", len(pids),
                )
                self._fanout_shared_to_provider_caches(vintage, state, pids, descs)
            return

        # Parallel shared scans: each job uses its own Redshift connection.
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._prefetch_shared_job, vintage, state, pids, descs): (vintage, state)
                for vintage, state in jobs
            }
            for fut in as_completed(futures):
                vintage, state = futures[fut]
                done += 1
                fut.result()  # raise on worker failure
                log.info(
                    "prefetch [%d/%d] SHARED done vintage=%s state=%s",
                    done, total, vintage, state or "all",
                )

    def _compose_provider_layer(
        self, provider_id: int, technology: str, vintage: str, state_list: list[str],
    ) -> Path:
        import pandas as pd

        out_dir = self.raw_dir / str(vintage) / str(provider_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        scope = "-".join(sorted(state_list))
        composed = (
            out_dir
            / f"{safe_service_name(technology)}_{self.cfg.backend}_{scope}_hex{self.hex_resolution}.parquet"
        )
        paths = [self._state_cache_path(out_dir, technology, state) for state in state_list]
        need_compose = (not self._cache_ready(composed)) or any(
            self._cache_ready(p) and p.stat().st_mtime > composed.stat().st_mtime for p in paths
        )
        if need_compose:
            frames = [pd.read_parquet(p) for p in paths if self._cache_ready(p)]
            if frames:
                combined = pd.concat(frames, ignore_index=True)
                if "h3" in combined.columns:
                    combined = combined.drop_duplicates(subset=["h3"], keep="last")
            else:
                combined = pd.DataFrame({
                    "h3": pd.Series(dtype="string"),
                    "signal_dbm": pd.Series(dtype="float64"),
                })
            tmp = composed.with_suffix(composed.suffix + ".part")
            combined.to_parquet(tmp, index=False)
            tmp.replace(composed)
        return composed

    def fetch(self, provider_id, technology, vintage) -> CoverageFile:  # pragma: no cover
        """Return the covered res-9 H3 cells for one (provider, service, build).

        Cache misses use a **direct** provider×service×state query (smallest
        Redshift transfer). Overnight :meth:`prefetch` warms caches via shared
        scans so workers typically never hit the warehouse.
        """
        pid = int(provider_id)
        states = self.cfg.states
        if states == "all":
            dest = self._state_cache_path(
                self.raw_dir / str(vintage) / str(pid), technology, None,
            )
            self._fetch_direct_state_slice(pid, technology, vintage, None, dest)
            return CoverageFile(
                provider_id, technology, vintage, dest,
                is_hex=True, hex_resolution=self.hex_resolution,
            )

        state_list = [str(s).zfill(2) for s in states]
        out_dir = self.raw_dir / str(vintage) / str(pid)
        for state in state_list:
            dest = self._state_cache_path(out_dir, technology, state)
            if not self._cache_ready(dest):
                self._fetch_direct_state_slice(pid, technology, vintage, state, dest)
        composed = self._compose_provider_layer(pid, technology, vintage, state_list)
        return CoverageFile(
            provider_id, technology, vintage, composed,
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
