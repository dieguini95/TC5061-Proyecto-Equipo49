"""Download, clean and fingerprint the monthly MiBici source files.

Run ``python -m mibici_ml.dataset --help``. Personal user attributes are kept
only in the ignored raw CSVs, never in processed Parquet files or the manifest.
"""

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Annotated
import unicodedata
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup
from loguru import logger
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
import typer
from urllib3.util.retry import Retry

from mibici_ml.config import (
    DATA_MANIFEST_PATH,
    END_MONTH,
    HTTP_USER_AGENT,
    NOMENCLATURA_URL,
    PORTAL_URL,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
    START_MONTH,
    TRIPS_PROCESSED_DIR,
)

app = typer.Typer(add_completion=False)
CLEANING_VERSION = "mibici-p1-v1"
TRIP_COLUMNS = ["trip_id", "start_time", "end_time", "origin_id", "destination_id"]
STATION_COLUMNS = ["station_id", "name", "lat", "lon", "capacity"]

TRIP_ALIASES = {
    "trip_id": ("viaje_id", "id_viaje", "trip_id", "id"),
    "start_time": ("inicio_del_viaje", "inicio_viaje", "fecha_inicio", "start_time"),
    "end_time": ("fin_del_viaje", "fin_viaje", "fecha_fin", "end_time"),
    "origin_id": ("origen_id", "id_origen", "estacion_origen", "origin_id"),
    "destination_id": ("destino_id", "id_destino", "estacion_destino", "destination_id"),
}
STATION_ALIASES = {
    "station_id": ("id", "numero", "estacion_id", "id_estacion", "station_id"),
    "name": ("name", "nombre", "nombre_estacion"),
    "lat": ("latitude", "latitud", "lat"),
    "lon": ("longitude", "longitud", "lon", "lng"),
    "capacity": ("dpcapacity", "capacidad", "capacity", "anclajes"),
}


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory, including the exact source bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_column_name(value: str) -> str:
    """Unify case, accents and separators without changing source values."""
    text = unicodedata.normalize("NFKD", str(value).strip().lstrip("\ufeff"))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _rename_columns(frame: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    normalized = [normalize_column_name(column) for column in frame.columns]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Columnas ambiguas despues de normalizar sus nombres.")
    result = frame.copy()
    result.columns = normalized
    mapping = {}
    for canonical, candidates in aliases.items():
        found = [name for name in candidates if name in result.columns]
        if len(found) != 1:
            raise ValueError(f"Se requiere una columna para {canonical}; encontradas: {found}.")
        mapping[found[0]] = canonical
    return result.rename(columns=mapping)


def read_source_csv(path: Path) -> tuple[pd.DataFrame, str]:
    """Read UTF-8/BOM or Windows-1252 CSVs, preserving exact row values."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with Path(path).open("r", encoding=encoding, newline="") as handle:
                sample = handle.read(65536)
            try:
                separator = csv.Sniffer().sniff(sample, delimiters=",;").delimiter
            except csv.Error:
                separator = ","
            frame = pd.read_csv(
                path, encoding=encoding, sep=separator, dtype="string", keep_default_na=False
            )
            return frame, encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo decodificar {Path(path).name} como UTF-8 ni Windows-1252.")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def _positive_integer(series: pd.Series) -> pd.Series:
    return (series.notna() & series.gt(0) & series.mod(1).eq(0)).fillna(False)


def clean_trips(frame: pd.DataFrame, month: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Prepare trips for P1 without personal attributes or catalogue filtering.

    Exact duplicates are removed before dropping user columns. Same-station,
    short positive and long positive trips remain valid departures. Exclusions
    are sequential and sum to raw-clean. Timestamps retain MiBici's local clock.
    """
    raw_rows = len(frame)
    exact_duplicates = int(frame.duplicated().sum())
    result = _rename_columns(frame.drop_duplicates(), TRIP_ALIASES)[TRIP_COLUMNS].copy()
    result["trip_id"] = result["trip_id"].astype("string").str.strip()
    for column in ("start_time", "end_time"):
        result[column] = pd.to_datetime(result[column], format="mixed", errors="coerce")
        if not pd.api.types.is_datetime64_any_dtype(result[column]):
            raise ValueError("Fechas con zonas horarias mixtas: revisar la fuente.")
        if result[column].dt.tz is not None:
            raise ValueError("Se esperan fechas locales de MiBici sin zona horaria explicita.")
    for column in ("origin_id", "destination_id"):
        result[column] = _numeric(result[column])
    qa = {"exact_duplicates_removed": exact_duplicates}
    invalid_time = result["start_time"].isna() | result["end_time"].isna()
    qa["invalid_timestamps_removed"] = int(invalid_time.sum())
    result = result.loc[~invalid_time].copy()
    nonpositive = result["end_time"].le(result["start_time"])
    qa["nonpositive_duration_removed"] = int(nonpositive.sum())
    result = result.loc[~nonpositive].copy()
    valid_station = _positive_integer(result["origin_id"]) & _positive_integer(
        result["destination_id"]
    )
    qa["invalid_station_ids_removed"] = int((~valid_station).sum())
    result = result.loc[valid_station].copy()
    outside_month = pd.Series(False, index=result.index)
    if month is not None:
        period = pd.Period(month.replace("_", "-"), freq="M")
        outside_month = result["start_time"].dt.to_period("M").ne(period)
    qa["starts_outside_source_month_removed"] = int(outside_month.sum())
    result = result.loc[~outside_month].copy()
    result[["origin_id", "destination_id"]] = result[["origin_id", "destination_id"]].astype(
        "int64"
    )
    qa["same_station_retained"] = int(result["origin_id"].eq(result["destination_id"]).sum())
    minutes = (result["end_time"] - result["start_time"]).dt.total_seconds() / 60
    qa["over_180_minutes_retained"] = int(minutes.gt(180).sum())
    qa["under_1_minute_retained"] = int(minutes.lt(1).sum())
    qa["duplicate_trip_ids_retained"] = int(result["trip_id"].duplicated().sum())
    qa["rows_removed_total"] = raw_rows - len(result)
    result = result.sort_values(["start_time", "trip_id"], kind="stable").reset_index(drop=True)
    return result[TRIP_COLUMNS], qa


def clean_stations(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize the catalogue, retaining zero/unknown capacities as metadata.

    Never filter historical trips through this current catalogue: previously
    operating station identifiers need not exist in the latest snapshot.
    """
    duplicates = int(frame.duplicated().sum())
    result = _rename_columns(frame.drop_duplicates(), STATION_ALIASES)[STATION_COLUMNS].copy()
    for column in ("station_id", "lat", "lon", "capacity"):
        result[column] = _numeric(result[column])
    valid_ids = _positive_integer(result["station_id"])
    removed = int((~valid_ids).sum())
    result = result.loc[valid_ids].copy()
    result["station_id"] = result["station_id"].astype("int64")
    if result["station_id"].duplicated().any():
        raise ValueError("El catalogo tiene identificadores repetidos con datos distintos.")
    result["name"] = result["name"].astype("string").str.strip()
    for column, lower, upper in (("lat", -90, 90), ("lon", -180, 180)):
        result.loc[~result[column].between(lower, upper), column] = float("nan")
        result[column] = result[column].astype("float64")
    result.loc[result["capacity"].lt(0), "capacity"] = float("nan")
    result["capacity"] = result["capacity"].astype("float64")
    qa = {
        "exact_duplicates_removed": duplicates,
        "invalid_station_ids_removed": removed,
        "missing_coordinates": int(result[["lat", "lon"]].isna().any(axis=1).sum()),
        "zero_capacity": int(result["capacity"].eq(0).sum()),
        "unknown_capacity": int(result["capacity"].isna().sum()),
    }
    return result.sort_values("station_id").reset_index(drop=True), qa


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": HTTP_USER_AGENT})
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def discover_source_urls(session: requests.Session, months: list[str]) -> dict[str, str]:
    """Resolve exact monthly filenames from the official portal; never guess IDs."""
    response = session.get(PORTAL_URL, timeout=(15, 120))
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    by_name = {}
    for anchor in soup.find_all("a", href=True):
        url = urljoin(PORTAL_URL, anchor["href"])
        filename = Path(unquote(urlparse(url).path)).name
        if re.fullmatch(r"datos_abiertos_\d{4}_\d{2}\.csv", filename):
            previous = by_name.get(filename)
            if previous is not None and previous != url:
                raise ValueError(f"El portal ofrece mas de una URL para {filename}.")
            by_name[filename] = url
    missing = [month for month in months if f"datos_abiertos_{month}.csv" not in by_name]
    if missing:
        raise ValueError(f"Meses no encontrados en el portal oficial: {', '.join(missing)}")
    return {month: by_name[f"datos_abiertos_{month}.csv"] for month in months}


def _check_hash(path: Path, expected: str | None) -> str:
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(
            f"SHA-256 inesperado en {path.name}: esperado {expected}, obtenido {actual}. "
            "No se sustituiran silenciosamente los datos. Revise el origen; --refresh "
            "descarga y acepta explicitamente una nueva version de las fuentes."
        )
    return actual


def _local_candidate(source_dir: Path | None, filename: str, expected: str | None) -> Path | None:
    if source_dir is None:
        return None
    canonical = Path(source_dir) / filename
    if canonical.is_file():
        _check_hash(canonical, expected)
        return canonical
    stem = Path(filename).stem
    candidates = [
        path
        for path in Path(source_dir).glob(f"{stem} (*).csv")
        if re.fullmatch(re.escape(stem) + r" \(\d+\)\.csv", path.name)
    ]
    candidates.sort(key=lambda path: int(re.search(r"\((\d+)\)", path.name).group(1)))
    if not candidates:
        return None
    hashes = {path: sha256_file(path) for path in candidates}
    if expected is not None:
        matching = [path for path in candidates if hashes[path] == expected]
        if not matching:
            _check_hash(candidates[0], expected)
        return matching[0]
    if len(set(hashes.values())) != 1:
        raise ValueError(f"Copias numeradas distintas de {filename}; indique un archivo canonico.")
    return candidates[0]


def _download(session: requests.Session, url: str, destination: Path) -> None:
    """Stream a source to a temporary file, retrying interrupted response bodies."""
    for attempt in range(3):
        try:
            with session.get(url, timeout=(15, 120), stream=True) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            return
        except requests.RequestException:
            destination.unlink(missing_ok=True)
            if attempt == 2:
                raise
            logger.warning("Descarga interrumpida; reintentando {}", destination.name)


def _obtain_raw(
    session: requests.Session,
    url: str,
    target: Path,
    expected: str | None,
    source_dir: Path | None,
    refresh: bool,
) -> str:
    if target.is_file() and not refresh:
        return _check_hash(target, expected)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        candidate = None if refresh else _local_candidate(source_dir, target.name, expected)
        if candidate is not None:
            logger.info("Reutilizando {}", candidate.name)
            shutil.copyfile(candidate, temporary)
        else:
            logger.info("Descargando {}", target.name)
            _download(session, url, temporary)
        actual = _check_hash(temporary, expected)
        temporary.replace(target)
        return actual
    finally:
        temporary.unlink(missing_ok=True)


def _write_parquet(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow", compression="snappy")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _historical_hashes() -> dict[str, str]:
    inventory = PROJ_ROOT / "reports" / "tables" / "inventario_meses.csv"
    if not inventory.is_file():
        return {}
    with inventory.open(encoding="utf-8-sig", newline="") as handle:
        return {row["mes"]: row["sha256"] for row in csv.DictReader(handle)}


def _month_range(start_month: str, end_month: str) -> list[str]:
    if not re.fullmatch(r"\d{4}-\d{2}", start_month) or not re.fullmatch(
        r"\d{4}-\d{2}", end_month
    ):
        raise ValueError("Las fechas deben usar el formato AAAA-MM.")
    start, end = pd.Period(start_month, freq="M"), pd.Period(end_month, freq="M")
    if start > end:
        raise ValueError("El mes inicial debe ser anterior o igual al mes final.")
    return [period.strftime("%Y_%m") for period in pd.period_range(start, end, freq="M")]


def _data_version(manifest: dict) -> str:
    """Version raw input and actual output bytes, excluding machine-specific paths."""
    identity = {
        "cleaning_version": manifest["cleaning_version"],
        "date_range": manifest["date_range"],
        "trips": [
            {key: entry[key] for key in ("month", "raw_sha256", "processed_sha256")}
            for entry in manifest["trips"]
        ],
        "stations": {key: manifest["stations"][key] for key in ("raw_sha256", "processed_sha256")},
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def ingest_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    manifest_path: Path = DATA_MANIFEST_PATH,
    start_month: str = START_MONTH,
    end_month: str = END_MONTH,
    source_dir: Path | None = None,
    refresh: bool = False,
) -> dict:
    """Ingest one month at a time and save a metadata-only, versioned manifest.

    Cached/copied files must match the prior manifest or historical inventory.
    ``refresh=True`` accepts updated downloads explicitly. ``raw_path`` and
    ``processed_path`` are relative to their respective root directories.
    """
    raw_dir, processed_dir, manifest_path = map(Path, (raw_dir, processed_dir, manifest_path))
    months = _month_range(start_month, end_month)
    if source_dir is not None and not Path(source_dir).is_dir():
        raise ValueError(f"No existe el directorio de fuentes: {source_dir}")
    previous = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {} if refresh else _historical_hashes()
    if not refresh:
        expected.update(
            {entry["month"]: entry["raw_sha256"] for entry in previous.get("trips", [])}
        )
    manifest = {
        "schema_version": 1,
        "cleaning_version": CLEANING_VERSION,
        "date_range": {"start_month": start_month, "end_month": end_month},
        "timestamp_convention": "MiBici local clock, timezone-naive",
        "paths_relative_to": {"raw_path": "raw_dir", "processed_path": "processed_dir"},
        "trips": [],
    }
    with _session() as session:
        urls = discover_source_urls(session, months)
        for month in months:
            raw_path = raw_dir / f"datos_abiertos_{month}.csv"
            raw_hash = _obtain_raw(
                session, urls[month], raw_path, expected.get(month), source_dir, refresh
            )
            original, encoding = read_source_csv(raw_path)
            clean, qa = clean_trips(original, month)
            if clean.empty:
                raise ValueError(f"El mes {month} no contiene viajes validos; no equivale a cero.")
            processed_path = (
                processed_dir
                / TRIPS_PROCESSED_DIR.relative_to(PROCESSED_DATA_DIR)
                / f"{month}.parquet"
            )
            entry = {
                "month": month,
                "source_url": urls[month],
                "raw_path": raw_path.relative_to(raw_dir).as_posix(),
                "processed_path": processed_path.relative_to(processed_dir).as_posix(),
                "raw_sha256": raw_hash,
                "processed_sha256": _write_parquet(clean, processed_path),
                "raw_bytes": raw_path.stat().st_size,
                "raw_rows": len(original),
                "clean_rows": len(clean),
                "encoding": encoding,
                "start_time_min": clean["start_time"].min().isoformat(),
                "start_time_max": clean["start_time"].max().isoformat(),
                "origin_stations": int(clean["origin_id"].nunique()),
                "qa": qa,
            }
            manifest["trips"].append(entry)
            logger.info(
                "{}: {:,} viajes originales, {:,} validos", month, len(original), len(clean)
            )
            del original, clean
        station_name = Path(unquote(urlparse(NOMENCLATURA_URL).path)).name
        station_raw = raw_dir / station_name
        station_expected = None if refresh else previous.get("stations", {}).get("raw_sha256")
        station_hash = _obtain_raw(
            session, NOMENCLATURA_URL, station_raw, station_expected, source_dir, refresh
        )
        original, encoding = read_source_csv(station_raw)
        stations, qa = clean_stations(original)
        if stations.empty:
            raise ValueError("El catalogo no contiene estaciones validas.")
        station_output = processed_dir / "stations.parquet"
        manifest["stations"] = {
            "source_url": NOMENCLATURA_URL,
            "raw_path": station_raw.relative_to(raw_dir).as_posix(),
            "processed_path": station_output.relative_to(processed_dir).as_posix(),
            "raw_sha256": station_hash,
            "processed_sha256": _write_parquet(stations, station_output),
            "raw_bytes": station_raw.stat().st_size,
            "raw_rows": len(original),
            "clean_rows": len(stations),
            "encoding": encoding,
            "qa": qa,
        }
    manifest["totals"] = {
        "months": len(months),
        "raw_trips": sum(entry["raw_rows"] for entry in manifest["trips"]),
        "clean_trips": sum(entry["clean_rows"] for entry in manifest["trips"]),
        "catalogue_stations": len(stations),
    }
    manifest["data_version"] = _data_version(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".part")
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    logger.success("Ingesta completa. Version de datos: {}", manifest["data_version"])
    return manifest


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    manifest_path: Path = DATA_MANIFEST_PATH,
    start_month: str = START_MONTH,
    end_month: str = END_MONTH,
    source_dir: Annotated[Path | None, typer.Option(help="Carpeta de CSV ya descargados.")] = None,
    refresh: Annotated[
        bool, typer.Option(help="Descargar y aceptar nuevos hashes de las fuentes.")
    ] = False,
):
    """Download/reuse CSVs, normalize them, and write Parquet plus a manifest.

    Sin subcomando (``python -m mibici_ml.dataset``) ejecuta la ingesta con estas opciones.
    """
    if ctx.invoked_subcommand is not None:
        return
    ingest_dataset(
        raw_dir, processed_dir, manifest_path, start_month, end_month, source_dir, refresh
    )


@app.command("ingest")
def ingest_command(
    start_month: Annotated[str, typer.Option(help="Primer mes AAAA-MM. Vacio = config.")] = "",
    end_month: Annotated[str, typer.Option(help="Ultimo mes AAAA-MM. Vacio = config.")] = "",
    source_dir: Annotated[str, typer.Option(help="CSV ya descargados. Vacio = descargar.")] = "",
    refresh: Annotated[
        str, typer.Option(help="true para descargar y aceptar hashes nuevos.")
    ] = "false",
) -> None:
    """Entry point de MLflow Projects: acepta parametros de texto y usa config si estan vacios.

    Equivale a ``main`` pero tolera que ``mlflow run`` pase cadenas vacias o ``true``/``false``.
    """
    ingest_dataset(
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        DATA_MANIFEST_PATH,
        start_month.strip() or START_MONTH,
        end_month.strip() or END_MONTH,
        Path(source_dir.strip()) if source_dir.strip() else None,
        refresh.strip().lower() in {"1", "true", "yes", "si", "sí"},
    )


if __name__ == "__main__":
    app()
