"""Validacion de datos con DeepChecks para las entradas de P1 y P2.

Ejecutar en el entorno de calidad (ver ``requirements-quality.txt``):

    .venv-quality/bin/python -m mibici_ml.data_validation

Lee los Parquet limpios de ``dataset.py`` (verificados contra el manifiesto), reconstruye
el panel de riesgo de P2 con las mismas funciones de ``features.py`` y ejecuta dos suites
de DeepChecks sobre muestras con semilla fija:

- ``data_integrity``: nulos, tipos, duplicados, valor unico y atipicos.
- ``train_test_validation``: drift de features y etiqueta, y nuevas categorias entre
  entrenamiento (oct-25 a jun-26) y prueba (sep-26).

El resumen JSON guarda solo agregados (conteos y nombres de chequeos). Los HTML pueden
contener filas de ejemplo y no se versionan.

Los modulos de DeepChecks se importan dentro de las funciones: el resto del paquete
(y sus pruebas no dependientes de DeepChecks) funciona en el entorno principal.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici_ml import config
from mibici_ml.config import (
    DATA_MANIFEST_PATH,
    DATA_VALIDATION_DIR,
    DATA_VALIDATION_SAMPLE_SIZE,
    DATA_VALIDATION_SEED,
    DATA_VALIDATION_SUMMARY_PATH,
    P2_HORIZON_HOURS,
    PROCESSED_DATA_DIR,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
)
from mibici_ml.features import (
    P2_FEATURE_COLUMNS,
    build_p2_features,
    build_p2_hourly_flows,
    build_p2_station_day_panel,
    p2_risk_labels,
)

app = typer.Typer(help="Validacion de datos con DeepChecks para P1 y P2 (MiBici).")

TRIP_COLUMNS = ("origin_id", "destination_id", "start_time", "end_time")
TRIP_FEATURES = ("duracion_min", "hora_inicio", "dia_semana", "origen", "destino")
TRIP_CAT_FEATURES = ("origen", "destino")
RISK_CAT_FEATURES = ("est",)
MAX_DETAIL_CHARS = 300
# Texto con el que DeepChecks indica que falta una columna (etiqueta, fecha o indice).
NOT_APPLICABLE_MARKER = "does not contain"
# Porcentaje maximo de nulos tolerado por variable (0: cualquier NaN se reporta como fallo).
MAX_NULL_PERCENT = 0
SUITE_NAMES = (
    "integridad_viajes",
    "integridad_riesgo_p2",
    "train_test_viajes",
    "train_test_riesgo_p2",
)


# ---------------------------------------------------------------------------
# Entrada verificada
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    """SHA-256 por bloques; no carga el archivo completo en memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(manifest_path: Path = DATA_MANIFEST_PATH) -> dict[str, Any]:
    """Lee el manifiesto de ingesta; falla con un mensaje accionable si no existe."""
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No existe el manifiesto {manifest_path}. Ejecute `python -m mibici_ml.dataset`."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("trips") or not manifest.get("data_version"):
        raise ValueError("Manifiesto sin archivos de viajes o sin data_version.")
    return manifest


def load_verified_trips(processed_dir: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    """Une los viajes limpios de los meses del manifiesto y verifica su hash SHA-256."""
    frames = []
    for entry in manifest["trips"]:
        path = processed_dir / entry["processed_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Faltan viajes limpios: {path}. Ejecute la ingesta.")
        if _sha256(path) != entry["processed_sha256"]:
            raise ValueError(f"Datos modificados: {path}. Ejecute la ingesta de nuevo.")
        frames.append(pd.read_parquet(path, columns=list(TRIP_COLUMNS)))
    trips = pd.concat(frames, ignore_index=True)
    if trips.empty:
        raise ValueError("No hay viajes limpios para validar.")
    return trips


def load_capacities(processed_dir: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    """Capacidad por estacion (ceros y ausentes como en ``risk.load_capacities``)."""
    stations = pd.read_parquet(processed_dir / manifest["stations"]["processed_path"])
    capacities = stations[["station_id", "capacity"]].rename(
        columns={"station_id": "est", "capacity": "capacidad"}
    )
    capacities["est"] = capacities["est"].astype("int64")
    capacities["capacidad"] = capacities["capacidad"].fillna(0).astype("int64")
    return capacities.drop_duplicates("est")


# ---------------------------------------------------------------------------
# Marcos de validacion
# ---------------------------------------------------------------------------


def temporal_split(ts: pd.Series) -> dict[str, np.ndarray]:
    """Entrenamiento y prueba con las ventanas de ``config.py``; sin traslape.

    Misma regla que ``risk.temporal_masks``: train oct-25 a jun-26 y prueba sep-26.
    """
    train = (
        (ts >= TRAIN_START) & (ts < pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1))
    ).to_numpy()
    test = ((ts >= TEST_START) & (ts < pd.Timestamp(TEST_END) + pd.Timedelta(days=1))).to_numpy()
    return {"train": train, "test": test}


def build_trip_frame(trips: pd.DataFrame) -> pd.DataFrame:
    """Variables de viaje para validar: duracion, hora, dia, origen y destino.

    Conserva ``ts`` (inicio del viaje) solo para partir en el tiempo; no es feature.
    """
    if trips.empty:
        raise ValueError("No hay viajes para construir el marco de validacion.")
    duration = (trips["end_time"] - trips["start_time"]).dt.total_seconds() / 60
    return pd.DataFrame(
        {
            "ts": trips["start_time"].to_numpy(),
            "duracion_min": duration.round(2).to_numpy(),
            "hora_inicio": trips["start_time"].dt.hour.to_numpy(),
            "dia_semana": trips["start_time"].dt.dayofweek.to_numpy(),
            "origen": trips["origin_id"].astype("int64").astype(str).to_numpy(),
            "destino": trips["destination_id"].astype("int64").astype(str).to_numpy(),
        }
    )


def build_risk_frame(trips: pd.DataFrame, capacities: pd.DataFrame) -> pd.DataFrame:
    """Panel de riesgo de P2 (features v1 y etiqueta) a partir de los viajes limpios.

    Usa las mismas funciones que ``risk.build_modeling_frame`` con el conjunto v1
    (``P2_FEATURE_COLUMNS``), que es el valor por defecto del pipeline.
    """
    flows = build_p2_hourly_flows(trips)
    panel, _ = build_p2_station_day_panel(flows, capacities)
    features = build_p2_features(panel).reset_index(drop=True)
    labels = p2_risk_labels(panel, P2_HORIZON_HOURS)
    frame = features[list(P2_FEATURE_COLUMNS)].copy()
    frame["est"] = panel["est"].astype("int64").astype(str).to_numpy()
    frame["label"] = labels
    frame["ts"] = panel["ts"].to_numpy()
    return frame


def _sample(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Muestra sin reemplazo con semilla fija; devuelve el marco completo si cabe."""
    if len(frame) <= sample_size:
        return frame.reset_index(drop=True)
    return frame.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def basic_findings(frame: pd.DataFrame, label: str | None = None) -> dict[str, Any]:
    """Conteos directos (sin DeepChecks): nulos, filas duplicadas y tasa de positivos."""
    data = frame.drop(columns=["ts"], errors="ignore")
    findings: dict[str, Any] = {
        "filas": len(data),
        "nulos": int(data.isna().sum().sum()),
        "filas_duplicadas": int(data.duplicated().sum()),
    }
    if label is not None and len(data):
        findings["tasa_positivos"] = round(float(data[label].mean()), 6)
    return findings


# ---------------------------------------------------------------------------
# Suites de DeepChecks
# ---------------------------------------------------------------------------


def _dataset(frame: pd.DataFrame, label: str | None, cat_features: tuple[str, ...]) -> Any:
    """``Dataset`` de DeepChecks; ``ts`` se declara como columna temporal (no es feature)."""
    from deepchecks.tabular import Dataset

    datetime_name = "ts" if "ts" in frame.columns else None
    return Dataset(
        frame,
        label=label,
        cat_features=list(cat_features),
        datetime_name=datetime_name,
    )


def integrity_suite() -> Any:
    """Suite de integridad: ``data_integrity`` de DeepChecks mas el porcentaje de nulos.

    ``data_integrity`` no mide la cantidad de NaN (``Mixed Nulls`` solo detecta
    representaciones mezcladas), por eso se agrega ``PercentOfNulls`` con umbral de 0%.
    """
    from deepchecks.tabular import Suite
    from deepchecks.tabular.checks import PercentOfNulls
    from deepchecks.tabular.suites import data_integrity

    base_checks = list(data_integrity().checks.values())
    nulls = PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(MAX_NULL_PERCENT)
    return Suite("Integridad de datos", *base_checks, nulls)


def run_integrity_suite(
    frame: pd.DataFrame, label: str | None = None, cat_features: tuple[str, ...] = ()
) -> Any:
    """Suite de integridad de DeepChecks sobre un marco (``ts`` opcional)."""
    return integrity_suite().run(_dataset(frame, label, cat_features))


def run_train_test_suite(
    train: pd.DataFrame,
    test: pd.DataFrame,
    label: str | None = None,
    cat_features: tuple[str, ...] = (),
) -> Any:
    """Suite ``train_test_validation``: drift, fugas y categorias nuevas entre train y test."""
    from deepchecks.tabular.suites import train_test_validation

    return train_test_validation().run(
        _dataset(train, label, cat_features), _dataset(test, label, cat_features)
    )


def run_validation_suites(
    trips_all: pd.DataFrame,
    trips_train: pd.DataFrame,
    trips_test: pd.DataFrame,
    risk_all: pd.DataFrame,
    risk_train: pd.DataFrame,
    risk_test: pd.DataFrame,
) -> dict[str, Any]:
    """Ejecuta las cuatro suites de DeepChecks y devuelve sus resultados por nombre.

    ``*_all`` alimenta la integridad; ``*_train`` y ``*_test`` la validacion train/test.
    Los marcos deben traer ``ts`` (columna temporal de DeepChecks, no es feature).
    """
    return {
        "integridad_viajes": run_integrity_suite(trips_all, None, TRIP_CAT_FEATURES),
        "integridad_riesgo_p2": run_integrity_suite(risk_all, "label", RISK_CAT_FEATURES),
        "train_test_viajes": run_train_test_suite(
            trips_train, trips_test, None, TRIP_CAT_FEATURES
        ),
        "train_test_riesgo_p2": run_train_test_suite(
            risk_train, risk_test, "label", RISK_CAT_FEATURES
        ),
    }


def summarize_suite(suite_result: Any) -> dict[str, Any]:
    """Resumen serializable: estado por chequeo y condiciones fallidas con detalle corto."""
    from deepchecks.core import CheckResult

    checks: list[dict[str, Any]] = []
    for item in suite_result.results:
        if isinstance(item, CheckResult):
            conditions = item.conditions_results
            failed = [c for c in conditions if not c.is_pass()]
            if not conditions:
                status = "sin_condiciones"
            elif failed:
                status = "fallido"
            else:
                status = "aprobado"
            checks.append(
                {
                    "chequeo": str(item.get_header()),
                    "estado": status,
                    "condiciones_fallidas": [
                        {"condicion": str(c.name), "detalle": str(c.details)[:MAX_DETAIL_CHARS]}
                        for c in failed
                    ],
                }
            )
        else:
            # Un chequeo sin la columna que necesita (etiqueta, fecha o indice) no aplica a
            # esa suite: se reporta como no aplicable, no como error del pipeline.
            reason = str(getattr(item, "exception", ""))[:MAX_DETAIL_CHARS]
            status = "no_aplicable" if NOT_APPLICABLE_MARKER in reason else "error"
            checks.append(
                {
                    "chequeo": str(getattr(item, "header", type(item).__name__)),
                    "estado": status,
                    "condiciones_fallidas": [],
                    "motivo": reason,
                }
            )
    counts = {
        status: sum(1 for check in checks if check["estado"] == status)
        for status in ("aprobado", "fallido", "sin_condiciones", "no_aplicable", "error")
    }
    return {
        "suite": str(suite_result.name),
        "chequeos_total": len(checks),
        **{f"chequeos_{status}": count for status, count in counts.items()},
        "chequeos": checks,
    }


# ---------------------------------------------------------------------------
# Orquestacion y salida
# ---------------------------------------------------------------------------


def run_validation(
    sample_size: int = DATA_VALIDATION_SAMPLE_SIZE,
    seed: int = DATA_VALIDATION_SEED,
    output_dir: Path = DATA_VALIDATION_DIR,
    summary_path: Path = DATA_VALIDATION_SUMMARY_PATH,
    processed_dir: Path = PROCESSED_DATA_DIR,
    manifest_path: Path = DATA_MANIFEST_PATH,
) -> dict[str, Any]:
    """Carga los datos verificados, ejecuta las suites, guarda HTML y resumen JSON."""
    if sample_size < 1:
        raise ValueError("sample_size debe ser un entero positivo.")
    manifest = read_manifest(manifest_path)
    trips = load_verified_trips(processed_dir, manifest)
    capacities = load_capacities(processed_dir, manifest)
    trip_frame = build_trip_frame(trips)
    risk_frame = build_risk_frame(trips, capacities)

    trip_split = temporal_split(trip_frame["ts"])
    risk_split = temporal_split(risk_frame["ts"])
    trips_all = _sample(trip_frame, sample_size, seed)
    trips_train = _sample(trip_frame[trip_split["train"]], sample_size, seed)
    trips_test = _sample(trip_frame[trip_split["test"]], sample_size, seed)
    risk_all = _sample(risk_frame, sample_size, seed)
    risk_train = _sample(risk_frame[risk_split["train"]], sample_size, seed)
    risk_test = _sample(risk_frame[risk_split["test"]], sample_size, seed)
    logger.info(f"Muestras (semilla {seed}): viajes {len(trips_all)}, riesgo {len(risk_all)}")

    results = run_validation_suites(
        trips_all, trips_train, trips_test, risk_all, risk_train, risk_test
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    suites = []
    for name in SUITE_NAMES:
        html_path = output_dir / f"{name}.html"
        results[name].save_as_html(str(html_path))
        suite = summarize_suite(results[name])
        suite["id"] = name
        suite["html"] = html_path.relative_to(config.PROJ_ROOT).as_posix()
        suites.append(suite)

    summary = {
        "data_version": manifest["data_version"],
        "semilla": seed,
        "tamano_muestra_maximo": sample_size,
        "ventanas": {
            "entrenamiento": [TRAIN_START, TRAIN_END],
            "prueba": [TEST_START, TEST_END],
        },
        "hallazgos_basicos": {
            "viajes_todos": basic_findings(trip_frame),
            "viajes_entrenamiento_muestra": basic_findings(trips_train),
            "viajes_prueba_muestra": basic_findings(trips_test),
            "riesgo_todos": basic_findings(risk_frame, "label"),
            "riesgo_entrenamiento_muestra": basic_findings(risk_train, "label"),
            "riesgo_prueba_muestra": basic_findings(risk_test, "label"),
        },
        "suites": suites,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"Resumen escrito en {summary_path}")
    return summary


@app.command()
def run(
    sample_size: Annotated[
        int, typer.Option(min=1, help="Filas maximas por muestra.")
    ] = DATA_VALIDATION_SAMPLE_SIZE,
    seed: Annotated[int, typer.Option(help="Semilla del muestreo.")] = DATA_VALIDATION_SEED,
    output_dir: Annotated[
        Path, typer.Option(help="Carpeta de reportes HTML.")
    ] = DATA_VALIDATION_DIR,
    summary_path: Annotated[
        Path, typer.Option(help="Ruta del resumen JSON agregado.")
    ] = DATA_VALIDATION_SUMMARY_PATH,
) -> None:
    """Ejecuta DeepChecks sobre viajes y riesgo de P2 y resume los resultados."""
    summary = run_validation(sample_size, seed, output_dir, summary_path)
    for suite in summary["suites"]:
        typer.echo(
            f"{suite['id']}: {suite['chequeos_total']} chequeos, "
            f"{suite['chequeos_aprobado']} aprobados, {suite['chequeos_fallido']} fallidos, "
            f"{suite['chequeos_no_aplicable']} no aplicables, "
            f"{suite['chequeos_error']} con error -> {suite['html']}"
        )
    typer.echo(f"Resumen: {summary_path}")


if __name__ == "__main__":
    app()
