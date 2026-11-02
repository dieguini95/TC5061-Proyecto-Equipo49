"""Calidad de la entrada real de P1; ejecutar en el entorno aislado de DeepChecks.

Reconstruye el panel mediante ``features.py``, verifica el manifiesto completo y
audita la misma cohorte de entrenamiento/evaluacion que los modelos de demanda.
Los errores estructurales o tecnicos detienen la ejecucion; drift y redundancias
son hallazgos observacionales, no una seleccion nueva de modelo ni un ajuste.
"""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
from typing import Annotated, Any

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici_ml import config
from mibici_ml.data_validation import MAX_NULL_PERCENT, summarize_suite
from mibici_ml.features import build_station_hour_panel, p1_model_features
from mibici_ml.modeling.demand import file_sha256, split_mask, verify_inputs

app = typer.Typer(help="Validacion de calidad de la cohorte estacion-hora de P1.")


def build_p1_validation_frame(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Valida el panel completo y selecciona la cohorte sin ocultar corrupcion.

    Las etiquetas de estaciones aun desconocidas y los rezagos sin historia
    permanecen ausentes por diseno. La mascara se contrasta con la disponibilidad
    temporal antes de seleccionar filas: un NaN introducido en una fila madura
    no puede convertirse silenciosamente en una exclusion de calentamiento.
    """
    required = {
        "station_id",
        "timestamp",
        "departures",
        "first_seen",
        "forecast_origin",
        "eligible",
        "evaluation_eligible",
        "mean_4weeks_coverage",
        *config.P1_FEATURE_COLUMNS,
    }
    missing = sorted(required - set(panel.columns))
    if missing or panel.empty:
        raise ValueError(f"Panel P1 vacio o incompleto; faltan columnas: {missing}")
    data = panel.sort_values(["station_id", "timestamp"]).copy()
    ts = pd.to_datetime(data["timestamp"], errors="raise")
    seen = pd.to_datetime(data["first_seen"], errors="raise")
    if ts.dt.tz is not None or seen.dt.tz is not None:
        raise ValueError("P1 requiere fechas locales sin zona horaria.")
    if ts.isna().any() or seen.isna().any() or not ts.eq(ts.dt.floor("h")).all():
        raise ValueError("El panel requiere fechas validas y horas exactas.")
    stations = pd.to_numeric(data["station_id"], errors="raise")
    if (
        stations.isna().any()
        or not (np.isfinite(stations) & stations.gt(0) & stations.mod(1).eq(0)).all()
    ):
        raise ValueError("Identificadores de estacion invalidos.")
    if data.duplicated(["station_id", "timestamp"]).any():
        raise ValueError("Hay claves estacion-hora duplicadas.")
    data["timestamp"] = ts
    gaps = data.groupby("station_id")["timestamp"].diff().dropna()
    if not gaps.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("El panel horario no es continuo.")
    if data.groupby("station_id")["first_seen"].nunique().ne(1).any():
        raise ValueError("first_seen debe ser constante por estacion.")
    origin = ts.dt.normalize()
    if not pd.to_datetime(data["forecast_origin"]).eq(origin).all():
        raise ValueError("La emision de P1 debe ser a medianoche.")
    known = seen.lt(origin)
    for name in ("eligible", "evaluation_eligible"):
        if not pd.api.types.is_bool_dtype(data[name]) or data[name].isna().any():
            raise ValueError(f"La mascara {name} debe ser booleana y no contener nulos.")
    if not data["eligible"].eq(known).all():
        raise ValueError("La mascara eligible no coincide con la primera observacion.")
    if data.loc[~known, "departures"].notna().any():
        raise ValueError("No deben inventarse etiquetas antes de conocer la estacion.")
    target = pd.to_numeric(data.loc[known, "departures"], errors="raise").astype(float)
    if not (np.isfinite(target) & target.ge(0) & target.mod(1).eq(0)).all():
        raise ValueError("El objetivo de P1 debe ser finito, entero y no negativo.")

    first_hour = data.groupby("station_id")["timestamp"].transform("min")

    def available(lag: int) -> pd.Series:
        source_time = ts - pd.Timedelta(hours=lag)
        return source_time.ge(first_hour) & seen.lt(source_time.dt.normalize())

    weeks = [
        available(config.P1_BASELINE_SEASONAL_LAG_HOURS * week)
        for week in range(1, config.P1_BASELINE_MEAN_WEEKS + 1)
    ]
    history_count = pd.concat(weeks, axis=1).sum(axis=1)
    if not data["mean_4weeks_coverage"].eq(history_count).all():
        raise ValueError("La cobertura semanal no coincide con la historia disponible.")
    expected = known & history_count.eq(config.P1_BASELINE_MEAN_WEEKS)
    for lag in config.P1_LAG_HOURS:
        expected &= available(lag)
    if not data["evaluation_eligible"].eq(expected).all():
        raise ValueError("La mascara evaluation_eligible oculta o admite historia incorrecta.")
    if not expected.any():
        raise ValueError("No hay filas elegibles con historia suficiente para P1.")
    cohort = data.loc[expected]
    predictors = p1_model_features(cohort)
    if predictors[["lag_24", "lag_48", "lag_168", "mean_4weeks"]].lt(0).any().any():
        raise ValueError("Los rezagos de conteos no pueden ser negativos.")
    calendar = (cohort["timestamp"].dt.hour, cohort["timestamp"].dt.dayofweek)
    if (
        not predictors["hour"].eq(calendar[0]).all()
        or not predictors["day_of_week"].eq(calendar[1]).all()
        or not predictors["is_weekend"].eq(calendar[1].ge(5).astype(float)).all()
    ):
        raise ValueError("Las variables de calendario no coinciden con la hora objetivo.")
    frame = predictors.copy()
    frame["departures"] = cohort["departures"].astype(float)
    frame["ts"] = cohort["timestamp"]
    coverage = {
        "filas_panel": len(data),
        "estaciones_panel": int(stations.nunique()),
        "filas_estacion_desconocida": int((~known).sum()),
        "filas_historia_insuficiente": int((known & ~expected).sum()),
        "filas_elegibles": len(frame),
        "claves_estacion_hora_duplicadas": 0,
        "objetivos_invalidos": 0,
        "predictores_no_finitos": 0,
        "vectores_predictores_repetidos": int(predictors.duplicated().sum()),
    }
    return frame.reset_index(drop=True), coverage


def split_p1_frame(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Usa las tres ventanas de P1, completas e independientes, sin barajar fechas."""
    windows = {
        "train": (config.TRAIN_START, config.TRAIN_END),
        "val": (config.VAL_START, config.VAL_END),
        "test": (config.TEST_START, config.TEST_END),
    }
    timestamps = frame.rename(columns={"ts": "timestamp"})
    masks = {name: split_mask(timestamps, *bounds) for name, bounds in windows.items()}
    if pd.concat(masks, axis=1).sum(axis=1).ne(1).any():
        raise ValueError("Las ventanas P1 se traslapan o dejan filas fuera de las particiones.")
    splits = {name: frame.loc[mask].reset_index(drop=True) for name, mask in masks.items()}
    if any(part.empty for part in splits.values()):
        raise ValueError("Las particiones train, val y test deben contener filas elegibles.")
    if not (splits["train"].ts.max() < splits["val"].ts.min() < splits["test"].ts.min()):
        raise ValueError("Las particiones de P1 no estan ordenadas temporalmente.")
    return splits


def sample_p1_frame(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Muestreo reproducible, sin reemplazo; no cambia el marco original."""
    if sample_size < 1:
        raise ValueError("sample_size debe ser positivo.")
    if frame.empty:
        raise ValueError("No se puede muestrear un marco vacio.")
    result = frame if len(frame) <= sample_size else frame.sample(n=sample_size, random_state=seed)
    return result.reset_index(drop=True).copy()


def p1_dataset(frame: pd.DataFrame) -> Any:
    """Dataset de regresion: solo las siete variables del modelo, sin identificadores."""
    from deepchecks.tabular import Dataset

    return Dataset(
        frame,
        features=list(config.P1_FEATURE_COLUMNS),
        label="departures",
        label_type="regression",
        cat_features=[],
        datetime_name="ts",
    )


def run_p1_suites(
    samples: dict[str, pd.DataFrame], seed: int = config.P1_DATA_VALIDATION_SEED
) -> dict[str, Any]:
    """Integridad y drift de la entrada P1; no reentrena P1 ni cambia el champion."""
    from deepchecks.tabular import Suite
    from deepchecks.tabular.checks import PercentOfNulls
    from deepchecks.tabular.suites import data_integrity, train_test_validation

    datasets = {name: p1_dataset(frame) for name, frame in samples.items()}
    integrity = Suite(
        "Integridad P1",
        *data_integrity(random_state=seed).checks.values(),
        PercentOfNulls(random_state=seed).add_condition_percent_of_nulls_not_greater_than(
            MAX_NULL_PERCENT
        ),
    )
    return {
        "p1_integridad": integrity.run(datasets["all"]),
        "p1_train_val": train_test_validation(random_state=seed).run(
            datasets["train"], datasets["val"]
        ),
        "p1_train_test": train_test_validation(random_state=seed).run(
            datasets["train"], datasets["test"]
        ),
    }


def summarize_p1_suite(result: Any) -> dict[str, Any]:
    """Distingue chequeos no aplicables a regresion de errores tecnicos reales.

    Reutiliza el resumen comun sin cambiar el comportamiento de P2. DeepChecks
    comunica la ausencia de metadatos opcionales con una excepcion especifica.
    Para los chequeos de clasificacion, 0.19.1 usa ``ModelValidationError`` con
    un mensaje exacto; otros errores de modelo se conservan como errores reales.
    """
    from deepchecks.core import CheckResult
    from deepchecks.core.errors import (
        DatasetValidationError,
        DeepchecksNotSupportedError,
        ModelValidationError,
    )

    summary = summarize_suite(result)
    for item, check in zip(result.results, summary["chequeos"], strict=True):
        if not isinstance(item, CheckResult):
            error = getattr(item, "exception", None)
            not_applicable = (
                isinstance(error, DeepchecksNotSupportedError)
                or (
                    isinstance(error, ModelValidationError)
                    and str(error) == "Check is irrelevant for regression tasks"
                )
                or (
                    isinstance(error, DatasetValidationError)
                    and str(error) == "Dataset does not contain an index"
                )
            )
            check["estado"] = "no_aplicable" if not_applicable else "error"
    for state in ("aprobado", "fallido", "sin_condiciones", "no_aplicable", "error"):
        summary[f"chequeos_{state}"] = sum(
            check["estado"] == state for check in summary["chequeos"]
        )
    return summary


def _provenance() -> dict[str, Any]:
    """Conserva identidad del codigo y versiones, sin rutas locales ni datos de usuario."""

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=config.PROJ_ROOT,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    packages = {}
    for name in ("deepchecks", "numpy", "pandas", "scikit-learn", "pyarrow"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "no_instalado"
    status = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "sha256_codigo": {
            name: file_sha256(config.PROJ_ROOT / name)
            for name in config.P1_DATA_VALIDATION_SOURCE_FILES
        },
        "python": platform.python_version(),
        "paquetes": packages,
    }


def run_validation(
    sample_size: int = config.P1_DATA_VALIDATION_SAMPLE_SIZE,
    seed: int = config.P1_DATA_VALIDATION_SEED,
    output_dir: Path = config.P1_DATA_VALIDATION_DIR,
    summary_path: Path = config.P1_DATA_VALIDATION_SUMMARY_PATH,
    processed_dir: Path = config.PROCESSED_DATA_DIR,
    manifest_path: Path = config.DATA_MANIFEST_PATH,
) -> dict[str, Any]:
    """Verifica datos, reconstruye P1 y guarda solo agregados versionables en JSON.

    Los HTML son evidencia local no versionada y pueden incluir ejemplos. Un
    chequeo fallido por drift/correlacion se documenta; un error tecnico produce
    salida no exitosa despues de conservar su resumen para diagnostico.
    """
    if sample_size < 1:
        raise ValueError("sample_size debe ser positivo.")
    manifest = verify_inputs(Path(manifest_path), Path(processed_dir))
    panel = build_station_hour_panel(
        Path(processed_dir) / "trips", config.START_MONTH, config.END_MONTH
    )
    frame, coverage = build_p1_validation_frame(panel)
    del panel
    splits = split_p1_frame(frame)
    samples = {
        name: sample_p1_frame(part, sample_size, seed)
        for name, part in {"all": frame, **splits}.items()
    }
    logger.info("P1: {} filas elegibles; muestras hasta {}", len(frame), sample_size)
    results = run_p1_suites(samples, seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suites = []
    for name, result in results.items():
        html_path = output_dir / f"{name}.html"
        result.save_as_html(str(html_path))
        summary = summarize_p1_suite(result)
        summary.update({"id": name, "html_archivo": html_path.name})
        suites.append(summary)
    summary = {
        "pregunta": "P1",
        "ejecutado_utc": datetime.now(UTC).isoformat(),
        "data_version": manifest["data_version"],
        "entrada_verificada": {"meses": len(manifest["trips"]), "catalogo_sha256": True},
        "semilla": seed,
        "tamano_muestra_maximo": sample_size,
        "features": list(config.P1_FEATURE_COLUMNS),
        "objetivo": "departures",
        "tipo_tarea": "regression",
        "cobertura": coverage,
        "particiones": {
            name: {
                "filas": len(part),
                "filas_muestra": len(samples[name]),
                "inicio": part.ts.min().isoformat(),
                "fin": part.ts.max().isoformat(),
            }
            for name, part in splits.items()
        },
        "procedencia": _provenance(),
        "suites": suites,
    }
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if any(suite["chequeos_error"] for suite in suites):
        raise RuntimeError("DeepChecks P1 tuvo errores tecnicos; revise el resumen JSON.")
    return summary


@app.command()
def run(
    sample_size: Annotated[int, typer.Option(min=1)] = config.P1_DATA_VALIDATION_SAMPLE_SIZE,
    seed: int = config.P1_DATA_VALIDATION_SEED,
    output_dir: Path = config.P1_DATA_VALIDATION_DIR,
    summary_path: Path = config.P1_DATA_VALIDATION_SUMMARY_PATH,
    processed_dir: Path = config.PROCESSED_DATA_DIR,
    manifest_path: Path = config.DATA_MANIFEST_PATH,
) -> None:
    """Valida P1 desde la ingesta, sin reentrenar los modelos de demanda."""
    summary = run_validation(
        sample_size, seed, output_dir, summary_path, processed_dir, manifest_path
    )
    typer.echo(f"P1 | data_version: {summary['data_version']}")
    typer.echo(f"Cobertura: {json.dumps(summary['cobertura'], ensure_ascii=False)}")
    for name, part in summary["particiones"].items():
        typer.echo(f"{name}: {part['filas']} filas; muestra {part['filas_muestra']}")
    for suite in summary["suites"]:
        typer.echo(
            f"{suite['id']}: {suite['chequeos_total']} chequeos; "
            f"{suite['chequeos_aprobado']} aprobados, {suite['chequeos_fallido']} fallidos, "
            f"{suite['chequeos_sin_condiciones']} sin condiciones, "
            f"{suite['chequeos_no_aplicable']} no aplicables, {suite['chequeos_error']} errores"
        )
    typer.echo(f"Resumen: {summary_path}")


if __name__ == "__main__":
    app()
