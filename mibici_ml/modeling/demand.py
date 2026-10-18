"""P1: baselines causales por estacion-hora y MLflow. Ejecutar como modulo."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Annotated
import zipfile

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from mibici_ml import config
from mibici_ml.dataset import _data_version
from mibici_ml.features import build_station_hour_panel, validate_p1_forecast_contract

app = typer.Typer()


def file_sha256(path: Path) -> str:
    """Identifica los bytes realmente usados antes de evaluar."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs(manifest_path: Path, processed_dir: Path) -> dict:
    """Rechaza archivos procesados distintos de la version documentada."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("data_version") or not manifest.get("trips"):
        raise ValueError("Manifiesto sin data_version o archivos de viajes.")
    expected = [
        p.strftime("%Y_%m")
        for p in pd.period_range(config.START_MONTH, config.END_MONTH, freq="M")
    ]
    if [entry["month"] for entry in manifest["trips"]] != expected:
        raise ValueError("El manifiesto debe enumerar exactamente todos los meses configurados.")
    if manifest.get("date_range") != {
        "start_month": config.START_MONTH,
        "end_month": config.END_MONTH,
    }:
        raise ValueError("El rango del manifiesto no coincide con config.py.")
    if _data_version(manifest) != manifest["data_version"]:
        raise ValueError("La identidad del manifiesto fue modificada.")
    for entry in manifest["trips"]:
        if entry["processed_path"] != f"trips/{entry['month']}.parquet":
            raise ValueError("Ruta procesada no coincide con el mes del manifiesto.")
        path = processed_dir / entry["processed_path"]
        if not path.is_file() or file_sha256(path) != entry["processed_sha256"]:
            raise ValueError(f"Datos ausentes o modificados: {path}. Ejecute la ingesta.")
    stations = manifest["stations"]
    if (
        stations["processed_path"] != "stations.parquet"
        or file_sha256(processed_dir / "stations.parquet") != stations["processed_sha256"]
    ):
        raise ValueError("El catalogo procesado no coincide con el manifiesto.")
    return manifest


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """MAE en viajes/h, RMSE y WAPE como fraccion; evita division por cada objetivo."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Se requieren vectores no vacios de igual longitud.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("No se admiten NaN ni infinito en la evaluacion.")
    error = predicted - actual
    denominator = float(actual.sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "wape": float(np.abs(error).sum() / denominator) if denominator > 0 else float("nan"),
        "bias": float(error.mean()),
        "actual_total": denominator,
    }


def split_mask(panel: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Incluye todo el ultimo dia; mismas fechas para todas las estaciones."""
    return (panel["timestamp"] >= pd.Timestamp(start)) & (
        panel["timestamp"] < pd.Timestamp(end) + pd.Timedelta(days=1)
    )


def evaluate_baselines(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compara referencias prefijadas en una cohorte comun de origen diario.

    Los dias ya observados de validacion/prueba alimentan rezagos de dias posteriores;
    ninguna variable usa informacion posterior a la medianoche de emision.
    """
    rows, coverage = [], {}
    periods = {
        "train": (config.TRAIN_START, config.TRAIN_END),
        "val": (config.VAL_START, config.VAL_END),
        "test": (config.TEST_START, config.TEST_END),
    }
    for split, (start, end) in periods.items():
        window = split_mask(panel, start, end)
        known = window & panel["eligible"]
        common = window & panel["evaluation_eligible"]
        coverage[split] = {
            "known_station_hours": int(known.sum()),
            "evaluated_station_hours": int(common.sum()),
            "excluded_warmup_hours": int((known & ~common).sum()),
            "n_stations": int(panel.loc[common, "station_id"].nunique()),
            "zero_fraction": float(panel.loc[common, "departures"].eq(0).mean()),
        }
        if split == "train":
            continue
        if not common.any():
            raise ValueError(f"No hay observaciones evaluables en {split}.")
        actual = panel.loc[common, "departures"].to_numpy(dtype=float)
        for model in config.P1_MODEL_NAMES:
            rows.append(
                {
                    "baseline": model,
                    "split": split,
                    **regression_metrics(actual, panel.loc[common, model].to_numpy(float)),
                    "n_rows": int(common.sum()),
                    "n_stations": coverage[split]["n_stations"],
                }
            )
    return pd.DataFrame(rows), coverage


def select_baseline(metrics: pd.DataFrame) -> str:
    """Seleccion por WAPE de validacion; desempate MAE y nombre, nunca por test."""
    validation = metrics.loc[metrics["split"].eq("val")]
    if validation["wape"].isna().any():
        raise ValueError("WAPE de validacion indefinido: no hay volumen positivo.")
    return str(validation.sort_values(["wape", "mae", "baseline"]).iloc[0]["baseline"])


def station_metrics(test: pd.DataFrame) -> pd.DataFrame:
    """Errores por estacion para detectar fallas ocultas por el promedio global."""
    records = []
    for model in config.P1_MODEL_NAMES:
        for station_id, frame in test.groupby("station_id", sort=True):
            records.append(
                {
                    "baseline": model,
                    "station_id": int(station_id),
                    "n_rows": len(frame),
                    **regression_metrics(frame.departures.to_numpy(), frame[model].to_numpy()),
                }
            )
    return pd.DataFrame(records)


def create_figures(metrics: pd.DataFrame, panel: pd.DataFrame) -> None:
    """Figuras agregadas sin usuarios ni registros individuales de viaje."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"seasonal_168h": "#315f9b", "mean_4weeks": "#da8b35"}
    labels = {"seasonal_168h": "Semana anterior", "mean_4weeks": "Media de 4 semanas"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, metric, title in zip(axes, ["mae", "wape"], ["MAE (viajes/h)", "WAPE (%)"]):
        for i, model in enumerate(config.P1_MODEL_NAMES):
            values = [
                float(
                    metrics.loc[metrics.baseline.eq(model) & metrics.split.eq(s), metric].iloc[0]
                )
                for s in ("val", "test")
            ]
            if metric == "wape":
                values = [100 * v for v in values]
            bars = ax.bar(
                np.arange(2) + (i - 0.5) * 0.34,
                values,
                0.34,
                color=colors[model],
                label=labels[model],
            )
            ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_xticks([0, 1], ["Validacion", "Prueba"])
        ax.set_title(title)
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.15)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("P1 | Baselines en la misma cohorte de estaciones-hora")
    fig.savefig(config.P1_FIGURES["comparison"], dpi=160)
    plt.close(fig)
    test = panel.loc[
        split_mask(panel, config.TEST_START, config.TEST_END) & panel.evaluation_eligible
    ].copy()
    test["date"] = test.timestamp.dt.normalize()
    daily = test.groupby("date")[["departures", *config.P1_MODEL_NAMES]].sum()
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    ax.plot(daily.index, daily.departures, color="#26323f", label="Observado", linewidth=2)
    for model in config.P1_MODEL_NAMES:
        ax.plot(daily.index, daily[model], color=colors[model], label=labels[model])
    ax.set(title="P1 | Totales diarios de la cohorte evaluable en prueba", ylabel="Viajes")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(config.P1_FIGURES["daily"], dpi=160)
    plt.close(fig)
    train = panel.loc[split_mask(panel, config.TRAIN_START, config.TRAIN_END) & panel.eligible]
    station = int(train.groupby("station_id").departures.sum().idxmax())
    example = test.loc[
        test.station_id.eq(station)
        & (test.timestamp < pd.Timestamp(config.TEST_START) + pd.Timedelta(days=7))
    ]
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    ax.plot(example.timestamp, example.departures, color="#26323f", label="Observado")
    for model in config.P1_MODEL_NAMES:
        ax.plot(
            example.timestamp, example[model], color=colors[model], label=labels[model], alpha=0.8
        )
    ax.set(title=f"P1 | Estacion {station}: primera semana de prueba", ylabel="Salidas por hora")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(config.P1_FIGURES["station"], dpi=160)
    plt.close(fig)


def code_provenance(root: Path) -> tuple[dict, list[Path]]:
    """Identifica commit base y codigo local exacto aunque aun no haya commit nuevo."""
    files = sorted((root / "mibici_ml").rglob("*.py"))
    files += [
        p for p in [root / "requirements-p1.lock.txt", root / "pyproject.toml"] if p.is_file()
    ]
    hashes = {p.relative_to(root).as_posix(): file_sha256(p) for p in files}
    version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    context = [p for p in [root / "README.md", root / "LICENSE"] if p.is_file()]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", True
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "source_version": version,
        "source_hashes": hashes,
        "context_hashes": {p.relative_to(root).as_posix(): file_sha256(p) for p in context},
    }, [*files, *context]


def log_runs(
    metrics: pd.DataFrame,
    manifest: dict,
    quality: dict,
    selected: str,
    tracking_uri: str,
    root: Path = config.PROJ_ROOT,
) -> list[dict]:
    """Registra ambos baselines y snapshot reproducible en el experimento del equipo."""
    validate_p1_forecast_contract()
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(config.EXPERIMENT_NAME)
    provenance, sources = code_provenance(root)
    runs = []
    with tempfile.TemporaryDirectory(prefix="p1-source-") as temporary:
        temporary = Path(temporary)
        archive = temporary / "code_snapshot.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in sources:
                bundle.write(path, path.relative_to(root).as_posix())
        provenance_path = temporary / "code_provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        for model in config.P1_MODEL_NAMES:
            run_name = "P1-estacional-168h" if model == "seasonal_168h" else "P1-media-4semanas"
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_params(
                    {
                        "model": model,
                        "random_seed": config.RANDOM_SEED,
                        "start_month": config.START_MONTH,
                        "end_month": config.END_MONTH,
                        "train_start": config.TRAIN_START,
                        "train_end": config.TRAIN_END,
                        "val_start": config.VAL_START,
                        "val_end": config.VAL_END,
                        "test_start": config.TEST_START,
                        "test_end": config.TEST_END,
                        "data_version": manifest["data_version"],
                        "source_version": provenance["source_version"],
                        "horizon_hours": config.P1_FORECAST_HORIZON_HOURS,
                        "forecast_origin_hour": config.P1_FORECAST_ORIGIN_HOUR,
                        "timezone": config.P1_TIMEZONE,
                        "lag_hours": ",".join(
                            str(config.P1_BASELINE_SEASONAL_LAG_HOURS * w)
                            for w in range(
                                1,
                                2
                                if model == "seasonal_168h"
                                else config.P1_BASELINE_MEAN_WEEKS + 1,
                            )
                        ),
                        "mean_weeks": 1
                        if model == "seasonal_168h"
                        else config.P1_BASELINE_MEAN_WEEKS,
                        "common_cohort": "four_prior_weeks_available_after_first_observed_day",
                        "label": "valid_trip_departures_per_station_hour",
                        "wape_units": "fraction",
                        "selection_metric": "val_wape",
                        "retraining": "none_historical_baseline",
                    }
                )
                mlflow.set_tags(
                    {
                        "equipo": "49",
                        "pregunta": "P1",
                        "git_commit": provenance["git_commit"],
                        "mlflow.source.git.commit": provenance["git_commit"],
                        "git_dirty": str(provenance["git_dirty"]).lower(),
                        "selected_on_validation": str(model == selected).lower(),
                        "evaluation": "daily_rolling_origin_fixed_baselines_same_cohort",
                        "scope": "local_tracking",
                        "source_snapshot": "code_snapshot.zip",
                    }
                )
                for row in metrics.loc[metrics.baseline.eq(model)].to_dict(orient="records"):
                    mlflow.log_metrics(
                        {
                            f"{row['split']}_{key}": float(row[key])
                            for key in (
                                "mae",
                                "rmse",
                                "wape",
                                "bias",
                                "n_rows",
                                "n_stations",
                                "actual_total",
                            )
                            if np.isfinite(row[key])
                        }
                    )
                mlflow.log_dict(manifest, "data_manifest.json")
                mlflow.log_dict(quality, "p1_quality.json")
                for path in (archive, provenance_path):
                    mlflow.log_artifact(str(path), artifact_path="reproducibility")
                for path in (config.P1_METRICS_PATH, config.P1_STATION_METRICS_PATH):
                    mlflow.log_artifact(str(path), artifact_path="tables")
                for path in config.P1_FIGURES.values():
                    mlflow.log_artifact(str(path), artifact_path="figures")
                runs.append(
                    {
                        "baseline": model,
                        "run_name": run_name,
                        "run_id": run.info.run_id,
                        "experiment_id": experiment.experiment_id,
                    }
                )
    return runs


@app.command()
def main(
    manifest_path: Annotated[
        Path, typer.Option(help="Manifiesto de dataset.py.")
    ] = config.DATA_MANIFEST_PATH,
    with_mlflow: Annotated[bool, typer.Option(help="Registrar experimentos locales.")] = True,
    tracking_uri: Annotated[
        str, typer.Option(help="Backend de MLflow.")
    ] = config.MLFLOW_TRACKING_URI,
) -> None:
    """Construye variables y evalua dos baselines prefijados; registra MLflow local."""
    validate_p1_forecast_contract()
    manifest = verify_inputs(manifest_path, config.PROCESSED_DATA_DIR)
    logger.info("Construyendo panel estacion-hora {} a {}", config.START_MONTH, config.END_MONTH)
    panel = build_station_hour_panel(config.TRIPS_PROCESSED_DIR)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(config.P1_PANEL_PATH, index=False)
    metrics, coverage = evaluate_baselines(panel)
    selected = select_baseline(metrics)
    metrics.to_csv(config.P1_METRICS_PATH, index=False)
    predictions = panel.loc[
        panel.evaluation_eligible & (panel.timestamp >= config.VAL_START),
        ["station_id", "timestamp", "forecast_origin", "departures", *config.P1_MODEL_NAMES],
    ]
    predictions.to_parquet(config.P1_PREDICTIONS_PATH, index=False)
    test = predictions.loc[split_mask(predictions, config.TEST_START, config.TEST_END)]
    station_metrics(test).to_csv(config.P1_STATION_METRICS_PATH, index=False)
    quality = {
        "data_version": manifest["data_version"],
        "raw_rows": sum(e["raw_rows"] for e in manifest["trips"]),
        "clean_rows": sum(e["clean_rows"] for e in manifest["trips"]),
        "months": [e["month"] for e in manifest["trips"]],
        "observed_origin_stations": int(panel.station_id.nunique()),
        "panel_rows": len(panel),
        "coverage": coverage,
        "selected_baseline": selected,
        "selection_metric": "val_wape",
        "source_granularity": "station_hour",
        "timezone": config.P1_TIMEZONE,
        "station_eligibility": "first_departure_observed_before_forecast_midnight",
        "evaluation_cohort": "four_previous_weekly_observations_available",
        "limitations": [
            "First observation is an activation proxy; closures are not observed.",
            "Trips measure served departures, not unsatisfied demand.",
            "Stations with less than four weeks of history are excluded and counted.",
            "Monthly archives permit retrospective replay, not a real-time operational feed.",
            "Baselines fixed before evaluation; learned candidate models are proposed only.",
        ],
    }
    config.P1_QUALITY_PATH.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    create_figures(metrics, panel)
    runs = log_runs(metrics, manifest, quality, selected, tracking_uri) if with_mlflow else []
    config.P1_RUNS_PATH.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    logger.info("Resultados:\n{}", metrics.to_string(index=False))
    logger.success("P1 listo. Seleccion por validacion: {}. Runs: {}", selected, len(runs))


if __name__ == "__main__":
    app()
