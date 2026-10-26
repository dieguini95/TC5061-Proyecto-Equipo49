"""P1 Etapa 2: candidatos anidados, seleccion temporal y Registry local.

Se entrenan modelos fijos con octubre-junio. La seleccion usa solo julio-agosto;
se escribe la decision antes de abrir septiembre. El test compara configuraciones
ya congeladas y nunca se utiliza para reajustar ni elegir el champion.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from typing import Annotated
from uuid import uuid4
import zipfile

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
import typer

from mibici_ml import config
from mibici_ml.features import build_station_hour_panel, p1_model_features
from mibici_ml.modeling import demand
from mibici_ml.modeling.p1_model import build_p1_model, get_effective_params
from mibici_ml.modeling.registry_p1 import register_champion

app = typer.Typer()


def parse_tuned_params(value: str) -> dict:
    """Lee clave=valor;clave=valor sin ejecutar texto recibido de la terminal."""
    if not value.strip():
        return {}
    result = {}
    for item in value.split(";"):
        key, separator, raw = item.partition("=")
        if not separator or not key.strip() or key.strip() in result:
            raise ValueError("Use hiperparametros unicos como clave=valor;clave=valor.")
        try:
            result[key.strip()] = json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Valor no numerico/JSON para {key}: {raw}") from exc
    return result


def configurations(model: str = "", tuned_json: str = "") -> list[dict]:
    """Dos referencias historicas y una lista explicita de candidatos prefijados."""
    overrides = parse_tuned_params(tuned_json)
    if model and model not in config.P1_CANDIDATE_GRID:
        raise ValueError(f"Modelo P1 desconocido: {model}. Use {list(config.P1_CANDIDATE_GRID)}.")
    if overrides and not model:
        raise ValueError("tuned_json requiere un modelo P1 explicito.")
    specs = [{"model_id": name, "model": name, "params": {}} for name in config.P1_MODEL_NAMES]
    grid = {model: [overrides]} if model else config.P1_CANDIDATE_GRID
    for family, candidates in grid.items():
        for number, params in enumerate(candidates, 1):
            get_effective_params(family, params)  # fail before expensive data preparation
            specs.append({"model_id": f"{family}_{number:02d}", "model": family, "params": params})
    return specs


def choose_on_validation(rows: list[dict]) -> str:
    """WAPE de validacion; MAE y nombre desempatan. No acepta metrica de prueba."""
    eligible = [row for row in rows if row["split"] == "val"]
    if not eligible or any(not np.isfinite(row["wape"]) for row in eligible):
        raise ValueError("Se necesita WAPE finito en validacion para seleccionar.")
    return min(eligible, key=lambda row: (row["wape"], row["mae"], row["model_id"]))["model_id"]


def _write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )


def _provenance(output: Path) -> dict:
    """Snapshot acotado de codigo y entorno; no copia datos, secretos ni modelos."""
    provenance, files = demand.code_provenance(config.PROJ_ROOT)
    for name in ("MLproject", "python_env.yaml", "requirements.txt"):
        path = config.PROJ_ROOT / name
        if path.is_file():
            files.append(path)
            provenance["source_hashes"][name] = demand.file_sha256(path)
    provenance["source_version"] = hashlib.sha256(
        json.dumps(provenance["source_hashes"], sort_keys=True).encode()
    ).hexdigest()
    provenance["python"] = platform.python_version()
    provenance["packages"] = {
        package: importlib.metadata.version(package)
        for package in (
            "mlflow",
            "scikit-learn",
            "numpy",
            "pandas",
            "pyarrow",
            "scipy",
            "skops",
            "joblib",
            "threadpoolctl",
            "python-dotenv",
            "loguru",
            "matplotlib",
            "typer",
            "tqdm",
        )
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "code_provenance.json", provenance)
    with zipfile.ZipFile(output / "code_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(config.PROJ_ROOT).as_posix())
    return provenance


def _tags(provenance: dict, manifest: dict, family: str, stage: str) -> dict:
    return {
        "team": "equipo49",
        "equipo": "49",
        "question": "P1",
        "pregunta": "P1",
        "model_family": family,
        "stage_of_work": stage,
        "git_commit": provenance["git_commit"],
        "mlflow.source.git.commit": provenance["git_commit"],
        "git_dirty": str(provenance["git_dirty"]).lower(),
        "source_version": provenance["source_version"],
        "data_version": manifest["data_version"],
        "evaluation": "daily_origin_00h_frozen_model_common_cohort",
    }


def _parameters(provenance: dict, manifest: dict) -> dict:
    return {
        "random_seed": config.RANDOM_SEED,
        "threads": config.P1_MODEL_THREADS,
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
        "feature_columns": ",".join(config.P1_FEATURE_COLUMNS),
        "selection_metric": "val_wape",
        "wape_units": "fraction",
        "retraining": "none_after_train_end",
        "early_stopping": False,
        "cohort": "four_previous_weeks_same_as_etapa1",
        "feature_version": config.P1_STAGE2_SOURCE_VERSION,
    }


def _metric_row(frame: pd.DataFrame, predicted: np.ndarray, spec: dict, split: str) -> dict:
    return {
        "model_id": spec["model_id"],
        "model": spec["model"],
        "split": split,
        **demand.regression_metrics(frame.departures.to_numpy(float), predicted),
        "n_rows": len(frame),
        "n_stations": int(frame.station_id.nunique()),
    }


def _log_metrics(row: dict, run_id: str) -> None:
    client = mlflow.MlflowClient()
    for key in ("mae", "rmse", "wape", "bias", "actual_total", "n_rows", "n_stations"):
        if np.isfinite(row[key]):
            client.log_metric(run_id, f"{row['split']}_{key}", float(row[key]))


def _diagnostics(frame: pd.DataFrame, predicted: np.ndarray, directory: Path, split: str) -> None:
    """Artefactos agregados de cada modelo; no incluye usuarios ni viajes individuales."""
    directory.mkdir(parents=True, exist_ok=True)
    actual = frame.departures.to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].hexbin(actual, predicted, gridsize=35, bins="log", mincnt=1, cmap="viridis")
    limit = max(actual.max(), predicted.max(), 1)
    axes[0].plot([0, limit], [0, limit], "--", color="tomato")
    axes[0].set(xlabel="Salidas observadas", ylabel="Salidas predichas", title=f"P1 | {split}")
    residual = predicted - actual
    axes[1].hist(residual, bins=65, color="#315f9b")
    axes[1].axvline(0, color="tomato", linestyle="--")
    axes[1].set(
        xlabel="Predicho - observado (viajes/h)", ylabel="Estaciones-hora", title="Residuos"
    )
    fig.savefig(directory / f"{split}_prediction_residuals.png", dpi=150)
    plt.close(fig)
    daily = pd.DataFrame(
        {
            "date": frame.timestamp.dt.normalize().to_numpy(),
            "actual": actual,
            "prediction": predicted,
        }
    )
    daily = daily.groupby("date")[["actual", "prediction"]].sum()
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.plot(daily.index, daily.actual, label="Observado", color="#26323f")
    ax.plot(daily.index, daily.prediction, label="Predicho", color="#da8b35")
    ax.set(title=f"P1 | {split}: volumen diario de la misma cohorte", ylabel="Viajes")
    ax.legend()
    fig.savefig(directory / f"{split}_daily_totals.png", dpi=150)
    plt.close(fig)


def _stability_rows(
    frame: pd.DataFrame, prediction: np.ndarray, spec: dict, split: str
) -> list[dict]:
    records = []
    for period in ("M", "D"):
        groups = frame.timestamp.dt.to_period(period)
        for value in groups.unique():
            mask = groups.eq(value).to_numpy()
            metrics = demand.regression_metrics(
                frame.departures.to_numpy(float)[mask], prediction[mask]
            )
            records.append(
                {
                    "model_id": spec["model_id"],
                    "split": split,
                    "period": str(value),
                    "frequency": period,
                    **metrics,
                }
            )
    return records


def _comparison_plot(metrics: pd.DataFrame, path: Path) -> None:
    pivot = metrics.pivot(index="model_id", columns="split", values="wape")
    ax = (pivot * 100).plot.barh(figsize=(10, 5), color=["#da8b35", "#315f9b"])
    ax.set(
        xlabel="WAPE (%) — menor es mejor",
        ylabel="",
        title="P1 | misma cohorte; seleccion solo en validacion",
    )
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def run_search(
    output_dir: Path | None = None,
    model: str = "",
    tuned_json: str = "",
    register_best: bool = False,
    with_mlflow: bool = True,
) -> dict:
    """Ejecuta un grid pequeno en todos los datos, sin muestreo ni ajuste con prueba.

    Los hijos cierran tras validacion; sus metricas de prueba se agregan despues
    de congelar la seleccion. Registro y alias se validan al terminar la busqueda.
    """
    specs = configurations(model, tuned_json)
    if register_best and not with_mlflow:
        raise ValueError("Registrar un modelo requiere with_mlflow=true.")
    demand.validate_p1_forecast_contract()
    manifest = demand.verify_inputs(config.DATA_MANIFEST_PATH, config.PROCESSED_DATA_DIR)
    output = (output_dir or config.P1_STAGE2_OUTPUT_DIR).resolve() / uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    provenance = _provenance(output / "config")
    panel = build_station_hour_panel(config.TRIPS_PROCESSED_DIR)
    frames, coverage = {}, {}
    for split, start, end in (
        ("train", config.TRAIN_START, config.TRAIN_END),
        ("val", config.VAL_START, config.VAL_END),
        ("test", config.TEST_START, config.TEST_END),
    ):
        window = demand.split_mask(panel, start, end)
        mask = window & panel.evaluation_eligible
        frames[split] = panel.loc[
            mask, ["station_id", "timestamp", "departures", *config.P1_FEATURE_COLUMNS]
        ].copy()
        if frames[split].empty:
            raise ValueError(f"No hay observaciones para {split}.")
        coverage[split] = {
            "n_rows": int(mask.sum()),
            "n_stations": int(frames[split].station_id.nunique()),
            "excluded_warmup_hours": int((window & panel.eligible & ~mask).sum()),
        }
    del panel
    x_train, x_val = p1_model_features(frames["train"]), p1_model_features(frames["val"])
    y_train = frames["train"].departures.to_numpy(float)
    example = x_val.head(5).copy()
    quality = {
        "data_version": manifest["data_version"],
        "coverage": coverage,
        "raw_trips": sum(row["raw_rows"] for row in manifest["trips"]),
        "clean_trips": sum(row["clean_rows"] for row in manifest["trips"]),
    }
    _write_json(output / "data" / "data_manifest.json", manifest)
    _write_json(output / "data" / "quality.json", quality)
    parameters = _parameters(provenance, manifest)
    _write_json(output / "config" / "experiment.json", {**parameters, "configurations": specs})
    datasets = {}
    if with_mlflow:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        experiment = mlflow.set_experiment(config.EXPERIMENT_NAME)
        # Dataset metadata/digest/schema, not the underlying rows, are stored by log_input.
        datasets["train"] = mlflow.data.from_pandas(
            x_train.assign(departures=y_train),
            source=config.DATA_MANIFEST_PATH.as_uri(),
            name="p1_station_hour_train",
            targets="departures",
        )
        datasets["val"] = mlflow.data.from_pandas(
            x_val.assign(departures=frames["val"].departures.to_numpy(float)),
            source=config.DATA_MANIFEST_PATH.as_uri(),
            name="p1_station_hour_val",
            targets="departures",
        )
        wrapper_id = os.environ.pop("MLFLOW_RUN_ID", None)
        if wrapper_id:
            wrapper = mlflow.MlflowClient().get_run(wrapper_id)
            if wrapper.info.experiment_id != experiment.experiment_id:
                raise ValueError("Use mlflow run --experiment-name mibici-equipo49 para P1.")
        parent_context = mlflow.start_run(
            run_id=wrapper_id, run_name="P1-etapa2-busqueda-temporal"
        )
    else:
        parent_context = nullcontext(None)
    rows, stability, fitted, run_ids, timings = [], [], {}, {}, {}
    with parent_context as parent:
        parent_id = parent.info.run_id if parent else None
        if with_mlflow:
            mlflow.set_tags(_tags(provenance, manifest, "grid", "search"))
            mlflow.set_tag("mlflow.runName", "P1-etapa2-busqueda-temporal")
            mlflow.log_params(parameters)
        for spec in specs:
            name = spec["model_id"]
            stage = "baseline" if spec["model"] in config.P1_MODEL_NAMES else "candidate"
            logger.info("P1: ajustando {} con {} filas", name, len(x_train))
            child_context = (
                mlflow.start_run(run_name=f"P1-{name}", nested=True)
                if with_mlflow
                else nullcontext(None)
            )
            with child_context as child:
                estimator = build_p1_model(spec["model"], spec["params"])
                started = time.perf_counter()
                estimator.fit(x_train, y_train)
                fit_seconds = time.perf_counter() - started
                started = time.perf_counter()
                prediction = estimator.predict(x_val)
                prediction_seconds = time.perf_counter() - started
                row = _metric_row(frames["val"], prediction, spec, "val")
                rows.append(row)
                stability.extend(_stability_rows(frames["val"], prediction, spec, "val"))
                timings[name] = {
                    "fit_seconds": fit_seconds,
                    "val_predict_seconds": prediction_seconds,
                    "val_predict_ms_per_1000": prediction_seconds / len(x_val) * 1_000_000,
                }
                fitted[name] = estimator
                child_dir = output / name
                _write_json(
                    child_dir / "config" / "model.json",
                    {
                        "model": spec["model"],
                        "hyperparameters": get_effective_params(spec["model"], spec["params"]),
                        **parameters,
                    },
                )
                _diagnostics(frames["val"], prediction, child_dir / "plots", "val")
                if child:
                    run_ids[name] = child.info.run_id
                    mlflow.set_tags(_tags(provenance, manifest, spec["model"], stage))
                    mlflow.log_params(
                        {
                            **parameters,
                            "model_id": name,
                            "model": spec["model"],
                            "hyperparameters": json.dumps(
                                get_effective_params(spec["model"], spec["params"]),
                                sort_keys=True,
                                default=str,
                            ),
                        }
                    )
                    mlflow.log_params(
                        {
                            f"hp.{key}": value
                            for key, value in get_effective_params(
                                spec["model"], spec["params"]
                            ).items()
                        }
                    )
                    _log_metrics(row, child.info.run_id)
                    mlflow.log_metrics(timings[name])
                    for context, dataset in datasets.items():
                        mlflow.log_input(
                            dataset, context="training" if context == "train" else "validation"
                        )
                    mlflow.sklearn.log_model(
                        estimator,
                        name="model",
                        signature=infer_signature(example, estimator.predict(example)),
                        input_example=example,
                        code_paths=[str(config.PROJ_ROOT / "mibici_ml")],
                        skops_trusted_types=[
                            "mibici_ml.modeling.p1_model.P1Regressor",
                            "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                        ],
                        pip_requirements=[
                            f"{key}=={value}" for key, value in provenance["packages"].items()
                        ],
                    )
                    mlflow.log_artifacts(str(output / "data"), "data")
                    mlflow.log_artifacts(str(output / "config"), "config")
                    mlflow.log_artifacts(str(child_dir / "config"), "config")
                    mlflow.log_artifacts(str(child_dir / "plots"), "plots")
                logger.info(
                    "P1 {}: val WAPE={:.6f}; entrenamiento {:.1f} s",
                    name,
                    row["wape"],
                    fit_seconds,
                )
        selected = choose_on_validation(rows)
        selection = {
            "selected_model_id": selected,
            "selection_metric": "val_wape",
            "rule": "minimum val_wape, then val_mae, then model_id; includes baselines",
            "test_used_for_selection": False,
            "refit_on_validation": False,
            "validation_results": rows.copy(),
        }
        # Persist this BEFORE computing any test predictions/metrics.
        _write_json(output / "config" / "selection.json", selection)
        if with_mlflow:
            mlflow.log_dict(selection, "config/selection.json")
            mlflow.set_tag("selected_model_id", selected)
        logger.info("P1 seleccion congelada: {}. Ahora se evalua prueba sin reajuste.", selected)
        x_test = p1_model_features(frames["test"])
        for spec in specs:
            name = spec["model_id"]
            prediction = fitted[name].predict(x_test)
            row = _metric_row(frames["test"], prediction, spec, "test")
            rows.append(row)
            stability.extend(_stability_rows(frames["test"], prediction, spec, "test"))
            _diagnostics(frames["test"], prediction, output / name / "plots", "test")
            if with_mlflow:
                _log_metrics(row, run_ids[name])
                client = mlflow.MlflowClient()
                client.set_tag(
                    run_ids[name], "selected_on_validation", str(name == selected).lower()
                )
                client.log_artifact(
                    run_ids[name], str(output / "config" / "selection.json"), "config"
                )
                client.log_artifact(
                    run_ids[name],
                    str(output / name / "plots" / "test_prediction_residuals.png"),
                    "plots",
                )
                client.log_artifact(
                    run_ids[name], str(output / name / "plots" / "test_daily_totals.png"), "plots"
                )
        metrics = pd.DataFrame(rows)
        baseline = metrics.loc[metrics.model_id.eq("mean_4weeks")].set_index("split")["wape"]
        metrics["wape_improvement_vs_mean4_pct"] = [
            100 * (1 - row.wape / baseline[row.split]) for row in metrics.itertuples()
        ]
        tables = output / "tables"
        tables.mkdir()
        metrics.to_csv(tables / "p1_etapa2_metrics.csv", index=False)
        pd.DataFrame(stability).to_csv(tables / "p1_etapa2_temporal_metrics.csv", index=False)
        pd.DataFrame.from_dict(timings, orient="index").rename_axis("model_id").to_csv(
            tables / "p1_etapa2_timings.csv"
        )
        _comparison_plot(metrics, output / "P1_etapa2_comparison.png")
        if with_mlflow:
            mlflow.log_artifacts(str(tables), "tables")
            mlflow.log_artifact(str(output / "P1_etapa2_comparison.png"), "plots")
            mlflow.log_artifacts(str(output / "data"), "data")
            mlflow.log_artifacts(str(output / "config"), "config")
    registry = None
    if register_best:
        registry = register_champion(
            run_ids[selected],
            example,
            fitted[selected].predict(example),
            tracking_uri=config.MLFLOW_TRACKING_URI,
        )
        mlflow.MlflowClient().log_dict(parent_id, registry, "registry/champion.json")
    result = {
        "parent_run_id": parent_id,
        "run_ids": run_ids,
        "selected_model_id": selected,
        "selected_config": next(spec for spec in specs if spec["model_id"] == selected),
        "registry": registry,
        "data_version": manifest["data_version"],
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "source_version": provenance["source_version"],
        "output_dir": str(output),
    }
    _write_json(output / "p1_etapa2_runs.json", result)
    if with_mlflow:
        mlflow.MlflowClient().log_dict(parent_id, result, "p1_etapa2_runs.json")
    logger.success("P1 Etapa 2 completo: {}", output)
    return result


@app.command()
def main(
    output_dir: Annotated[str, typer.Option(help="Directorio base; vacio usa config.py.")] = "",
    register_best: Annotated[
        str, typer.Option(help="true: validar y asignar champion local.")
    ] = "false",
) -> None:
    if register_best.lower() not in {"true", "false"}:
        raise typer.BadParameter("register_best debe ser true o false.")
    run_search(
        # MLflow Projects on Windows can pass a literal pair of quotes for "".
        output_dir=Path(output_dir) if output_dir not in ("", '""', "''") else None,
        register_best=register_best.lower() == "true",
    )


if __name__ == "__main__":
    app()
