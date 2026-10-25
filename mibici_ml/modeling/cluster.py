"""Exploratory P3 station clustering from the versioned EDA summary."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import time
from typing import Annotated, Any

from loguru import logger
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import mlflow
from mlflow import sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer

from mibici_ml.config import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    P3_N_INIT,
    PROJ_ROOT,
    RANDOM_SEED,
    TABLES_DIR,
)

app = typer.Typer(no_args_is_help=True, help="Baseline exploratorio P3 para estaciones MiBici.")

REQUIRED_COLUMNS = (
    "est",
    "nombre",
    "lat",
    "lon",
    "capacidad",
    "salidas_12m",
    "llegadas_12m",
)
MODEL_FEATURES = (
    "log1p_salidas_12m",
    "log1p_llegadas_12m",
    "capacidad_modelo",
)
ASSIGNMENT_COLUMNS = (
    "est",
    "nombre",
    "lat",
    "lon",
    "capacidad",
    "capacidad_modelo",
    "capacidad_imputada",
    "salidas_12m",
    "llegadas_12m",
    "cluster",
)
ARTIFACT_NAMES = (
    "p3_metricas_por_k.csv",
    "p3_comparacion_k.png",
    "p3_asignaciones_cluster.csv",
    "p3_perfiles_cluster.csv",
    "p3_perfiles_cluster.png",
    "p3_mapa_clusters.png",
    "p3_calidad_datos.csv",
)
N_INIT = P3_N_INIT


def load_station_summary(input_path: Path) -> pd.DataFrame:
    """Load the versioned station-level summary CSV."""
    try:
        stations = pd.read_csv(input_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(f"No se pudo leer el resumen de estaciones: {input_path}: {error}") from error
    logger.info("Resumen de estaciones cargado desde {}", input_path)
    return stations


def _numeric_values(stations: pd.DataFrame, column: str) -> pd.Series:
    """Return a finite numeric column or raise a column-specific error."""
    try:
        values = pd.to_numeric(stations[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"La columna '{column}' debe contener valores numericos.") from error
    numeric = values.to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"La columna '{column}' contiene valores faltantes o no finitos.")
    return values


def validate_station_summary(stations: pd.DataFrame) -> None:
    """Validate identifiers, numeric model inputs, and map coordinates."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in stations.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(missing_columns)}.")
    if stations["est"].isna().any():
        raise ValueError("La columna 'est' contiene identificadores faltantes.")
    if stations["est"].duplicated().any():
        raise ValueError("La columna 'est' debe contener identificadores unicos.")
    if stations.empty:
        raise ValueError("El resumen de estaciones no puede estar vacio.")
    if stations["nombre"].isna().any():
        raise ValueError("La columna 'nombre' contiene valores faltantes.")

    for column in ("lat", "lon", "capacidad", "salidas_12m", "llegadas_12m"):
        _numeric_values(stations, column)

    if (pd.to_numeric(stations["capacidad"]) < 0).any():
        raise ValueError("La columna 'capacidad' no puede contener valores negativos.")
    positive_capacity = pd.to_numeric(stations["capacidad"])
    if not (positive_capacity > 0).any():
        raise ValueError(
            "Se requiere al menos una capacidad positiva para calcular la mediana de imputacion."
        )
    for column in ("salidas_12m", "llegadas_12m"):
        if (pd.to_numeric(stations[column]) < 0).any():
            raise ValueError(f"La columna '{column}' no puede contener valores negativos.")

    zero_capacity_count = int((pd.to_numeric(stations["capacidad"]) == 0).sum())
    logger.info(
        "Validacion correcta: {} estaciones; {} requieren imputacion de capacidad.",
        len(stations),
        zero_capacity_count,
    )


def compute_file_sha256(input_path: Path) -> str:
    """Compute a file SHA-256 digest in bounded-memory blocks."""
    digest = hashlib.sha256()
    try:
        with input_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"No se pudo calcular SHA-256 para {input_path}: {error}") from error
    return digest.hexdigest()


def get_git_commit() -> str:
    """Return the current repository commit without modifying Git state."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJ_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"No se pudo obtener el commit Git desde {PROJ_ROOT}.") from error
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError(f"Git no devolvio un commit desde {PROJ_ROOT}.")
    return commit


def prepare_station_data(
    stations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], float]:
    """Copy station data, impute zero capacity, and prepare the three features."""
    validate_station_summary(stations)
    enriched = stations.copy(deep=True)
    capacity = pd.to_numeric(enriched["capacidad"]).astype(float)
    positive_capacity_median = float(capacity.loc[capacity > 0].median())
    if not np.isfinite(positive_capacity_median) or positive_capacity_median <= 0:
        raise ValueError("La mediana de capacidades positivas debe ser finita y mayor que cero.")

    zero_capacity = capacity.eq(0)
    enriched["capacidad_imputada"] = zero_capacity.astype(bool)
    enriched["capacidad_modelo"] = capacity.mask(zero_capacity, positive_capacity_median)
    features = pd.DataFrame(
        {
            "log1p_salidas_12m": np.log1p(
                pd.to_numeric(enriched["salidas_12m"]).astype(float)
            ),
            "log1p_llegadas_12m": np.log1p(
                pd.to_numeric(enriched["llegadas_12m"]).astype(float)
            ),
            "capacidad_modelo": enriched["capacidad_modelo"].astype(float),
        },
        index=enriched.index,
    )
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("La matriz de features contiene valores faltantes o no finitos.")
    if (enriched["capacidad_modelo"] <= 0).any():
        raise ValueError("'capacidad_modelo' debe ser estrictamente positiva.")
    return enriched, features, list(MODEL_FEATURES), positive_capacity_median


def prepare_cluster_features(stations: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return the exact allowed model features and their stable names."""
    _, features, feature_names, _ = prepare_station_data(stations)
    return features, feature_names


def build_pipeline(final_k: int, random_seed: int, n_init: int = N_INIT) -> Pipeline:
    """Build a reproducible StandardScaler and KMeans pipeline."""
    if final_k < 2:
        raise ValueError("final_k debe ser al menos 2.")
    if n_init < 1:
        raise ValueError("n_init debe ser positivo.")
    return Pipeline(
        steps=(
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=final_k, random_state=random_seed, n_init=n_init),
            ),
        )
    )


def _validate_k_range(k_min: int, k_max: int, n_samples: int) -> None:
    if k_min < 2:
        raise ValueError("k_min debe ser al menos 2.")
    if k_max < k_min:
        raise ValueError("k_max debe ser mayor o igual que k_min.")
    if k_max >= n_samples:
        raise ValueError("Cada k debe ser menor que el numero de estaciones.")


def evaluate_kmeans_candidates(
    features: pd.DataFrame,
    k_min: int = 3,
    k_max: int = 8,
    random_seed: int = RANDOM_SEED,
    n_init: int = N_INIT,
) -> pd.DataFrame:
    """Evaluate KMeans candidates without selecting a final k or logging MLflow."""
    _validate_k_range(k_min, k_max, len(features))
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("La matriz de features debe ser finita y no contener faltantes.")

    rows = [
        evaluate_kmeans_candidate(features, k, random_seed, n_init)[0]
        for k in range(k_min, k_max + 1)
    ]
    return pd.DataFrame(rows).sort_values("k", ignore_index=True)


def evaluate_kmeans_candidate(
    features: pd.DataFrame,
    k: int,
    random_seed: int = RANDOM_SEED,
    n_init: int = N_INIT,
) -> tuple[dict[str, int | float], Pipeline, np.ndarray]:
    """Fit and score one KMeans configuration with the shared P3 protocol."""
    _validate_k_range(k, k, len(features))
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("La matriz de features debe ser finita y no contener faltantes.")

    pipeline = build_pipeline(k, random_seed, n_init)
    fit_start = time.perf_counter()
    labels = pipeline.fit_predict(features)
    fit_seconds = time.perf_counter() - fit_start
    cluster_sizes = pd.Series(labels).value_counts()
    if len(cluster_sizes) != k:
        raise ValueError(f"KMeans produjo {len(cluster_sizes)} clusters para k={k}.")

    predict_start = time.perf_counter()
    predicted_labels = pipeline.predict(features)
    predict_seconds = time.perf_counter() - predict_start
    if not np.array_equal(labels, predicted_labels):
        raise ValueError("KMeans no reprodujo las etiquetas al predecir los datos de ajuste.")

    scaled_features = pipeline.named_steps["scaler"].transform(features)
    metrics: dict[str, int | float] = {
        "k": k,
        "inertia": float(pipeline.named_steps["kmeans"].inertia_),
        "silhouette_score": float(silhouette_score(scaled_features, labels)),
        "calinski_harabasz_score": float(
            calinski_harabasz_score(scaled_features, labels)
        ),
        "davies_bouldin_score": float(davies_bouldin_score(scaled_features, labels)),
        "min_cluster_size": int(cluster_sizes.min()),
        "max_cluster_size": int(cluster_sizes.max()),
        "n_stations": len(features),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError(f"KMeans produjo metricas no finitas para k={k}.")
    return metrics, pipeline, np.asarray(labels, dtype=int)


def fit_final_pipeline(
    features: pd.DataFrame,
    final_k: int,
    random_seed: int = RANDOM_SEED,
    n_init: int = N_INIT,
) -> tuple[Pipeline, np.ndarray]:
    """Fit a reproducible final pipeline for the explicitly chosen k."""
    _validate_k_range(final_k, final_k, len(features))
    pipeline = build_pipeline(final_k, random_seed, n_init)
    labels = pipeline.fit_predict(features)
    if len(np.unique(labels)) != final_k:
        raise ValueError(f"KMeans produjo menos de {final_k} clusters.")
    return pipeline, labels


def build_station_assignments(stations: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Build one auditable cluster assignment row per input station."""
    if len(stations) != len(labels):
        raise ValueError("Debe existir exactamente una etiqueta por estacion.")
    if "capacidad_modelo" not in stations or "capacidad_imputada" not in stations:
        raise ValueError("Prepare las capacidades antes de construir asignaciones.")
    assignments = stations.loc[:, list(ASSIGNMENT_COLUMNS[:-1])].copy()
    assignments["cluster"] = np.asarray(labels, dtype=int)
    if assignments["est"].duplicated().any():
        raise ValueError("Las asignaciones contienen identificadores de estacion duplicados.")
    return assignments.loc[:, list(ASSIGNMENT_COLUMNS)]


def build_cluster_profiles(assignments: pd.DataFrame) -> pd.DataFrame:
    """Summarize station counts, capacities, and original usage by cluster."""
    profiles = (
        assignments.groupby("cluster", as_index=False)
        .agg(
            n_stations=("est", "size"),
            capacidad_modelo_media=("capacidad_modelo", "mean"),
            capacidad_modelo_mediana=("capacidad_modelo", "median"),
            salidas_12m_media=("salidas_12m", "mean"),
            salidas_12m_mediana=("salidas_12m", "median"),
            llegadas_12m_media=("llegadas_12m", "mean"),
            llegadas_12m_mediana=("llegadas_12m", "median"),
            salidas_12m_suma=("salidas_12m", "sum"),
            llegadas_12m_suma=("llegadas_12m", "sum"),
        )
        .sort_values("cluster", ignore_index=True)
    )
    return profiles


def build_data_quality_summary(
    enriched: pd.DataFrame, features: pd.DataFrame, positive_capacity_median: float
) -> pd.DataFrame:
    """Summarize validated row counts and the controlled capacity imputation."""
    return pd.DataFrame(
        [
            {
                "n_stations": len(enriched),
                "n_zero_capacity": int((enriched["capacidad"] == 0).sum()),
                "n_imputed_capacity": int(enriched["capacidad_imputada"].sum()),
                "positive_capacity_median": float(positive_capacity_median),
                "n_missing_model_features": int(features.isna().sum().sum()),
                "n_duplicate_station_ids": int(enriched["est"].duplicated().sum()),
            }
        ]
    )


def _save_figure(figure: Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_k_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    """Plot each candidate metric on its own axis to preserve readable scales."""
    metric_columns = (
        ("inertia", "Inercia"),
        ("silhouette_score", "Silueta"),
        ("calinski_harabasz_score", "Calinski-Harabasz"),
        ("davies_bouldin_score", "Davies-Bouldin"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (column, title) in zip(axes.flat, metric_columns, strict=True):
        axis.plot(metrics["k"], metrics[column], marker="o")
        axis.set_title(title)
        axis.set_xlabel("Numero de clusters (k)")
        axis.set_ylabel(title)
        axis.set_xticks(metrics["k"])
        axis.grid(alpha=0.25)
    figure.suptitle("Comparacion de configuraciones KMeans")
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_cluster_profiles(profiles: pd.DataFrame, output_path: Path) -> None:
    """Plot original-scale means for station usage and modeled capacity."""
    profile_columns = (
        ("salidas_12m_media", "Salidas por estacion (media, 12 meses)"),
        ("llegadas_12m_media", "Llegadas por estacion (media, 12 meses)"),
        ("capacidad_modelo_media", "Capacidad modelada (media)"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 5))
    for axis, (column, title) in zip(axes, profile_columns, strict=True):
        axis.bar(profiles["cluster"].astype(str), profiles[column])
        axis.set_title(title)
        axis.set_xlabel("Cluster")
        axis.set_ylabel("Valor original medio")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Perfiles agregados por cluster")
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_station_clusters(assignments: pd.DataFrame, output_path: Path) -> None:
    """Plot station coordinates colored by cluster without external map tiles."""
    figure, axis = plt.subplots(figsize=(9, 8))
    clusters = sorted(assignments["cluster"].unique())
    scatter = axis.scatter(
        assignments["lon"],
        assignments["lat"],
        c=assignments["cluster"],
        cmap="tab10",
        vmin=min(clusters) - 0.5,
        vmax=max(clusters) + 0.5,
        s=24,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.25,
    )
    colorbar = figure.colorbar(scatter, ax=axis, ticks=clusters)
    colorbar.set_label("Cluster")
    axis.set_xlabel("Longitud")
    axis.set_ylabel("Latitud")
    axis.set_title("Ubicacion de estaciones por cluster")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, output_path)


def save_artifacts(
    output_dir: Path,
    metrics: pd.DataFrame,
    assignments: pd.DataFrame,
    profiles: pd.DataFrame,
    quality: pd.DataFrame,
) -> dict[str, Path]:
    """Write the seven requested artifacts only under the selected directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / name for name in ARTIFACT_NAMES}
    metrics.to_csv(paths["p3_metricas_por_k.csv"], index=False)
    assignments.to_csv(paths["p3_asignaciones_cluster.csv"], index=False)
    profiles.to_csv(paths["p3_perfiles_cluster.csv"], index=False)
    quality.to_csv(paths["p3_calidad_datos.csv"], index=False)
    plot_k_comparison(metrics, paths["p3_comparacion_k.png"])
    plot_cluster_profiles(profiles, paths["p3_perfiles_cluster.png"])
    plot_station_clusters(assignments, paths["p3_mapa_clusters.png"])
    return paths


def log_p3_run(
    pipeline: Pipeline,
    final_k: int,
    k_min: int,
    k_max: int,
    random_seed: int,
    feature_names: list[str],
    input_path: Path,
    input_sha256: str,
    positive_capacity_median: float,
    final_metrics: pd.Series,
    quality: pd.DataFrame,
    artifacts: dict[str, Path],
    experiment_name: str = EXPERIMENT_NAME,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> str:
    """Log a completed P3 exploratory baseline and return its MLflow run ID."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    commit = get_git_commit()
    quality_row = quality.iloc[0]
    params: dict[str, Any] = {
        "algorithm": "KMeans",
        "final_k": final_k,
        "k_min": k_min,
        "k_max": k_max,
        "random_seed": random_seed,
        "n_init": N_INIT,
        "feature_names": ",".join(feature_names),
        "transform": "log1p_counts",
        "scaler": "StandardScaler",
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "period": "2025-10_to_2026-09",
        "capacity_zero_policy": "median_positive_imputation",
        "capacity_feature": "capacidad_modelo",
        "positive_capacity_median": float(positive_capacity_median),
    }
    metric_names = (
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
        "inertia",
        "min_cluster_size",
        "max_cluster_size",
    )
    run_metrics = {name: float(final_metrics[name]) for name in metric_names}
    run_metrics.update(
        {
            "n_stations": float(quality_row["n_stations"]),
            "n_zero_capacity": float(quality_row["n_zero_capacity"]),
            "n_imputed_capacity": float(quality_row["n_imputed_capacity"]),
        }
    )
    tags = {
        "team": "equipo49",
        "question": "P3",
        "run_type": "exploratory_baseline",
        "data_scope": "eda_aggregate_full_12m",
        "temporal_holdout": "false",
        "limitation": "no_station_hour_profiles",
        "git_commit": commit,
    }
    with mlflow.start_run(run_name="P3-kmeans-exploratory-baseline") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(run_metrics)
        mlflow.set_tags(tags)
        for artifact_path in artifacts.values():
            mlflow.log_artifact(str(artifact_path))
        mlflow_sklearn.log_model(pipeline, name="p3_pipeline")
        return run.info.run_id


def _evaluate_and_save(
    input_path: Path,
    output_dir: Path,
    k_min: int,
    k_max: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], float]:
    """Load station data, prepare features, and write candidate-only outputs."""
    stations = load_station_summary(input_path)
    _, features, feature_names, positive_capacity_median = prepare_station_data(stations)
    metrics = evaluate_kmeans_candidates(features, k_min, k_max, random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "p3_metricas_por_k.csv"
    plot_path = output_dir / "p3_comparacion_k.png"
    metrics.to_csv(metrics_path, index=False)
    plot_k_comparison(metrics, plot_path)
    logger.info("Evaluacion guardada en {}", output_dir)
    return metrics, features, feature_names, positive_capacity_median


@app.command("evaluate")
def evaluate_command(
    *,
    input_path: Annotated[
        Path, typer.Option(help="CSV agregado versionado de estaciones.")
    ] = TABLES_DIR / "estaciones_resumen.csv",
    output_dir: Annotated[
        Path, typer.Option(help="Directorio para metricas y figura de comparacion.")
    ],
    k_min: Annotated[int, typer.Option(min=2, help="Primer k que se evaluara, inclusive.")] = 3,
    k_max: Annotated[int, typer.Option(min=2, help="Ultimo k que se evaluara, inclusive.")] = 8,
    random_seed: Annotated[int, typer.Option(help="Semilla reproducible de KMeans.")] = RANDOM_SEED,
) -> None:
    """Evaluate candidate k values without selecting a final k or logging MLflow."""
    _evaluate_and_save(input_path, output_dir, k_min, k_max, random_seed)


@app.command("run")
def run_command(
    *,
    input_path: Annotated[
        Path, typer.Option(help="CSV agregado versionado de estaciones.")
    ] = TABLES_DIR / "estaciones_resumen.csv",
    output_dir: Annotated[Path, typer.Option(help="Directorio exclusivo para los siete artefactos.")],
    final_k: Annotated[
        int,
        typer.Option(
            min=2,
            help="k elegido tras revisar evaluate; no se selecciona automaticamente.",
        ),
    ],
    k_min: Annotated[int, typer.Option(min=2, help="Primer k que se evaluara, inclusive.")] = 3,
    k_max: Annotated[int, typer.Option(min=2, help="Ultimo k que se evaluara, inclusive.")] = 8,
    random_seed: Annotated[int, typer.Option(help="Semilla reproducible de KMeans.")] = RANDOM_SEED,
    with_mlflow: Annotated[bool, typer.Option(help="Registrar el baseline en MLflow local.")] = False,
) -> None:
    """Fit the explicitly selected baseline and optionally log an MLflow run."""
    if not k_min <= final_k <= k_max:
        raise typer.BadParameter("final_k debe estar dentro del rango k_min..k_max.")
    stations = load_station_summary(input_path)
    enriched, features, feature_names, positive_capacity_median = prepare_station_data(stations)
    metrics = evaluate_kmeans_candidates(features, k_min, k_max, random_seed)
    pipeline, labels = fit_final_pipeline(features, final_k, random_seed)
    assignments = build_station_assignments(enriched, labels)
    profiles = build_cluster_profiles(assignments)
    quality = build_data_quality_summary(enriched, features, positive_capacity_median)
    artifacts = save_artifacts(output_dir, metrics, assignments, profiles, quality)
    if with_mlflow:
        final_metrics = metrics.loc[metrics["k"] == final_k].iloc[0]
        run_id = log_p3_run(
            pipeline=pipeline,
            final_k=final_k,
            k_min=k_min,
            k_max=k_max,
            random_seed=random_seed,
            feature_names=feature_names,
            input_path=input_path,
            input_sha256=compute_file_sha256(input_path),
            positive_capacity_median=positive_capacity_median,
            final_metrics=final_metrics,
            quality=quality,
            artifacts=artifacts,
        )
        logger.success("Run MLflow P3 registrado: {}", run_id)
    logger.success("Baseline P3 guardado en {}", output_dir)


if __name__ == "__main__":
    app()
