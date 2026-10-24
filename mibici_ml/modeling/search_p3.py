"""Comparable nested KMeans search, artifacts, and optional P3 registration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Annotated, Any

import mlflow
import mlflow.data
import numpy as np
import pandas as pd
import typer

from mibici_ml.config import (
    MLFLOW_TRACKING_URI,
    P3_BASELINE_K,
    P3_CANDIDATE_K,
    P3_CHAMPION_ALIAS,
    P3_INPUT_PATH,
    P3_MODEL_FEATURES,
    P3_N_INIT,
    P3_PERIOD,
    P3_REGISTRY_MODEL_NAME,
    P3_SEARCH_EXPERIMENT_NAME,
    RANDOM_SEED,
)
from mibici_ml.modeling import cluster
from mibici_ml.modeling.p3_model import (
    P3ClusterModel,
    log_p3_pyfunc_model,
    p3_input_example,
)
from mibici_ml.modeling.registry_p3 import (
    assign_champion_alias,
    p3_model_description,
    register_p3_model_from_run,
    set_model_version_tags,
    validate_model_inference,
)

app = typer.Typer(no_args_is_help=True, help="Nested search reproducible para P3.")
SELECTION_METRIC = "silhouette_score"
MIN_CLUSTER_SIZE = 2


@dataclass(frozen=True)
class P3SearchResult:
    parent_run_id: str
    child_run_ids: dict[int, str]
    candidate_run_ids: dict[int, str]
    best_candidate_k: int
    best_candidate_run_id: str
    recommended_k: int
    recommended_run_id: str
    selected_k: int
    selected_run_id: str
    data_version: str
    input_sha256: str
    model_version: str | None


def select_best_candidate(metrics: pd.DataFrame) -> pd.Series:
    """Select a valid run deterministically, with documented tie-breaks."""
    required = {
        "k",
        "silhouette_score",
        "davies_bouldin_score",
        "min_cluster_size",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"Faltan metricas para seleccionar candidato: {', '.join(missing)}.")
    valid = metrics.loc[
        (metrics["min_cluster_size"] >= MIN_CLUSTER_SIZE)
        & np.isfinite(metrics["silhouette_score"])
        & np.isfinite(metrics["davies_bouldin_score"])
    ]
    if valid.empty:
        raise ValueError(
            "No existe candidato elegible: cada cluster debe contener al menos 2 estaciones."
        )
    return valid.sort_values(
        ["silhouette_score", "davies_bouldin_score", "k"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]


def _relative_source_path(input_path: Path) -> str:
    return os.path.relpath(input_path.resolve(), cluster.PROJ_ROOT)


def _build_manifest(
    input_path: Path,
    input_sha256: str,
    stations: pd.DataFrame,
    git_commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "question": "P3",
        "source_path": _relative_source_path(input_path),
        "sha256": input_sha256,
        "row_count": len(stations),
        "column_names": list(stations.columns),
        "period": P3_PERIOD,
        "aggregation": "12_month_station_summary",
        "contains_personal_data": False,
        "limitations": [
            "Agregados de 12 meses completos; no hay holdout temporal.",
            "No representa perfiles horarios por estación.",
            "Coordenadas se usan solo para el mapa, no como features.",
            "Capacidad cero se imputa con la mediana positiva del input; decisión provisional.",
        ],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
    }


def _write_candidate_artifacts(
    output_dir: Path,
    enriched: pd.DataFrame,
    features: pd.DataFrame,
    labels: np.ndarray,
    metrics: dict[str, int | float],
    positive_capacity_median: float,
    run_config: dict[str, Any],
) -> dict[str, Path]:
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"
    config_dir = output_dir / "config"
    for directory in (plots_dir, data_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)

    assignments = cluster.build_station_assignments(enriched, labels)
    profiles = cluster.build_cluster_profiles(assignments)
    quality = cluster.build_data_quality_summary(
        enriched,
        features,
        positive_capacity_median,
    )
    metric_frame = pd.DataFrame([metrics])
    metric_path = data_dir / "p3_metricas_candidato.csv"
    profiles_path = data_dir / "p3_perfiles_cluster.csv"
    quality_path = data_dir / "p3_calidad_datos.csv"
    metric_frame.to_csv(metric_path, index=False)
    profiles.to_csv(profiles_path, index=False)
    quality.to_csv(quality_path, index=False)
    cluster.plot_station_clusters(assignments, plots_dir / "p3_mapa_clusters.png")
    cluster.plot_cluster_profiles(profiles, plots_dir / "p3_perfiles_cluster.png")
    config_path = config_dir / "p3_run_config.json"
    config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    return {
        "plots": plots_dir,
        "data": data_dir,
        "config": config_dir,
    }


def run_search(
    input_path: Path | str,
    output_dir: Path,
    candidate_k: Sequence[int],
    random_seed: int,
    n_init: int,
    experiment_name: str,
    tracking_uri: str,
    register_best: bool,
    registry_model_name: str,
    champion_alias: str,
    selected_k_override: int | None = None,
) -> P3SearchResult:
    """Run baseline plus nested candidates on one fixed, hashed P3 input."""
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    candidate_values = tuple(int(k) for k in candidate_k)
    if not candidate_values or len(candidate_values) != len(set(candidate_values)):
        raise ValueError("candidate_k debe contener valores unicos y al menos un candidato.")
    if any(k < 2 for k in candidate_values):
        raise ValueError("Cada valor de candidate_k debe ser al menos 2.")
    if P3_BASELINE_K in candidate_values:
        raise ValueError("candidate_k contiene el baseline; k=5 se registra como referencia del padre.")
    if n_init < 1:
        raise ValueError("n_init debe ser positivo.")
    if register_best and selected_k_override is None:
        raise ValueError(
            "Revise la comparacion y especifique selected_k_override antes de registrar."
        )

    stations = cluster.load_station_summary(input_path)
    enriched, features, feature_names, positive_capacity_median = cluster.prepare_station_data(
        stations
    )
    input_sha256 = cluster.compute_file_sha256(input_path)
    data_version = f"sha256:{input_sha256}"
    git_commit = cluster.get_git_commit()
    manifest = _build_manifest(input_path, input_sha256, stations, git_commit)
    manifest_path = output_dir / "parent" / "data" / "p3_input_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    raw_model_features = stations.loc[:, list(P3_MODEL_FEATURES)].copy()
    dataset = mlflow.data.from_pandas(
        raw_model_features,
        source=_relative_source_path(input_path),
        name="p3_raw_station_features",
        digest=input_sha256[:36],
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    child_run_ids: dict[int, str] = {}
    child_metrics: dict[int, dict[str, int | float]] = {}
    candidate_run_ids: dict[int, str] = {}
    with mlflow.start_run(run_name="P3-kmeans-nested-search") as parent_run:
        parent_run_id = parent_run.info.run_id
        parent_tags = {
            "team": "equipo49",
            "question": "P3",
            "model_family": "kmeans",
            "stage_of_work": "search",
            "git_commit": git_commit,
            "data_scope": "eda_aggregate_full_12m",
            "temporal_holdout": "false",
            "input_sha256": input_sha256,
            "data_version": data_version,
            "baseline_k": str(P3_BASELINE_K),
        }
        mlflow.set_tags(parent_tags)
        mlflow.log_params(
            {
                "n_candidates": len(candidate_values),
                "baseline_k": P3_BASELINE_K,
                "random_seed": random_seed,
                "n_init": n_init,
                "input_path": _relative_source_path(input_path),
                "input_sha256": input_sha256,
                "data_version": data_version,
            }
        )

        configurations = [(P3_BASELINE_K, True), *((k, False) for k in candidate_values)]
        quality_row = cluster.build_data_quality_summary(
            enriched,
            features,
            positive_capacity_median,
        ).iloc[0]
        for k, is_baseline in configurations:
            metrics, pipeline, labels = cluster.evaluate_kmeans_candidate(
                features,
                k,
                random_seed,
                n_init,
            )
            model = P3ClusterModel(pipeline, positive_capacity_median, k)
            run_config = {
                "question": "P3",
                "k": k,
                "baseline": is_baseline,
                "random_seed": random_seed,
                "n_init": n_init,
                "feature_names": feature_names,
                "transform": "log1p_counts",
                "scaler": "StandardScaler",
                "capacity_zero_policy": "median_positive_imputation",
                "positive_capacity_median": positive_capacity_median,
                "input_path": _relative_source_path(input_path),
                "input_sha256": input_sha256,
                "data_version": data_version,
                "period": P3_PERIOD,
            }
            artifact_dirs = _write_candidate_artifacts(
                output_dir / ("baseline-k5" if is_baseline else f"candidate-k{k}"),
                enriched,
                features,
                labels,
                metrics,
                positive_capacity_median,
                run_config,
            )
            with mlflow.start_run(
                run_name=(
                    "P3-kmeans-k5-baseline"
                    if is_baseline
                    else f"P3-kmeans-k{k}-candidate"
                ),
                nested=True,
            ) as child_run:
                child_run_ids[k] = child_run.info.run_id
                if not is_baseline:
                    candidate_run_ids[k] = child_run.info.run_id
                child_tags = {
                    "team": "equipo49",
                    "question": "P3",
                    "model_family": "kmeans",
                    "stage_of_work": "baseline" if is_baseline else "candidate",
                    "git_commit": git_commit,
                    "parent_search": parent_run_id,
                    "data_scope": "eda_aggregate_full_12m",
                    "temporal_holdout": "false",
                    "input_sha256": input_sha256,
                    "data_version": data_version,
                }
                child_tags["baseline_k" if is_baseline else "candidate_k"] = str(k)
                mlflow.set_tags(child_tags)
                child_params = {
                    "k": k,
                    "random_seed": random_seed,
                    "n_init": n_init,
                    "feature_names": ",".join(feature_names),
                    "transform": "log1p_counts",
                    "scaler": "StandardScaler",
                    "capacity_zero_policy": "median_positive_imputation",
                    "positive_capacity_median": positive_capacity_median,
                    "input_path": _relative_source_path(input_path),
                    "input_sha256": input_sha256,
                    "data_version": data_version,
                    "period": P3_PERIOD,
                }
                mlflow.log_params(child_params)
                run_metrics = {
                    **{key: float(value) for key, value in metrics.items()},
                    "n_zero_capacity": float(quality_row["n_zero_capacity"]),
                    "n_imputed_capacity": float(quality_row["n_imputed_capacity"]),
                }
                mlflow.log_metrics(run_metrics)
                mlflow.log_input(dataset, context="training")
                for name, directory in artifact_dirs.items():
                    mlflow.log_artifacts(str(directory), artifact_path=name)
                mlflow.log_artifact(str(manifest_path), artifact_path="data")
                log_p3_pyfunc_model(model, name="model")
            child_metrics[k] = metrics

        comparison = pd.DataFrame(
            [
                {
                    **child_metrics[k],
                    "run_id": child_run_ids[k],
                    "is_baseline": k == P3_BASELINE_K,
                }
                for k, _ in configurations
            ]
        )
        candidate_frame = comparison.loc[~comparison["is_baseline"]]
        best_candidate = select_best_candidate(candidate_frame)
        recommended = select_best_candidate(comparison)
        best_candidate_k = int(best_candidate["k"])
        best_candidate_run_id = candidate_run_ids[best_candidate_k]
        recommended_k = int(recommended["k"])
        recommended_run_id = str(recommended["run_id"])
        if selected_k_override is not None:
            if selected_k_override not in child_metrics:
                raise ValueError(
                    "selected_k_override debe ser el baseline o uno de los candidate_k ejecutados."
                )
            if child_metrics[selected_k_override]["min_cluster_size"] < MIN_CLUSTER_SIZE:
                raise ValueError(
                    "El k seleccionado manualmente tiene un cluster con menos de 2 estaciones."
                )
            selected_k = selected_k_override
        else:
            selected_k = recommended_k
        selected_run_id = child_run_ids[selected_k]
        selected_metric = float(child_metrics[selected_k][SELECTION_METRIC])

        comparison_path = output_dir / "parent" / "data" / "p3_comparacion_candidatos.csv"
        comparison.to_csv(comparison_path, index=False)
        parent_config = {
            "candidate_k": list(candidate_values),
            "baseline_k": P3_BASELINE_K,
            "random_seed": random_seed,
            "n_init": n_init,
            "selection_metric": SELECTION_METRIC,
            "selection_rule": (
                "eligible min_cluster_size >= 2; silhouette descending; "
                "Davies-Bouldin ascending; smaller k ascending"
            ),
            "best_candidate_k": best_candidate_k,
            "recommended_k": recommended_k,
            "selected_k_override": selected_k_override,
            "selected_k": selected_k,
            "best_candidate_run_id": best_candidate_run_id,
            "recommended_run_id": recommended_run_id,
            "selected_run_id": selected_run_id,
            "data_version": data_version,
            "input_sha256": input_sha256,
        }
        parent_config_path = output_dir / "parent" / "config" / "p3_search_config.json"
        parent_config_path.parent.mkdir(parents=True, exist_ok=True)
        parent_config_path.write_text(
            json.dumps(parent_config, indent=2) + "\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(manifest_path), artifact_path="data")
        mlflow.log_artifact(str(comparison_path), artifact_path="data")
        mlflow.log_artifact(str(parent_config_path), artifact_path="config")
        mlflow.log_metrics(
            {
                "n_candidates": float(len(candidate_values)),
                "baseline_k": float(P3_BASELINE_K),
                "best_candidate_k": float(best_candidate_k),
                "recommended_k": float(recommended_k),
                "selected_k": float(selected_k),
                "selection_metric_value": float(recommended[SELECTION_METRIC]),
            }
        )
        mlflow.set_tags(
            {
                "best_candidate_run_id": best_candidate_run_id,
                "recommended_run_id": recommended_run_id,
                "selected_run_id": selected_run_id,
                "selection_metric": SELECTION_METRIC,
                "selected_k": str(selected_k),
            }
        )

    version_number: str | None = None
    if register_best:
        description = p3_model_description(
            selected_k,
            SELECTION_METRIC,
            selected_metric,
        )
        version = register_p3_model_from_run(
            selected_run_id,
            model_artifact_path="model",
            registry_model_name=registry_model_name,
            tracking_uri=tracking_uri,
            description=description,
        )
        version_tags = {
            "question": "P3",
            "team": "equipo49",
            "model_family": "kmeans",
            "git_commit": git_commit,
            "data_version": data_version,
            "input_sha256": input_sha256,
            "source_run_id": selected_run_id,
            "parent_search_run_id": parent_run_id,
            "selected_k": str(selected_k),
            "selection_metric": SELECTION_METRIC,
            "stage_of_work": "champion",
        }
        set_model_version_tags(
            registry_model_name,
            version.version,
            version_tags,
            tracking_uri=tracking_uri,
        )
        predictions = validate_model_inference(
            f"models:/{registry_model_name}/{version.version}",
            p3_input_example(),
            selected_k,
            tracking_uri=tracking_uri,
        )
        if predictions.empty:
            raise ValueError("La validacion del modelo registrado produjo una salida vacia.")
        assign_champion_alias(
            registry_model_name,
            version.version,
            champion_alias,
            tracking_uri=tracking_uri,
        )
        version_number = str(version.version)

    return P3SearchResult(
        parent_run_id=parent_run_id,
        child_run_ids=child_run_ids,
        candidate_run_ids=candidate_run_ids,
        best_candidate_k=best_candidate_k,
        best_candidate_run_id=best_candidate_run_id,
        recommended_k=recommended_k,
        recommended_run_id=recommended_run_id,
        selected_k=selected_k,
        selected_run_id=selected_run_id,
        data_version=data_version,
        input_sha256=input_sha256,
        model_version=version_number,
    )


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise typer.BadParameter("El valor debe ser true o false.")
    return normalized == "true"


@app.command("run")
def run_command(
    input_path: Annotated[str, typer.Option(help="CSV agregado P3; vacio = config.py.")] = "",
    output_dir: Annotated[str, typer.Option(help="Directorio de trabajo; vacio = /tmp unico.")] = "",
    k_values: Annotated[str, typer.Option(help="Ks candidatos separados por coma; vacio = config.py.")] = "",
    random_seed: Annotated[str, typer.Option(help="Semilla; vacio = config.py.")] = "",
    n_init: Annotated[str, typer.Option(help="n_init KMeans; vacio = config.py.")] = "",
    experiment_name: Annotated[str, typer.Option(help="Nombre del experimento MLflow.")] = P3_SEARCH_EXPERIMENT_NAME,
    tracking_uri: Annotated[str, typer.Option(help="URI de tracking MLflow.")] = MLFLOW_TRACKING_URI,
    register_best: Annotated[str, typer.Option(help="Registrar y promover tras validar: true/false.")] = "false",
    registry_model_name: Annotated[str, typer.Option(help="Nombre fijo en Model Registry; vacio = config.py.")] = "",
    champion_alias: Annotated[str, typer.Option(help="Alias de producción; vacio = config.py.")] = "",
    selected_k: Annotated[str, typer.Option(help="Override manual de k; vacio = recomendación.")] = "",
) -> None:
    """Run a nested P3 search; Registry promotion is opt-in and validated."""
    candidate_values = (
        [int(item.strip()) for item in k_values.split(",") if item.strip()]
        if k_values
        else list(P3_CANDIDATE_K)
    )
    run_search(
        input_path=input_path or P3_INPUT_PATH,
        output_dir=Path(output_dir) if output_dir else Path(
            tempfile.mkdtemp(prefix="mibici-p3-search-")
        ),
        candidate_k=candidate_values,
        random_seed=int(random_seed) if random_seed else RANDOM_SEED,
        n_init=int(n_init) if n_init else P3_N_INIT,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        register_best=_parse_bool(register_best),
        registry_model_name=registry_model_name or P3_REGISTRY_MODEL_NAME,
        champion_alias=champion_alias or P3_CHAMPION_ALIAS,
        selected_k_override=int(selected_k) if selected_k else None,
    )


if __name__ == "__main__":
    app()
