"""Explicit P3 Model Registry operations and inference checks."""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
import numpy as np
import pandas as pd

from mibici_ml.config import P3_CHAMPION_ALIAS, P3_REGISTRY_MODEL_NAME


def p3_model_description(
    selected_k: int,
    selection_metric: str,
    metric_value: float,
) -> str:
    """Describe the model scope, input, selection, and known limitations."""
    return (
        f"Objetivo: agrupar estaciones MiBici por volumen agregado y capacidad (k={selected_k}). "
        "Features crudas: capacidad, salidas_12m y llegadas_12m; transforma conteos con log1p, "
        "imputa capacidad cero con la mediana positiva aprendida y aplica StandardScaler + KMeans. "
        "Fuente: reports/tables/estaciones_resumen.csv, agregado de 12 meses "
        "(2025-10 a 2026-09). Seleccion recomendado por "
        f"{selection_metric}={metric_value:.8g}, con minimo de 2 estaciones por cluster; "
        "Davies-Bouldin y menor k desempatan. No hay holdout temporal ni se afirma "
        "generalizacion temporal; los agregados no representan perfiles horarios."
    )


def register_p3_model_from_run(
    run_id: str,
    model_artifact_path: str = "model",
    registry_model_name: str = P3_REGISTRY_MODEL_NAME,
    tracking_uri: str | None = None,
    description: str | None = None,
    tags: dict[str, str] | None = None,
) -> ModelVersion:
    """Register a logged P3 model version without assigning an alias."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    version = mlflow.register_model(
        model_uri=model_uri,
        name=registry_model_name,
        await_registration_for=0,
    )
    if description:
        set_registered_model_description(
            registry_model_name,
            description,
            tracking_uri=tracking_uri,
        )
    if tags:
        set_model_version_tags(
            registry_model_name,
            version.version,
            tags,
            tracking_uri=tracking_uri,
        )
    return version


def set_registered_model_description(
    registry_model_name: str,
    description: str,
    tracking_uri: str | None = None,
) -> RegisteredModel:
    """Set the description on the registered model, not a specific version."""
    client = MlflowClient(tracking_uri=tracking_uri)
    return client.update_registered_model(registry_model_name, description=description)


def set_model_version_tags(
    registry_model_name: str,
    version: str | int,
    tags: dict[str, str],
    tracking_uri: str | None = None,
) -> None:
    """Set traceability tags on one registered model version."""
    client = MlflowClient(tracking_uri=tracking_uri)
    for key, value in tags.items():
        client.set_model_version_tag(
            registry_model_name,
            str(version),
            key,
            str(value),
        )


def assign_champion_alias(
    registry_model_name: str = P3_REGISTRY_MODEL_NAME,
    version: str | int = 1,
    champion_alias: str = P3_CHAMPION_ALIAS,
    tracking_uri: str | None = None,
) -> None:
    """Explicitly move an alias; rollback is the same call with an older version."""
    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_registered_model_alias(registry_model_name, champion_alias, str(version))


def load_champion_model(
    registry_model_name: str = P3_REGISTRY_MODEL_NAME,
    champion_alias: str = P3_CHAMPION_ALIAS,
    tracking_uri: str | None = None,
) -> Any:
    """Load the current alias through the MLflow pyfunc inference interface."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pyfunc.load_model(f"models:/{registry_model_name}@{champion_alias}")


def validate_model_inference(
    model_uri: str,
    input_example: pd.DataFrame,
    selected_k: int,
    tracking_uri: str | None = None,
) -> pd.DataFrame:
    """Load a concrete model URI and verify finite integer labels in [0, k)."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(input_example)
    if not isinstance(predictions, pd.DataFrame) or "cluster" not in predictions:
        raise ValueError("La inferencia P3 debe devolver un DataFrame con columna 'cluster'.")
    labels = predictions["cluster"].to_numpy(dtype=float)
    if not np.isfinite(labels).all():
        raise ValueError("La inferencia P3 produjo etiquetas no finitas.")
    if not np.equal(labels, np.floor(labels)).all():
        raise ValueError("La inferencia P3 debe devolver etiquetas enteras.")
    if ((labels < 0) | (labels >= selected_k)).any():
        raise ValueError("La inferencia P3 produjo etiquetas fuera del rango esperado.")
    return predictions


def validate_champion_inference(
    input_example: pd.DataFrame,
    selected_k: int,
    registry_model_name: str = P3_REGISTRY_MODEL_NAME,
    champion_alias: str = P3_CHAMPION_ALIAS,
    tracking_uri: str | None = None,
) -> pd.DataFrame:
    """Check the aliased model after a deliberate promotion."""
    model_uri = f"models:/{registry_model_name}@{champion_alias}"
    return validate_model_inference(
        model_uri,
        input_example,
        selected_k,
        tracking_uri=tracking_uri,
    )
