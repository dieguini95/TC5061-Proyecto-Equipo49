"""Operaciones explicitas del Model Registry para P2 (riesgo de estacion).

Mismo patron que ``registry_p3``: registrar la version desde el run, describirla,
guardar su linaje como tags, validar la inferencia y solo entonces mover el alias.
"""

from __future__ import annotations

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
import numpy as np
import pandas as pd

from mibici_ml.config import P2_CHAMPION_ALIAS, P2_REGISTRY_MODEL_NAME


def p2_model_description(model_name: str, threshold: float) -> str:
    """Describe alcance, entrada, seleccion y limitaciones del champion de P2."""
    return (
        "Objetivo: probabilidad de que una estacion MiBici quede vacia o llena en las "
        "siguientes 2 horas. Modelo "
        f"{model_name} con features v2 (estado observado sin fuga futura). Puntuar con "
        f"predict_proba y umbral de validacion {threshold:.4f}, no con predict (corte 0.5). "
        "Seleccion por PR-AUC de validacion; prueba abierta una sola vez. Limitacion: "
        "inventario aproximado solo con viajes."
    )


def register_p2_model_from_run(
    run_id: str,
    model_artifact_path: str = "model",
    registry_model_name: str = P2_REGISTRY_MODEL_NAME,
    tracking_uri: str | None = None,
    description: str | None = None,
    tags: dict[str, str] | None = None,
) -> ModelVersion:
    """Registra una version desde un run sin asignar alias."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/{model_artifact_path}",
        name=registry_model_name,
        await_registration_for=0,
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    if description:
        client.update_registered_model(registry_model_name, description=description)
    for key, value in (tags or {}).items():
        client.set_model_version_tag(registry_model_name, str(version.version), key, str(value))
    return version


def validate_p2_model_inference(
    model_uri: str,
    input_example: pd.DataFrame,
    tracking_uri: str | None = None,
) -> np.ndarray:
    """Carga una URI concreta y verifica probabilidades finitas en [0, 1]."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(model_uri)
    scores = np.asarray(model.predict(input_example), dtype=float).ravel()
    if not np.isfinite(scores).all():
        raise ValueError("La inferencia P2 produjo probabilidades no finitas.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("La inferencia P2 produjo valores fuera de [0, 1].")
    return scores


def assign_p2_alias(
    version: str | int,
    alias: str = P2_CHAMPION_ALIAS,
    registry_model_name: str = P2_REGISTRY_MODEL_NAME,
    tracking_uri: str | None = None,
) -> None:
    """Mueve un alias de forma explicita; el rollback es volver a llamar con otra version."""
    MlflowClient(tracking_uri=tracking_uri).set_registered_model_alias(
        registry_model_name, alias, str(version)
    )
