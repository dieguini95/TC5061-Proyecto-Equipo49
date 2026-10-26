"""Register and validate a P1 model before deliberately promoting its alias."""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import numpy as np
import pandas as pd

from mibici_ml import config


def _prediction_vector(values: Any, n_rows: int, label: str) -> np.ndarray:
    """Require one finite, nonnegative expected departure count per input row."""
    predictions = np.asarray(values, dtype=float)
    if predictions.shape != (n_rows,):
        raise ValueError(f"{label}: se requiere un vector de {n_rows} predicciones.")
    if not np.isfinite(predictions).all():
        raise ValueError(f"{label}: las predicciones deben ser finitas.")
    if (predictions < 0).any():
        raise ValueError(f"{label}: las salidas previstas no pueden ser negativas.")
    return predictions


def validate_model_inference(
    model_uri: str,
    input_example: pd.DataFrame,
    expected_predictions: np.ndarray,
    tracking_uri: str | None = None,
) -> np.ndarray:
    """Check native sklearn and pyfunc inference against the evaluated predictions.

    The comparison uses the exact engineered feature rows, not raw trip records.
    Both interfaces must retain the regression contract when loading by URI.
    """
    if not isinstance(input_example, pd.DataFrame) or input_example.empty:
        raise ValueError("El ejemplo de inferencia debe ser un DataFrame no vacio.")
    expected = _prediction_vector(expected_predictions, len(input_example), "Referencia")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    native = mlflow.sklearn.load_model(model_uri)
    native_predictions = _prediction_vector(
        native.predict(input_example), len(input_example), "sklearn"
    )
    generic = mlflow.pyfunc.load_model(model_uri)
    predictions = _prediction_vector(generic.predict(input_example), len(input_example), "pyfunc")
    for interface, actual in (("sklearn", native_predictions), ("pyfunc", predictions)):
        if not np.allclose(actual, expected, rtol=1e-7, atol=1e-9):
            raise ValueError(
                f"{interface}: la inferencia no coincide con las predicciones evaluadas."
            )
    return predictions


def load_champion_model(tracking_uri: str | None = None) -> Any:
    """Load the P1 champion by its stable alias through the pyfunc interface."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pyfunc.load_model(
        f"models:/{config.P1_REGISTRY_MODEL_NAME}@{config.P1_CHAMPION_ALIAS}"
    )


def _alias_version(client: MlflowClient, name: str, alias: str) -> Any | None:
    """Return an existing alias, distinguishing absence from a backend failure."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException as error:
        if error.error_code not in {"RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"}:
            raise
        return None


def _versions_from_run(client: MlflowClient, name: str, run_id: str) -> list[Any]:
    """Find a previous registration, including paginated Registry responses."""
    versions, token = [], None
    while True:
        page = client.search_model_versions(
            filter_string=f"name = '{name}'", max_results=1000, page_token=token
        )
        versions.extend(version for version in page if version.run_id == run_id)
        token = page.token
        if not token:
            return versions


def _run_lineage(run: Any) -> dict[str, str]:
    """Require the code and data identity of a completed source run."""
    if run.info.status != "FINISHED":
        raise ValueError("Solo se puede promover un run FINISHED.")
    question = run.data.tags.get("question") or run.data.tags.get("pregunta")
    if question is not None and question.upper() != "P1":
        raise ValueError("El run de origen debe corresponder a P1.")
    lineage = {}
    for key in ("git_commit", "data_version", "source_version"):
        value = run.data.params.get(key) or run.data.tags.get(key)
        normalized = str(value).strip() if value is not None else ""
        if not normalized or normalized.lower() in {"unknown", "unavailable", "none"}:
            raise ValueError(f"El run carece de linaje reproducible: {key}.")
        lineage[key] = str(value)
    if "git_dirty" in run.data.tags:
        lineage["git_dirty"] = run.data.tags["git_dirty"]
    return lineage


def register_champion(
    run_id: str,
    input_example: pd.DataFrame,
    expected_predictions: np.ndarray,
    tracking_uri: str | None = None,
) -> dict[str, str]:
    """Register a run's ``model`` artifact and promote only a verified version.

    Repeated calls for the same run reuse its version. A failed validation before
    promotion leaves the old alias intact; failure after promotion restores it.
    Registration is explicit and does not train or select another model.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    lineage = _run_lineage(run)
    name, alias = config.P1_REGISTRY_MODEL_NAME, config.P1_CHAMPION_ALIAS
    # Fail before Registry mutation if the source artifact or expected output is invalid.
    validate_model_inference(
        f"runs:/{run_id}/model", input_example, expected_predictions, tracking_uri
    )
    previous = _alias_version(client, name, alias)
    existing = _versions_from_run(client, name, run_id)
    if previous is not None and previous.run_id == run_id:
        version = previous
    elif existing:
        version = min(existing, key=lambda candidate: int(candidate.version))
    else:
        version = mlflow.register_model(f"runs:/{run_id}/model", name, await_registration_for=30)
    version_number = str(version.version)
    version_uri = f"models:/{name}/{version_number}"
    alias_uri = f"models:/{name}@{alias}"
    validate_model_inference(version_uri, input_example, expected_predictions, tracking_uri)

    description = (
        "P1: pronostico de salidas observadas por estacion-hora en MiBici. "
        "Emision a medianoche para las 24 horas siguientes. La entrada contiene "
        "variables de calendario y rezagos preparados por el pipeline, disponibles "
        "al emitir el pronostico; no son registros crudos de viajes. Seleccion por "
        "WAPE de validacion, con prueba temporal reservada. No estima inventario "
        "ni demanda insatisfecha. "
        f"Run de origen: {run_id}. Columnas: {', '.join(map(str, input_example.columns))}."
    )
    tags = {
        "team": "equipo49",
        "equipo": "49",
        "question": "P1",
        "source_run_id": run_id,
        "selection_metric": "val_wape",
        "stage_of_work": (
            "champion"
            if previous is not None and str(previous.version) == version_number
            else "validated_candidate"
        ),
        **lineage,
    }
    if previous is not None and str(previous.version) != version_number:
        tags["previous_champion_version"] = str(previous.version)
    client.update_model_version(name, version_number, description=description)
    for key, value in tags.items():
        client.set_model_version_tag(name, version_number, key, value)
    client.set_registered_model_alias(name, alias, version_number)
    try:
        validate_model_inference(alias_uri, input_example, expected_predictions, tracking_uri)
    except Exception:
        if previous is None:
            client.delete_registered_model_alias(name, alias)
        else:
            client.set_registered_model_alias(name, alias, str(previous.version))
        raise
    client.update_registered_model(name, description=description)
    client.set_model_version_tag(name, version_number, "stage_of_work", "champion")
    return {
        "name": name,
        "version": version_number,
        "alias": alias,
        "model_uri": alias_uri,
        "run_id": run_id,
    }
