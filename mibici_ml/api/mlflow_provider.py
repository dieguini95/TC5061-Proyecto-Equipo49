"""MLflow-backed provider for the P3 clustering model.

Two sources, selected by configuration:

- ``MIBICI_MODEL_PATH``: model directory exported from the Registry with
  ``make serving-export`` (inside the container). It needs no tracking server
  and no ``mlflow.db``; the sidecar ``model_metadata.json`` provides the
  registry name, alias and numeric version.
- otherwise: the alias URI in the MLflow Model Registry
  (``MLFLOW_TRACKING_URI``), used for local development.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mibici_ml.api.errors import ModelLoadError, ModelNotReadyError, ModelPredictionError
from mibici_ml.api.providers import FEATURE_NAMES
from mibici_ml.api.settings import APISettings
from mibici_ml.config import MLFLOW_TRACKING_URI, SERVING_METADATA_FILE


def normalize_labels(output: Any) -> list[Any]:
    """Convert a pyfunc output to a one-dimensional Python list of labels.

    Accepts a NumPy array, a pandas Series or a single-column DataFrame.
    Ambiguous multidimensional or multi-column results are rejected. Integer
    validation is done by the application, which owns the response contract.
    """
    if isinstance(output, pd.DataFrame):
        if output.shape[1] != 1:
            raise ModelPredictionError("La salida del modelo tiene varias columnas.")
        values = output.iloc[:, 0].to_numpy()
    elif isinstance(output, pd.Series):
        values = output.to_numpy()
    else:
        values = np.asarray(output)
    if values.ndim != 1:
        raise ModelPredictionError("La salida del modelo no es unidimensional.")
    return values.tolist()


class MLflowModelProvider:
    """Load the P3 champion and predict through the MLflow pyfunc interface."""

    def __init__(self, settings: APISettings, *, tracking_uri: str | None = None) -> None:
        self._settings = settings
        self._tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
        self._model: Any = None
        self._version: str | None = None

    def load(self) -> None:
        try:
            if self._settings.model_path:
                self._load_exported()
            else:
                self._load_from_registry()
        except ModelLoadError:
            raise
        except Exception as error:
            self._model = None
            raise ModelLoadError from error

    def _load_exported(self) -> None:
        import mlflow

        model_dir = Path(str(self._settings.model_path))
        metadata_path = model_dir.parent / SERVING_METADATA_FILE
        if not model_dir.is_dir() or not metadata_path.is_file():
            raise ModelLoadError
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("registry_model_name") != self._settings.model_name
            or metadata.get("alias") != self._settings.model_alias
        ):
            raise ModelLoadError
        self._model = mlflow.pyfunc.load_model(str(model_dir))
        self._version = str(metadata["version"])

    def _load_from_registry(self) -> None:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(self._tracking_uri)
        self._model = mlflow.pyfunc.load_model(self._settings.model_uri)
        version = MlflowClient(tracking_uri=self._tracking_uri).get_model_version_by_alias(
            self._settings.model_name,
            self._settings.model_alias,
        )
        self._version = str(version.version)

    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, frame: pd.DataFrame) -> Sequence[int]:
        if self._model is None:
            raise ModelNotReadyError
        return normalize_labels(self._model.predict(frame))

    def metadata(self) -> dict[str, object]:
        return {
            "model_name": self._settings.model_name,
            "model_uri": self._settings.model_uri,
            "model_alias": self._settings.model_alias,
            "model_version": self._version,
            "feature_names": list(FEATURE_NAMES),
            "ready": self.is_ready(),
        }
