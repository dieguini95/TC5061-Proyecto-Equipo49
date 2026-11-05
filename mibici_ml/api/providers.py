"""Injectable model provider contract and explicit development providers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import pandas as pd

from mibici_ml.api.errors import ModelLoadError, ModelNotReadyError, ModelPredictionError
from mibici_ml.api.settings import APISettings

FEATURE_NAMES = ("capacidad", "salidas_12m", "llegadas_12m")


@runtime_checkable
class ModelProvider(Protocol):
    """Provide model lifecycle, readiness, prediction, and safe metadata.

    Implementations must normalize model output to a one-dimensional
    ``Sequence[int]``. An MLflow adapter must convert NumPy arrays, pandas
    Series, and a single-column DataFrame to a Python sequence, and reject
    ambiguous multidimensional or multi-column results.
    """

    def load(self) -> None: ...

    def is_ready(self) -> bool: ...

    def predict(self, frame: pd.DataFrame) -> Sequence[int]: ...

    def metadata(self) -> dict[str, object]: ...


class UnconfiguredModelProvider:
    """Safe default that reports unavailable until a real loader is integrated."""

    def __init__(self, settings: APISettings) -> None:
        self._settings = settings

    def load(self) -> None:
        raise ModelLoadError("A real MLflowModelProvider has not been configured.")

    def is_ready(self) -> bool:
        return False

    def predict(self, frame: pd.DataFrame) -> Sequence[int]:
        del frame
        raise ModelNotReadyError

    def metadata(self) -> dict[str, object]:
        return {
            "model_name": self._settings.model_name,
            "model_uri": self._settings.model_uri,
            "model_alias": self._settings.model_alias,
            "model_version": None,
            "feature_names": list(FEATURE_NAMES),
            "ready": False,
        }


class FakeModelProvider:
    """Deterministic provider for tests and explicitly selected local development."""

    def __init__(
        self,
        *,
        model_name: str = "mibici-equipo49-p3-grupos",
        model_uri: str = "models:/mibici-equipo49-p3-grupos@champion",
        model_alias: str | None = "champion",
        model_version: str | None = None,
        initially_ready: bool = False,
        fail_load: bool = False,
        fail_prediction: bool = False,
    ) -> None:
        self._model_name = model_name
        self._model_uri = model_uri
        self._model_alias = model_alias
        self._model_version = model_version
        self._ready = initially_ready
        self._fail_load = fail_load
        self._fail_prediction = fail_prediction

    def load(self) -> None:
        if self._fail_load:
            raise ModelLoadError("Fake provider load failure.")
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, frame: pd.DataFrame) -> Sequence[int]:
        if not self._ready:
            raise ModelNotReadyError
        if self._fail_prediction:
            raise ModelPredictionError("Fake provider inference failure.")
        if tuple(frame.columns) != FEATURE_NAMES:
            raise ModelPredictionError("Unexpected feature columns.")
        return [int(float(row[0]) % 3) for row in frame.itertuples(index=False, name=None)]

    def metadata(self) -> dict[str, object]:
        return {
            "model_name": self._model_name,
            "model_uri": self._model_uri,
            "model_alias": self._model_alias,
            "model_version": self._model_version,
            "feature_names": list(FEATURE_NAMES),
            "ready": self._ready,
        }


def configured_provider(settings: APISettings) -> ModelProvider:
    if settings.provider == "fake":
        return FakeModelProvider(
            model_name=settings.model_name,
            model_uri=settings.model_uri,
            model_alias=settings.model_alias,
        )
    if settings.provider == "mlflow":
        # Import diferido: el provider falso y las pruebas no cargan MLflow.
        from mibici_ml.api.mlflow_provider import MLflowModelProvider

        return MLflowModelProvider(settings)
    return UnconfiguredModelProvider(settings)
