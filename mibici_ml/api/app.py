"""FastAPI application for the P3 clustering model."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import asynccontextmanager
import math
from numbers import Real
from typing import Any

from fastapi import FastAPI, Request
from loguru import logger
import pandas as pd

from mibici_ml.api.errors import (
    BatchSizeLimitError,
    ModelLoadError,
    ModelNotReadyError,
    ModelPredictionError,
    PredictionShapeError,
    install_exception_handlers,
)
from mibici_ml.api.providers import FEATURE_NAMES, ModelProvider, configured_provider
from mibici_ml.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictionResponse,
    ReadyResponse,
    StationInput,
)
from mibici_ml.api.settings import APISettings

API_VERSION = "0.0.1"


def _provider_predictions(values: Any, expected_count: int) -> list[int]:
    if isinstance(values, (str, bytes, dict)) or not isinstance(values, Sequence):
        raise PredictionShapeError("Model output must be a sequence.")
    if len(values) != expected_count:
        raise PredictionShapeError("Model output count does not match the input.")

    clusters = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise PredictionShapeError("Cluster labels must be numeric integers.")
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or not numeric_value.is_integer():
            raise PredictionShapeError("Cluster labels must be finite integers.")
        cluster = int(numeric_value)
        if cluster < 0:
            raise PredictionShapeError("Cluster labels cannot be negative.")
        clusters.append(cluster)
    return clusters


def _input_frame(items: list[StationInput]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "capacidad": item.capacidad,
                "salidas_12m": item.salidas_12m,
                "llegadas_12m": item.llegadas_12m,
            }
            for item in items
        ],
        columns=FEATURE_NAMES,
    )


def _safe_metadata(provider: ModelProvider) -> ModelMetadataResponse:
    try:
        return ModelMetadataResponse.model_validate(provider.metadata())
    except Exception as error:
        raise ModelPredictionError from error


def create_app(
    provider: ModelProvider | None = None,
    settings: APISettings | None = None,
) -> FastAPI:
    api_settings = settings or APISettings.from_env()
    model_provider = provider if provider is not None else configured_provider(api_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            model_provider.load()
        except Exception:  # noqa: BLE001 - provider load must not take health down.
            application.state.model_load_failed = True
            logger.error("Provider startup load failed (MODEL_LOAD_FAILED).")
        yield

    application = FastAPI(
        title="MiBici P3 Serving API",
        version=API_VERSION,
        description="REST API for station clustering using the P3 model contract.",
        lifespan=lifespan,
    )
    application.state.model_provider = model_provider
    application.state.api_settings = api_settings
    application.state.model_load_failed = False
    install_exception_handlers(application)

    @application.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        return HealthResponse(version=API_VERSION)

    @application.get(
        "/ready",
        response_model=ReadyResponse,
        responses={503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
        tags=["service"],
    )
    def ready(request: Request) -> ReadyResponse:
        current_provider: ModelProvider = request.app.state.model_provider
        _ensure_provider_ready(request, current_provider)
        metadata = _safe_metadata(current_provider)
        if not metadata.ready:
            raise ModelNotReadyError
        return ReadyResponse(
            model_name=metadata.model_name,
            model_alias=metadata.model_alias,
        )

    @application.get(
        "/model",
        response_model=ModelMetadataResponse,
        responses={503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
        tags=["model"],
    )
    def model_metadata(request: Request) -> ModelMetadataResponse:
        current_provider: ModelProvider = request.app.state.model_provider
        _ensure_provider_ready(request, current_provider)
        return _safe_metadata(current_provider)

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        responses={
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["prediction"],
    )
    def predict(item: StationInput, request: Request) -> PredictionResponse:
        current_provider: ModelProvider = request.app.state.model_provider
        _ensure_provider_ready(request, current_provider)
        try:
            clusters = _provider_predictions(current_provider.predict(_input_frame([item])), 1)
        except ModelPredictionError:
            raise
        except Exception as error:
            raise ModelPredictionError from error
        metadata = _safe_metadata(current_provider)
        return PredictionResponse(
            station_id=item.station_id,
            cluster=clusters[0],
            model_name=metadata.model_name,
            model_alias=metadata.model_alias,
        )

    @application.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        responses={
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["prediction"],
    )
    def predict_batch(
        batch: BatchPredictionRequest, request: Request
    ) -> BatchPredictionResponse:
        current_provider: ModelProvider = request.app.state.model_provider
        _ensure_provider_ready(request, current_provider)
        settings_for_request: APISettings = request.app.state.api_settings
        if len(batch.items) > settings_for_request.max_batch_size:
            raise BatchSizeLimitError
        try:
            clusters = _provider_predictions(
                current_provider.predict(_input_frame(batch.items)),
                len(batch.items),
            )
        except ModelPredictionError:
            raise
        except Exception as error:
            raise ModelPredictionError from error
        metadata = _safe_metadata(current_provider)
        predictions = [
            PredictionResponse(
                station_id=item.station_id,
                cluster=cluster,
                model_name=metadata.model_name,
                model_alias=metadata.model_alias,
            )
            for item, cluster in zip(batch.items, clusters, strict=True)
        ]
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            model_name=metadata.model_name,
            model_alias=metadata.model_alias,
        )

    return application


def _is_provider_ready(provider: ModelProvider) -> bool:
    try:
        ready = provider.is_ready()
    except Exception as error:
        raise ModelPredictionError from error
    if not isinstance(ready, bool):
        raise ModelPredictionError
    return ready


def _ensure_provider_ready(request: Request, provider: ModelProvider) -> None:
    if request.app.state.model_load_failed:
        raise ModelLoadError
    if not _is_provider_ready(provider):
        raise ModelNotReadyError


app = create_app()
