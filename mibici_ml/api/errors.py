"""Safe, consistent errors exposed by the serving API."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger


class ModelNotReadyError(Exception):
    """Raised when inference is requested before a provider is ready."""


class ModelLoadError(Exception):
    """Raised when an explicitly selected provider cannot load its model."""


class ModelPredictionError(Exception):
    """Raised when model inference fails."""


class PredictionShapeError(ModelPredictionError):
    """Raised when model output does not match the input batch."""


class BatchSizeLimitError(Exception):
    """Raised when a batch exceeds the configured serving limit."""


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        del request, error
        return _error_response(
            422,
            "REQUEST_VALIDATION_ERROR",
            "La solicitud no cumple el contrato.",
        )

    @app.exception_handler(ModelNotReadyError)
    async def model_not_ready_handler(
        request: Request, error: ModelNotReadyError
    ) -> JSONResponse:
        del request, error
        return _error_response(503, "MODEL_NOT_READY", "El modelo no está disponible.")

    @app.exception_handler(ModelLoadError)
    async def model_load_handler(
        request: Request, error: ModelLoadError
    ) -> JSONResponse:
        del request, error
        logger.error("Model load failed (MODEL_LOAD_FAILED).")
        return _error_response(503, "MODEL_LOAD_FAILED", "No se pudo cargar el modelo.")

    @app.exception_handler(BatchSizeLimitError)
    async def batch_limit_handler(
        request: Request, error: BatchSizeLimitError
    ) -> JSONResponse:
        del request, error
        return _error_response(
            422,
            "BATCH_SIZE_EXCEEDED",
            "El lote excede el máximo permitido.",
        )

    @app.exception_handler(ModelPredictionError)
    async def model_prediction_handler(
        request: Request, error: ModelPredictionError
    ) -> JSONResponse:
        del request, error
        logger.error("Model inference failed (MODEL_PREDICTION_FAILED).")
        return _error_response(
            500,
            "MODEL_PREDICTION_FAILED",
            "No se pudo generar la predicción.",
        )
