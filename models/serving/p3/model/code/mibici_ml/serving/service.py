"""Fabrica comun de servicios de inferencia (una app FastAPI por pregunta).

Cada pregunta define un ``ServiceSpec`` con su contrato de entrada y salida y la
funcion que convierte la puntuacion del modelo en la respuesta. La fabrica aporta
el ciclo de vida del modelo, las rutas comunes, el lote y los codigos de error.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, create_model

from mibici_ml.config import SERVING_MAX_BATCH
from mibici_ml.serving.bundle import BundleError, ServingBundle, load_bundle

SERVICE_VERSION = "1.0.0"


@dataclass(frozen=True)
class ServiceSpec:
    """Contrato de una pregunta: variables, esquemas y conversion de la puntuacion."""

    question: str
    title: str
    description: str
    feature_names: tuple[str, ...]
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    to_output: Callable[[float, ServingBundle], BaseModel]
    bundle_dir: Path
    max_batch: int = SERVING_MAX_BATCH


def _finite_safe(value: Any) -> Any:
    """Reemplaza NaN e infinitos por texto para que el error 422 sea JSON valido."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _finite_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_safe(item) for item in value]
    return value


def _check_contract(bundle: ServingBundle, spec: ServiceSpec) -> None:
    if bundle.feature_names != spec.feature_names:
        raise BundleError(
            "Las variables del bundle no coinciden con el esquema de la API. "
            "Reexporta el modelo o actualiza el esquema de entrada."
        )


def _batch_models(spec: ServiceSpec) -> tuple[type[BaseModel], type[BaseModel]]:
    request = create_model(
        f"{spec.question.upper()}BatchRequest",
        registros=(
            list[spec.input_model],  # type: ignore[name-defined]
            Field(min_length=1, max_length=spec.max_batch),
        ),
        __config__=ConfigDict(extra="forbid"),
    )
    response = create_model(
        f"{spec.question.upper()}BatchResponse",
        predicciones=(list[spec.output_model], ...),  # type: ignore[name-defined]
        n=(int, ...),
    )
    return request, response


def create_service_app(spec: ServiceSpec) -> FastAPI:
    """Crea la app de una pregunta. El modelo se carga al arrancar; si falla, ``/ready`` da 503."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.bundle = None
        app.state.load_error = None
        try:
            bundle = load_bundle(spec.bundle_dir)
            _check_contract(bundle, spec)
            app.state.bundle = bundle
            logger.info(f"Modelo cargado: {bundle.model_label} version {bundle.version}")
        except (BundleError, OSError, ValueError) as exc:
            app.state.load_error = str(exc)
            logger.error(f"No se pudo cargar el modelo de {spec.question}: {exc}")
        yield

    app = FastAPI(
        title=spec.title,
        description=spec.description,
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )
    base = f"/v1/{spec.question}"
    # Alias locales: FastAPI resuelve las anotaciones en el ambito de la funcion.
    InputModel = spec.input_model
    OutputModel = spec.output_model
    BatchRequest, BatchResponse = _batch_models(spec)

    def _loaded(request: Request) -> ServingBundle:
        bundle = request.app.state.bundle
        if bundle is None:
            raise HTTPException(status_code=503, detail="Modelo no disponible.")
        return bundle

    def _score(bundle: ServingBundle, rows: list[BaseModel]) -> np.ndarray:
        matrix = np.array(
            [[float(getattr(row, name)) for name in spec.feature_names] for row in rows],
            dtype=float,
        )
        try:
            return bundle.score(matrix)
        except ValueError as exc:
            logger.error(f"Fallo al puntuar {spec.question}: {exc}")
            raise HTTPException(status_code=500, detail="No se pudo calcular la prediccion.")

    @app.exception_handler(RequestValidationError)
    async def _invalid_input(request: Request, exc: RequestValidationError) -> JSONResponse:
        # El 'input' del error puede traer NaN/inf, que no son JSON valido: se convierten a texto.
        detail = jsonable_encoder(_finite_safe(exc.errors()))
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.get("/health", tags=["servicio"], summary="Proceso vivo")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["servicio"], summary="Modelo cargado y listo para predecir")
    def ready(request: Request) -> dict[str, str]:
        bundle = _loaded(request)
        return {"status": "ready", "modelo": bundle.model_label, "version": bundle.version}

    @app.get(f"{base}/model", tags=[spec.question], summary="Metadatos del modelo servido")
    def model_info(request: Request) -> dict[str, Any]:
        bundle = _loaded(request)
        meta = bundle.metadata
        return {
            "modelo": bundle.model_label,
            "version": bundle.version,
            "alias": meta["alias"],
            "score_kind": meta["score_kind"],
            "umbral": bundle.threshold,
            "feature_names": list(bundle.feature_names),
            "data_version": meta.get("data_version"),
            "model_family": meta.get("model_family"),
        }

    @app.post(
        f"{base}/predict",
        response_model=OutputModel,
        tags=[spec.question],
        summary="Prediccion para un registro",
    )
    def predict(record: InputModel, request: Request) -> BaseModel:
        bundle = _loaded(request)
        score = _score(bundle, [record])[0]
        return spec.to_output(float(score), bundle)

    @app.post(
        f"{base}/predict/batch",
        response_model=BatchResponse,
        tags=[spec.question],
        summary=f"Prediccion para hasta {spec.max_batch} registros",
    )
    def predict_batch(body: BatchRequest, request: Request) -> dict[str, Any]:
        bundle = _loaded(request)
        rows = list(body.registros)
        scores = _score(bundle, rows)
        outputs = [spec.to_output(float(s), bundle) for s in scores]
        return {"predicciones": outputs, "n": len(outputs)}

    return app
