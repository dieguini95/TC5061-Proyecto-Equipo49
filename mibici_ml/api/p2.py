"""Segundo modelo de la API: riesgo de que una estacion quede vacia o llena (P2).

Se monta como router en ``/v1/p2`` cuando ``create_app`` recibe ``p2_bundle_dir``.
El modelo es un bundle skops (``model.skops`` y ``model_metadata.json``), asi que
la ruta de inferencia no depende de MLflow. La puntuacion es una probabilidad;
``riesgo_alto`` aplica el umbral de validacion que queda en los metadatos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Request
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mibici_ml.api.errors import ModelLoadError, ModelNotReadyError, ModelPredictionError
from mibici_ml.config import SERVING_MAX_BATCH
from mibici_ml.serving.bundle import BundleError, ServingBundle, load_bundle

# Mismo orden que features.P2_FEATURE_COLUMNS_V2; test_serving_contract lo verifica.
P2_FEATURE_NAMES = (
    "hora",
    "dia_semana",
    "fin_de_semana",
    "capacidad",
    "flujo_acumulado",
    "flujo_acumulado_sobre_c",
    "llegadas_1h",
    "salidas_1h",
    "llegadas_2h",
    "salidas_2h",
    "salidas_misma_hora_ayer",
    "riesgo_observado_h",
    "riesgo_observado_previo_h",
    "tendencia_flujo_3h",
)

Finite = Annotated[float, Field(allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class RegistroP2(BaseModel):
    """Estado de una estacion en el instante t, solo con informacion hasta t."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "hora": 8,
                    "dia_semana": 2,
                    "fin_de_semana": 0,
                    "capacidad": 30,
                    "flujo_acumulado": -4,
                    "flujo_acumulado_sobre_c": -0.1333,
                    "llegadas_1h": 3,
                    "salidas_1h": 7,
                    "llegadas_2h": 5,
                    "salidas_2h": 12,
                    "salidas_misma_hora_ayer": 9,
                    "riesgo_observado_h": 0,
                    "riesgo_observado_previo_h": 0,
                    "tendencia_flujo_3h": -0.2,
                }
            ]
        },
    )

    hora: int = Field(ge=0, le=23, description="Hora local de inicio de la observacion.")
    dia_semana: int = Field(ge=0, le=6, description="0 = lunes ... 6 = domingo.")
    fin_de_semana: int = Field(ge=0, le=1, description="1 si dia_semana es 5 o 6.")
    capacidad: float = Field(gt=0, allow_inf_nan=False, description="Docks de la estacion.")
    flujo_acumulado: Finite = Field(description="Salidas menos llegadas acumuladas del dia.")
    flujo_acumulado_sobre_c: Finite = Field(description="flujo_acumulado / capacidad.")
    llegadas_1h: NonNegative
    salidas_1h: NonNegative
    llegadas_2h: NonNegative
    salidas_2h: NonNegative
    salidas_misma_hora_ayer: NonNegative
    riesgo_observado_h: int = Field(ge=0, le=1, description="1 si en t ya estaba vacia o llena.")
    riesgo_observado_previo_h: int = Field(ge=0, le=1, description="Mismo estado H horas antes.")
    tendencia_flujo_3h: Finite = Field(description="Cambio del flujo en 3 h, sobre capacidad.")

    @model_validator(mode="after")
    def _fin_de_semana_coherente(self) -> RegistroP2:
        if self.fin_de_semana != int(self.dia_semana >= 5):
            raise ValueError("fin_de_semana debe ser 1 solo si dia_semana es 5 o 6.")
        return self


class PrediccionP2(BaseModel):
    riesgo_prob: float = Field(ge=0, le=1, description="Probabilidad de vacia o llena en 2 h.")
    riesgo_alto: bool = Field(description="True si riesgo_prob alcanza el umbral de validacion.")
    umbral: float = Field(description="Umbral de validacion del champion.")
    modelo: str = Field(description="Nombre@alias del modelo servido.")
    version: str = Field(description="Version del modelo en el Registry.")


class LoteP2(BaseModel):
    items: list[RegistroP2] = Field(min_length=1, max_length=SERVING_MAX_BATCH)


class ResultadoLoteP2(BaseModel):
    predicciones: list[PrediccionP2]
    count: int


class ModeloP2(BaseModel):
    modelo: str
    version: str
    umbral: float
    score_kind: str
    feature_names: list[str]
    ready: Literal[True] = True


def load_p2_bundle(bundle_dir: Path) -> ServingBundle:
    """Load the P2 bundle and verify that its feature contract is the API contract."""
    bundle = load_bundle(bundle_dir)
    if bundle.metadata.get("question") != "p2":
        raise BundleError("El bundle no corresponde a P2.")
    if tuple(bundle.feature_names) != P2_FEATURE_NAMES:
        raise BundleError("El contrato de variables del bundle no coincide con la API.")
    if bundle.threshold is None:
        raise BundleError("El bundle de P2 no declara umbral de validacion.")
    return bundle


def _score_rows(bundle: ServingBundle, rows: list[RegistroP2]) -> list[PrediccionP2]:
    matrix = np.array(
        [[getattr(row, name) for name in bundle.feature_names] for row in rows],
        dtype=float,
    )
    try:
        scores = bundle.score(matrix)
        threshold = bundle.threshold
        if threshold is None:
            raise ValueError("Sin umbral de validacion.")
        return [
            PrediccionP2(
                riesgo_prob=float(score),
                riesgo_alto=float(score) >= threshold,
                umbral=threshold,
                modelo=bundle.model_label,
                version=bundle.version,
            )
            for score in scores
        ]
    except Exception as error:
        raise ModelPredictionError from error


def build_p2_router() -> APIRouter:
    """Build the ``/v1/p2`` router; the bundle is read from ``app.state``."""
    router = APIRouter(prefix="/v1/p2", tags=["p2"])

    def _ready_bundle(request: Request) -> ServingBundle:
        if request.app.state.p2_load_failed:
            raise ModelLoadError
        bundle = request.app.state.p2_bundle
        if bundle is None:
            raise ModelNotReadyError
        return bundle

    @router.get("/ready")
    def ready(request: Request) -> dict[str, object]:
        _ready_bundle(request)
        return {"status": "ready", "ready": True}

    @router.get("/model", response_model=ModeloP2)
    def model(request: Request) -> ModeloP2:
        bundle = _ready_bundle(request)
        return ModeloP2(
            modelo=bundle.model_label,
            version=bundle.version,
            umbral=bundle.threshold or 0.0,
            score_kind=str(bundle.metadata["score_kind"]),
            feature_names=list(bundle.feature_names),
        )

    @router.post("/predict", response_model=PrediccionP2)
    def predict(item: RegistroP2, request: Request) -> PrediccionP2:
        bundle = _ready_bundle(request)
        return _score_rows(bundle, [item])[0]

    @router.post("/predict/batch", response_model=ResultadoLoteP2)
    def predict_batch(lote: LoteP2, request: Request) -> ResultadoLoteP2:
        bundle = _ready_bundle(request)
        predictions = _score_rows(bundle, lote.items)
        return ResultadoLoteP2(predicciones=predictions, count=len(predictions))

    return router
