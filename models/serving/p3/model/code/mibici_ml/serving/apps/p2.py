"""Servicio de inferencia de P2: riesgo de que una estacion quede vacia o llena.

Uso: ``uvicorn mibici_ml.serving.apps.p2:app``. El modelo se lee de
``models/serving/p2`` (o de ``MIBICI_SERVING_ROOT``). La puntuacion es una
probabilidad (``predict_proba``); la bandera ``riesgo_alto`` aplica el umbral que
se fijo en validacion y que queda en los metadatos del Registry.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mibici_ml.config import SERVING_ROOT_DIR
from mibici_ml.serving.bundle import ServingBundle
from mibici_ml.serving.service import ServiceSpec, create_service_app

# Mismo orden que features.P2_FEATURE_COLUMNS_V2; un test verifica la igualdad.
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
    def _fin_de_semana_coherente(self) -> "RegistroP2":
        if self.fin_de_semana != int(self.dia_semana >= 5):
            raise ValueError("fin_de_semana debe ser 1 solo si dia_semana es 5 o 6.")
        return self


class PrediccionP2(BaseModel):
    riesgo_prob: float = Field(ge=0, le=1, description="Probabilidad de vacia o llena en 2 h.")
    riesgo_alto: bool = Field(description="True si riesgo_prob alcanza el umbral de validacion.")
    umbral: float = Field(description="Umbral de validacion del champion.")
    modelo: str = Field(description="Nombre@alias del modelo servido.")
    version: str = Field(description="Version del modelo en el Registry.")


def _to_output(score: float, bundle: ServingBundle) -> PrediccionP2:
    threshold = bundle.threshold
    if threshold is None:
        raise ValueError("El bundle de P2 no declara umbral de validacion.")
    return PrediccionP2(
        riesgo_prob=score,
        riesgo_alto=score >= threshold,
        umbral=threshold,
        modelo=bundle.model_label,
        version=bundle.version,
    )


P2_SPEC = ServiceSpec(
    question="p2",
    title="MiBici - Riesgo de estacion vacia o llena (P2)",
    description=(
        "Probabilidad de que una estacion MiBici quede vacia o llena en las siguientes "
        "2 horas. Envia el estado observado hasta t; la API devuelve la probabilidad y la "
        "bandera de umbral del champion de `mibici-equipo49-p2-riesgo`."
    ),
    feature_names=P2_FEATURE_NAMES,
    input_model=RegistroP2,
    output_model=PrediccionP2,
    to_output=_to_output,
    bundle_dir=SERVING_ROOT_DIR / "p2",
)

app = create_service_app(P2_SPEC)
