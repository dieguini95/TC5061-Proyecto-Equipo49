"""Pydantic v2 request and response schemas for P3 serving."""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mibici_ml.api.settings import DEFAULT_MAX_BATCH_SIZE


class StrictAPIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StationInput(StrictAPIModel):
    station_id: int | str | None = None
    capacidad: float
    salidas_12m: float
    llegadas_12m: float

    @field_validator(
        "capacidad",
        "salidas_12m",
        "llegadas_12m",
        mode="before",
    )
    @classmethod
    def reject_boolean_features(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError(  # noqa: TRY004 - Pydantic maps this to HTTP 422.
                "Los valores numéricos no pueden ser booleanos."
            )
        return value

    @field_validator("capacidad", "salidas_12m", "llegadas_12m")
    @classmethod
    def validate_finite_nonnegative(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("El valor debe ser finito.")
        if value < 0:
            raise ValueError("El valor no puede ser negativo.")
        return value

    @field_validator("station_id")
    @classmethod
    def validate_station_id(cls, value: int | str | None) -> int | str | None:
        if isinstance(value, str) and not value.strip():
            raise ValueError("station_id no puede estar vacío.")
        return value


class PredictionResponse(StrictAPIModel):
    station_id: int | str | None
    cluster: int
    model_name: str
    model_alias: str | None
    status: Literal["ok"] = "ok"


class BatchPredictionRequest(StrictAPIModel):
    items: list[StationInput] = Field(min_length=1, max_length=DEFAULT_MAX_BATCH_SIZE)


class BatchPredictionResponse(StrictAPIModel):
    predictions: list[PredictionResponse]
    count: int
    model_name: str
    model_alias: str | None
    status: Literal["ok"] = "ok"


class HealthResponse(StrictAPIModel):
    status: Literal["ok"] = "ok"
    service: str = "mibici-p3-api"
    version: str


class ReadyResponse(StrictAPIModel):
    status: Literal["ready"] = "ready"
    model_name: str
    model_alias: str | None
    ready: Literal[True] = True


class ModelMetadataResponse(StrictAPIModel):
    model_name: str
    model_uri: str
    model_alias: str | None
    model_version: str | None
    feature_names: list[str]
    ready: bool

    @field_validator("model_name", "model_alias")
    @classmethod
    def validate_model_identifiers(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError("Model identifiers must be safe names.")
        return value

    @field_validator("model_uri")
    @classmethod
    def validate_model_uri(cls, value: str) -> str:
        if not re.fullmatch(
            r"models:/[A-Za-z0-9][A-Za-z0-9_.-]*@[A-Za-z0-9][A-Za-z0-9_.-]*",
            value,
        ):
            raise ValueError("The model URI must be a safe Registry alias URI.")
        return value

    @field_validator("model_version")
    @classmethod
    def validate_model_version(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[0-9]+", value):
            raise ValueError("Model version must be a numeric identifier.")
        return value

    @field_validator("feature_names")
    @classmethod
    def validate_feature_names(cls, value: list[str]) -> list[str]:
        if value != ["capacidad", "salidas_12m", "llegadas_12m"]:
            raise ValueError("Unexpected model feature contract.")
        return value


class ErrorDetail(StrictAPIModel):
    code: str
    message: str


class ErrorResponse(StrictAPIModel):
    error: ErrorDetail
