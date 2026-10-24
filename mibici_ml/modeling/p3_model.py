"""Raw-input contract and MLflow flavor logging for the P3 clustering model."""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from mibici_ml.config import P3_MODEL_FEATURES, PROJ_ROOT
from mibici_ml.modeling.cluster import (
    MODEL_FEATURES,
    fit_final_pipeline,
    prepare_station_data,
)


class P3ClusterModel(mlflow.pyfunc.PythonModel):
    """KMeans inference wrapper accepting the three raw station features."""

    def __init__(
        self,
        pipeline: Pipeline,
        positive_capacity_median: float,
        selected_k: int,
    ) -> None:
        self.pipeline = pipeline
        self.positive_capacity_median = float(positive_capacity_median)
        self.selected_k = int(selected_k)
        if not np.isfinite(self.positive_capacity_median) or self.positive_capacity_median <= 0:
            raise ValueError("La mediana positiva de capacidad debe ser finita y positiva.")
        if self.selected_k < 2:
            raise ValueError("selected_k debe ser al menos 2.")

    @classmethod
    def fit(
        cls,
        stations: pd.DataFrame,
        selected_k: int,
        random_seed: int,
        n_init: int,
    ) -> P3ClusterModel:
        """Fit the historical feature transform and KMeans pipeline."""
        _, features, _, positive_capacity_median = prepare_station_data(stations)
        pipeline, _ = fit_final_pipeline(
            features,
            final_k=selected_k,
            random_seed=random_seed,
            n_init=n_init,
        )
        return cls(pipeline, positive_capacity_median, selected_k)

    def _transform_input(self, model_input: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("La entrada del modelo P3 debe ser un pandas DataFrame.")
        missing = [name for name in P3_MODEL_FEATURES if name not in model_input.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas para inferencia: {', '.join(missing)}.")
        if model_input.empty:
            raise ValueError("La entrada del modelo P3 no puede estar vacia.")

        raw = model_input.loc[:, list(P3_MODEL_FEATURES)].copy()
        for column in P3_MODEL_FEATURES:
            try:
                raw[column] = pd.to_numeric(raw[column], errors="raise").astype(float)
            except (TypeError, ValueError) as error:
                raise ValueError(f"La columna '{column}' debe contener valores numericos.") from error
            if not np.isfinite(raw[column].to_numpy(dtype=float)).all():
                raise ValueError(f"La columna '{column}' contiene valores faltantes o no finitos.")
            if (raw[column] < 0).any():
                raise ValueError(f"La columna '{column}' no puede contener valores negativos.")

        capacity = raw["capacidad"].mask(raw["capacidad"].eq(0), self.positive_capacity_median)
        transformed = pd.DataFrame(
            {
                MODEL_FEATURES[0]: np.log1p(raw["salidas_12m"]),
                MODEL_FEATURES[1]: np.log1p(raw["llegadas_12m"]),
                MODEL_FEATURES[2]: capacity,
            },
            index=raw.index,
        )
        if not np.isfinite(transformed.to_numpy(dtype=float)).all():
            raise ValueError("La transformacion de entrada produjo valores no finitos.")
        return transformed

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Return integer cluster labels for validated raw station features."""
        del context, params
        transformed = self._transform_input(model_input)
        labels = np.asarray(self.pipeline.predict(transformed), dtype=int)
        if ((labels < 0) | (labels >= self.selected_k)).any():
            raise ValueError("KMeans produjo etiquetas fuera del rango de clusters configurado.")
        return pd.DataFrame({"cluster": labels}, index=model_input.index)


def p3_input_example() -> pd.DataFrame:
    """Return a small synthetic example matching the public inference contract."""
    return pd.DataFrame(
        {
            "capacidad": [12.0, 20.0, 30.0],
            "salidas_12m": [80.0, 240.0, 640.0],
            "llegadas_12m": [90.0, 230.0, 620.0],
        }
    )


def log_p3_pyfunc_model(model: P3ClusterModel, name: str) -> ModelInfo:
    """Log a signed pyfunc model with its raw-feature example and source code."""
    example = p3_input_example()
    signature = infer_signature(example, model.predict(None, example))
    return mlflow.pyfunc.log_model(
        name=name,
        python_model=model,
        code_paths=[str(PROJ_ROOT / "mibici_ml")],
        signature=signature,
        input_example=example,
    )
