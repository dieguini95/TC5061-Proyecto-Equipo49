"""Estimadores serializables de P1 sobre el mismo contrato causal de variables.

Las transformaciones de demanda viven en ``features.py``. Este módulo recibe
únicamente sus columnas numéricas, ajusta cada candidato con la muestra que le
entrega el pipeline y conserva el recorte a cero dentro del modelo guardado.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from mibici_ml import config
from mibici_ml.features import p1_model_features

BASELINE_COLUMNS = {"seasonal_168h": "lag_168", "mean_4weeks": "mean_4weeks"}
_ALLOWED_PARAMS = {
    "ridge": frozenset({"alpha"}),
    "hist_poisson": frozenset(
        {
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
            "max_depth",
        }
    ),
    "hist_squared": frozenset(
        {
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
            "max_depth",
        }
    ),
}


def _resolved_params(model_name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Aplica solo overrides conocidos; los defaults se leen de la configuración."""
    if params is not None and not isinstance(params, dict):
        raise TypeError("Los hiperparámetros de P1 deben ser un diccionario.")
    overrides = {} if params is None else dict(params)
    if model_name in BASELINE_COLUMNS:
        if overrides:
            raise ValueError(f"El baseline {model_name} no admite hiperparámetros.")
        return {}
    if model_name not in _ALLOWED_PARAMS:
        supported = ", ".join([*BASELINE_COLUMNS, *_ALLOWED_PARAMS])
        raise ValueError(f"Modelo P1 desconocido: {model_name}. Opciones: {supported}.")
    unknown = set(overrides) - _ALLOWED_PARAMS[model_name]
    if unknown:
        raise ValueError(
            f"Hiperparámetros no admitidos para {model_name}: {', '.join(sorted(unknown))}"
        )
    defaults = dict(config.P1_MODEL_DEFAULTS[model_name])
    if set(defaults) - _ALLOWED_PARAMS[model_name]:
        raise ValueError(f"P1_MODEL_DEFAULTS contiene parámetros no admitidos para {model_name}.")
    return {**defaults, **overrides}


def _regressor(model_name: str, resolved: dict[str, Any]) -> BaseEstimator:
    """Crea el estimador con la pérdida, semilla y partición interna fijadas."""
    if model_name == "ridge":
        return Ridge(**resolved)
    return HistGradientBoostingRegressor(
        loss="poisson" if model_name == "hist_poisson" else "squared_error",
        early_stopping=False,
        random_state=config.RANDOM_SEED,
        **resolved,
    )


def get_effective_params(model_name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Devuelve los parámetros efectivos del estimador para registrar su ejecución.

    Incluye los defaults de scikit-learn de la versión fijada, la pérdida de cada
    familia y ``early_stopping=False``. Ese último valor evita que el estimador
    cree una partición aleatoria distinta de la validación temporal del proyecto.
    """
    resolved = _resolved_params(model_name, params)
    if model_name in BASELINE_COLUMNS:
        return {"source_column": BASELINE_COLUMNS[model_name]}
    return _regressor(model_name, resolved).get_params(deep=False)


class P1Regressor(RegressorMixin, BaseEstimator):
    """Modelo P1 clonable con selección de variables y predicciones no negativas.

    Ridge codifica hora y día como categorías, ajustadas solo en entrenamiento,
    y escala el resto de columnas. La matriz permanece dispersa para no expandir
    millones de filas a una tabla densa. Los modelos histogramados usan la misma
    entrada numérica y desactivan la parada temprana con partición aleatoria.
    """

    def __init__(self, model_name: str = "mean_4weeks", params: dict[str, Any] | None = None):
        self.model_name = model_name
        self.params = params

    def fit(self, X: pd.DataFrame, y: Any = None) -> P1Regressor:
        """Ajusta únicamente con las filas proporcionadas; los baselines no se entrenan."""
        resolved = _resolved_params(self.model_name, self.params)
        features = p1_model_features(X)
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.estimator_ = None
        if self.model_name not in BASELINE_COLUMNS:
            if y is None:
                raise ValueError("El ajuste de un candidato P1 requiere la variable objetivo.")
            target = np.asarray(y, dtype=float)
            if target.ndim != 1 or len(target) != len(features):
                raise ValueError("El objetivo P1 debe ser un vector con una etiqueta por fila.")
            if not np.isfinite(target).all() or (target < 0).any():
                raise ValueError("El objetivo P1 debe contener conteos finitos no negativos.")
            if self.model_name == "hist_poisson" and not (target > 0).any():
                raise ValueError("La pérdida Poisson requiere algún conteo positivo al entrenar.")
            estimator = _regressor(self.model_name, resolved)
            if self.model_name == "ridge":
                categorical = ["hour", "day_of_week"]
                numeric = [name for name in features.columns if name not in categorical]
                preprocessing = ColumnTransformer(
                    [
                        (
                            "calendar",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                            categorical,
                        ),
                        ("numeric", StandardScaler(), numeric),
                    ],
                    sparse_threshold=1.0,
                )
                estimator = Pipeline([("features", preprocessing), ("regressor", estimator)])
            with threadpool_limits(limits=config.P1_MODEL_THREADS):
                estimator.fit(features, target)
            self.estimator_ = estimator
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice con el contrato guardado y aplica el mismo recorte al cargar el modelo."""
        check_is_fitted(self, attributes=["is_fitted_"])
        features = p1_model_features(X)
        if not np.array_equal(features.columns.to_numpy(), self.feature_names_in_):
            raise ValueError("Las columnas de P1 cambiaron respecto al modelo ajustado.")
        if self.model_name in BASELINE_COLUMNS:
            predictions = features[BASELINE_COLUMNS[self.model_name]].to_numpy(dtype=float)
        else:
            with threadpool_limits(limits=config.P1_MODEL_THREADS):
                predictions = np.asarray(self.estimator_.predict(features), dtype=float)
        if predictions.ndim != 1 or not np.isfinite(predictions).all():
            raise ValueError("El modelo P1 produjo predicciones ausentes o no finitas.")
        return np.maximum(predictions, 0.0)


def build_p1_model(model_name: str, params: dict[str, Any] | None = None) -> P1Regressor:
    """Construye un modelo nuevo sin entrenarlo ni consultar validación o prueba."""
    _resolved_params(model_name, params)
    return P1Regressor(model_name=model_name, params=params)
