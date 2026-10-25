"""Pruebas de las utilidades explicitas del Registry de P2."""

import numpy as np
import pandas as pd
import pytest

from mibici_ml import config
from mibici_ml.modeling import registry_p2


def test_p2_registry_name_and_alias_are_unique_and_stable():
    assert config.P2_REGISTRY_MODEL_NAME == "mibici-equipo49-p2-riesgo"
    assert config.P2_CHAMPION_ALIAS == "champion"
    assert config.P2_CHALLENGER_ALIAS == "challenger"


def test_p2_description_states_scoring_contract():
    text = registry_p2.p2_model_description("hist_gb", 0.2002)
    assert "predict_proba" in text
    assert "0.2002" in text
    assert "2 horas" in text


def test_validate_p2_inference_rejects_out_of_range(monkeypatch):
    class Bad:
        def predict(self, frame):
            return np.array([0.5, 1.7])

    monkeypatch.setattr(registry_p2.mlflow.pyfunc, "load_model", lambda uri: Bad())
    with pytest.raises(ValueError, match="fuera de"):
        registry_p2.validate_p2_model_inference("models:/x@champion", pd.DataFrame({"a": [1, 2]}))


def test_validate_p2_inference_rejects_non_finite(monkeypatch):
    class Bad:
        def predict(self, frame):
            return np.array([0.1, np.nan])

    monkeypatch.setattr(registry_p2.mlflow.pyfunc, "load_model", lambda uri: Bad())
    with pytest.raises(ValueError, match="no finitas"):
        registry_p2.validate_p2_model_inference("models:/x@champion", pd.DataFrame({"a": [1, 2]}))


def test_validate_p2_inference_accepts_probabilities(monkeypatch):
    class Good:
        def predict(self, frame):
            return np.array([0.05, 0.4])

    monkeypatch.setattr(registry_p2.mlflow.pyfunc, "load_model", lambda uri: Good())
    scores = registry_p2.validate_p2_model_inference("models:/x@champion", pd.DataFrame({"a": [1, 2]}))
    assert scores.tolist() == [0.05, 0.4]
