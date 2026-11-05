"""MLflowModelProvider: Registry alias, exported directory, failures and output contract."""

from __future__ import annotations

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from mibici_ml.api.app import create_app
from mibici_ml.api.errors import ModelLoadError, ModelNotReadyError, ModelPredictionError
from mibici_ml.api.mlflow_provider import MLflowModelProvider, normalize_labels
from mibici_ml.api.settings import APISettings

NAME = "mibici-equipo49-p3-grupos"
URI = f"models:/{NAME}@champion"


def _settings(alias: str = "champion", model_path: str | None = None) -> APISettings:
    return APISettings(
        provider="mlflow",
        model_name=NAME,
        model_alias=alias,
        model_uri=f"models:/{NAME}@{alias}",
        model_path=model_path,
    )


def _frame(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["capacidad", "salidas_12m", "llegadas_12m"]).astype(float)


def test_registry_provider_loads_alias_and_predicts_integer_labels(registry_store):
    provider = MLflowModelProvider(_settings(), tracking_uri=registry_store.tracking_uri)
    assert not provider.is_ready()
    with pytest.raises(ModelNotReadyError):
        provider.predict(_frame([[30.0, 1.0, 1.0]]))

    provider.load()

    assert provider.is_ready()
    metadata = provider.metadata()
    assert metadata["model_version"] == "1"
    assert metadata["model_uri"] == URI
    assert metadata["ready"] is True
    labels = provider.predict(
        _frame([[30.0, 1200.0, 900.0], [12.0, 80.0, 90.0], [20.0, 1.0, 1.0]])
    )
    assert list(labels) == [0, 0, 2]
    assert all(isinstance(label, int) for label in labels)


def test_exported_directory_serves_without_any_registry(exported_p3_dir, tmp_path):
    settings = _settings(model_path=str(exported_p3_dir))
    # Tracking URI apuntando a una base inexistente: la carga no debe consultarla.
    provider = MLflowModelProvider(settings, tracking_uri=f"sqlite:///{tmp_path / 'no.db'}")

    provider.load()

    assert provider.is_ready()
    assert provider.metadata()["model_version"] == "1"
    assert list(provider.predict(_frame([[20.0, 1.0, 1.0]]))) == [2]


def test_exported_directory_with_other_alias_is_load_error(exported_p3_dir):
    provider = MLflowModelProvider(
        _settings(alias="staging", model_path=str(exported_p3_dir)),
    )
    with pytest.raises(ModelLoadError):
        provider.load()
    assert not provider.is_ready()


def test_missing_exported_directory_is_load_error(tmp_path):
    provider = MLflowModelProvider(_settings(model_path=str(tmp_path / "sin-exportar" / "model")))
    with pytest.raises(ModelLoadError):
        provider.load()
    assert not provider.is_ready()


def test_registry_alias_that_does_not_exist_is_load_error(registry_store):
    provider = MLflowModelProvider(
        _settings(alias="staging"), tracking_uri=registry_store.tracking_uri
    )
    with pytest.raises(ModelLoadError):
        provider.load()
    assert not provider.is_ready()


def test_app_serves_registry_model_through_the_shared_endpoints(registry_store):
    settings = _settings()
    provider = MLflowModelProvider(settings, tracking_uri=registry_store.tracking_uri)
    with TestClient(create_app(provider=provider, settings=settings)) as client:
        assert client.get("/ready").status_code == 200
        single = client.post(
            "/predict",
            json={"station_id": "EC-7", "capacidad": 20, "salidas_12m": 1, "llegadas_12m": 2},
        )
        batch = client.post(
            "/predict/batch",
            json={
                "items": [
                    {"capacidad": 30, "salidas_12m": 1200, "llegadas_12m": 900},
                    {"capacidad": 12, "salidas_12m": 80, "llegadas_12m": 90},
                ]
            },
        )
    assert single.status_code == 200
    assert single.json()["cluster"] == 2
    assert single.json()["model_alias"] == "champion"
    assert batch.status_code == 200
    assert [item["cluster"] for item in batch.json()["predictions"]] == [0, 0]


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (np.array([1, 2]), [1, 2]),
        (pd.Series([0, 2]), [0, 2]),
        (pd.DataFrame({"cluster": [1, 0]}), [1, 0]),
    ],
)
def test_normalize_labels_returns_one_dimensional_list(output, expected):
    assert normalize_labels(output) == expected


@pytest.mark.parametrize(
    "output",
    [
        pd.DataFrame({"a": [1], "b": [2]}),
        np.array([[1], [2]]),
    ],
)
def test_normalize_labels_rejects_ambiguous_shapes(output):
    with pytest.raises(ModelPredictionError):
        normalize_labels(output)
