from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from mibici_ml.api.app import create_app
from mibici_ml.api.errors import ModelLoadError, ModelPredictionError
from mibici_ml.api.mlflow_provider import MLflowModelProvider
from mibici_ml.api.providers import (
    FakeModelProvider,
    configured_provider,
)
from mibici_ml.api.settings import APISettings


class BadOutputProvider(FakeModelProvider):
    def __init__(self, result: object) -> None:
        super().__init__(initially_ready=True)
        self.result = result

    def predict(self, frame: pd.DataFrame):
        del frame
        return self.result


def test_fake_provider_load_controls_readiness() -> None:
    provider = FakeModelProvider()
    assert not provider.is_ready()
    provider.load()
    assert provider.is_ready()
    assert provider.metadata()["ready"] is True


def test_known_provider_load_failure_keeps_health_and_reports_failure() -> None:
    provider = FakeModelProvider(fail_load=True)
    with TestClient(create_app(provider=provider)) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_LOAD_FAILED"


def test_unexpected_provider_load_failure_is_safe_and_does_not_stop_health() -> None:
    class UnexpectedLoadFailureProvider(FakeModelProvider):
        def load(self) -> None:
            raise RuntimeError("internal /private/path tracking_uri=http://internal")

    with TestClient(create_app(provider=UnexpectedLoadFailureProvider())) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_LOAD_FAILED"
    for forbidden in ("/private/path", "tracking_uri", "http://internal"):
        assert forbidden not in response.text.lower()


def test_mlflow_provider_mode_reports_missing_export_safely(tmp_path) -> None:
    # Modo mlflow sin modelo disponible: el provider real se configura, pero sin
    # fallback al modelo falso y sin tocar el Registry local.
    settings = APISettings(
        provider="mlflow",
        model_name="mibici-equipo49-p3-grupos",
        model_alias="champion",
        model_uri="models:/mibici-equipo49-p3-grupos@champion",
        model_path=str(tmp_path / "sin-exportar" / "model"),
    )
    provider = configured_provider(settings)
    assert isinstance(provider, MLflowModelProvider)
    assert not isinstance(provider, FakeModelProvider)
    with pytest.raises(ModelLoadError):
        provider.load()
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_LOAD_FAILED"


def test_inference_failure_returns_safe_500() -> None:
    provider = FakeModelProvider(initially_ready=True, fail_prediction=True)
    with TestClient(create_app(provider=provider)) as client:
        response = client.post(
            "/predict",
            json={"capacidad": 1, "salidas_12m": 2, "llegadas_12m": 3},
        )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "MODEL_PREDICTION_FAILED"
    assert "fake provider" not in response.text.lower()


def test_prediction_length_mismatch_returns_safe_500() -> None:
    with TestClient(create_app(provider=BadOutputProvider([1, 2]))) as client:
        response = client.post(
            "/predict",
            json={"capacidad": 1, "salidas_12m": 2, "llegadas_12m": 3},
        )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "MODEL_PREDICTION_FAILED"


def test_batch_prediction_length_mismatch_returns_safe_500() -> None:
    with TestClient(create_app(provider=BadOutputProvider([1]))) as client:
        response = client.post(
            "/predict/batch",
            json={
                "items": [
                    {"station_id": 1, "capacidad": 1, "salidas_12m": 2, "llegadas_12m": 3},
                    {"station_id": 2, "capacidad": 2, "salidas_12m": 3, "llegadas_12m": 4},
                ]
            },
        )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "MODEL_PREDICTION_FAILED"


@pytest.mark.parametrize("result", [["not-numeric"], [float("nan")], [1.5], [-1], "1"])
def test_invalid_model_output_returns_safe_500(result: object) -> None:
    with TestClient(create_app(provider=BadOutputProvider(result))) as client:
        response = client.post(
            "/predict",
            json={"capacidad": 1, "salidas_12m": 2, "llegadas_12m": 3},
        )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "MODEL_PREDICTION_FAILED"
    assert "traceback" not in response.text.lower()


def test_prediction_error_does_not_leak_internal_uri_or_paths() -> None:
    class LeakingFailureProvider(FakeModelProvider):
        def predict(self, frame: pd.DataFrame):
            del frame
            raise ModelPredictionError(
                "internal /home/private models:/secret tracking_uri=http://internal"
            )

    with TestClient(create_app(provider=LeakingFailureProvider(initially_ready=True))) as client:
        response = client.post(
            "/predict",
            json={"capacidad": 1, "salidas_12m": 2, "llegadas_12m": 3},
        )
    assert response.status_code == 500
    for forbidden in ("/home/private", "models:/secret", "tracking_uri", "traceback"):
        assert forbidden not in response.text.lower()


def test_metadata_cannot_inject_unapproved_fields() -> None:
    class UnsafeMetadataProvider(FakeModelProvider):
        def metadata(self) -> dict[str, object]:
            return {**super().metadata(), "tracking_uri": "internal"}

    with TestClient(create_app(provider=UnsafeMetadataProvider(initially_ready=True))) as client:
        response = client.get("/model")
    assert response.status_code == 500
