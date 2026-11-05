from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mibici_ml.api.app import create_app
from mibici_ml.api.providers import FakeModelProvider


def test_create_app_returns_fastapi(app: FastAPI) -> None:
    assert isinstance(app, FastAPI)


def test_default_app_does_not_select_fake_provider() -> None:
    default_app = create_app()
    assert not isinstance(default_app.state.model_provider, FakeModelProvider)


def test_health_is_process_health_not_model_readiness(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "mibici-p3-api",
        "version": "0.0.1",
    }


def test_health_remains_ok_when_provider_is_not_ready(unready_client: TestClient) -> None:
    response = unready_client.get("/health")
    assert response.status_code == 200
    assert "ready" not in response.json()


def test_openapi_exposes_expected_endpoints(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert {"/health", "/ready", "/model", "/predict", "/predict/batch"} <= paths.keys()
