from __future__ import annotations

from fastapi.testclient import TestClient

from mibici_ml.api.providers import FEATURE_NAMES


def test_ready_returns_model_identity_when_loaded(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_name": "mibici-equipo49-p3-grupos",
        "model_alias": "champion",
        "ready": True,
    }


def test_ready_returns_safe_503_before_load(unready_client: TestClient) -> None:
    response = unready_client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "MODEL_NOT_READY"
    assert "traceback" not in response.text.lower()


def test_model_metadata_has_only_safe_allowlisted_fields(client: TestClient) -> None:
    response = client.get("/model")
    assert response.status_code == 200
    assert response.json() == {
        "model_name": "mibici-equipo49-p3-grupos",
        "model_uri": "models:/mibici-equipo49-p3-grupos@champion",
        "model_alias": "champion",
        "model_version": None,
        "feature_names": list(FEATURE_NAMES),
        "ready": True,
    }
    assert "tracking_uri" not in response.text.lower()
    assert "password" not in response.text.lower()


def test_model_metadata_is_unavailable_until_provider_is_ready(
    unready_client: TestClient,
) -> None:
    response = unready_client.get("/model")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_NOT_READY"


def test_model_metadata_rejects_tracking_urls_or_local_paths(client: TestClient) -> None:
    provider = client.app.state.model_provider
    provider.metadata = lambda: {
        "model_name": "mibici-equipo49-p3-grupos",
        "model_uri": "http://internal/path",
        "model_alias": "champion",
        "model_version": None,
        "feature_names": list(FEATURE_NAMES),
        "ready": True,
    }
    response = client.get("/model")
    assert response.status_code == 500
    assert "http://internal" not in response.text
