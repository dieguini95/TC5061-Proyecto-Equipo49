from __future__ import annotations

from fastapi.testclient import TestClient

from mibici_ml.api.providers import FEATURE_NAMES


def test_predict_accepts_integer_station_id_and_preserves_feature_order(
    recording_client: TestClient, recording_provider, request_body: dict[str, object]
) -> None:
    response = recording_client.post("/predict", json=request_body)
    assert response.status_code == 200
    result = response.json()
    assert result["station_id"] == 123
    assert isinstance(result["cluster"], int)
    frame = recording_provider.received_frames[-1]
    assert tuple(frame.columns) == FEATURE_NAMES
    assert "station_id" not in frame.columns


def test_predict_accepts_string_station_id(recording_client: TestClient) -> None:
    response = recording_client.post(
        "/predict",
        json={
            "station_id": "station-123",
            "capacidad": 1.5,
            "salidas_12m": 2.25,
            "llegadas_12m": 3.75,
        },
    )
    assert response.status_code == 200
    assert response.json()["station_id"] == "station-123"


def test_predict_accepts_missing_station_id_and_zero_features(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"capacidad": 0, "salidas_12m": 0, "llegadas_12m": 0},
    )
    assert response.status_code == 200
    assert response.json()["station_id"] is None
    assert response.json()["cluster"] == 0


def test_predict_returns_not_ready_without_loaded_provider(unready_client: TestClient) -> None:
    response = unready_client.post(
        "/predict",
        json={"capacidad": 1, "salidas_12m": 2, "llegadas_12m": 3},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_NOT_READY"
