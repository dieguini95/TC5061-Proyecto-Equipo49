from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capacidad", -0.01),
        ("salidas_12m", -1),
        ("llegadas_12m", -2),
        ("capacidad", float("nan")),
        ("salidas_12m", float("inf")),
        ("llegadas_12m", float("-inf")),
        ("capacidad", "not-a-number"),
        ("llegadas_12m", None),
    ],
)
def test_predict_rejects_invalid_features(
    client: TestClient,
    request_body: dict[str, object],
    field: str,
    value: object,
) -> None:
    request_body[field] = value
    response = client.post(
        "/predict",
        content=json.dumps(request_body, allow_nan=True),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REQUEST_VALIDATION_ERROR"
    assert "input" not in response.text.lower()


def test_predict_rejects_missing_feature(client: TestClient) -> None:
    response = client.post("/predict", json={"capacidad": 1, "salidas_12m": 2})
    assert response.status_code == 422


@pytest.mark.parametrize("field", ["capacidad", "salidas_12m", "llegadas_12m"])
def test_predict_rejects_boolean_features(
    client: TestClient,
    request_body: dict[str, object],
    field: str,
) -> None:
    request_body[field] = True
    response = client.post("/predict", json=request_body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REQUEST_VALIDATION_ERROR"


def test_predict_rejects_extra_field(client: TestClient, request_body: dict[str, object]) -> None:
    request_body["extra"] = "forbidden"
    response = client.post("/predict", json=request_body)
    assert response.status_code == 422


@pytest.mark.parametrize("station_id", ["", "   "])
def test_predict_rejects_empty_string_station_id(
    client: TestClient, request_body: dict[str, object], station_id: str
) -> None:
    request_body["station_id"] = station_id
    assert client.post("/predict", json=request_body).status_code == 422


def test_batch_rejects_extra_fields(client: TestClient, request_body: dict[str, object]) -> None:
    request_body["extra"] = True
    response = client.post("/predict/batch", json={"items": [request_body]})
    assert response.status_code == 422
