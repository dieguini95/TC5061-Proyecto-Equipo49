from __future__ import annotations

from fastapi.testclient import TestClient

from mibici_ml.api.providers import FEATURE_NAMES


def _item(station_id: int) -> dict[str, object]:
    return {
        "station_id": station_id,
        "capacidad": float(station_id),
        "salidas_12m": float(station_id + 1),
        "llegadas_12m": float(station_id + 2),
    }


def test_batch_preserves_order_ids_count_and_feature_contract(
    recording_client: TestClient, recording_provider
) -> None:
    response = recording_client.post(
        "/predict/batch",
        json={"items": [_item(22), _item(11), _item(33)]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert [row["station_id"] for row in body["predictions"]] == [22, 11, 33]
    assert [row["cluster"] for row in body["predictions"]] == [1, 2, 0]
    assert tuple(recording_provider.received_frames[-1].columns) == FEATURE_NAMES


def test_batch_accepts_one_item(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"items": [_item(7)]})
    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_batch_accepts_default_maximum_of_100(client: TestClient) -> None:
    response = client.post(
        "/predict/batch",
        json={"items": [_item(index) for index in range(100)]},
    )
    assert response.status_code == 200
    assert response.json()["count"] == 100


def test_batch_rejects_empty_and_over_100_items(client: TestClient) -> None:
    empty = client.post("/predict/batch", json={"items": []})
    too_many = client.post(
        "/predict/batch",
        json={"items": [_item(index) for index in range(101)]},
    )
    assert empty.status_code == 422
    assert too_many.status_code == 422


def test_batch_obeys_a_lower_configured_limit() -> None:
    from mibici_ml.api.app import create_app
    from mibici_ml.api.providers import FakeModelProvider
    from mibici_ml.api.settings import APISettings

    app = create_app(
        provider=FakeModelProvider(initially_ready=True),
        settings=APISettings(
            provider=None,
            model_uri="models:/mibici-equipo49-p3-grupos@champion",
            model_name="mibici-equipo49-p3-grupos",
            model_alias="champion",
            max_batch_size=1,
        ),
    )
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"items": [_item(1), _item(2)]})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "BATCH_SIZE_EXCEEDED"
