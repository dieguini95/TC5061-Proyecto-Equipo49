"""P2 router mounted under /v1/p2 next to the P3 endpoints."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mibici_ml.api.app import create_app
from mibici_ml.api.p2 import P2_FEATURE_NAMES
from mibici_ml.api.providers import FakeModelProvider
from mibici_ml.serving.bundle import load_bundle, save_bundle

THRESHOLD = 0.3

REGISTRO = {
    "hora": 8,
    "dia_semana": 2,
    "fin_de_semana": 0,
    "capacidad": 30,
    "flujo_acumulado": -4,
    "flujo_acumulado_sobre_c": -0.1333,
    "llegadas_1h": 3,
    "salidas_1h": 7,
    "llegadas_2h": 5,
    "salidas_2h": 12,
    "salidas_misma_hora_ayer": 9,
    "riesgo_observado_h": 0,
    "riesgo_observado_previo_h": 0,
    "tendencia_flujo_3h": -0.2,
}


def _metadata(feature_names=P2_FEATURE_NAMES) -> dict:
    return {
        "question": "p2",
        "registry_model_name": "mibici-equipo49-p2-riesgo",
        "alias": "champion",
        "version": "1",
        "model_id": "m-test",
        "source_run_id": "run-test",
        "model_family": "linear",
        "score_kind": "predict_proba_positive",
        "feature_names": list(feature_names),
        "threshold": THRESHOLD,
        "threshold_source": "test",
        "data_version": "test",
        "git_commit": "test",
        "skops_trusted_types": [],
    }


@pytest.fixture
def p2_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(400, len(P2_FEATURE_NAMES)))
    labels = (features[:, 11] + rng.normal(size=400) > 0).astype(int)
    model = LogisticRegression(max_iter=500).fit(features, labels)
    return save_bundle(tmp_path / "p2", model, _metadata())


@pytest.fixture
def client(p2_dir: Path) -> TestClient:
    app = create_app(provider=FakeModelProvider(initially_ready=True), p2_bundle_dir=p2_dir)
    with TestClient(app) as test_client:
        yield test_client


def test_ready_and_model_expose_contract_and_threshold(client):
    assert client.get("/v1/p2/ready").status_code == 200
    model = client.get("/v1/p2/model").json()
    assert model["modelo"] == "mibici-equipo49-p2-riesgo@champion"
    assert model["version"] == "1"
    assert model["umbral"] == THRESHOLD
    assert model["feature_names"] == list(P2_FEATURE_NAMES)


def test_predict_returns_probability_and_threshold_flag(client, p2_dir):
    response = client.post("/v1/p2/predict", json=REGISTRO)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["riesgo_prob"] <= 1.0
    assert body["riesgo_alto"] == (body["riesgo_prob"] >= THRESHOLD)
    assert body["umbral"] == THRESHOLD
    row = np.array([[REGISTRO[name] for name in P2_FEATURE_NAMES]], dtype=float)
    expected = load_bundle(p2_dir).score(row)[0]
    assert body["riesgo_prob"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "mutation",
    [
        {"hora": 24},
        {"dia_semana": 7},
        {"capacidad": 0},
        {"riesgo_observado_h": 2},
        {"llegadas_1h": -1},
        {"dia_semana": 5, "fin_de_semana": 0},
        {"campo_extra": 1},
    ],
)
def test_predict_rejects_invalid_records_with_422(client, mutation):
    payload = {**REGISTRO, **mutation}
    response = client.post("/v1/p2/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REQUEST_VALIDATION_ERROR"


def test_predict_rejects_missing_field_and_nan_without_500(client):
    missing = copy.deepcopy(REGISTRO)
    missing.pop("hora")
    assert client.post("/v1/p2/predict", json=missing).status_code == 422

    nan_body = json.dumps({**REGISTRO, "capacidad": float("nan")})
    response = client.post(
        "/v1/p2/predict", content=nan_body, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


def test_batch_predicts_each_record_and_limits_size(client):
    response = client.post(
        "/v1/p2/predict/batch",
        json={"items": [REGISTRO, {**REGISTRO, "dia_semana": 5, "fin_de_semana": 1}]},
    )
    assert response.status_code == 200
    assert response.json()["count"] == 2

    assert client.post("/v1/p2/predict/batch", json={"items": []}).status_code == 422
    too_many = {"items": [REGISTRO] * 501}
    assert client.post("/v1/p2/predict/batch", json=too_many).status_code == 422


def test_missing_bundle_is_503_while_p3_and_health_stay_up(tmp_path):
    app = create_app(
        provider=FakeModelProvider(initially_ready=True),
        p2_bundle_dir=tmp_path / "sin-modelo",
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        response = client.get("/v1/p2/ready")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "MODEL_LOAD_FAILED"
        assert client.post("/v1/p2/predict", json=REGISTRO).status_code == 503


def test_feature_contract_mismatch_is_503(tmp_path):
    rng = np.random.default_rng(7)
    features = rng.normal(size=(200, len(P2_FEATURE_NAMES)))
    model = LogisticRegression(max_iter=300).fit(features, rng.integers(0, 2, size=200))
    directory = save_bundle(
        tmp_path / "p2",
        model,
        _metadata(feature_names=tuple(reversed(P2_FEATURE_NAMES))),
    )
    app = create_app(provider=FakeModelProvider(initially_ready=True), p2_bundle_dir=directory)
    with TestClient(app) as client:
        assert client.get("/v1/p2/ready").status_code == 503


def test_p2_is_not_mounted_without_a_bundle_dir():
    with TestClient(create_app(provider=FakeModelProvider(initially_ready=True))) as client:
        assert client.post("/v1/p2/predict", json=REGISTRO).status_code == 404
