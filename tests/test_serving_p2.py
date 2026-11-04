"""Pruebas de la API de P2 con un bundle sintetico (no requiere el Registry)."""

import copy
from dataclasses import replace
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
import skops.io as sio

from mibici_ml.serving import bundle as bundle_module
from mibici_ml.serving.apps.p2 import P2_FEATURE_NAMES, P2_SPEC
from mibici_ml.serving.service import create_service_app

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


@pytest.fixture(scope="module")
def fitted_model():
    rng = np.random.default_rng(42)
    matrix = rng.random((300, len(P2_FEATURE_NAMES)))
    labels = (matrix[:, 0] > 0.7).astype(int)
    return HistGradientBoostingClassifier(max_iter=10, random_state=42).fit(matrix, labels)


def _write_bundle(directory, model, feature_names=P2_FEATURE_NAMES, threshold=THRESHOLD):
    trusted = sio.get_untrusted_types(data=sio.dumps(model))
    metadata = {
        "question": "p2",
        "registry_model_name": "mibici-equipo49-p2-riesgo",
        "alias": "champion",
        "version": "7",
        "model_id": "m-test",
        "source_run_id": "run-test",
        "model_family": "gradient_boosting",
        "score_kind": bundle_module.SCORE_PREDICT_PROBA,
        "feature_names": list(feature_names),
        "threshold": threshold,
        "data_version": "test",
        "git_commit": "test",
        "sklearn_version": "1.9.1",
        "skops_trusted_types": trusted,
    }
    bundle_module.save_bundle(directory, model, metadata)
    return directory


def _client(directory):
    return TestClient(create_service_app(replace(P2_SPEC, bundle_dir=directory)))


@pytest.fixture
def client(tmp_path, fitted_model):
    _write_bundle(tmp_path / "p2", fitted_model)
    with _client(tmp_path / "p2") as test_client:
        yield test_client


def test_health_and_ready_report_loaded_model(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["modelo"] == "mibici-equipo49-p2-riesgo@champion"


def test_model_endpoint_exposes_contract_and_threshold(client):
    body = client.get("/v1/p2/model").json()
    assert body["umbral"] == pytest.approx(THRESHOLD)
    assert body["feature_names"] == list(P2_FEATURE_NAMES)
    assert body["score_kind"] == "predict_proba_positive"


def test_predict_returns_probability_and_threshold_flag(client, fitted_model):
    response = client.post("/v1/p2/predict", json=REGISTRO)
    assert response.status_code == 200
    body = response.json()
    row = np.array([[float(REGISTRO[name]) for name in P2_FEATURE_NAMES]])
    expected = fitted_model.predict_proba(row)[0, 1]
    assert body["riesgo_prob"] == pytest.approx(expected)
    assert 0.0 <= body["riesgo_prob"] <= 1.0
    assert body["riesgo_alto"] is (body["riesgo_prob"] >= THRESHOLD)
    assert body["umbral"] == pytest.approx(THRESHOLD)
    assert body["modelo"] == "mibici-equipo49-p2-riesgo@champion"
    assert body["version"] == "7"


@pytest.mark.parametrize(
    "mutation",
    [
        {"hora": 24},
        {"hora": -1},
        {"dia_semana": 7},
        {"fin_de_semana": 1},  # dia 2 no es fin de semana
        {"capacidad": 0},
        {"llegadas_1h": -1},
        {"riesgo_observado_h": 2},
        {"riesgo_observado_previo_h": -1},
        {"hora": "temprano"},
        {"hora": None},
        {"extra_campo": 1},
    ],
)
def test_predict_rejects_invalid_records_with_422(client, mutation):
    payload = copy.deepcopy(REGISTRO)
    payload.update(mutation)
    assert client.post("/v1/p2/predict", json=payload).status_code == 422


def test_predict_rejects_missing_field_and_non_finite_values(client):
    payload = copy.deepcopy(REGISTRO)
    del payload["salidas_2h"]
    assert client.post("/v1/p2/predict", json=payload).status_code == 422
    raw = json.dumps(REGISTRO).replace("-4", "NaN")
    response = client.post(
        "/v1/p2/predict", content=raw, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


def test_flujo_acumulado_may_be_negative(client):
    assert client.post("/v1/p2/predict", json=REGISTRO).status_code == 200


def test_batch_predicts_each_record(client):
    other = dict(REGISTRO, hora=20, dia_semana=6, fin_de_semana=1)
    response = client.post("/v1/p2/predict/batch", json={"registros": [REGISTRO, other]})
    assert response.status_code == 200
    body = response.json()
    assert body["n"] == 2
    assert len(body["predicciones"]) == 2
    single = client.post("/v1/p2/predict", json=other).json()
    assert body["predicciones"][1]["riesgo_prob"] == pytest.approx(single["riesgo_prob"])


def test_batch_limits_size_and_rejects_empty(client):
    assert client.post("/v1/p2/predict/batch", json={"registros": []}).status_code == 422
    too_many = {"registros": [REGISTRO] * (P2_SPEC.max_batch + 1)}
    assert client.post("/v1/p2/predict/batch", json=too_many).status_code == 422


def test_missing_bundle_returns_503_but_health_stays_up(tmp_path):
    with _client(tmp_path / "vacio") as test_client:
        assert test_client.get("/health").status_code == 200
        assert test_client.get("/ready").status_code == 503
        assert test_client.post("/v1/p2/predict", json=REGISTRO).status_code == 503
        assert test_client.get("/v1/p2/model").status_code == 503


def test_feature_contract_mismatch_returns_503(tmp_path, fitted_model):
    shuffled = tuple(reversed(P2_FEATURE_NAMES))
    _write_bundle(tmp_path / "p2", fitted_model, feature_names=shuffled)
    with _client(tmp_path / "p2") as test_client:
        assert test_client.get("/ready").status_code == 503
        assert test_client.post("/v1/p2/predict", json=REGISTRO).status_code == 503
