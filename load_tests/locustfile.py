"""Escenarios de carga de la API de MiBici (Actividad 9).

Mezcla operativa (MiBiciMixedUser), con espera de 0.5 a 1.5 s entre peticiones:
    60 %  POST /v1/p2/predict         riesgo de estación (P2), 1 predicción por petición
    20 %  POST /predict               cluster de estación (P3), 1 predicción por petición
    15 %  POST /v1/p2/predict/batch   lote de 20 estaciones (P2), 20 predicciones por petición
     3 %  GET  /health
     2 %  GET  /v1/p2/ready

Escenarios de optimización por lotes (S4, sin espera entre peticiones):
    P2UnitUser    solo POST /v1/p2/predict (línea base unitaria)
    P2BatchUser   solo POST /v1/p2/predict/batch de 20 estaciones

Uso (desde la raíz del repositorio):
    locust -f load_tests/locustfile.py MiBiciMixedUser --headless -u 40 -r 5 -t 90s \
        --host http://127.0.0.1:8000 --reset-stats --csv reports/tables/a9_locust_S2

Variables de entorno: LOAD_SEED (semilla de los payloads, 42 por defecto) y LOCUST_HOST.
Las respuestas se validan: un código distinto de 200 o un cuerpo sin la clave esperada
cuenta como fallo en las estadísticas de Locust.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from locust import HttpUser, between, constant, task
from payloads import (
    DEFAULT_SEED,
    P2_BATCH_SIZE,
    make_rng,
    p2_batch,
    p2_record,
    p3_station,
)

SEED = int(os.environ.get("LOAD_SEED", str(DEFAULT_SEED)))
_USER_INDEX = itertools.count()

P2_PREDICT = "/v1/p2/predict"
P2_BATCH = "/v1/p2/predict/batch"
P3_PREDICT = "/predict"


def _check_json_key(response, key: str, expected_status: int = 200) -> None:
    """Marca la respuesta como éxito solo si el código y la clave esperada coinciden."""
    if response.status_code != expected_status:
        response.failure(f"HTTP {response.status_code}")
        return
    try:
        body = response.json()
    except ValueError:
        response.failure("respuesta no JSON")
        return
    if key not in body:
        response.failure(f"falta la clave {key}")
        return
    response.success()


def _check_batch(response, expected: int) -> None:
    if response.status_code != 200:
        response.failure(f"HTTP {response.status_code}")
        return
    try:
        body = response.json()
    except ValueError:
        response.failure("respuesta no JSON")
        return
    if body.get("count") != expected or len(body.get("predicciones", [])) != expected:
        response.failure("el lote no devolvió el número esperado de predicciones")
        return
    response.success()


class _MiBiciUser(HttpUser):
    """Base común: un generador de payloads determinista por usuario virtual."""

    abstract = True

    def on_start(self) -> None:
        self.rng = make_rng(SEED, next(_USER_INDEX))

    def _post_p2(self) -> None:
        payload = p2_record(self.rng)
        with self.client.post(
            P2_PREDICT, json=payload, name=f"POST {P2_PREDICT}", catch_response=True
        ) as response:
            _check_json_key(response, "riesgo_prob")

    def _post_p2_batch(self) -> None:
        payload = p2_batch(self.rng, P2_BATCH_SIZE)
        with self.client.post(
            P2_BATCH,
            json=payload,
            name=f"POST {P2_BATCH} (lote de {P2_BATCH_SIZE})",
            catch_response=True,
        ) as response:
            _check_batch(response, P2_BATCH_SIZE)


class MiBiciMixedUser(_MiBiciUser):
    """Mezcla operativa para S1, S2 y S3."""

    wait_time = between(0.5, 1.5)

    @task(60)
    def p2_predict(self) -> None:
        self._post_p2()

    @task(20)
    def p3_predict(self) -> None:
        payload = p3_station(self.rng)
        with self.client.post(
            P3_PREDICT, json=payload, name=f"POST {P3_PREDICT}", catch_response=True
        ) as response:
            _check_json_key(response, "cluster")

    @task(15)
    def p2_batch(self) -> None:
        self._post_p2_batch()

    @task(3)
    def health(self) -> None:
        with self.client.get("/health", name="GET /health", catch_response=True) as response:
            _check_json_key(response, "status")

    @task(2)
    def p2_ready(self) -> None:
        with self.client.get(
            "/v1/p2/ready", name="GET /v1/p2/ready", catch_response=True
        ) as response:
            _check_json_key(response, "ready")


class P2UnitUser(_MiBiciUser):
    """S4a: línea base unitaria, sin espera entre peticiones."""

    wait_time = constant(0)

    @task
    def predict(self) -> None:
        self._post_p2()


class P2BatchUser(_MiBiciUser):
    """S4b: lotes de 20 estaciones, sin espera entre peticiones."""

    wait_time = constant(0)

    @task
    def batch(self) -> None:
        self._post_p2_batch()
