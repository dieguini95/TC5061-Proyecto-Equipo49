"""Generadores de payloads para la prueba de carga de MiBici (Actividad 9).

Módulo sin dependencias de Locust: lo importan locustfile.py (en el generador de carga) y
tests/test_compose_and_loadtest.py (en CI). Los valores siguen los contratos de la API:
    - P2: mibici_ml.api.p2.RegistroP2 (14 variables, fin_de_semana coherente con dia_semana).
    - P3: mibici_ml.api.schemas.StationInput (capacidad y flujos no negativos).

Con la misma semilla y el mismo índice de usuario, los payloads son idénticos entre corridas.
"""

from __future__ import annotations

import random

DEFAULT_SEED = 42
P2_BATCH_SIZE = 20
P3_STATION_COUNT = 484  # estaciones del catálogo 2026-09 (references/nomenclatura_2026_09.csv)


def make_rng(seed: int, user_index: int) -> random.Random:
    """Generador determinista por usuario virtual."""
    return random.Random(seed * 100_003 + user_index)


def p2_record(rng: random.Random) -> dict[str, int | float]:
    """Estado de una estación en el instante t (13 a 14 variables de P2)."""
    capacidad = rng.randint(12, 40)
    dia_semana = rng.randint(0, 6)
    flujo_acumulado = rng.randint(-capacidad, capacidad)
    llegadas_1h = rng.randint(0, 10)
    salidas_1h = rng.randint(0, 10)
    return {
        "hora": rng.randint(0, 23),
        "dia_semana": dia_semana,
        "fin_de_semana": int(dia_semana >= 5),
        "capacidad": capacidad,
        "flujo_acumulado": flujo_acumulado,
        "flujo_acumulado_sobre_c": round(flujo_acumulado / capacidad, 4),
        "llegadas_1h": llegadas_1h,
        "salidas_1h": salidas_1h,
        "llegadas_2h": llegadas_1h + rng.randint(0, 10),
        "salidas_2h": salidas_1h + rng.randint(0, 10),
        "salidas_misma_hora_ayer": rng.randint(0, 12),
        # Clase positiva rara (~0.5 % en los datos): se simula con 1 %.
        "riesgo_observado_h": int(rng.random() < 0.01),
        "riesgo_observado_previo_h": int(rng.random() < 0.01),
        "tendencia_flujo_3h": round(rng.uniform(-0.5, 0.5), 4),
    }


def p2_batch(rng: random.Random, size: int = P2_BATCH_SIZE) -> dict[str, list]:
    """Lote de estaciones para /v1/p2/predict/batch."""
    return {"items": [p2_record(rng) for _ in range(size)]}


def p3_station(rng: random.Random) -> dict[str, int | float]:
    """Entrada de clustering P3 (StationInput)."""
    return {
        "station_id": rng.randint(1, P3_STATION_COUNT),
        "capacidad": float(rng.randint(12, 40)),
        "salidas_12m": round(rng.uniform(0.0, 5000.0), 1),
        "llegadas_12m": round(rng.uniform(0.0, 5000.0), 1),
    }
