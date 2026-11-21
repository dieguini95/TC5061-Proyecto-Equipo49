#!/usr/bin/env python3
"""Smoke-test the local P2/P3 serving container using only the standard library."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import math
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

POLL_INTERVAL_SECONDS = 0.5
MAX_READINESS_REQUEST_SECONDS = 5.0
P3_STATION_ID = 123
P3_PAYLOAD = {
    "station_id": P3_STATION_ID,
    "capacidad": 20.0,
    "salidas_12m": 1200.0,
    "llegadas_12m": 900.0,
}
P2_PAYLOAD = {
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


class SmokeTestError(Exception):
    """A safe, user-facing smoke-test failure."""


class _RetryableSmokeTestError(SmokeTestError):
    """A connection or server error that may clear during model startup."""


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def _loopback_base_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as error:
        raise argparse.ArgumentTypeError("base URL is invalid") from error
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise argparse.ArgumentTypeError(
            "base URL must be a local HTTP loopback address without credentials or path"
        )
    return value.rstrip("/")


def _positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be a positive number") from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive number")
    return timeout


def _request_json(
    base_url: str,
    path: str,
    timeout: float,
    *,
    payload: dict[str, int | float | str] | None = None,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"

    request = Request(f"{base_url}{path}", data=data, headers=headers, method=method)
    try:
        with build_opener(_NoRedirectHandler).open(request, timeout=timeout) as response:
            status = response.getcode()
            raw_body = response.read()
    except HTTPError as error:
        if 500 <= error.code < 600:
            raise _RetryableSmokeTestError(f"{path} returned HTTP {error.code}") from None
        raise SmokeTestError(f"{path} returned HTTP {error.code}") from None
    except TimeoutError:
        raise _RetryableSmokeTestError(f"{path} request timed out") from None
    except URLError:
        raise _RetryableSmokeTestError(f"could not connect to {path}") from None
    except OSError:
        raise _RetryableSmokeTestError(f"could not connect to {path}") from None

    if not 200 <= status < 300:
        if 500 <= status < 600:
            raise _RetryableSmokeTestError(f"{path} returned HTTP {status}")
        raise SmokeTestError(f"{path} returned HTTP {status}")
    try:
        body = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SmokeTestError(f"{path} returned invalid JSON") from None
    if not isinstance(body, dict):
        raise SmokeTestError(f"{path} returned JSON that is not an object")
    return body


def _require(condition: bool, path: str, message: str) -> None:
    if not condition:
        raise SmokeTestError(f"{path} returned an invalid response: {message}")


def _validate_health(body: dict[str, Any], path: str) -> None:
    _require(body.get("status") == "ok", path, "missing or invalid status")
    _require(
        isinstance(body.get("service"), str) and bool(body["service"]),
        path,
        "missing service",
    )
    _require(
        isinstance(body.get("version"), str) and bool(body["version"]),
        path,
        "missing version",
    )


def _validate_p3_ready(body: dict[str, Any], path: str) -> None:
    _require(body.get("status") == "ready", path, "missing or invalid status")
    _require(body.get("ready") is True, path, "ready must be true")
    _require(
        isinstance(body.get("model_name"), str) and bool(body["model_name"]),
        path,
        "missing model_name",
    )
    _require(
        "model_alias" in body
        and (body["model_alias"] is None or isinstance(body["model_alias"], str)),
        path,
        "missing or invalid model_alias",
    )


def _validate_p2_ready(body: dict[str, Any], path: str) -> None:
    _require(body.get("status") == "ready", path, "missing or invalid status")
    _require(body.get("ready") is True, path, "ready must be true")


def _validate_p3_prediction(body: dict[str, Any], path: str) -> None:
    _require(body.get("status") == "ok", path, "missing or invalid status")
    _require(body.get("station_id") == P3_STATION_ID, path, "station_id does not match")
    _require(type(body.get("cluster")) is int, path, "cluster must be an integer")
    _require(
        isinstance(body.get("model_name"), str) and bool(body["model_name"]),
        path,
        "missing model_name",
    )
    _require(
        "model_alias" in body
        and (body["model_alias"] is None or isinstance(body["model_alias"], str)),
        path,
        "missing or invalid model_alias",
    )


def _validate_p2_prediction(body: dict[str, Any], path: str) -> None:
    risk_probability = body.get("riesgo_prob")
    threshold = body.get("umbral")
    _require(
        isinstance(risk_probability, (int, float))
        and not isinstance(risk_probability, bool)
        and math.isfinite(risk_probability)
        and 0 <= risk_probability <= 1,
        path,
        "riesgo_prob must be between 0 and 1",
    )
    _require(type(body.get("riesgo_alto")) is bool, path, "riesgo_alto must be boolean")
    _require(
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and math.isfinite(threshold)
        and 0 <= threshold <= 1,
        path,
        "umbral must be between 0 and 1",
    )
    _require(
        body["riesgo_alto"] == (risk_probability >= threshold),
        path,
        "riesgo_alto does not match riesgo_prob and umbral",
    )
    _require(
        isinstance(body.get("modelo"), str) and bool(body["modelo"]),
        path,
        "missing modelo",
    )
    _require(
        isinstance(body.get("version"), str) and bool(body["version"]),
        path,
        "missing version",
    )


def _wait_for_endpoint(
    base_url: str,
    path: str,
    validator: Callable[[dict[str, Any], str], None],
    deadline: float,
    timeout: float,
) -> None:
    last_error = "no response"
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SmokeTestError(f"{path} did not become ready within {timeout:g}s ({last_error})")
        try:
            body = _request_json(
                base_url,
                path,
                min(remaining, MAX_READINESS_REQUEST_SECONDS),
            )
        except _RetryableSmokeTestError as error:
            last_error = str(error)
        else:
            validator(body, path)
            return

        sleep_seconds = min(POLL_INTERVAL_SECONDS, max(0.0, deadline - time.monotonic()))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def run_smoke(base_url: str, timeout: float) -> None:
    readiness_deadline = time.monotonic() + timeout
    checks = (
        ("/health", _validate_health),
        ("/ready", _validate_p3_ready),
        ("/v1/p2/ready", _validate_p2_ready),
    )
    for path, validator in checks:
        _wait_for_endpoint(base_url, path, validator, readiness_deadline, timeout)
        print(f"OK {path}")

    p3_path = "/predict"
    p3_result = _request_json(base_url, p3_path, timeout, payload=P3_PAYLOAD)
    _validate_p3_prediction(p3_result, p3_path)
    print(f"OK {p3_path}")

    p2_path = "/v1/p2/predict"
    p2_result = _request_json(base_url, p2_path, timeout, payload=P2_PAYLOAD)
    _validate_p2_prediction(p2_result, p2_path)
    print(f"OK {p2_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, type=_loopback_base_url)
    parser.add_argument("--timeout", type=_positive_timeout, default=120.0)
    args = parser.parse_args(argv)
    try:
        run_smoke(args.base_url, args.timeout)
    except SmokeTestError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
