from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Self
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest
from scripts import ci_smoke_serving as smoke


class FakeResponse:
    def __init__(self, body: object, status: int = 200) -> None:
        self.status = status
        self.raw_body = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self.raw_body


def _success_routes() -> dict[tuple[str, str], object]:
    return {
        ("GET", "/health"): {
            "status": "ok",
            "service": "mibici-p3-api",
            "version": "0.0.1",
        },
        ("GET", "/ready"): {
            "status": "ready",
            "model_name": "p3-model",
            "model_alias": "champion",
            "ready": True,
        },
        ("GET", "/v1/p2/ready"): {"status": "ready", "ready": True},
        ("POST", "/predict"): {
            "station_id": smoke.P3_STATION_ID,
            "cluster": 2,
            "model_name": "p3-model",
            "model_alias": "champion",
            "status": "ok",
        },
        ("POST", "/v1/p2/predict"): {
            "riesgo_prob": 0.25,
            "riesgo_alto": True,
            "umbral": 0.2,
            "modelo": "p2-model@champion",
            "version": "1",
        },
    }


def _install_fake_opener(monkeypatch, routes):
    calls = []

    def open_request(request, timeout):
        path = urlsplit(request.full_url).path
        method = request.get_method()
        calls.append((method, path, timeout, request.data))
        response = routes[(method, path)]
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            response = response(request, timeout)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, FakeResponse):
            return response
        return FakeResponse(response)

    opener = SimpleNamespace(open=open_request)
    monkeypatch.setattr(smoke, "build_opener", lambda *_args: opener)
    return calls


def test_smoke_checks_health_both_readiness_endpoints_and_p2_p3_predictions(monkeypatch, capsys):
    routes = _success_routes()
    p2_ready_response = routes[("GET", "/v1/p2/ready")]
    p2_ready_calls = 0

    def p2_ready_with_startup_retry(_request, _timeout):
        nonlocal p2_ready_calls
        p2_ready_calls += 1
        if p2_ready_calls == 1:
            return HTTPError(
                "http://127.0.0.1/v1/p2/ready",
                503,
                "loading",
                hdrs=None,
                fp=io.BytesIO(b"not included in output"),
            )
        return FakeResponse(p2_ready_response)

    routes[("GET", "/v1/p2/ready")] = p2_ready_with_startup_retry
    calls = _install_fake_opener(monkeypatch, routes)

    assert smoke.main(["--base-url", "http://127.0.0.1:8000", "--timeout", "2"]) == 0

    output = capsys.readouterr().out
    assert "OK /health" in output
    assert "OK /ready" in output
    assert "OK /v1/p2/ready" in output
    assert "OK /predict" in output
    assert "OK /v1/p2/predict" in output
    assert p2_ready_calls == 2
    assert ("POST", "/predict") in {(method, path) for method, path, _, _ in calls}
    assert ("POST", "/v1/p2/predict") in {(method, path) for method, path, _, _ in calls}
    p3_request = next(
        data for method, path, _, data in calls if method == "POST" and path == "/predict"
    )
    p2_request = next(
        data for method, path, _, data in calls if method == "POST" and path == "/v1/p2/predict"
    )
    assert json.loads(p3_request) == smoke.P3_PAYLOAD
    assert json.loads(p2_request) == smoke.P2_PAYLOAD


def test_http_error_returns_nonzero_without_printing_response_body(monkeypatch, capsys):
    _install_fake_opener(
        monkeypatch,
        {
            ("GET", "/health"): HTTPError(
                "http://127.0.0.1/health",
                404,
                "not found",
                hdrs=None,
                fp=io.BytesIO(b"secret response body"),
            )
        },
    )

    assert smoke.main(["--base-url", "http://127.0.0.1:8000", "--timeout", "1"]) == 1
    output = capsys.readouterr().err
    assert "/health returned HTTP 404" in output
    assert "secret response body" not in output


def test_invalid_json_is_reported_without_echoing_body(monkeypatch, capsys):
    _install_fake_opener(
        monkeypatch,
        {("GET", "/health"): FakeResponse(b"not-json")},
    )

    assert smoke.main(["--base-url", "http://127.0.0.1:8000", "--timeout", "1"]) == 1
    output = capsys.readouterr().err
    assert "/health returned invalid JSON" in output
    assert "not-json" not in output


def test_readiness_requires_model_fields(monkeypatch, capsys):
    routes = _success_routes()
    routes[("GET", "/ready")] = {"status": "ready", "ready": True}
    _install_fake_opener(monkeypatch, routes)

    assert smoke.main(["--base-url", "http://127.0.0.1:8000", "--timeout", "1"]) == 1
    assert "missing model_name" in capsys.readouterr().err


def test_readiness_requires_nullable_model_alias_key():
    with pytest.raises(smoke.SmokeTestError, match="missing or invalid model_alias"):
        smoke._validate_p3_ready(
            {"status": "ready", "ready": True, "model_name": "p3-model"},
            "/ready",
        )


def test_timeout_is_bounded_and_error_does_not_expose_exception_text(monkeypatch, capsys):
    clock = [0.0]
    monkeypatch.setattr(smoke.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(smoke.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    _install_fake_opener(
        monkeypatch,
        {("GET", "/health"): TimeoutError("sensitive connection detail")},
    )

    assert smoke.main(["--base-url", "http://127.0.0.1:8000", "--timeout", "1"]) == 1
    output = capsys.readouterr().err
    assert "did not become ready within 1s" in output
    assert "sensitive connection detail" not in output


def test_external_base_url_is_rejected_before_network_access():
    with pytest.raises(SystemExit) as error:
        smoke.main(["--base-url", "https://example.com"])
    assert error.value.code == 2
