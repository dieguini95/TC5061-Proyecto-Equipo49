from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mibici_ml.api.app import create_app
from mibici_ml.api.providers import FakeModelProvider


@pytest.fixture
def provider() -> FakeModelProvider:
    return FakeModelProvider(initially_ready=True)


@pytest.fixture
def app(provider: FakeModelProvider) -> FastAPI:
    return create_app(provider=provider)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def unready_client() -> TestClient:
    class UnreadyProvider(FakeModelProvider):
        def load(self) -> None:
            return None

    app = create_app(provider=UnreadyProvider())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def request_body() -> dict[str, object]:
    return {
        "station_id": 123,
        "capacidad": 20.0,
        "salidas_12m": 1200.0,
        "llegadas_12m": 900.0,
    }


class RecordingProvider(FakeModelProvider):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(initially_ready=True, **kwargs)
        self.received_frames = []

    def predict(self, frame):
        self.received_frames.append(frame.copy(deep=True))
        return super().predict(frame)


@pytest.fixture
def recording_provider() -> RecordingProvider:
    return RecordingProvider()


@pytest.fixture
def recording_client(recording_provider: RecordingProvider) -> TestClient:
    with TestClient(create_app(provider=recording_provider)) as test_client:
        yield test_client
