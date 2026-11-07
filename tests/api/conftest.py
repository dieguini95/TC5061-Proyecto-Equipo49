from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import mlflow
from mlflow import MlflowClient
import pandas as pd
import pytest

from mibici_ml.api.app import create_app
from mibici_ml.api.providers import FakeModelProvider
from mibici_ml.serving.export_pyfunc import export_pyfunc_champion


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


# --- Registry temporal para pruebas del provider MLflow (nunca toca mlflow.db) ---

P3_REGISTRY_NAME = "mibici-equipo49-p3-grupos"


class _ClusterStub(mlflow.pyfunc.PythonModel):
    """Stand-in for P3ClusterModel: deterministic cluster = int(capacidad) % 3."""

    def predict(self, context, model_input, params=None):
        del context, params
        labels = model_input["capacidad"].astype(int) % 3
        return pd.DataFrame({"cluster": labels.to_numpy()}, index=model_input.index)


@dataclass(frozen=True)
class RegistryStore:
    tracking_uri: str
    name: str = P3_REGISTRY_NAME


@pytest.fixture
def registry_store(tmp_path: Path):
    """Temporary MLflow Registry with the stub registered and aliased as ``champion``."""
    previous = mlflow.get_tracking_uri()
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    try:
        experiment_id = mlflow.create_experiment(
            "api-tests", artifact_location=(tmp_path / "artifacts").as_uri()
        )
        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.pyfunc.log_model(
                name="model",
                python_model=_ClusterStub(),
                pip_requirements=["pandas"],
            )
            run_id = run.info.run_id
        version = mlflow.register_model(f"runs:/{run_id}/model", P3_REGISTRY_NAME)
        MlflowClient(tracking_uri=tracking_uri).set_registered_model_alias(
            P3_REGISTRY_NAME, "champion", version.version
        )
        yield RegistryStore(tracking_uri=tracking_uri)
    finally:
        mlflow.set_tracking_uri(previous)


@pytest.fixture
def exported_p3_dir(registry_store: RegistryStore, tmp_path: Path) -> Path:
    """Export the stub champion with the real exporter; returns ``.../p3/model``."""
    bundle_dir = export_pyfunc_champion(
        "p3",
        output_root=tmp_path / "serving",
        tracking_uri=registry_store.tracking_uri,
    )
    return bundle_dir / "model"
