"""P1 Registry contract, idempotency, and promotion failure checks."""

import tempfile

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from mibici_ml import config
from mibici_ml.modeling import registry_p1


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "P1_REGISTRY_MODEL_NAME", "test-p1-demanda", raising=False)
    monkeypatch.setattr(config, "P1_CHAMPION_ALIAS", "champion", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    previous_uri = mlflow.get_tracking_uri()
    tracking_uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment(
        "p1-registry-test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    yield client, tracking_uri, experiment_id
    mlflow.set_tracking_uri(previous_uri)


def _logged_model(registry, value=2.5, missing_lineage=None):
    _, _, experiment_id = registry
    frame = pd.DataFrame({"lag_168": [1.0, 2.0, 3.0], "hour": [8.0, 9.0, 10.0]})
    model = DummyRegressor(strategy="constant", constant=value).fit(frame, [1.0, 2.0, 3.0])
    expected = model.predict(frame)
    with mlflow.start_run(experiment_id=experiment_id) as run:
        params = {"data_version": "data-fixture-v1", "source_version": "source-fixture-v1"}
        tags = {"question": "P1", "git_commit": "a" * 40, "git_dirty": "false"}
        params.pop(missing_lineage, None)
        tags.pop(missing_lineage, None)
        mlflow.log_params(params)
        mlflow.set_tags(tags)
        mlflow.log_metric("val_wape", 0.5)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(frame, expected),
            input_example=frame,
            pip_requirements=[],
        )
        run_id = run.info.run_id
    return run_id, frame, expected


def test_registers_validates_loads_alias_and_reuses_version(registry):
    client, uri, _ = registry
    run_id, frame, expected = _logged_model(registry)
    first = registry_p1.register_champion(run_id, frame, expected, uri)
    second = registry_p1.register_champion(run_id, frame, expected, uri)
    assert first == second
    assert first == {
        "name": config.P1_REGISTRY_MODEL_NAME,
        "version": "1",
        "alias": "champion",
        "model_uri": f"models:/{config.P1_REGISTRY_MODEL_NAME}@champion",
        "run_id": run_id,
    }
    versions = client.search_model_versions(f"name = '{config.P1_REGISTRY_MODEL_NAME}'")
    assert len(versions) == 1
    stored = client.get_model_version(config.P1_REGISTRY_MODEL_NAME, first["version"])
    assert stored.run_id == run_id
    assert stored.tags["source_run_id"] == run_id
    assert stored.tags["git_commit"] == "a" * 40
    assert stored.tags["data_version"] == "data-fixture-v1"
    assert stored.tags["source_version"] == "source-fixture-v1"
    assert stored.tags["question"] == "P1"
    assert "medianoche" in stored.description
    assert "WAPE de validacion" in stored.description
    loaded = registry_p1.load_champion_model(uri)
    np.testing.assert_allclose(loaded.predict(frame), expected)


def test_new_champion_retains_previous_version_in_lineage(registry):
    client, uri, _ = registry
    first = _logged_model(registry, 2.0)
    registry_p1.register_champion(*first, tracking_uri=uri)
    second = _logged_model(registry, 4.0)
    promoted = registry_p1.register_champion(*second, tracking_uri=uri)
    version = client.get_model_version(config.P1_REGISTRY_MODEL_NAME, promoted["version"])
    assert promoted["version"] == "2"
    assert version.tags["previous_champion_version"] == "1"


def test_version_validation_failure_keeps_previous_alias(registry, monkeypatch):
    client, uri, _ = registry
    registry_p1.register_champion(*_logged_model(registry, 2.0), tracking_uri=uri)
    second = _logged_model(registry, 4.0)
    original = registry_p1.validate_model_inference

    def reject_new_version(model_uri, *args, **kwargs):
        if model_uri == f"models:/{config.P1_REGISTRY_MODEL_NAME}/2":
            raise ValueError("forced version validation failure")
        return original(model_uri, *args, **kwargs)

    monkeypatch.setattr(registry_p1, "validate_model_inference", reject_new_version)
    with pytest.raises(ValueError, match="forced version"):
        registry_p1.register_champion(*second, tracking_uri=uri)
    current = client.get_model_version_by_alias(config.P1_REGISTRY_MODEL_NAME, "champion")
    assert str(current.version) == "1"


def test_alias_validation_failure_rolls_back_previous_alias(registry, monkeypatch):
    client, uri, _ = registry
    registry_p1.register_champion(*_logged_model(registry, 2.0), tracking_uri=uri)
    second = _logged_model(registry, 4.0)
    original = registry_p1.validate_model_inference

    def reject_promoted_alias(model_uri, *args, **kwargs):
        if model_uri.endswith("@champion"):
            raise ValueError("forced alias validation failure")
        return original(model_uri, *args, **kwargs)

    monkeypatch.setattr(registry_p1, "validate_model_inference", reject_promoted_alias)
    with pytest.raises(ValueError, match="forced alias"):
        registry_p1.register_champion(*second, tracking_uri=uri)
    current = client.get_model_version_by_alias(config.P1_REGISTRY_MODEL_NAME, "champion")
    assert str(current.version) == "1"
    failed = client.get_model_version(config.P1_REGISTRY_MODEL_NAME, "2")
    assert failed.tags["stage_of_work"] == "validated_candidate"


def test_failed_first_alias_validation_removes_unvalidated_alias(registry, monkeypatch):
    client, uri, _ = registry
    source = _logged_model(registry)
    original = registry_p1.validate_model_inference

    def reject_alias(model_uri, *args, **kwargs):
        if model_uri.endswith("@champion"):
            raise ValueError("forced alias validation failure")
        return original(model_uri, *args, **kwargs)

    monkeypatch.setattr(registry_p1, "validate_model_inference", reject_alias)
    with pytest.raises(ValueError, match="forced alias"):
        registry_p1.register_champion(*source, tracking_uri=uri)
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias(config.P1_REGISTRY_MODEL_NAME, "champion")


@pytest.mark.parametrize("missing", ["git_commit", "data_version", "source_version"])
def test_missing_lineage_is_rejected_before_registration(registry, missing):
    client, uri, _ = registry
    source = _logged_model(registry, missing_lineage=missing)
    with pytest.raises(ValueError, match=missing):
        registry_p1.register_champion(*source, tracking_uri=uri)
    assert not client.search_model_versions(f"name = '{config.P1_REGISTRY_MODEL_NAME}'")


def test_unfinished_run_is_rejected_before_loading_or_registering(registry):
    client, uri, experiment_id = registry
    run = client.create_run(experiment_id, tags={"question": "P1"})
    with pytest.raises(ValueError, match="FINISHED"):
        registry_p1.register_champion(
            run.info.run_id, pd.DataFrame({"hour": [8.0]}), np.array([1.0]), uri
        )
    assert not client.search_model_versions(f"name = '{config.P1_REGISTRY_MODEL_NAME}'")


@pytest.mark.parametrize(
    "values,message",
    [([np.nan, 1.0], "finitas"), ([-0.1, 1.0], "negativas"), ([[1.0], [2.0]], "vector")],
)
def test_invalid_inference_output_is_rejected(monkeypatch, values, message):
    class Invalid:
        def predict(self, frame):
            return np.asarray(values)

    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: Invalid())
    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: Invalid())
    with pytest.raises(ValueError, match=message):
        registry_p1.validate_model_inference(
            "models:/fixture/1", pd.DataFrame({"hour": [8.0, 9.0]}), np.array([1.0, 2.0])
        )


def test_pyfunc_must_match_evaluated_native_predictions(monkeypatch):
    class Native:
        def predict(self, frame):
            return np.array([1.0, 2.0])

    class IncorrectPyfunc:
        def predict(self, frame):
            return np.array([0.0, 1.0])

    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: Native())
    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: IncorrectPyfunc())
    with pytest.raises(ValueError, match="pyfunc.*no coincide"):
        registry_p1.validate_model_inference(
            "models:/fixture/1", pd.DataFrame({"hour": [8.0, 9.0]}), np.array([1.0, 2.0])
        )
