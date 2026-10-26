"""Contratos de seleccion, MLflow Project y artefactos P1 en datos sinteticos."""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mibici_ml import config
from mibici_ml.modeling import search_p1, train


def test_selection_is_invariant_to_test_metrics():
    rows = [
        {"model_id": "a", "split": "val", "wape": .3, "mae": 1.0},
        {"model_id": "b", "split": "val", "wape": .4, "mae": .5},
        {"model_id": "b", "split": "test", "wape": 0, "mae": 0},
        {"model_id": "a", "split": "test", "wape": 99, "mae": 99},
    ]
    assert search_p1.choose_on_validation(rows) == "a"
    with pytest.raises(ValueError, match="WAPE"):
        search_p1.choose_on_validation([dict(rows[0], wape=float("nan"))])


def test_configuration_rejects_unknown_and_unsafe_overrides():
    with pytest.raises(ValueError):
        search_p1.configurations("nonexistent")
    with pytest.raises(ValueError):
        search_p1.configurations("", "max_iter=50")
    with pytest.raises(ValueError):
        search_p1.configurations("hist_poisson", "early_stopping=true")
    with pytest.raises(ValueError):
        search_p1.parse_tuned_params("alpha=1;alpha=2")
    assert search_p1.parse_tuned_params("max_iter=50;learning_rate=0.08") == {
        "max_iter": 50, "learning_rate": .08
    }
    specs = search_p1.configurations("hist_poisson", "max_iter=50")
    assert len(specs) == 3  # two original baselines plus explicit candidate
    assert specs[-1]["params"] == {"max_iter": 50}


def test_train_dispatch_preserves_p1_parameters(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(search_p1, "run_search", lambda **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(train.app, [
        "--question", "p1", "--model", "ridge", "--tuned-json", "alpha=2.0",
        "--output-dir", str(tmp_path), "--register-best", "true",
    ])
    assert result.exit_code == 0, result.output
    assert calls == [{"output_dir": tmp_path, "model": "ridge", "tuned_json": "alpha=2.0",
                      "register_best": True, "with_mlflow": True}]


def _panel():
    frames = []
    for start in (config.TRAIN_START, config.VAL_START, config.TEST_START):
        timestamps = pd.date_range(start, periods=48, freq="h")
        lag = (np.arange(48) % 9).astype(float)
        frames.append(pd.DataFrame({
            "station_id": 1, "timestamp": timestamps, "eligible": True,
            "evaluation_eligible": True, "departures": lag + 1,
            "lag_24": lag, "lag_48": lag, "lag_168": lag,
            "mean_4weeks": lag, "hour": timestamps.hour,
            "day_of_week": timestamps.dayofweek, "is_weekend": timestamps.dayofweek >= 5,
        }))
    return pd.concat(frames, ignore_index=True)


def test_shared_train_defaults_preserve_other_questions(monkeypatch):
    p1_calls, p2_calls, p3_calls = [], [], []
    monkeypatch.setattr(search_p1, "run_search", lambda **kw: p1_calls.append(kw))
    monkeypatch.setattr(train.risk, "run_command", lambda **kw: p2_calls.append(kw))
    monkeypatch.setattr(train.cluster, "run_command", lambda **kw: p3_calls.append(kw))
    runner = CliRunner()
    assert runner.invoke(train.app, ["--question", "p1"]).exit_code == 0
    assert runner.invoke(train.app, ["--question", "p2", "--model", "hist_gb"]).exit_code == 0
    assert runner.invoke(train.app, ["--question", "p3", "--final-k", "5"]).exit_code == 0
    assert p1_calls[0]["output_dir"] is None
    assert p2_calls[0]["output_dir"] == config.MLFLOW_TRAIN_OUTPUT_DIR
    assert p3_calls[0]["output_dir"] == config.MLFLOW_TRAIN_OUTPUT_DIR


def test_end_to_end_nested_models_artifacts_and_registry(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(config, "EXPERIMENT_NAME", "p1-etapa2-contract-equipo49")
    monkeypatch.setattr(config, "P1_REGISTRY_MODEL_NAME", "p1-contract-demanda")
    monkeypatch.setattr(config, "P1_CANDIDATE_GRID", {
        "ridge": [{"alpha": 1.0}],
        "hist_poisson": [{"max_iter": 2, "min_samples_leaf": 2}],
        "hist_squared": [{"max_iter": 2, "min_samples_leaf": 2}],
    })
    manifest = {"data_version": "synthetic-v1", "trips": [{"raw_rows": 200, "clean_rows": 200}]}
    monkeypatch.setattr(search_p1.demand, "verify_inputs", lambda *args: manifest)
    monkeypatch.setattr(search_p1, "build_station_hour_panel", lambda *args: _panel())
    result = search_p1.run_search(output_dir=tmp_path / "output", register_best=True)
    client = mlflow.MlflowClient(tracking_uri=uri)
    parent = client.get_run(result["parent_run_id"])
    assert parent.info.status == "FINISHED"
    assert len(result["run_ids"]) == 5
    for run_id in result["run_ids"].values():
        run = client.get_run(run_id)
        assert run.data.tags["mlflow.parentRunId"] == parent.info.run_id
        assert run.data.tags["question"] == "P1"
        assert run.data.params["data_version"] == "synthetic-v1"
        assert "val_wape" in run.data.metrics and "test_wape" in run.data.metrics
        assert len(run.inputs.dataset_inputs) == 2
        plots = {artifact.path for artifact in client.list_artifacts(run_id, "plots")}
        assert "plots/val_prediction_residuals.png" in plots
        assert "plots/test_prediction_residuals.png" in plots
        model_uri = f"runs:/{run_id}/model"
        loaded = mlflow.pyfunc.load_model(model_uri)
        assert loaded.metadata.signature is not None
        predicted = loaded.predict(search_p1.p1_model_features(_panel()).head(3))
        assert np.isfinite(predicted).all() and (predicted >= 0).all()
    selected = client.get_run(result["run_ids"][result["selected_model_id"]])
    assert selected.data.tags["selected_on_validation"] == "true"
    registry = result["registry"]
    alias = client.get_model_version_by_alias(registry["name"], registry["alias"])
    assert alias.run_id == selected.info.run_id
    selection = json.loads((Path(result["output_dir"]) / "config" / "selection.json").read_text())
    assert selection["test_used_for_selection"] is False
    assert all(row["split"] == "val" for row in selection["validation_results"])
