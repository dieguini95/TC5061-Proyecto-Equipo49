"""Pruebas del despachador de entrenamiento y de la busqueda anidada (sin datos reales)."""

import pytest
import typer
from typer.testing import CliRunner

from mibici_ml import config
from mibici_ml.modeling import search, train


def test_grid_combinations_match_config_grid():
    # El numero esperado sale del producto cartesiano del grid, no de ejes fijos.
    for model_name, grid in config.P2_SEARCH_GRID.items():
        combos = search.grid_combinations(model_name)
        expected = 1
        for values in grid.values():
            expected *= len(values)
        assert len(combos) == expected
        assert all(set(c) == set(grid) for c in combos)


def test_build_model_applies_search_hyperparameters():
    model = search.risk.build_model("hist_gb", 42, {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15})
    assert model.learning_rate == 0.05
    assert model.max_iter == 300
    assert model.max_leaf_nodes == 15


def test_build_model_logistic_applies_c_without_changing_defaults():
    pipeline = search.risk.build_model("logistica", 42, {"C": 0.1})
    assert pipeline.named_steps["clf"].C == 0.1
    assert pipeline.named_steps["clf"].class_weight == "balanced"
    assert search.risk.build_model("logistica", 42).named_steps["clf"].C == 1.0


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        search.risk.build_model("no_existe", 42)


def test_train_rejects_unknown_question(tmp_path):
    runner = CliRunner()
    result = runner.invoke(train.app, ["--question", "p9", "--output-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "p1, p2, p3" in result.output


def test_train_p2_rejects_model_outside_candidates(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        train.app, ["--question", "p2", "--model", "mayoritaria", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code != 0
    assert "logistica, hist_gb" in result.output


def test_train_p3_requires_final_k(tmp_path):
    runner = CliRunner()
    result = runner.invoke(train.app, ["--question", "p3", "--output-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "--final-k" in result.output


def test_registered_name_uses_config_prefix():
    assert config.P2_REGISTERED_MODEL_PREFIX == "mibici-equipo49-p2-riesgo"


def test_tracking_uri_defaults_to_sqlite_project_db(monkeypatch):
    import importlib

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    reloaded = importlib.reload(config)
    try:
        assert reloaded.MLFLOW_TRACKING_URI.startswith("sqlite:///")
        assert reloaded.MLFLOW_TRACKING_URI.endswith("/mlflow.db")
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_release_wrapper_run_ends_active_run_and_env_var(tmp_path, monkeypatch):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("wrapper-release-test")
    wrapper = mlflow.start_run(run_name="wrapper")
    monkeypatch.setenv("MLFLOW_RUN_ID", wrapper.info.run_id)
    train._release_wrapper_run()
    assert mlflow.active_run() is None
    assert "MLFLOW_RUN_ID" not in __import__("os").environ


def test_release_wrapper_run_is_noop_without_active_run(monkeypatch):
    import mlflow

    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    assert mlflow.active_run() is None
    train._release_wrapper_run()
    assert mlflow.active_run() is None


def test_as_bool_parses_mlflow_string_flags():
    assert train._as_bool("true") is True
    assert train._as_bool("True") is True
    assert train._as_bool("false") is False
    assert train._as_bool("") is False
