from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytest
import yaml

from mibici_ml.config import (
    P3_BASELINE_K,
    P3_CANDIDATE_K,
    P3_CHAMPION_ALIAS,
    P3_INPUT_PATH,
    P3_MODEL_FEATURES,
    P3_REGISTRY_MODEL_NAME,
)
from mibici_ml.modeling import cluster
from mibici_ml.modeling.p3_model import P3ClusterModel, log_p3_pyfunc_model, p3_input_example
from mibici_ml.modeling.search_p3 import run_search, select_best_candidate


def sample_stations() -> pd.DataFrame:
    group = np.repeat(np.arange(4), 10)
    within = np.tile(np.arange(10), 4)
    return pd.DataFrame(
        {
            "est": np.arange(1, 41),
            "nombre": [f"Estacion {station}" for station in range(1, 41)],
            "lat": 20.6 + np.arange(40) / 10000,
            "lon": -103.4 + np.arange(40) / 10000,
            "capacidad": np.where(np.arange(40) == 0, 0, (group + 1) * 8 + within),
            "salidas_12m": np.expm1(group * 1.5 + within * 0.015).round(3),
            "llegadas_12m": np.expm1(group * 1.4 + within * 0.02).round(3),
            "pct_dias_no_factibles": np.nan,
        }
    )


def temporary_experiment(tmp_path, experiment_name: str) -> str:
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    client = MlflowClient(tracking_uri=tracking_uri)
    client.create_experiment(
        experiment_name,
        artifact_location=(tmp_path / "artifacts" / experiment_name).as_uri(),
    )
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def test_p3_configuration_is_centralized():
    assert P3_BASELINE_K == 5
    assert P3_CANDIDATE_K == (3, 4, 6, 7, 8)
    assert P3_MODEL_FEATURES == ("capacidad", "salidas_12m", "llegadas_12m")
    assert P3_REGISTRY_MODEL_NAME == "mibici-equipo49-p3-grupos"
    assert P3_CHAMPION_ALIAS == "champion"


def test_p3_model_accepts_raw_features_and_imputes_zero_capacity():
    model = P3ClusterModel.fit(sample_stations(), selected_k=3, random_seed=42, n_init=10)
    inputs = pd.DataFrame(
        {
            "capacidad": [0.0, 20.0],
            "salidas_12m": [10.0, 200.0],
            "llegadas_12m": [20.0, 220.0],
        }
    )
    predictions = model.predict(None, inputs)
    assert predictions["cluster"].between(0, 2).all()
    assert model._transform_input(inputs).loc[0, "capacidad_modelo"] == model.positive_capacity_median


@pytest.mark.parametrize(
    "inputs, message",
    [
        (pd.DataFrame({"capacidad": [1], "salidas_12m": [2]}), "Faltan columnas"),
        (
            pd.DataFrame({"capacidad": [-1], "salidas_12m": [2], "llegadas_12m": [3]}),
            "negativos",
        ),
        (
            pd.DataFrame({"capacidad": [1], "salidas_12m": [-2], "llegadas_12m": [3]}),
            "negativos",
        ),
        (
            pd.DataFrame({"capacidad": [1], "salidas_12m": [2], "llegadas_12m": [np.nan]}),
            "no finitos",
        ),
        (
            pd.DataFrame({"capacidad": [np.inf], "salidas_12m": [2], "llegadas_12m": [3]}),
            "no finitos",
        ),
    ],
)
def test_p3_model_rejects_invalid_inference_inputs(inputs, message):
    model = P3ClusterModel.fit(sample_stations(), selected_k=3, random_seed=42, n_init=10)
    with pytest.raises(ValueError, match=message):
        model.predict(None, inputs)


def test_p3_pyfunc_signature_example_and_loaded_prediction_round_trip(tmp_path):
    tracking_uri = temporary_experiment(tmp_path, "p3-model-roundtrip")
    mlflow.set_experiment("p3-model-roundtrip")
    model = P3ClusterModel.fit(sample_stations(), selected_k=3, random_seed=42, n_init=10)
    example = p3_input_example()
    with mlflow.start_run():
        info = log_p3_pyfunc_model(model, name="model")

    loaded = mlflow.pyfunc.load_model(info.model_uri)
    assert loaded.metadata.signature.inputs is not None
    assert loaded.metadata.signature.outputs is not None
    assert loaded.metadata.signature.inputs.to_dict() == [
        {"type": "double", "name": "capacidad", "required": True},
        {"type": "double", "name": "salidas_12m", "required": True},
        {"type": "double", "name": "llegadas_12m", "required": True},
    ]
    pd.testing.assert_frame_equal(loaded.predict(example), model.predict(None, example))
    assert tracking_uri.startswith("sqlite:///")


def test_nested_search_tracks_five_children_and_is_reproducible(tmp_path):
    input_path = tmp_path / "stations.csv"
    sample_stations().to_csv(input_path, index=False)
    tracking_uri = temporary_experiment(tmp_path, "p3-search-one")
    first = run_search(
        input_path=input_path,
        output_dir=tmp_path / "output-one",
        candidate_k=P3_CANDIDATE_K,
        random_seed=42,
        n_init=10,
        experiment_name="p3-search-one",
        tracking_uri=tracking_uri,
        register_best=False,
        registry_model_name=P3_REGISTRY_MODEL_NAME,
        champion_alias=P3_CHAMPION_ALIAS,
    )
    temporary_experiment(tmp_path, "p3-search-two")
    second = run_search(
        input_path=input_path,
        output_dir=tmp_path / "output-two",
        candidate_k=P3_CANDIDATE_K,
        random_seed=42,
        n_init=10,
        experiment_name="p3-search-two",
        tracking_uri=tracking_uri,
        register_best=False,
        registry_model_name=P3_REGISTRY_MODEL_NAME,
        champion_alias=P3_CHAMPION_ALIAS,
    )

    assert len(first.child_run_ids) == 6
    assert len(first.candidate_run_ids) == 5
    assert first.recommended_k == second.recommended_k
    assert first.input_sha256 == second.input_sha256
    assert first.data_version == f"sha256:{first.input_sha256}"
    client = MlflowClient(tracking_uri=tracking_uri)
    parent = client.get_run(first.parent_run_id)
    assert parent.data.tags["team"] == "equipo49"
    assert parent.data.tags["question"] == "P3"
    assert parent.data.tags["model_family"] == "kmeans"
    assert parent.data.tags["stage_of_work"] == "search"
    assert parent.data.tags["baseline_k"] == "5"
    assert parent.data.tags["input_sha256"] == first.input_sha256
    assert parent.data.tags["data_version"] == first.data_version
    assert parent.data.tags["best_candidate_run_id"] == first.best_candidate_run_id
    assert parent.data.tags["recommended_run_id"] == first.recommended_run_id
    assert parent.data.tags["selected_run_id"] == first.selected_run_id
    assert parent.data.tags["git_commit"]
    assert parent.data.metrics["best_candidate_k"] == first.best_candidate_k
    assert parent.data.metrics["recommended_k"] == first.recommended_k
    assert len(parent.inputs.dataset_inputs) == 0

    for k, run_id in first.child_run_ids.items():
        child = client.get_run(run_id)
        assert child.data.tags["mlflow.parentRunId"] == first.parent_run_id
        assert child.data.tags["team"] == "equipo49"
        assert child.data.tags["question"] == "P3"
        assert child.data.tags["model_family"] == "kmeans"
        is_baseline = k == P3_BASELINE_K
        assert child.data.tags["stage_of_work"] == (
            "baseline" if is_baseline else "candidate"
        )
        assert child.data.tags["data_scope"] == "eda_aggregate_full_12m"
        assert child.data.tags["temporal_holdout"] == "false"
        assert child.data.tags["baseline_k" if is_baseline else "candidate_k"] == str(k)
        assert child.data.tags["git_commit"]
        assert child.data.tags["parent_search"] == first.parent_run_id
        assert child.data.tags["input_sha256"] == first.input_sha256
        assert child.data.tags["data_version"] == first.data_version
        assert child.data.params["k"] == str(k)
        assert child.data.params["n_init"] == "10"
        assert child.data.params["feature_names"]
        assert child.data.params["transform"] == "log1p_counts"
        assert child.data.params["scaler"] == "StandardScaler"
        assert child.data.params["capacity_zero_policy"] == "median_positive_imputation"
        assert child.data.params["positive_capacity_median"]
        assert child.data.params["input_path"]
        assert child.data.params["period"]
        assert np.isfinite(list(child.data.metrics.values())).all()
        assert len(child.inputs.dataset_inputs) == 1
        dataset_source = child.inputs.dataset_inputs[0].dataset.source
        source_uri = json.loads(dataset_source)["uri"]
        assert input_path.name in source_uri
        assert not Path(source_uri).is_absolute()
        second_child = client.get_run(second.child_run_ids[k])
        assert child.data.metrics["silhouette_score"] == second_child.data.metrics["silhouette_score"]
        assert child.data.metrics["inertia"] == pytest.approx(
            second_child.data.metrics["inertia"]
        )
        assert child.data.tags["mlflow.runName"] == (
            "P3-kmeans-k5-baseline"
            if is_baseline
            else f"P3-kmeans-k{k}-candidate"
        )
        artifact_paths = {
            artifact.path for artifact in client.list_artifacts(run_id)
        }
        assert {"plots", "data", "config"}.issubset(artifact_paths)
        plot_paths = {
            artifact.path for artifact in client.list_artifacts(run_id, "plots")
        }
        assert {
            "plots/p3_mapa_clusters.png",
            "plots/p3_perfiles_cluster.png",
        }.issubset(plot_paths)
        data_paths = {artifact.path for artifact in client.list_artifacts(run_id, "data")}
        assert {
            "data/p3_input_manifest.json",
            "data/p3_metricas_candidato.csv",
            "data/p3_perfiles_cluster.csv",
            "data/p3_calidad_datos.csv",
        }.issubset(data_paths)
        config_paths = {
            artifact.path for artifact in client.list_artifacts(run_id, "config")
        }
        assert "config/p3_run_config.json" in config_paths
        assert not any("estaciones_resumen.csv" in path for path in data_paths)
        client.download_artifacts(run_id, "data/p3_input_manifest.json", str(tmp_path))
        manifest_file = tmp_path / "p3_input_manifest.json"
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert manifest["sha256"] == first.input_sha256
        assert manifest["row_count"] == 40
        assert manifest["column_names"] == sample_stations().columns.tolist()
        assert manifest["contains_personal_data"] is False
        assert "rows" not in manifest
        assert all(artifact.file_size > 0 for artifact in client.list_artifacts(run_id, "plots"))
        assert all(artifact.file_size > 0 for artifact in client.list_artifacts(run_id, "data"))

    client.download_artifacts(
        first.parent_run_id,
        "data/p3_comparacion_candidatos.csv",
        str(tmp_path),
    )
    comparison = pd.read_csv(tmp_path / "p3_comparacion_candidatos.csv")
    assert len(comparison) == 6
    assert comparison["is_baseline"].sum() == 1
    assert set(comparison["k"]) == {3, 4, 5, 6, 7, 8}
    assert set(comparison["run_id"]) == set(first.child_run_ids.values())


def test_candidate_selection_uses_minimum_size_then_tie_breaks():
    metrics = pd.DataFrame(
        [
            {"k": 3, "silhouette_score": 0.6, "davies_bouldin_score": 0.7, "min_cluster_size": 1},
            {"k": 4, "silhouette_score": 0.5, "davies_bouldin_score": 0.6, "min_cluster_size": 2},
            {"k": 6, "silhouette_score": 0.5, "davies_bouldin_score": 0.6, "min_cluster_size": 2},
        ]
    )
    assert int(select_best_candidate(metrics)["k"]) == 4


@pytest.mark.parametrize(
    ("baseline_silhouette", "candidate_silhouette", "expected_k"),
    [(0.8, 0.5, 5), (0.4, 0.7, 3)],
)
def test_global_recommendation_compares_baseline_and_candidate(
    baseline_silhouette,
    candidate_silhouette,
    expected_k,
):
    metrics = pd.DataFrame(
        [
            {
                "k": 5,
                "silhouette_score": baseline_silhouette,
                "davies_bouldin_score": 0.5,
                "min_cluster_size": 2,
                "is_baseline": True,
            },
            {
                "k": 3,
                "silhouette_score": candidate_silhouette,
                "davies_bouldin_score": 0.6,
                "min_cluster_size": 2,
                "is_baseline": False,
            },
        ]
    )
    assert int(select_best_candidate(metrics)["k"]) == expected_k


def test_manual_override_selects_candidate_and_run_id_even_when_baseline_recommended(
    tmp_path,
    monkeypatch,
):
    original_evaluate = cluster.evaluate_kmeans_candidate

    def controlled_evaluate(features, k, random_seed, n_init):
        metrics, pipeline, labels = original_evaluate(features, k, random_seed, n_init)
        metrics["silhouette_score"] = 0.9 if k == P3_BASELINE_K else 0.4
        metrics["min_cluster_size"] = max(metrics["min_cluster_size"], 2)
        return metrics, pipeline, labels

    monkeypatch.setattr(cluster, "evaluate_kmeans_candidate", controlled_evaluate)
    input_path = tmp_path / "stations.csv"
    sample_stations().to_csv(input_path, index=False)
    tracking_uri = temporary_experiment(tmp_path, "p3-search-override")
    result = run_search(
        input_path=input_path,
        output_dir=tmp_path / "override-output",
        candidate_k=(3,),
        random_seed=42,
        n_init=10,
        experiment_name="p3-search-override",
        tracking_uri=tracking_uri,
        register_best=False,
        registry_model_name=P3_REGISTRY_MODEL_NAME,
        champion_alias=P3_CHAMPION_ALIAS,
        selected_k_override=3,
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    assert result.recommended_k == P3_BASELINE_K
    assert result.recommended_run_id == result.child_run_ids[P3_BASELINE_K]
    assert result.selected_k == 3
    assert result.selected_run_id == result.child_run_ids[3]
    assert result.best_candidate_k == 3
    assert result.best_candidate_run_id == result.candidate_run_ids[3]
    parent = client.get_run(result.parent_run_id)
    assert parent.data.tags["selected_run_id"] == result.selected_run_id


def test_selected_baseline_can_be_registered_and_promoted(tmp_path, monkeypatch):
    original_evaluate = cluster.evaluate_kmeans_candidate

    def baseline_wins(features, k, random_seed, n_init):
        metrics, pipeline, labels = original_evaluate(features, k, random_seed, n_init)
        metrics["silhouette_score"] = 0.9 if k == P3_BASELINE_K else 0.4
        metrics["min_cluster_size"] = max(metrics["min_cluster_size"], 2)
        return metrics, pipeline, labels

    monkeypatch.setattr(cluster, "evaluate_kmeans_candidate", baseline_wins)
    input_path = tmp_path / "stations.csv"
    sample_stations().to_csv(input_path, index=False)
    tracking_uri = temporary_experiment(tmp_path, "p3-search-register-baseline")
    result = run_search(
        input_path=input_path,
        output_dir=tmp_path / "registered-baseline-output",
        candidate_k=(3,),
        random_seed=42,
        n_init=10,
        experiment_name="p3-search-register-baseline",
        tracking_uri=tracking_uri,
        register_best=True,
        registry_model_name=P3_REGISTRY_MODEL_NAME,
        champion_alias=P3_CHAMPION_ALIAS,
        selected_k_override=P3_BASELINE_K,
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    version = client.get_model_version(P3_REGISTRY_MODEL_NAME, result.model_version)
    alias_version = client.get_model_version_by_alias(
        P3_REGISTRY_MODEL_NAME,
        P3_CHAMPION_ALIAS,
    )

    assert result.recommended_k == P3_BASELINE_K
    assert result.selected_k == P3_BASELINE_K
    assert result.selected_run_id == result.child_run_ids[P3_BASELINE_K]
    assert version.run_id == result.selected_run_id
    assert version.tags["source_run_id"] == result.selected_run_id
    assert version.tags["parent_search_run_id"] == result.parent_run_id
    assert version.tags["stage_of_work"] == "champion"
    assert str(alias_version.version) == result.model_version


def test_registration_requires_an_explicit_reviewed_selection(tmp_path):
    with pytest.raises(ValueError, match="Revise la comparacion"):
        run_search(
            input_path=tmp_path / "not-read.csv",
            output_dir=tmp_path / "unused",
            candidate_k=(3,),
            random_seed=42,
            n_init=10,
            experiment_name="unused",
            tracking_uri="sqlite://",
            register_best=True,
            registry_model_name=P3_REGISTRY_MODEL_NAME,
            champion_alias=P3_CHAMPION_ALIAS,
        )


def test_mlproject_has_additive_p3_search_and_preserves_p2_search():
    project = yaml.safe_load((P3_INPUT_PATH.parents[2] / "MLproject").read_text(encoding="utf-8"))
    entries = project["entry_points"]
    assert entries["search"]["command"] == "python -m mibici_ml.modeling.search"
    assert entries["search"]["parameters"] == {}
    p3_search = entries["search_p3"]
    assert "python -m mibici_ml.modeling.search_p3 --input-path" in p3_search["command"]
    assert p3_search["parameters"]["register_best"]["default"] == "false"
    assert p3_search["parameters"]["output_dir"]["default"] == ""
    assert set(P3_CANDIDATE_K) == {3, 4, 6, 7, 8}
    assert P3_BASELINE_K == 5
