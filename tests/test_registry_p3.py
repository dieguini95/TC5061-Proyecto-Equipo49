from __future__ import annotations

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd

from mibici_ml.config import P3_CHAMPION_ALIAS, P3_REGISTRY_MODEL_NAME
from mibici_ml.modeling.p3_model import P3ClusterModel, log_p3_pyfunc_model, p3_input_example
from mibici_ml.modeling.registry_p3 import (
    assign_champion_alias,
    load_champion_model,
    p3_model_description,
    register_p3_model_from_run,
    validate_champion_inference,
    validate_model_inference,
)


def sample_stations() -> pd.DataFrame:
    group = np.repeat(np.arange(4), 8)
    within = np.tile(np.arange(8), 4)
    return pd.DataFrame(
        {
            "est": np.arange(1, 33),
            "nombre": [f"Estacion {station}" for station in range(1, 33)],
            "lat": 20.6 + np.arange(32) / 10000,
            "lon": -103.4 + np.arange(32) / 10000,
            "capacidad": np.where(np.arange(32) == 0, 0, (group + 1) * 8 + within),
            "salidas_12m": np.expm1(group * 1.5 + within * 0.015).round(3),
            "llegadas_12m": np.expm1(group * 1.4 + within * 0.02).round(3),
            "pct_dias_no_factibles": np.nan,
        }
    )


def test_registry_version_is_validated_before_champion_assignment(tmp_path):
    tracking_uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    client = MlflowClient(tracking_uri=tracking_uri)
    client.create_experiment(
        "p3-registry-test",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("p3-registry-test")
    model = P3ClusterModel.fit(sample_stations(), selected_k=3, random_seed=42, n_init=10)
    with mlflow.start_run() as run:
        source_run_id = run.info.run_id
        log_p3_pyfunc_model(model, name="model")

    description = p3_model_description(3, "silhouette_score", 0.5)
    version = register_p3_model_from_run(
        source_run_id,
        registry_model_name=P3_REGISTRY_MODEL_NAME,
        tracking_uri=tracking_uri,
        description=description,
        tags={
            "question": "P3",
            "team": "equipo49",
            "model_family": "kmeans",
            "git_commit": "test-commit",
            "data_version": "sha256:test",
            "input_sha256": "test",
            "source_run_id": source_run_id,
            "parent_search_run_id": "parent-test",
            "selected_k": "3",
            "selection_metric": "silhouette_score",
            "stage_of_work": "champion",
        },
    )

    version_uri = f"models:/{P3_REGISTRY_MODEL_NAME}/{version.version}"
    expected = validate_model_inference(
        version_uri,
        p3_input_example(),
        selected_k=3,
        tracking_uri=tracking_uri,
    )
    assert expected["cluster"].between(0, 2).all()
    assign_champion_alias(
        P3_REGISTRY_MODEL_NAME,
        version.version,
        P3_CHAMPION_ALIAS,
        tracking_uri=tracking_uri,
    )
    loaded = load_champion_model(
        P3_REGISTRY_MODEL_NAME,
        P3_CHAMPION_ALIAS,
        tracking_uri=tracking_uri,
    )
    pd.testing.assert_frame_equal(loaded.predict(p3_input_example()), expected)
    actual = validate_champion_inference(
        p3_input_example(),
        selected_k=3,
        registry_model_name=P3_REGISTRY_MODEL_NAME,
        champion_alias=P3_CHAMPION_ALIAS,
        tracking_uri=tracking_uri,
    )
    pd.testing.assert_frame_equal(actual, expected)

    registered_model = client.get_registered_model(P3_REGISTRY_MODEL_NAME)
    stored_version = client.get_model_version(P3_REGISTRY_MODEL_NAME, version.version)
    assert registered_model.description == description
    assert stored_version.run_id == source_run_id
    assert stored_version.tags["source_run_id"] == source_run_id
    assert stored_version.tags["parent_search_run_id"] == "parent-test"
    assert stored_version.tags["stage_of_work"] == "champion"
    assert client.get_model_version_by_alias(
        P3_REGISTRY_MODEL_NAME,
        P3_CHAMPION_ALIAS,
    ).version == version.version
