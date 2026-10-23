from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from mibici_ml.config import TABLES_DIR
from mibici_ml.modeling.cluster import (
    ARTIFACT_NAMES,
    MODEL_FEATURES,
    build_cluster_profiles,
    build_data_quality_summary,
    build_station_assignments,
    compute_file_sha256,
    evaluate_kmeans_candidates,
    fit_final_pipeline,
    load_station_summary,
    log_p3_run,
    plot_k_comparison,
    prepare_cluster_features,
    prepare_station_data,
    save_artifacts,
    validate_station_summary,
)


def sample_stations() -> pd.DataFrame:
    """Create a small deterministic fixture for unit tests only."""
    return pd.DataFrame(
        {
            "est": np.arange(1, 13),
            "nombre": [f"Estacion {station}" for station in range(1, 13)],
            "lat": 20.60 + np.arange(12) / 1000,
            "lon": -103.40 + np.arange(12) / 1000,
            "capacidad": [0, 0, 10, 12, 14, 16, 20, 22, 24, 26, 28, 30],
            "salidas_12m": [0, 1, 20, 30, 50, 80, 120, 180, 260, 370, 520, 720],
            "llegadas_12m": [1, 0, 25, 35, 60, 90, 140, 200, 290, 410, 570, 800],
            "pct_dias_no_factibles": [np.nan] * 12,
        }
    )


def prepared_sample() -> tuple[pd.DataFrame, pd.DataFrame, list[str], float]:
    return prepare_station_data(sample_stations())


def make_artifacts(output_dir, random_seed: int = 42):
    enriched, features, feature_names, capacity_median = prepared_sample()
    metrics = evaluate_kmeans_candidates(features, 3, 4, random_seed)
    pipeline, labels = fit_final_pipeline(features, 3, random_seed)
    assignments = build_station_assignments(enriched, labels)
    profiles = build_cluster_profiles(assignments)
    quality = build_data_quality_summary(enriched, features, capacity_median)
    paths = save_artifacts(output_dir, metrics, assignments, profiles, quality)
    return (
        enriched,
        features,
        feature_names,
        capacity_median,
        metrics,
        pipeline,
        assignments,
        profiles,
        quality,
        paths,
    )


def test_missing_required_columns_raise_clear_error():
    stations = sample_stations().drop(columns="lat")
    with pytest.raises(ValueError, match="lat"):
        validate_station_summary(stations)


def test_duplicate_station_id_is_rejected():
    stations = sample_stations()
    stations.loc[1, "est"] = stations.loc[0, "est"]
    with pytest.raises(ValueError, match="unicos"):
        validate_station_summary(stations)


def test_missing_station_id_is_rejected():
    stations = sample_stations()
    stations.loc[0, "est"] = np.nan
    with pytest.raises(ValueError, match="faltantes"):
        validate_station_summary(stations)


def test_negative_capacity_is_rejected():
    stations = sample_stations()
    stations.loc[0, "capacidad"] = -1
    with pytest.raises(ValueError, match="negativos"):
        validate_station_summary(stations)


def test_zero_capacity_is_accepted_and_logged(caplog):
    stations = sample_stations()
    validate_station_summary(stations)
    assert (stations["capacidad"] == 0).sum() == 2


def test_zero_capacity_uses_median_of_positive_capacities_only():
    stations = sample_stations()
    positive_median = stations.loc[stations["capacidad"] > 0, "capacidad"].median()
    enriched, _, _, calculated_median = prepare_station_data(stations)
    assert calculated_median == positive_median
    assert enriched.loc[enriched["capacidad"] == 0, "capacidad_modelo"].eq(
        positive_median
    ).all()


def test_original_capacity_is_preserved_without_mutating_input():
    stations = sample_stations()
    original = stations.copy(deep=True)
    enriched, _, _, _ = prepare_station_data(stations)
    pd.testing.assert_frame_equal(stations, original)
    pd.testing.assert_series_equal(enriched["capacidad"], original["capacidad"])


def test_capacity_imputed_marks_exactly_zero_rows():
    stations = sample_stations()
    enriched, _, _, _ = prepare_station_data(stations)
    expected = stations["capacidad"].eq(0)
    expected.name = "capacidad_imputada"
    pd.testing.assert_series_equal(enriched["capacidad_imputada"], expected)


def test_all_zero_capacity_is_rejected():
    stations = sample_stations()
    stations["capacidad"] = 0
    with pytest.raises(ValueError, match="al menos una capacidad positiva"):
        validate_station_summary(stations)


def test_model_capacity_is_positive_and_finite():
    enriched, _, _, _ = prepared_sample()
    values = enriched["capacidad_modelo"].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert (values > 0).all()


@pytest.mark.parametrize("column", ["salidas_12m", "llegadas_12m"])
def test_negative_usage_counts_are_rejected(column):
    stations = sample_stations()
    stations.loc[0, column] = -1
    with pytest.raises(ValueError, match="negativos"):
        validate_station_summary(stations)


def test_features_have_exact_names_and_exclude_metadata():
    features, names = prepare_cluster_features(sample_stations())
    assert names == list(MODEL_FEATURES)
    assert list(features.columns) == list(MODEL_FEATURES)
    assert "capacidad_imputada" not in features
    assert not {"est", "nombre", "lat", "lon", "pct_dias_no_factibles"}.intersection(
        features.columns
    )
    assert np.isfinite(features.to_numpy(dtype=float)).all()
    assert not features.isna().any().any()


def test_kmeans_labels_are_reproducible_for_same_seed():
    _, features, _, _ = prepared_sample()
    _, first_labels = fit_final_pipeline(features, 3, random_seed=17)
    _, second_labels = fit_final_pipeline(features, 3, random_seed=17)
    np.testing.assert_array_equal(first_labels, second_labels)


def test_assignments_have_one_row_per_station_and_required_columns():
    enriched, features, _, _ = prepared_sample()
    _, labels = fit_final_pipeline(features, 3)
    assignments = build_station_assignments(enriched, labels)
    assert len(assignments) == len(enriched)
    assert assignments["est"].is_unique
    assert list(assignments.columns) == [
        "est",
        "nombre",
        "lat",
        "lon",
        "capacidad",
        "capacidad_modelo",
        "capacidad_imputada",
        "salidas_12m",
        "llegadas_12m",
        "cluster",
    ]


def test_candidate_k_below_two_is_rejected():
    _, features, _, _ = prepared_sample()
    with pytest.raises(ValueError, match="k_min"):
        evaluate_kmeans_candidates(features, 1, 3)


def test_candidate_k_equal_to_sample_count_is_rejected():
    _, features, _, _ = prepared_sample()
    with pytest.raises(ValueError, match="menor que el numero"):
        evaluate_kmeans_candidates(features, 2, len(features))


def test_candidate_metrics_are_finite_for_each_k():
    _, features, _, _ = prepared_sample()
    metrics = evaluate_kmeans_candidates(features, 3, 5)
    metric_columns = [
        "inertia",
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
    ]
    assert metrics["k"].tolist() == [3, 4, 5]
    assert np.isfinite(metrics[metric_columns].to_numpy(dtype=float)).all()
    assert (metrics["min_cluster_size"] > 0).all()
    assert (metrics["max_cluster_size"] >= metrics["min_cluster_size"]).all()


def test_profiles_are_sorted_and_cluster_counts_are_consistent():
    enriched, features, _, _ = prepared_sample()
    _, labels = fit_final_pipeline(features, 3)
    assignments = build_station_assignments(enriched, labels)
    profiles = build_cluster_profiles(assignments)
    assert profiles["cluster"].tolist() == sorted(profiles["cluster"].tolist())
    assert profiles["n_stations"].sum() == len(assignments)
    assert profiles["cluster"].is_unique
    assert {
        "capacidad_modelo_media",
        "capacidad_modelo_mediana",
        "salidas_12m_media",
        "salidas_12m_mediana",
        "llegadas_12m_media",
        "llegadas_12m_mediana",
        "salidas_12m_suma",
        "llegadas_12m_suma",
    }.issubset(profiles.columns)


def test_sha256_is_deterministic(tmp_path):
    from mibici_ml.modeling.cluster import compute_file_sha256

    input_path = tmp_path / "input.csv"
    input_path.write_text("est,capacidad\n1,10\n", encoding="utf-8")
    assert compute_file_sha256(input_path) == compute_file_sha256(input_path)
    assert len(compute_file_sha256(input_path)) == 64


def test_real_station_table_contract_and_assignments():
    input_path = TABLES_DIR / "estaciones_resumen.csv"
    stations = load_station_summary(input_path)
    expected_columns = [
        "est",
        "nombre",
        "lat",
        "lon",
        "capacidad",
        "salidas_12m",
        "llegadas_12m",
        "pct_dias_no_factibles",
    ]
    assert stations.shape == (484, 8)
    assert stations.columns.tolist() == expected_columns
    assert stations["est"].is_unique
    assert (stations["capacidad"] == 0).sum() == 4

    enriched, features, _, positive_median = prepare_station_data(stations)
    assert enriched["capacidad_imputada"].sum() == 4
    assert positive_median == stations.loc[stations["capacidad"] > 0, "capacidad"].median()
    assert features.shape == (484, 3)
    assert not features.isna().any().any()
    assert np.isfinite(features.to_numpy(dtype=float)).all()

    _, labels = fit_final_pipeline(features, 3)
    assignments = build_station_assignments(enriched, labels)
    assert len(assignments) == 484
    assert assignments["est"].is_unique
    assert assignments["capacidad_imputada"].sum() == 4


def test_save_artifacts_writes_seven_nonempty_files(tmp_path):
    *_, paths = make_artifacts(tmp_path)
    assert set(paths) == set(ARTIFACT_NAMES)
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def test_plot_k_comparison_writes_nonempty_png(tmp_path):
    _, features, _, _ = prepared_sample()
    metrics = evaluate_kmeans_candidates(features, 3, 4)
    output_path = tmp_path / "nested" / "comparison.png"
    plot_k_comparison(metrics, output_path)
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_sha256_and_quality_artifact_are_based_on_real_values(tmp_path):
    input_path = TABLES_DIR / "estaciones_resumen.csv"
    assert compute_file_sha256(input_path) == compute_file_sha256(input_path)
    enriched, features, _, capacity_median, _, _, _, _, quality, paths = make_artifacts(
        tmp_path
    )
    quality_row = quality.iloc[0]
    assert quality_row["n_stations"] == len(enriched)
    assert quality_row["n_zero_capacity"] == enriched["capacidad"].eq(0).sum()
    assert quality_row["n_imputed_capacity"] == enriched["capacidad_imputada"].sum()
    assert quality_row["positive_capacity_median"] == capacity_median
    assert quality_row["n_missing_model_features"] == features.isna().sum().sum()
    assert quality_row["n_duplicate_station_ids"] == 0
    assert paths["p3_calidad_datos.csv"].is_file()


def test_mlflow_logs_metadata_metrics_and_artifacts_in_tmp_path(tmp_path, monkeypatch):
    (
        _,
        features,
        feature_names,
        capacity_median,
        metrics,
        pipeline,
        _,
        _,
        quality,
        artifacts,
    ) = make_artifacts(tmp_path / "artifacts")
    final_metrics = metrics.loc[metrics["k"] == 3].iloc[0]
    input_path = tmp_path / "input.csv"
    sample_stations().to_csv(input_path, index=False)
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    run_id = log_p3_run(
        pipeline=pipeline,
        final_k=3,
        k_min=3,
        k_max=4,
        random_seed=42,
        feature_names=feature_names,
        input_path=input_path,
        input_sha256=compute_file_sha256(input_path),
        positive_capacity_median=capacity_median,
        final_metrics=final_metrics,
        quality=quality,
        artifacts=artifacts,
        experiment_name="p3-test-temporary",
        tracking_uri=tracking_uri,
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert run.data.params["capacity_zero_policy"] == "median_positive_imputation"
    assert run.data.params["capacity_feature"] == "capacidad_modelo"
    assert float(run.data.params["positive_capacity_median"]) == capacity_median
    assert run.data.metrics["n_zero_capacity"] == 2
    assert run.data.metrics["n_imputed_capacity"] == 2
    assert run.data.tags["question"] == "P3"
    assert run.data.tags["temporal_holdout"] == "false"
    assert run.data.tags["git_commit"]
    artifact_names = {artifact.path for artifact in client.list_artifacts(run_id)}
    assert set(ARTIFACT_NAMES).issubset(artifact_names)
    assert run_id
    assert (tmp_path / "mlflow.db").exists()
