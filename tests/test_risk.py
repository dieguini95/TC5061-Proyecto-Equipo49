from __future__ import annotations

import json

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytest

from mibici_ml.config import P2_HORIZON_HOURS
from mibici_ml.features import (
    P2_FEATURE_COLUMNS,
    build_p2_features,
    build_p2_hourly_flows,
    build_p2_station_day_panel,
    p2_risk_labels,
)
from mibici_ml.modeling.risk import (
    FEATURE_COLUMNS,
    _stage_of_work,
    best_f1_threshold,
    build_model,
    compute_quality_summary,
    evaluate_all,
    fit_and_score,
    load_verified_manifest,
    log_p2_run,
    persistence_scores,
    precision_at_k,
    recall_at_precision,
    score_split,
    temporal_masks,
)


def _trips(rows: list[tuple[str, str, str, int, int]]) -> pd.DataFrame:
    """Viajes limpios minimos: (trip_id, inicio, fin, origen, destino)."""
    frame = pd.DataFrame(
        rows, columns=["trip_id", "start_time", "end_time", "origin_id", "destination_id"]
    )
    frame["start_time"] = pd.to_datetime(frame["start_time"])
    frame["end_time"] = pd.to_datetime(frame["end_time"])
    return frame


def _capacities(mapping: dict[int, int]) -> pd.DataFrame:
    return pd.DataFrame({"est": list(mapping), "capacidad": list(mapping.values())})


def _panel_from_trips(trips: pd.DataFrame, capacities: dict[int, int]):
    flows = build_p2_hourly_flows(trips)
    return build_p2_station_day_panel(flows, _capacities(capacities))


def _write_manifest(root, *, trip_hash_ok: bool) -> tuple:
    """Manifiesto minimo con Parquet reales; ``data_version`` se calcula como en dataset.py."""
    from mibici_ml.dataset import _data_version
    from mibici_ml.modeling.risk import file_sha256

    processed = root / "processed"
    (processed / "trips").mkdir(parents=True)
    trip_path = processed / "trips" / "2026_01.parquet"
    pd.DataFrame({"origin_id": [1]}).to_parquet(trip_path)
    stations_path = processed / "stations.parquet"
    pd.DataFrame({"station_id": [1], "capacity": [10]}).to_parquet(stations_path)
    trip_hash = file_sha256(trip_path)
    if not trip_hash_ok:
        trip_hash = "0" * 64
    manifest = {
        "cleaning_version": "test",
        "date_range": ["2026-01", "2026-01"],
        "trips": [
            {
                "month": "2026_01",
                "raw_sha256": "raw",
                "processed_path": "trips/2026_01.parquet",
                "processed_sha256": trip_hash,
            }
        ],
        "stations": {
            "raw_sha256": "raw",
            "processed_path": "stations.parquet",
            "processed_sha256": file_sha256(stations_path),
        },
    }
    manifest["data_version"] = _data_version(manifest)
    manifest_path = root / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, processed


def _features_for_active_set(panel: pd.DataFrame) -> pd.DataFrame:
    """Features como las arma el pipeline: v1, mas las columnas v2 si P2_FEATURE_SET=v2."""
    from mibici_ml.features import P2_FEATURE_COLUMNS_V2, add_p2_v2_features

    features = build_p2_features(panel)
    if FEATURE_COLUMNS == P2_FEATURE_COLUMNS_V2:
        v2 = add_p2_v2_features(panel)
        extra = [c for c in P2_FEATURE_COLUMNS_V2 if c not in features.columns]
        features = pd.concat([features, v2[extra].reset_index(drop=True)], axis=1)
    return features


def test_build_p2_hourly_flows_counts_departures_and_arrivals_by_hour():
    trips = _trips(
        [
            (1, "2026-01-01 00:10", "2026-01-01 00:50", 10, 20),
            (2, "2026-01-01 00:20", "2026-01-01 01:05", 10, 20),
            (3, "2026-01-01 01:30", "2026-01-01 01:40", 20, 10),
        ]
    )
    flows = build_p2_hourly_flows(trips).set_index(["est", "ts"])
    assert flows.loc[(10, pd.Timestamp("2026-01-01 00:00")), "salidas"] == 2
    assert flows.loc[(20, pd.Timestamp("2026-01-01 00:00")), "llegadas"] == 1
    assert flows.loc[(20, pd.Timestamp("2026-01-01 01:00")), "llegadas"] == 1


def test_build_p2_hourly_flows_rejects_missing_columns():
    with pytest.raises(ValueError, match="Faltan columnas"):
        build_p2_hourly_flows(pd.DataFrame({"origin_id": [1]}))


def test_feasibility_filters_infeasible_station_days():
    # Capacidad 2 y 3 llegadas seguidas sin salidas: F sube a 3 > C, el dia es infactible.
    trips = _trips(
        [
            (1, "2026-01-01 00:10", "2026-01-01 01:00", 2, 1),
            (2, "2026-01-01 00:20", "2026-01-01 01:10", 3, 1),
            (3, "2026-01-01 00:30", "2026-01-01 01:20", 4, 1),
            (4, "2026-01-02 00:10", "2026-01-02 00:20", 1, 2),
        ]
    )
    panel, bounds = _panel_from_trips(trips, {1: 2, 2: 5, 3: 5, 4: 5})
    day1 = bounds[(bounds["est"] == 1) & (bounds["day"] == pd.Timestamp("2026-01-01"))]
    assert not bool(day1["factible"].iloc[0])
    assert not ((panel["est"] == 1) & (panel["day"] == pd.Timestamp("2026-01-01"))).any()


def test_panel_uses_midpoint_inventory_and_positive_capacity_only():
    trips = _trips([(1, "2026-01-01 00:10", "2026-01-01 00:20", 1, 2)])
    panel, _ = _panel_from_trips(trips, {1: 10, 2: 10, 9: 0})
    assert 9 not in set(panel["est"])
    assert panel["inventario"].between(0, 10).all()


def test_label_marks_empty_or_full_within_horizon_same_day_only():
    panel = pd.DataFrame(
        {
            "est": [1, 1, 1, 1],
            "day": pd.to_datetime(["2026-01-01"] * 3 + ["2026-01-02"]),
            "inventario": [5.0, 0.0, 5.0, 5.0],
            "capacidad": [10, 10, 10, 10],
        }
    )
    labels = p2_risk_labels(panel, horizon=1)
    assert labels[0] == 1  # la fila siguiente esta vacia
    assert labels[2] == 0  # la fila siguiente no es de riesgo
    assert labels[3] == 0  # sin fila siguiente en el mismo dia


def test_label_detects_full_station():
    # La etiqueta mira hacia adelante: la fila 0 (inventario 3) ve la fila 1 llena -> 1.
    # La fila 1 no tiene fila siguiente en el dia, asi que no se etiqueta como riesgo.
    panel = pd.DataFrame(
        {
            "est": [1, 1],
            "day": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "inventario": [3.0, 10.0],
            "capacidad": [10, 10],
        }
    )
    assert p2_risk_labels(panel, horizon=1).tolist() == [1, 0]


def test_label_horizon_is_the_configured_window():
    panel = pd.DataFrame(
        {
            "est": [1, 1, 1],
            "day": pd.to_datetime(["2026-01-01"] * 3),
            "inventario": [5.0, 5.0, 0.0],
            "capacidad": [10, 10, 10],
        }
    )
    assert p2_risk_labels(panel, horizon=1).tolist() == [0, 1, 0]
    assert p2_risk_labels(panel, horizon=2).tolist() == [1, 1, 0]


def _feature_panel(n: int = 30) -> pd.DataFrame:
    hours = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "est": 1,
            "ts": hours,
            "day": hours.normalize(),
            "salidas": np.arange(n) % 3,
            "llegadas": np.arange(n) % 2,
            "capacidad": 10,
            "flujo_acumulado": np.arange(n, dtype=float),
        }
    )


def test_features_do_not_depend_on_future_rows():
    base = _feature_panel()
    altered = base.copy()
    altered.loc[25:, ["salidas", "llegadas", "flujo_acumulado"]] = 999
    feats_a = build_p2_features(base)
    feats_b = build_p2_features(altered)
    cut = 20
    pd.testing.assert_frame_equal(
        feats_a.loc[:cut, list(P2_FEATURE_COLUMNS)].reset_index(drop=True),
        feats_b.loc[:cut, list(P2_FEATURE_COLUMNS)].reset_index(drop=True),
    )


def test_feature_columns_match_declared_contract():
    feats = build_p2_features(_feature_panel())
    assert set(P2_FEATURE_COLUMNS).issubset(feats.columns)


def test_temporal_masks_are_disjoint_and_ordered():
    hours = pd.date_range("2025-10-01", "2026-09-30 23:00", freq="h")
    panel = pd.DataFrame({"ts": hours})
    masks = temporal_masks(panel)
    assert not (masks["train"] & masks["val"]).any()
    assert not (masks["val"] & masks["test"]).any()
    assert panel.loc[masks["train"], "ts"].max() < panel.loc[masks["val"], "ts"].min()
    assert panel.loc[masks["val"], "ts"].max() < panel.loc[masks["test"], "ts"].min()


def test_persistence_uses_lagged_label_without_leaking_current_hour():
    panel = pd.DataFrame(
        {
            "est": [1] * 6,
            "day": pd.to_datetime(["2026-01-01"] * 6),
            "ts": pd.date_range("2026-01-01", periods=6, freq="h"),
        }
    )
    labels = np.array([1, 0, 0, 1, 0, 0])
    scores = persistence_scores(panel, labels)
    assert scores[P2_HORIZON_HOURS] == labels[0]
    assert scores[0] == 0.0


def test_precision_at_k_top_fraction_per_hour():
    ts = pd.Series(pd.to_datetime(["2026-01-01 00:00"] * 4 + ["2026-01-01 01:00"] * 4))
    scores = np.array([0.9, 0.1, 0.2, 0.3, 0.8, 0.7, 0.1, 0.2])
    labels = np.array([1, 0, 0, 0, 1, 0, 0, 0])
    value = precision_at_k(scores, labels, ts, k_fraction=0.25)
    assert value == pytest.approx(1.0)


def test_recall_at_precision_respects_minimum():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.4, 0.1])
    assert recall_at_precision(labels, scores, min_precision=0.5) == pytest.approx(1.0)


def test_recall_at_precision_handles_unreachable_precision():
    labels = np.array([0, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.4, 0.1])
    assert recall_at_precision(labels, scores, min_precision=0.99) == pytest.approx(0.0)


def test_score_split_returns_prefixed_keys():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    ts = pd.Series(pd.date_range("2026-01-01", periods=4, freq="h"))
    metrics = score_split(labels, scores, ts, threshold=0.5, prefix="test_")
    assert metrics["test_pr_auc"] == pytest.approx(1.0)
    assert all(key.startswith("test_") for key in metrics)


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_model("no-existe", random_seed=1)


def test_evaluate_all_runs_on_synthetic_panel_without_test_leak():
    hours = pd.date_range("2025-10-01", "2026-09-30 23:00", freq="h")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        {
            "est": 1,
            "ts": hours,
            "day": hours.normalize(),
            "label": rng.integers(0, 2, len(hours)),
            "salidas": rng.integers(0, 3, len(hours)),
            "llegadas": rng.integers(0, 3, len(hours)),
            "inventario": 5.0,
            "capacidad": 10,
            "flujo_acumulado": 0.0,
        }
    )
    features = _features_for_active_set(panel)
    metrics, context, _ = evaluate_all(panel, features, random_seed=7)
    assert set(metrics["modelo"]) == {
        "mayoritaria",
        "persistencia",
        "logistica",
        "hist_gb",
        "extra_trees",
    }
    assert all("test_" not in column for column in metrics.columns)
    assert context["masks"]["test"].sum() > 0


def test_load_verified_manifest_rejects_month_set_not_configured(tmp_path):
    # El manifiesto de prueba enumera un solo mes; P2 exige los 12 meses configurados.
    manifest_path, processed = _write_manifest(tmp_path, trip_hash_ok=True)
    with pytest.raises(ValueError, match="meses configurados"):
        load_verified_manifest(manifest_path, processed)


def test_load_verified_manifest_rejects_modified_trip_file(tmp_path):
    manifest_path, processed = _write_manifest(tmp_path, trip_hash_ok=False)
    with pytest.raises(ValueError):
        load_verified_manifest(manifest_path, processed)


def test_station_hash_mismatch_is_recorded_not_fatal(tmp_path, monkeypatch):
    # Un catalogo con hash distinto no detiene P2; queda registrado para MLflow.
    from mibici_ml import config
    from mibici_ml.dataset import _data_version

    manifest_path, processed = _write_manifest(tmp_path, trip_hash_ok=True)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["stations"]["processed_sha256"] = "f" * 64
    data["data_version"] = _data_version(data)
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(config, "START_MONTH", "2026-01")
    monkeypatch.setattr(config, "END_MONTH", "2026-01")
    manifest = load_verified_manifest(manifest_path, processed)
    assert manifest["_stations_hash_check"]["match"] is False


def test_log_p2_run_writes_tags_metrics_and_manifest_artifact(tmp_path):
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    artifact = tmp_path / "metricas.csv"
    artifact.write_text("a\n1\n", encoding="utf-8")
    quality = pd.DataFrame(
        [
            {
                "viajes_validos": 10,
                "estacion_dia_no_factibles": 1,
                "pct_estacion_dia_no_factibles": 9.0,
                "filas_panel_modelado": 100,
                "tasa_positivos_panel": 0.2,
            }
        ]
    )
    manifest = {"data_version": "abc123", "trips": [], "stations": {}}
    run_id = log_p2_run(
        run_name="P2-test",
        run_type="model_selection_validation",
        model_name="mayoritaria",
        random_seed=1,
        params_extra={"split_scope": "validation_only"},
        metrics={"val_pr_auc": 0.5},
        quality=quality,
        artifacts={"metricas": artifact},
        model=None,
        temporal_holdout=False,
        manifest=manifest,
        experiment_name="test-p2",
        tracking_uri=tracking_uri,
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert run.data.tags["question"] == "P2"
    assert run.data.tags["temporal_holdout"] == "false"
    assert run.data.tags["data_version"] == "abc123"
    assert run.data.tags["git_commit"]
    assert run.data.tags["git_dirty"] in {"true", "false"}
    assert run.data.params["data_version"] == "abc123"
    assert run.data.params["horizon_hours"] == str(P2_HORIZON_HOURS)
    assert run.data.metrics["val_pr_auc"] == pytest.approx(0.5)
    artifact_paths = {a.path for a in client.list_artifacts(run_id, "data")}
    assert "data/data_manifest.json" in artifact_paths
    assert "data/p2_calidad_datos.csv" in artifact_paths
    assert "config/run_config.json" in {a.path for a in client.list_artifacts(run_id, "config")}
    assert run.data.tags["stage_of_work"] == "candidate"
    assert run.data.tags["model_family"] == "baseline"


def test_log_p2_run_logs_signature_input_and_plots_subfolder(tmp_path):
    from sklearn.dummy import DummyClassifier

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    features = pd.DataFrame({col: np.arange(40, dtype=float) % 5 for col in FEATURE_COLUMNS})
    labels = np.tile([0, 1], 20)
    model = DummyClassifier(strategy="prior").fit(features, labels)
    plot = tmp_path / "p2_curva_pr.png"
    plot.write_bytes(b"png")
    quality = pd.DataFrame(
        [
            {
                "viajes_validos": 1,
                "estacion_dia_no_factibles": 0,
                "pct_estacion_dia_no_factibles": 0.0,
                "filas_panel_modelado": 40,
                "tasa_positivos_panel": 0.5,
            }
        ]
    )
    run_id = log_p2_run(
        run_name="P2-sig-test",
        run_type="model_selection_validation",
        model_name="hist_gb",
        random_seed=1,
        params_extra={"split_scope": "validation_only"},
        metrics={"val_pr_auc": 0.1},
        quality=quality,
        artifacts={"curva_pr": plot},
        model=model,
        temporal_holdout=False,
        manifest={"data_version": "abc123", "trips": [], "stations": {}},
        experiment_name="test-p2-sig",
        tracking_uri=tracking_uri,
        features=features,
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert run.data.tags["model_family"] == "gradient_boosting"
    assert run.data.tags["stage_of_work"] == "candidate"
    assert "plots/p2_curva_pr.png" in {a.path for a in client.list_artifacts(run_id, "plots")}
    model_info = client.list_artifacts(run_id, "p2_hist_gb")
    assert model_info, "el modelo debe quedar guardado"
    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/p2_hist_gb")
    assert loaded.metadata.signature is not None
    # La firma refleja exactamente las columnas con que se entreno (set activo).
    assert list(loaded.metadata.signature.inputs.input_names()) == list(FEATURE_COLUMNS)


def test_search_stage_of_work_tag_is_search():
    assert _stage_of_work("search_candidate") == "search"
    assert _stage_of_work("final_test") == "final"


def test_build_model_supports_l1_logistic_and_extra_trees():
    l1 = build_model("logistica", random_seed=1, params={"C": 1.0, "penalty": "l1"})
    assert l1.named_steps["clf"].solver == "liblinear"
    assert l1.named_steps["clf"].penalty == "l1"
    trees = build_model("extra_trees", random_seed=1, params={"min_samples_leaf": 20})
    assert trees.n_estimators == 300
    assert trees.min_samples_leaf == 20


def test_v2_features_do_not_depend_on_future_rows():
    from mibici_ml.features import P2_FEATURE_COLUMNS_V2, add_p2_v2_features

    base = build_p2_features(_feature_panel())
    base["inventario"] = 5.0
    base["day"] = pd.to_datetime(base["ts"]).dt.normalize()
    altered = base.copy()
    altered.loc[25:, ["salidas", "llegadas", "flujo_acumulado", "inventario"]] = 999
    v2_a = add_p2_v2_features(base)
    v2_b = add_p2_v2_features(altered)
    cut = 20
    pd.testing.assert_frame_equal(
        v2_a.loc[:cut, list(P2_FEATURE_COLUMNS_V2)].reset_index(drop=True),
        v2_b.loc[:cut, list(P2_FEATURE_COLUMNS_V2)].reset_index(drop=True),
    )


def test_feature_set_v1_is_default_and_unchanged():
    from mibici_ml.features import P2_FEATURE_COLUMNS
    from mibici_ml.modeling import risk

    assert risk.FEATURE_SETS["v1"] == P2_FEATURE_COLUMNS
    assert len(P2_FEATURE_COLUMNS) == 11


def test_parse_tuned_handles_numbers_and_empty_items():
    from mibici_ml.modeling.risk import parse_tuned

    assert parse_tuned("learning_rate=0.03;max_iter=600;;max_leaf_nodes=15") == {
        "learning_rate": 0.03,
        "max_iter": 600,
        "max_leaf_nodes": 15,
    }


def test_risk_cli_registers_evaluate_and_run_commands():
    from typer.testing import CliRunner

    from mibici_ml.modeling import risk

    runner = CliRunner()
    result = runner.invoke(risk.app, ["--help"])
    assert result.exit_code == 0
    assert "evaluate" in result.output
    assert "run" in result.output


def _synthetic_risk_panel() -> pd.DataFrame:
    """Panel anual por horas con etiqueta aleatoria fija (semilla 0) para pruebas de reproducibilidad."""
    hours = pd.date_range("2025-10-01", "2026-09-30 23:00", freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "est": 1,
            "ts": hours,
            "day": hours.normalize(),
            "label": rng.integers(0, 2, len(hours)),
            "salidas": rng.integers(0, 3, len(hours)),
            "llegadas": rng.integers(0, 3, len(hours)),
            "inventario": 5.0,
            "capacidad": 10,
            "flujo_acumulado": 0.0,
        }
    )


def test_fit_and_score_same_seed_gives_identical_scores():
    panel = _synthetic_risk_panel()
    features = _features_for_active_set(panel)
    labels = panel["label"].to_numpy()
    masks = temporal_masks(panel)
    metrics_a, scores_a, _ = fit_and_score(
        "extra_trees", features, labels, panel, masks, random_seed=7
    )
    metrics_b, scores_b, _ = fit_and_score(
        "extra_trees", features, labels, panel, masks, random_seed=7
    )
    # Tolerancia de punto flotante: la suma de arboles en paralelo puede reordenarse y
    # cambiar el ultimo bit (~1e-16). La reproducibilidad se mide con rtol=1e-12.
    np.testing.assert_allclose(scores_a["val"], scores_b["val"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scores_a["test"], scores_b["test"], rtol=1e-12, atol=1e-12)
    assert metrics_a == pytest.approx(metrics_b, rel=1e-12)


def test_fit_and_score_different_seed_changes_scores_as_control():
    panel = _synthetic_risk_panel()
    features = _features_for_active_set(panel)
    labels = panel["label"].to_numpy()
    masks = temporal_masks(panel)
    _, scores_a, _ = fit_and_score("extra_trees", features, labels, panel, masks, random_seed=7)
    _, scores_b, _ = fit_and_score("extra_trees", features, labels, panel, masks, random_seed=8)
    assert not np.array_equal(scores_a["val"], scores_b["val"])


def test_evaluate_all_metrics_are_identical_across_runs_with_same_seed():
    panel = _synthetic_risk_panel()
    features = _features_for_active_set(panel)
    first, _, _ = evaluate_all(panel, features, random_seed=7)
    second, _, _ = evaluate_all(panel, features, random_seed=7)
    pd.testing.assert_frame_equal(first, second, check_exact=False, rtol=1e-12)


def test_best_f1_threshold_handles_single_class_and_tied_scores():
    labels = np.zeros(6, dtype=int)
    scores = np.array([0.2, 0.4, 0.6, 0.8, 0.1, 0.3])
    threshold = best_f1_threshold(labels, scores)
    assert isinstance(threshold, float)
    assert scores.min() <= threshold <= scores.max()

    tied_scores = np.full(6, 0.5)
    tied_labels = np.array([1, 0, 1, 0, 1, 0])
    assert best_f1_threshold(tied_labels, tied_scores) == pytest.approx(0.5)


def test_quality_summary_handles_empty_and_populated_panels():
    manifest = {"data_version": "v-test"}
    trips = pd.DataFrame({"origin_id": [1, 2]})
    capacities = pd.DataFrame({"est": [1, 2], "capacidad": [0, 10]})

    empty = compute_quality_summary(
        manifest,
        trips,
        pd.DataFrame({"factible": pd.Series([], dtype=bool)}),
        capacities,
        pd.DataFrame({"label": pd.Series([], dtype=int)}),
    ).iloc[0]
    assert empty["estacion_dia_total"] == 0
    assert empty["pct_estacion_dia_no_factibles"] == 0.0
    assert empty["tasa_positivos_panel"] == 0.0
    assert empty["estaciones_capacidad_cero"] == 1

    populated = compute_quality_summary(
        manifest,
        trips,
        pd.DataFrame({"factible": [True, True, True, False]}),
        capacities,
        pd.DataFrame({"label": [1, 0, 0, 0]}),
    ).iloc[0]
    assert populated["estacion_dia_no_factibles"] == 1
    assert populated["pct_estacion_dia_no_factibles"] == pytest.approx(25.0)
    assert populated["tasa_positivos_panel"] == pytest.approx(0.25)
