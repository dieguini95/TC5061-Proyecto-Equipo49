from __future__ import annotations

from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytest

from mibici_ml.config import P2_HORIZON_HOURS
from mibici_ml.modeling.risk import (
    FEATURE_COLUMNS,
    _risk_labels,
    build_features,
    build_hourly_flows,
    build_model,
    build_station_day_panel,
    clean_trips,
    evaluate_all,
    load_station_catalog,
    log_p2_run,
    persistence_scores,
    precision_at_k,
    read_trip_file,
    recall_at_precision,
    score_split,
    temporal_masks,
)


def _trips(rows: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    """Viajes minimos: (viaje_id, inicio, fin, origen, destino)."""
    frame = pd.DataFrame(
        rows, columns=["Viaje_Id", "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]
    )
    frame["Usuario_Id"] = "1"
    frame["Genero"] = "M"
    frame["Año_de_nacimiento"] = "1990"
    return frame[
        [
            "Viaje_Id",
            "Usuario_Id",
            "Genero",
            "Año_de_nacimiento",
            "Inicio_del_viaje",
            "Fin_del_viaje",
            "Origen_Id",
            "Destino_Id",
        ]
    ]


def _panel_from_trips(trips: pd.DataFrame, capacities: dict[str, int]):
    cleaned, _ = clean_trips(trips, set(capacities))
    flows = build_hourly_flows(cleaned)
    catalog = pd.DataFrame({"est": list(capacities), "capacidad": list(capacities.values())})
    return build_station_day_panel(flows, catalog)


def test_read_trip_file_decodes_cp1252_and_checks_columns(tmp_path):
    path = tmp_path / "datos_abiertos_2026_01.csv"
    text = '"Viaje_Id","Usuario_Id","Genero","Año_de_nacimiento","Inicio_del_viaje","Fin_del_viaje","Origen_Id","Destino_Id"\n"1","2","M","1990","2026-01-01 00:00:00","2026-01-01 00:10:00","3","4"\n'
    path.write_bytes(text.encode("cp1252"))
    frame = read_trip_file(path)
    assert next(iter(frame.columns)) == "Viaje_Id"
    assert len(frame) == 1


def test_read_trip_file_rejects_wrong_column_count(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columnas"):
        read_trip_file(path)


def test_clean_trips_removes_duplicates_bad_dates_and_unknown_stations():
    trips = _trips(
        [
            ("1", "2026-01-01 00:00:00", "2026-01-01 00:10:00", "A", "B"),
            ("1", "2026-01-01 00:00:00", "2026-01-01 00:10:00", "A", "B"),
            ("2", "2026-01-01 00:20:00", "2026-01-01 00:10:00", "A", "B"),
            ("3", "no-fecha", "2026-01-01 00:10:00", "A", "B"),
            ("4", "2026-01-01 01:00:00", "2026-01-01 01:05:00", "A", "ZZZ"),
        ]
    )
    cleaned, counts = clean_trips(trips, {"A", "B"})
    assert counts["viajes_duplicados_eliminados"] == 1
    assert counts["viajes_fechas_invalidas"] == 2
    assert counts["viajes_fuera_catalogo"] == 1
    assert list(cleaned["Viaje_Id"]) == ["1"]


def test_load_station_catalog_treats_zero_capacity_as_zero(tmp_path):
    path = tmp_path / "nomenclatura.csv"
    path.write_text('"id","name","dpcapacity"\n"1","x","15"\n"2","y","0"\n', encoding="utf-8")
    catalog = load_station_catalog(path)
    assert dict(zip(catalog["est"], catalog["capacidad"])) == {"1": 15, "2": 0}


def test_feasibility_filters_infeasible_station_days():
    # Capacidad 2, llegan 3 bicis en la hora 1 sin salidas: F pasa de 0 a 3 > C, dia infactible.
    trips = _trips(
        [
            ("1", "2026-01-01 00:10:00", "2026-01-01 01:00:00", "B", "A"),
            ("2", "2026-01-01 00:20:00", "2026-01-01 01:10:00", "C", "A"),
            ("3", "2026-01-01 00:30:00", "2026-01-01 01:20:00", "D", "A"),
            ("4", "2026-01-02 00:10:00", "2026-01-02 00:20:00", "A", "B"),
        ]
    )
    panel, bounds = _panel_from_trips(trips, {"A": 2, "B": 5, "C": 5, "D": 5})
    day1_a = bounds[(bounds["est"] == "A") & (bounds["day"] == pd.Timestamp("2026-01-01"))]
    assert not bool(day1_a["factible"].iloc[0])
    assert not ((panel["est"] == "A") & (panel["day"] == pd.Timestamp("2026-01-01"))).any()


def test_label_marks_empty_or_full_within_horizon_only_same_day():
    panel = pd.DataFrame(
        {
            "est": ["A"] * 4,
            "day": pd.to_datetime(["2026-01-01"] * 3 + ["2026-01-02"]),
            "inventario": [5.0, 0.0, 5.0, 5.0],
            "capacidad": [10, 10, 10, 10],
        }
    )
    labels = _risk_labels(panel, horizon=1)
    # Fila 0 mira la fila 1 (vacia) -> 1; fila 1 esta vacia misma -> 1 (su siguiente no es 0); fila 2 mira 3 -> 0
    assert labels[0] == 1
    assert labels[2] == 0
    assert labels[3] == 0  # sin fila siguiente en el mismo dia


def test_label_detects_full_station():
    panel = pd.DataFrame(
        {
            "est": ["A", "A"],
            "day": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "inventario": [3.0, 10.0],
            "capacidad": [10, 10],
        }
    )
    assert _risk_labels(panel, horizon=1).tolist() == [1, 0]


def test_features_do_not_depend_on_future_rows():
    hours = pd.date_range("2026-01-01", periods=30, freq="h")
    base = pd.DataFrame(
        {
            "est": "A",
            "ts": hours,
            "day": hours.normalize(),
            "salidas": np.arange(30) % 3,
            "llegadas": np.arange(30) % 2,
            "capacidad": 10,
            "flujo_acumulado": np.arange(30, dtype=float),
        }
    )
    altered = base.copy()
    altered.loc[25:, ["salidas", "llegadas", "flujo_acumulado"]] = 999
    feats_a = build_features(base)
    feats_b = build_features(altered)
    cut = 20
    pd.testing.assert_frame_equal(
        feats_a.loc[:cut, list(FEATURE_COLUMNS)].reset_index(drop=True),
        feats_b.loc[:cut, list(FEATURE_COLUMNS)].reset_index(drop=True),
    )


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
            "est": ["A"] * 6,
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
            "est": "A",
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
    features = build_features(panel)
    metrics, context, _ = evaluate_all(panel, features, random_seed=7)
    assert set(metrics["modelo"]) == {"mayoritaria", "persistencia", "logistica", "hist_gb"}
    assert all("test_" not in column for column in metrics.columns)
    assert context["masks"]["test"].sum() > 0


def test_log_p2_run_writes_tags_and_metrics(tmp_path):
    tracking_uri = (tmp_path / "mlruns").as_uri()
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
        input_hashes={"viajes": "abc"},
        experiment_name="test-p2",
        tracking_uri=tracking_uri,
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert run.data.tags["question"] == "P2"
    assert run.data.tags["temporal_holdout"] == "false"
    assert run.data.metrics["val_pr_auc"] == pytest.approx(0.5)
    assert run.data.params["horizon_hours"] == str(P2_HORIZON_HOURS)
