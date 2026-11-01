"""Pruebas de la validacion de datos (DeepChecks) con datos sinteticos y deterministas.

Las pruebas que requieren DeepChecks se omiten con ``importorskip`` en el entorno
principal. Ejecutarlas con el entorno de calidad:

    .venv-quality/bin/python -m pytest -q tests/test_data_validation.py
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from mibici_ml.config import TEST_END, TEST_START, TRAIN_END, TRAIN_START
from mibici_ml.data_validation import (
    basic_findings,
    build_risk_frame,
    build_trip_frame,
    load_verified_trips,
    read_manifest,
    run_integrity_suite,
    run_train_test_suite,
    summarize_suite,
    temporal_split,
)
from mibici_ml.features import P2_FEATURE_COLUMNS


def _synthetic_frame(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Marco limpio con columna temporal, una numerica, una categorica y etiqueta binaria."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=n, freq="h"),
            "x": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
            "label": rng.integers(0, 2, size=n),
        }
    )


def _statuses(summary: dict) -> dict[str, str]:
    return {check["chequeo"]: check["estado"] for check in summary["chequeos"]}


def _trips_fixture() -> pd.DataFrame:
    """Viajes minimos y balanceados entre dos estaciones para que el panel sea factible."""
    start = pd.Timestamp("2026-01-05 06:00")
    rows = []
    for i in range(40):
        origin, destination = (1, 2) if i % 2 == 0 else (2, 1)
        begin = start + pd.Timedelta(minutes=7 * i)
        rows.append(
            {
                "origin_id": origin,
                "destination_id": destination,
                "start_time": begin,
                "end_time": begin + pd.Timedelta(minutes=5),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pruebas sin DeepChecks (entorno principal)
# ---------------------------------------------------------------------------


def test_temporal_split_is_disjoint_and_follows_configured_windows():
    ts = pd.Series(pd.date_range(TRAIN_START, TEST_END, freq="h"))
    split = temporal_split(ts)
    assert not (split["train"] & split["test"]).any()
    assert ts[split["train"]].min() == pd.Timestamp(TRAIN_START)
    assert ts[split["train"]].max() <= pd.Timestamp(TRAIN_END) + pd.Timedelta(hours=23)
    assert ts[split["test"]].min() == pd.Timestamp(TEST_START)
    assert ts[split["test"]].max() <= pd.Timestamp(TEST_END) + pd.Timedelta(hours=23)


def test_build_trip_frame_derives_duration_hour_day_and_categories():
    trips = pd.DataFrame(
        {
            "origin_id": [7, 8],
            "destination_id": [9, 9],
            "start_time": pd.to_datetime(["2026-01-05 08:30", "2026-01-10 23:00"]),
            "end_time": pd.to_datetime(["2026-01-05 08:45", "2026-01-10 23:30"]),
        }
    )
    frame = build_trip_frame(trips)
    assert frame["duracion_min"].tolist() == [15.0, 30.0]
    assert frame["hora_inicio"].tolist() == [8, 23]
    assert frame["dia_semana"].tolist() == [0, 5]
    assert frame["origen"].tolist() == ["7", "8"]
    assert frame["destino"].tolist() == ["9", "9"]


def test_build_trip_frame_rejects_empty_input():
    empty = _trips_fixture().iloc[0:0]
    with pytest.raises(ValueError):
        build_trip_frame(empty)


def test_basic_findings_counts_nulls_duplicates_and_positive_rate():
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=4, freq="h"),
            "x": [1.0, 1.0, np.nan, 2.0],
            "label": [1, 1, 0, 0],
        }
    )
    findings = basic_findings(frame, label="label")
    assert findings["filas"] == 4
    assert findings["nulos"] == 1
    # Sin ts, las filas 0 y 1 son identicas (x=1.0, label=1): un duplicado.
    assert findings["filas_duplicadas"] == 1
    assert findings["tasa_positivos"] == pytest.approx(0.5)


def test_sampling_is_bounded_and_reproducible_with_same_seed():
    from mibici_ml.data_validation import _sample

    frame = _synthetic_frame(n=500)
    first = _sample(frame, sample_size=100, seed=42)
    second = _sample(frame, sample_size=100, seed=42)
    assert len(first) == 100
    pd.testing.assert_frame_equal(first, second)
    assert len(_sample(frame, sample_size=1000, seed=42)) == 500


def test_read_manifest_missing_file_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset"):
        read_manifest(tmp_path / "no_existe.json")


def test_load_verified_trips_rejects_missing_and_modified_files(tmp_path):
    processed = tmp_path / "processed"
    (processed / "trips").mkdir(parents=True)
    trips_path = processed / "trips" / "2026_01.parquet"
    _trips_fixture().to_parquet(trips_path, index=False)
    good_hash = hashlib.sha256(trips_path.read_bytes()).hexdigest()
    manifest = {
        "data_version": "x",
        "trips": [
            {
                "month": "2026_01",
                "processed_path": "trips/2026_01.parquet",
                "processed_sha256": good_hash,
            }
        ],
    }
    assert len(load_verified_trips(processed, manifest)) == 40

    manifest["trips"][0]["processed_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="modificados"):
        load_verified_trips(processed, manifest)

    manifest["trips"][0]["processed_path"] = "trips/2026_09.parquet"
    with pytest.raises(FileNotFoundError, match="ingesta"):
        load_verified_trips(processed, manifest)


def test_build_risk_frame_follows_p2_feature_contract_and_binary_labels():
    trips = _trips_fixture()
    capacities = pd.DataFrame({"est": [1, 2], "capacidad": [10, 10]})
    frame = build_risk_frame(trips, capacities)
    assert set(P2_FEATURE_COLUMNS).issubset(frame.columns)
    assert set(frame["label"].unique()).issubset({0, 1})
    assert frame[list(P2_FEATURE_COLUMNS)].isna().sum().sum() == 0
    assert set(frame["est"]) == {"1", "2"}


def test_run_validation_rejects_non_positive_sample_size(tmp_path):
    from mibici_ml.data_validation import run_validation

    with pytest.raises(ValueError, match="positivo"):
        run_validation(sample_size=0, output_dir=tmp_path, summary_path=tmp_path / "s.json")


# ---------------------------------------------------------------------------
# Pruebas con DeepChecks (entorno de calidad)
# ---------------------------------------------------------------------------


def test_clean_data_has_no_failed_integrity_checks():
    pytest.importorskip("deepchecks")
    frame = _synthetic_frame(seed=1)[["ts", "x", "cat"]]
    summary = summarize_suite(run_integrity_suite(frame, None, ("cat",)))
    assert summary["chequeos_fallido"] == 0
    assert summary["chequeos_error"] == 0


def test_nulls_are_flagged_by_integrity_suite():
    pytest.importorskip("deepchecks")
    frame = _synthetic_frame(seed=2)[["ts", "x", "cat"]]
    frame.loc[frame.index[::2], "x"] = np.nan
    summary = summarize_suite(run_integrity_suite(frame, None, ("cat",)))
    flagged = [c["chequeo"] for c in summary["chequeos"] if c["estado"] == "fallido"]
    assert any("Null" in name for name in flagged), flagged


def test_duplicated_rows_are_counted_and_flagged():
    pytest.importorskip("deepchecks")
    base = _synthetic_frame(n=400, seed=3)[["ts", "x", "cat"]]
    frame = pd.concat([base, base], ignore_index=True)
    assert basic_findings(frame)["filas_duplicadas"] >= 400
    summary = summarize_suite(run_integrity_suite(frame, None, ("cat",)))
    flagged = [c["chequeo"] for c in summary["chequeos"] if c["estado"] == "fallido"]
    assert any("Duplicat" in name for name in flagged), flagged


def test_feature_drift_is_flagged_between_train_and_test():
    pytest.importorskip("deepchecks")
    train = _synthetic_frame(n=600, seed=4)[["ts", "x", "cat"]]
    test = _synthetic_frame(n=600, seed=5)[["ts", "x", "cat"]]
    test["x"] = test["x"] + 3.0
    test["ts"] = test["ts"] + pd.Timedelta(days=400)
    summary = summarize_suite(run_train_test_suite(train, test, None, ("cat",)))
    flagged = [c["chequeo"] for c in summary["chequeos"] if c["estado"] == "fallido"]
    assert any("Drift" in name for name in flagged), flagged


def test_missing_label_is_reported_as_not_applicable_not_as_error():
    pytest.importorskip("deepchecks")
    frame = _synthetic_frame(seed=6)[["ts", "x", "cat"]]
    summary = summarize_suite(run_integrity_suite(frame, None, ("cat",)))
    statuses = _statuses(summary)
    label_checks = [name for name in statuses if "Label" in name]
    assert label_checks, statuses
    assert all(statuses[name] == "no_aplicable" for name in label_checks)
    assert summary["chequeos_error"] == 0


def test_temporal_overlap_between_train_and_test_is_flagged():
    pytest.importorskip("deepchecks")
    train = _synthetic_frame(n=500, seed=7)
    test = _synthetic_frame(n=500, seed=8)
    test["ts"] = train["ts"].to_numpy()
    summary = summarize_suite(run_train_test_suite(train, test, "label", ("cat",)))
    flagged = [c["chequeo"] for c in summary["chequeos"] if c["estado"] == "fallido"]
    assert any("Leakage" in name for name in flagged), flagged


def test_summary_is_json_serializable_and_counts_are_consistent():
    pytest.importorskip("deepchecks")
    frame = _synthetic_frame(seed=9)[["ts", "x", "cat", "label"]]
    summary = summarize_suite(run_integrity_suite(frame, "label", ("cat",)))
    encoded = json.dumps(summary, ensure_ascii=False)
    assert json.loads(encoded)["chequeos_total"] == len(summary["chequeos"])
    counted = sum(
        summary[key]
        for key in (
            "chequeos_aprobado",
            "chequeos_fallido",
            "chequeos_sin_condiciones",
            "chequeos_no_aplicable",
            "chequeos_error",
        )
    )
    assert counted == summary["chequeos_total"]


def test_same_input_gives_same_integrity_summary():
    pytest.importorskip("deepchecks")
    frame = _synthetic_frame(seed=10)[["ts", "x", "cat", "label"]]
    first = summarize_suite(run_integrity_suite(frame, "label", ("cat",)))
    second = summarize_suite(run_integrity_suite(frame.copy(), "label", ("cat",)))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
