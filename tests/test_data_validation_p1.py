"""Calidad de la entrada real de P1: cobertura, contrato temporal y DeepChecks.

Los casos de DeepChecks se ejecutan en ``.venv-quality``. Las demas pruebas
tambien funcionan con el entorno principal, sin entrenar ni publicar modelos.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mibici_ml import config
from mibici_ml import data_validation_p1 as validation
from mibici_ml.dataset import _data_version
from mibici_ml.features import add_p1_features, complete_station_hour_panel


@pytest.fixture()
def panel():
    """Una estacion anual y otra nueva sin las cuatro semanas de historia."""
    hours = pd.date_range(config.TRAIN_START, config.TEST_END + " 23:00", freq="h")
    counts = pd.DataFrame(
        {"station_id": 1, "timestamp": hours, "departures": 1 + np.arange(len(hours)) % 11}
    )
    late = pd.DataFrame(
        {"station_id": [2], "timestamp": [pd.Timestamp("2026-09-15 10:00")], "departures": [2]}
    )
    return add_p1_features(
        complete_station_hour_panel(
            pd.concat([counts, late], ignore_index=True), config.START_MONTH, config.END_MONTH
        )
    )


def _first_eligible(panel):
    return panel.index[panel.evaluation_eligible][0]


def test_real_panel_preserves_expected_warmup_and_exact_model_input(panel):
    before = panel.copy(deep=True)
    frame, coverage = validation.build_p1_validation_frame(panel)
    assert set(frame.columns) == {*config.P1_FEATURE_COLUMNS, "departures", "ts"}
    assert len(frame) == int(panel.evaluation_eligible.sum())
    assert frame.ts.min() == pd.Timestamp("2025-10-30")
    assert frame.ts.max() == pd.Timestamp("2026-09-30 23:00")
    assert not panel.loc[panel.station_id.eq(2), "evaluation_eligible"].any()
    assert panel.loc[~panel.evaluation_eligible, "mean_4weeks"].isna().any()
    assert frame[list(config.P1_FEATURE_COLUMNS)].dtypes.eq(np.dtype("float64")).all()
    assert np.isfinite(frame.drop(columns="ts").to_numpy()).all()
    assert frame.departures.ge(0).all()
    assert coverage["filas_panel"] == len(panel)
    assert coverage["estaciones_panel"] == 2
    assert coverage["filas_elegibles"] == len(frame)
    assert (
        coverage["filas_elegibles"]
        + coverage["filas_estacion_desconocida"]
        + coverage["filas_historia_insuficiente"]
        == coverage["filas_panel"]
    )
    pd.testing.assert_frame_equal(panel, before)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1.0, 0.5])
def test_invalid_eligible_target_fails_instead_of_dropping_observation(panel, value):
    panel["departures"] = panel.departures.astype("float64")
    panel.loc[_first_eligible(panel), "departures"] = value
    with pytest.raises(ValueError):
        validation.build_p1_validation_frame(panel)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_eligible_history_is_an_error_not_expected_warmup(panel, value):
    panel.loc[_first_eligible(panel), "lag_24"] = value
    with pytest.raises(ValueError):
        validation.build_p1_validation_frame(panel)


@pytest.mark.parametrize("column,value", [("hour", 24), ("day_of_week", 7), ("is_weekend", 0.5)])
def test_calendar_variables_must_match_the_target_timestamp(panel, column, value):
    panel[column] = panel[column].astype(float)
    panel.loc[_first_eligible(panel), column] = value
    with pytest.raises(ValueError):
        validation.build_p1_validation_frame(panel)


def test_negative_historical_counts_are_rejected(panel):
    panel.loc[_first_eligible(panel), "lag_24"] = -1
    with pytest.raises(ValueError):
        validation.build_p1_validation_frame(panel)


@pytest.mark.parametrize("kind", ["duplicate", "gap", "fractional_hour"])
def test_invalid_station_hour_grid_is_rejected(panel, kind):
    row = _first_eligible(panel) + 10
    if kind == "duplicate":
        panel = pd.concat([panel, panel.loc[[row]]], ignore_index=True)
    elif kind == "gap":
        panel = panel.drop(index=row)
    else:
        panel.loc[row, "timestamp"] += pd.Timedelta(minutes=15)
    with pytest.raises(ValueError):
        validation.build_p1_validation_frame(panel)


def test_forecast_origin_cannot_be_shifted_into_the_forecast_day(panel):
    panel.loc[_first_eligible(panel), "forecast_origin"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError):
        validation.build_p1_validation_frame(panel)


@pytest.mark.parametrize("column", ["eligible", "evaluation_eligible"])
def test_missing_eligibility_metadata_cannot_silently_remove_rows(panel, column):
    panel[column] = panel[column].astype("boolean")
    panel.loc[_first_eligible(panel), column] = pd.NA
    with pytest.raises(ValueError):
        validation.build_p1_validation_frame(panel)


def test_complete_history_cannot_be_excluded_by_changing_only_the_flag(panel):
    panel.loc[_first_eligible(panel), "evaluation_eligible"] = False
    with pytest.raises(ValueError):
        validation.build_p1_validation_frame(panel)


def test_no_evaluable_history_is_rejected_with_no_empty_deepchecks_dataset(panel):
    early = panel.loc[panel.timestamp.lt("2025-10-20")].copy()
    with pytest.raises(ValueError):
        validation.build_p1_validation_frame(early)


def test_temporal_splits_include_last_hour_and_preserve_all_eligible_rows(panel):
    frame, _ = validation.build_p1_validation_frame(panel)
    splits = validation.split_p1_frame(frame)
    expected = {
        "train": (pd.Timestamp("2025-10-30"), pd.Timestamp("2026-06-30 23:00")),
        "val": (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-31 23:00")),
        "test": (pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-30 23:00")),
    }
    assert set(splits) == set(expected)
    assert sum(map(len, splits.values())) == len(frame)
    for name, (start, end) in expected.items():
        assert splits[name].ts.min() == start
        assert splits[name].ts.max() == end
    assert splits["train"].ts.max() < splits["val"].ts.min()
    assert splits["val"].ts.max() < splits["test"].ts.min()


def test_missing_validation_window_is_rejected(panel):
    frame, _ = validation.build_p1_validation_frame(panel)
    frame = frame.loc[~frame.ts.between("2026-07-01", "2026-08-31 23:00")]
    with pytest.raises(ValueError):
        validation.split_p1_frame(frame)


def test_p1_sampling_is_repeatable_without_mutating_input(panel):
    frame, _ = validation.build_p1_validation_frame(panel)
    before = frame.copy(deep=True)
    first = validation.sample_p1_frame(frame, 100, 42)
    second = validation.sample_p1_frame(frame, 100, 42)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(frame, before)
    assert len(first) == 100
    assert len(validation.sample_p1_frame(frame, len(frame) + 1, 42)) == len(frame)


@pytest.mark.parametrize("sample_size", [0, -1])
def test_invalid_sample_size_fails_before_loading_data(tmp_path, sample_size):
    with pytest.raises(ValueError):
        validation.run_validation(
            sample_size=sample_size,
            processed_dir=tmp_path,
            manifest_path=tmp_path / "does-not-exist.json",
            output_dir=tmp_path / "html",
            summary_path=tmp_path / "summary.json",
        )
    assert not (tmp_path / "html").exists()
    assert not (tmp_path / "summary.json").exists()


def _manifest_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "START_MONTH", "2026-09")
    monkeypatch.setattr(config, "END_MONTH", "2026-09")
    processed = tmp_path / "processed"
    (processed / "trips").mkdir(parents=True)
    trip = processed / "trips" / "2026_09.parquet"
    station = processed / "stations.parquet"
    trip.write_bytes(b"verified-trip-fixture")
    station.write_bytes(b"verified-station-fixture")
    manifest = {
        "cleaning_version": "test",
        "date_range": {"start_month": "2026-09", "end_month": "2026-09"},
        "trips": [
            {
                "month": "2026_09",
                "raw_sha256": "raw-trip",
                "processed_path": "trips/2026_09.parquet",
                "processed_sha256": hashlib.sha256(trip.read_bytes()).hexdigest(),
            }
        ],
        "stations": {
            "raw_sha256": "raw-stations",
            "processed_path": "stations.parquet",
            "processed_sha256": hashlib.sha256(station.read_bytes()).hexdigest(),
        },
    }
    manifest["data_version"] = _data_version(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return processed, path, manifest


@pytest.mark.parametrize("tamper", ["catalogue", "manifest_identity"])
def test_p1_validation_rejects_wrong_identity_before_building_panel(tmp_path, monkeypatch, tamper):
    processed, manifest_path, manifest = _manifest_fixture(tmp_path, monkeypatch)
    if tamper == "catalogue":
        (processed / "stations.parquet").write_bytes(b"changed-capacity")
    else:
        manifest["data_version"] = "different-version"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        validation,
        "build_station_hour_panel",
        lambda *args, **kwargs: pytest.fail("Invalid data reached feature building"),
    )
    with pytest.raises(ValueError):
        validation.run_validation(
            processed_dir=processed,
            manifest_path=manifest_path,
            output_dir=tmp_path / "html",
            summary_path=tmp_path / "summary.json",
        )
    assert not (tmp_path / "summary.json").exists()


def _regression_frame(n=700, seed=7, start="2025-11-01"):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame(
        {
            "lag_24": rng.integers(0, 20, n).astype(float),
            "lag_48": rng.integers(0, 20, n).astype(float),
            "lag_168": rng.integers(0, 20, n).astype(float),
            "mean_4weeks": rng.uniform(0, 20, n),
            "hour": timestamps.hour.astype(float),
            "day_of_week": timestamps.dayofweek.astype(float),
            "is_weekend": (timestamps.dayofweek >= 5).astype(float),
            "departures": rng.integers(0, 4, n).astype(float),
            "ts": timestamps,
        }
    )


def test_deepchecks_counts_remain_regression_and_metadata_never_becomes_a_feature():
    pytest.importorskip("deepchecks")
    frame = _regression_frame()
    frame["station_id"] = 999
    dataset = validation.p1_dataset(frame)
    assert dataset.label_type.value == "regression"
    assert list(dataset.features) == list(config.P1_FEATURE_COLUMNS)
    assert not {"station_id", "ts", "departures"}.intersection(dataset.features)


def test_deepchecks_p1_detects_introduced_nulls_and_train_test_drift():
    pytest.importorskip("deepchecks")
    train = _regression_frame(seed=7)
    val = _regression_frame(seed=8, start="2026-07-01")
    test = _regression_frame(seed=9, start="2026-09-01")
    test["lag_24"] += 1000
    all_rows = pd.concat([train, val, test], ignore_index=True)
    all_rows.loc[all_rows.index[::2], "lag_48"] = np.nan
    results = validation.run_p1_suites({"all": all_rows, "train": train, "val": val, "test": test})
    summaries = [validation.summarize_p1_suite(result) for result in results.values()]
    assert len(summaries) == 3
    assert all(summary["chequeos_error"] == 0 for summary in summaries)
    failures = [
        check["chequeo"]
        for summary in summaries
        for check in summary["chequeos"]
        if check["estado"] == "fallido"
    ]
    assert any("Null" in name for name in failures), failures
    assert any("Drift" in name for name in failures), failures


def test_regression_inapplicability_is_distinct_from_real_execution_errors():
    pytest.importorskip("deepchecks")
    from deepchecks.core import CheckFailure
    from deepchecks.core.errors import (
        DatasetValidationError,
        DeepchecksNotSupportedError,
        ModelValidationError,
    )
    from deepchecks.tabular.checks import PercentOfNulls

    errors = [
        DeepchecksNotSupportedError("Dataset does not contain an index"),
        DatasetValidationError("Dataset does not contain an index"),
        ModelValidationError("Check is irrelevant for regression tasks"),
        ModelValidationError("Model failed during execution"),
        RuntimeError("Database does not contain a required artifact"),
    ]
    result = SimpleNamespace(
        name="Synthetic failures",
        results=[CheckFailure(PercentOfNulls(), error) for error in errors],
    )
    summary = validation.summarize_p1_suite(result)
    assert summary["chequeos_total"] == 5
    assert summary["chequeos_no_aplicable"] == 3
    assert summary["chequeos_error"] == 2
    assert [check["estado"] for check in summary["chequeos"]] == [
        "no_aplicable",
        "no_aplicable",
        "no_aplicable",
        "error",
        "error",
    ]


@pytest.mark.parametrize("technical_error", [False, True])
def test_orchestration_preserves_aggregate_diagnostics_and_supports_external_output(
    panel, tmp_path, monkeypatch, technical_error
):
    """A failure must not look successful; portable HTML paths must not expose local paths."""

    class Result:
        def save_as_html(self, path):
            Path(path).write_text("<html>synthetic test report</html>", encoding="utf-8")

    def suites(samples, seed):
        assert set(samples) == {"all", "train", "val", "test"}
        assert all(len(part) == 50 for part in samples.values())
        assert seed == 42
        return {"p1_integridad": Result()}

    monkeypatch.setattr(
        validation,
        "verify_inputs",
        lambda *args: {
            "data_version": "verified-fixture",
            "trips": [{}] * 12,
        },
    )
    monkeypatch.setattr(validation, "build_station_hour_panel", lambda *args: panel)
    monkeypatch.setattr(validation, "run_p1_suites", suites)
    monkeypatch.setattr(validation, "_provenance", lambda: {"git_dirty": True})
    monkeypatch.setattr(
        validation,
        "summarize_p1_suite",
        lambda result: {
            "chequeos_total": 1,
            "chequeos_aprobado": 0 if technical_error else 1,
            "chequeos_error": int(technical_error),
            "chequeos": [],
        },
    )
    output_dir = tmp_path / "outside-repository"
    summary_path = tmp_path / "summary.json"
    kwargs = {
        "sample_size": 50,
        "seed": 42,
        "output_dir": output_dir,
        "summary_path": summary_path,
    }
    if technical_error:
        with pytest.raises(RuntimeError, match="tecnicos"):
            validation.run_validation(**kwargs)
    else:
        validation.run_validation(**kwargs)
    text = summary_path.read_text(encoding="utf-8")
    saved = json.loads(text)
    assert str(tmp_path) not in text
    assert tmp_path.name not in text
    assert saved["data_version"] == "verified-fixture"
    assert saved["suites"][0]["chequeos_error"] == int(technical_error)
    assert saved["suites"][0]["html_archivo"] == "p1_integridad.html"
    assert (output_dir / "p1_integridad.html").is_file()
