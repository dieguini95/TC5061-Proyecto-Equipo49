import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from mibici_ml.modeling import demand


def test_metrics_include_zero_demand_without_dividing_by_individual_targets():
    metrics = demand.regression_metrics(np.array([0, 2, 4]), np.array([1, 2, 1]))
    assert metrics["mae"] == pytest.approx(4 / 3)
    assert metrics["wape"] == pytest.approx(4 / 6)
    assert metrics["rmse"] == pytest.approx(np.sqrt(10 / 3))
    assert np.isnan(demand.regression_metrics(np.zeros(2), np.ones(2))["wape"])


def test_invalid_predictions_fail_instead_of_silently_changing_cohort():
    with pytest.raises(ValueError):
        demand.regression_metrics(np.array([1, 2]), np.array([1, np.nan]))


def test_baseline_selection_does_not_use_test_results():
    frame = pd.DataFrame(
        [
            {"baseline": "seasonal_168h", "split": "val", "wape": 0.4, "mae": 1.0},
            {"baseline": "mean_4weeks", "split": "val", "wape": 0.3, "mae": 0.9},
            {"baseline": "seasonal_168h", "split": "test", "wape": 0.01, "mae": 0.01},
            {"baseline": "mean_4weeks", "split": "test", "wape": 9.0, "mae": 9.0},
        ]
    )
    assert demand.select_baseline(frame) == "mean_4weeks"


def test_split_includes_end_date_evening_and_excludes_next_midnight():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-06-30 23:00", "2026-07-01 00:00", "2026-08-31 23:00", "2026-09-01 00:00"]
            )
        }
    )
    assert demand.split_mask(frame, "2026-07-01", "2026-08-31").tolist() == [
        False,
        True,
        True,
        False,
    ]


def test_input_hash_rejects_tampered_processed_data(tmp_path, monkeypatch):
    monkeypatch.setattr(demand.config, "START_MONTH", "2026-09")
    monkeypatch.setattr(demand.config, "END_MONTH", "2026-09")
    processed = tmp_path / "processed"
    (processed / "trips").mkdir(parents=True)
    parquet = processed / "trips" / "2026_09.parquet"
    parquet.write_bytes(b"version_one")
    station_file = processed / "stations.parquet"
    station_file.write_bytes(b"station_fixture")
    manifest = {
        "cleaning_version": "fixture-v1",
        "date_range": {"start_month": "2026-09", "end_month": "2026-09"},
        "trips": [
            {
                "month": "2026_09",
                "raw_sha256": "fixture",
                "processed_path": "trips/2026_09.parquet",
                "processed_sha256": demand.file_sha256(parquet),
            }
        ],
        "stations": {
            "raw_sha256": "fixture",
            "processed_path": "stations.parquet",
            "processed_sha256": demand.file_sha256(station_file),
        },
    }
    manifest["data_version"] = demand._data_version(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    demand.verify_inputs(path, processed)
    parquet.write_bytes(b"version_two")
    with pytest.raises(ValueError, match="modificados"):
        demand.verify_inputs(path, processed)


def test_mlflow_contains_metrics_tags_artifacts_and_exact_source(tmp_path, monkeypatch):
    monkeypatch.setattr(demand.tempfile, "tempdir", str(tmp_path))
    import mlflow

    source_root = tmp_path / "source"
    source = source_root / "mibici_ml" / "test.py"
    source.parent.mkdir(parents=True)
    source.write_text("# exact source\n", encoding="utf-8")
    source_root.joinpath(".env").write_text("SECRET=not-for-artifacts", encoding="utf-8")
    source_root.joinpath("README.md").write_text("# Project\n", encoding="utf-8")
    source_root.joinpath("LICENSE").write_text("MIT fixture\n", encoding="utf-8")
    for attribute in ("P1_METRICS_PATH", "P1_STATION_METRICS_PATH"):
        artifact = tmp_path / f"{attribute}.csv"
        artifact.write_text("aggregate,only\n1,2\n", encoding="utf-8")
        monkeypatch.setattr(demand.config, attribute, artifact)
    figure = tmp_path / "figure.txt"
    figure.write_text("aggregate figure fixture", encoding="utf-8")
    monkeypatch.setattr(demand.config, "P1_FIGURES", {"fixture": figure})
    monkeypatch.setattr(demand.config, "EXPERIMENT_NAME", "p1-contract-equipo49")
    rows = [
        {
            "baseline": model,
            "split": split,
            "mae": 1.0,
            "rmse": 1.2,
            "wape": 0.5,
            "bias": 0.0,
            "n_rows": 2,
            "n_stations": 1,
            "actual_total": 4.0,
        }
        for model in demand.config.P1_MODEL_NAMES
        for split in ("val", "test")
    ]
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    runs = demand.log_runs(
        pd.DataFrame(rows),
        {"data_version": "fixture-v1"},
        {"fixture": True},
        "mean_4weeks",
        uri,
        root=source_root,
    )
    assert len(runs) == 2
    client = mlflow.MlflowClient(tracking_uri=uri)
    for result in runs:
        run = client.get_run(result["run_id"])
        assert run.info.status == "FINISHED"
        assert run.data.tags["pregunta"] == "P1"
        assert run.data.tags["equipo"] == "49"
        assert run.data.params["data_version"] == "fixture-v1"
        assert run.data.metrics["test_mae"] == 1.0
        artifact = client.download_artifacts(run.info.run_id, "reproducibility/code_snapshot.zip")
        with zipfile.ZipFile(Path(artifact)) as bundle:
            assert set(bundle.namelist()) == {"mibici_ml/test.py", "README.md", "LICENSE"}
            assert bundle.read("mibici_ml/test.py") == source.read_bytes()
            assert bundle.read("README.md") == source_root.joinpath("README.md").read_bytes()


def test_documentation_context_is_fingerprinted_without_changing_code_version(tmp_path):
    module = tmp_path / "mibici_ml" / "example.py"
    module.parent.mkdir()
    module.write_text("# fixture\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# Initial documentation\n", encoding="utf-8")
    first, _ = demand.code_provenance(tmp_path)
    readme.write_text("# Updated run identifiers\n", encoding="utf-8")
    second, _ = demand.code_provenance(tmp_path)
    assert first["source_version"] == second["source_version"]
    assert first["context_hashes"] != second["context_hashes"]


def test_invalid_forecast_fails_before_reading_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(demand.config, "P1_FORECAST_HORIZON_HOURS", 12)
    monkeypatch.setattr(
        demand, "verify_inputs", lambda *args: pytest.fail("Must validate forecast before data")
    )
    with pytest.raises(ValueError, match="24 h"):
        demand.main(tmp_path / "missing.json", False, f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")


def test_invalid_forecast_fails_before_creating_tracking_store(tmp_path, monkeypatch):
    monkeypatch.setattr(demand.config, "P1_FORECAST_ORIGIN_HOUR", 6)
    store = tmp_path / "mlflow.db"
    with pytest.raises(ValueError, match="24 h"):
        demand.log_runs(pd.DataFrame(), {}, {}, "mean_4weeks", f"sqlite:///{store.as_posix()}")
    assert not store.exists()
