from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip(
    "evidently",
    reason="CLI integration tests require the isolated monitoring environment.",
)
from mibici_ml.monitoring import cli, outputs
from mibici_ml.monitoring.outputs import OUTPUT_NAMES


def test_cli_runs_all_p2_features_and_keeps_outputs_aggregate_only(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    feature_names = cli.load_p2_feature_names()
    assert len(feature_names) == 14
    random = np.random.default_rng(42)
    reference_dates = pd.date_range("2025-10-01", periods=40, freq="D")
    current_dates = pd.date_range("2026-09-01", periods=30, freq="D")
    panel: dict[str, object] = {"ts": reference_dates.append(current_dates)}
    for index, feature in enumerate(feature_names):
        panel[feature] = np.concatenate(
            (
                random.normal(index, 0.25, len(reference_dates)),
                random.normal(index + 0.75, 0.25, len(current_dates)),
            )
        )
    row_count = len(reference_dates) + len(current_dates)
    panel.update(
        {
            "est": [f"SYNTH-EST-CANARY-{row}" for row in range(row_count)],
            "station_id": [f"SYNTH-STATION-CANARY-{row}" for row in range(row_count)],
            "nombre": [f"SYNTH-NAME-CANARY-{row}" for row in range(row_count)],
            "lat": [20.123456789 + row / 1_000_000 for row in range(row_count)],
            "lon": [-103.987654321 - row / 1_000_000 for row in range(row_count)],
            "label": [row % 2 for row in range(row_count)],
            "data_version": ["synthetic-version-canary"] * row_count,
            "artifact_uri": [
                f"synthetic://SYNTHETIC-TOKEN-CANARY-{row}" for row in range(row_count)
            ],
            "source_run_id": [f"SYNTHETIC-RUN-CANARY-{row}" for row in range(row_count)],
            "source_file": ["SYNTHETIC-RAW-ROW-CANARY.csv"] * row_count,
            "source_path": ["/synthetic/private/panel.parquet"] * row_count,
            "mlflow_tracking_uri": ["MLFLOW_TRACKING_URI-CANARY"] * row_count,
        }
    )
    input_path = tmp_path / "synthetic_p2_panel.csv"
    output_dir = tmp_path / "reports" / "drift" / "p2"
    pd.DataFrame(panel).to_csv(input_path, index=False)
    monkeypatch.setattr(cli, "DEFAULT_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(outputs, "ALLOWED_OUTPUT_ROOT", tmp_path / "reports" / "drift")
    monkeypatch.setattr(cli, "_git_commit", lambda: "synthetic-commit")

    result = cli.main(["--input", str(input_path)])

    assert result == 0
    output_paths = tuple(output_dir / name for name in OUTPUT_NAMES)
    assert {path.name for path in output_dir.iterdir()} == set(OUTPUT_NAMES)
    assert all(path.resolve().is_relative_to(tmp_path.resolve()) for path in output_paths)
    assert (output_dir / "drift_report.html").stat().st_size > 0
    console = capsys.readouterr().out
    assert "role=historical_test" in console

    html = (output_dir / "drift_report.html").read_text(encoding="utf-8")
    report_json = (output_dir / "drift_report.json").read_text(encoding="utf-8")
    csv_text = (output_dir / "drift_columns.csv").read_text(encoding="utf-8")
    metadata_text = (output_dir / "metadata.json").read_text(encoding="utf-8")
    outputs_text = (html, report_json, csv_text, metadata_text)
    sensitive_values = (
        "SYNTH-EST-CANARY",
        "SYNTH-STATION-CANARY",
        "SYNTH-NAME-CANARY",
        "20.123456789",
        "-103.987654321",
        "SYNTHETIC-TOKEN-CANARY",
        "SYNTHETIC-RUN-CANARY",
        "/synthetic/private/panel.parquet",
        "MLFLOW_TRACKING_URI",
        "SYNTHETIC-RAW-ROW-CANARY",
        "2025-10-17",
        "2026-09-17",
        str(input_path),
        "/home/ruan",
        "D1LZGW44",
    )
    for sensitive_value in sensitive_values:
        assert sensitive_value not in console
        assert sensitive_value not in html
        assert sensitive_value not in report_json
        assert sensitive_value not in csv_text
        assert sensitive_value not in metadata_text

    report = json.loads(report_json)
    summary = report["summary"]
    assert summary["total_columns"] == 14
    assert [item["feature"] for item in summary["columns"]] == list(feature_names)
    assert summary["reference"]["start"] == "2025-10-01"
    assert summary["current"]["start"] == "2026-09-01"
    assert summary["current"]["role"] == "historical_test"
    assert summary["reference_rows"] == 40
    assert summary["current_rows"] == 30
    assert "data_version" not in summary
    assert "label" not in summary

    metadata = json.loads(metadata_text)
    assert metadata["model"] == "P2"
    assert metadata["data_version"] == "synthetic-version-canary"
    assert metadata["feature_names"] == list(feature_names)
    assert metadata["reference"]["start"] == "2025-10-01"
    assert metadata["current"]["role"] == "historical_test"
    assert metadata["reference_source_rows"] == 40
    assert metadata["current_source_rows"] == 30
    assert metadata["random_seed"] == 42
    assert str(input_path) not in metadata_text

    with (output_dir / "drift_columns.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["feature"] for row in rows] == list(feature_names)
    assert len(rows) == 14


def test_cli_requires_explicit_overwrite_for_existing_outputs(
    tmp_path: Path,
    monkeypatch,
    synthetic_panel: pd.DataFrame,
    capsys,
) -> None:
    input_path = tmp_path / "synthetic_monitoring_fixture.csv"
    output_dir = tmp_path / "reports" / "drift" / "p2"
    synthetic_panel.to_csv(input_path, index=False)
    output_dir.mkdir(parents=True)
    (output_dir / "metadata.json").write_text("existing", encoding="utf-8")
    monkeypatch.setattr(cli, "load_p2_feature_names", lambda: ("feature_a", "feature_b"))
    monkeypatch.setattr(cli, "DEFAULT_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(outputs, "ALLOWED_OUTPUT_ROOT", tmp_path / "reports" / "drift")

    result = cli.main(["--input", str(input_path)])

    assert result == 2
    assert "use --overwrite" in capsys.readouterr().err
