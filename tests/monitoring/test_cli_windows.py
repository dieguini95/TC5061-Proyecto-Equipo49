"""CLI de monitoreo: ventanas configurables, rol out_of_time y subcarpeta por corrida.

Datos sinteticos; no usa ``data/``, ``mlruns/`` ni ``reports/`` reales.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("evidently")

from mibici_ml.monitoring import cli, outputs
from mibici_ml.monitoring.errors import MonitoringError
from mibici_ml.monitoring.schemas import WindowDefinition


def test_run_output_dir_uses_safe_subfolder_only(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DEFAULT_OUTPUT_DIR", tmp_path / "p2")

    assert cli._run_output_dir(None) == tmp_path / "p2"
    assert cli._run_output_dir("r1") == tmp_path / "p2" / "r1"
    for unsafe in ("../escape", "R1", "a/b", "", "x" * 33, ".hidden"):
        with pytest.raises(MonitoringError):
            cli._run_output_dir(unsafe)


def test_out_of_time_is_a_valid_current_role_and_unknown_roles_are_rejected():
    window = WindowDefinition(
        name="current", start="2026-10-01", end="2026-10-31", role="out_of_time"
    )
    assert window.role == "out_of_time"
    with pytest.raises(MonitoringError):
        WindowDefinition(name="current", start="2026-10-01", end="2026-10-31", role="production")


def test_cli_compares_training_reference_with_out_of_time_october(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    feature_names = cli.load_p2_feature_names()
    rng = np.random.default_rng(7)
    reference_dates = pd.date_range("2025-10-01", periods=40, freq="D")
    current_dates = pd.date_range("2026-10-01", periods=30, freq="D")
    dates = reference_dates.append(current_dates)
    panel: dict[str, object] = {"ts": dates, "data_version": ["synthetic-version"] * len(dates)}
    for index, feature in enumerate(feature_names):
        panel[feature] = np.concatenate(
            (
                rng.normal(index, 0.25, len(reference_dates)),
                rng.normal(index, 0.25, len(current_dates)),
            )
        )
    input_path = tmp_path / "synthetic_panel.csv"
    pd.DataFrame(panel).to_csv(input_path, index=False)
    root = tmp_path / "reports" / "drift"
    monkeypatch.setattr(cli, "DEFAULT_OUTPUT_DIR", root / "p2")
    monkeypatch.setattr(outputs, "ALLOWED_OUTPUT_ROOT", root)
    monkeypatch.setattr(cli, "_git_commit", lambda: "synthetic-commit")

    result = cli.main(
        [
            "--input",
            str(input_path),
            "--reference-start",
            "2025-10-01",
            "--reference-end",
            "2025-11-09",
            "--current-start",
            "2026-10-01",
            "--current-end",
            "2026-10-30",
            "--current-role",
            "out_of_time",
            "--run-name",
            "r1",
        ]
    )

    assert result == 0
    output_dir = root / "p2" / "r1"
    assert (output_dir / "drift_report.html").stat().st_size > 0
    console = capsys.readouterr().out
    assert "role=out_of_time" in console
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    limitations = " ".join(metadata["limitations"])
    assert "out-of-time period after training" in limitations
    assert "historical_test" not in limitations
