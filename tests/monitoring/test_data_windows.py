from __future__ import annotations

import pandas as pd
import pytest

from mibici_ml.monitoring.data_windows import prepare_windows
from mibici_ml.monitoring.errors import MonitoringError
from mibici_ml.monitoring.schemas import DriftConfig, WindowDefinition

REFERENCE = WindowDefinition("reference", "2025-10-01", "2026-06-30", "reference")
CURRENT = WindowDefinition("current", "2026-09-01", "2026-09-30", "historical_test")


def _config(**overrides: object) -> DriftConfig:
    values: dict[str, object] = {
        "feature_names": ("feature_a", "feature_b"),
        "excluded_columns": ("station_id", "nombre", "lat", "lon"),
        "sample_limit": 150_000,
        "random_seed": 42,
    }
    values.update(overrides)
    return DriftConfig(**values)


def test_splits_inclusive_windows_and_excludes_sensitive_columns(synthetic_panel: pd.DataFrame) -> None:
    prepared = prepare_windows(
        synthetic_panel,
        REFERENCE,
        CURRENT,
        _config(),
        expected_data_version="synthetic-v1",
    )

    assert prepared.reference_source_rows == 40
    assert prepared.current_source_rows == 30
    assert list(prepared.reference.columns) == ["feature_a", "feature_b"]
    assert list(prepared.current.columns) == ["feature_a", "feature_b"]
    assert prepared.data_version == "synthetic-v1"


def test_sampling_is_deterministic(synthetic_panel: pd.DataFrame) -> None:
    config = _config(sample_limit=5, random_seed=7)
    first = prepare_windows(synthetic_panel, REFERENCE, CURRENT, config)
    second = prepare_windows(synthetic_panel, REFERENCE, CURRENT, config)

    pd.testing.assert_frame_equal(first.reference, second.reference)
    pd.testing.assert_frame_equal(first.current, second.current)
    assert len(first.reference) == len(first.current) == 5


@pytest.mark.parametrize(
    ("reference", "current"),
    [
        (
            WindowDefinition("reference", "2025-10-01", "2026-06-30", "historical_test"),
            CURRENT,
        ),
        (
            REFERENCE,
            WindowDefinition("current", "2026-09-01", "2026-09-30", "reference"),
        ),
    ],
)
def test_rejects_wrong_window_roles(
    synthetic_panel: pd.DataFrame,
    reference: WindowDefinition,
    current: WindowDefinition,
) -> None:
    with pytest.raises(MonitoringError):
        prepare_windows(synthetic_panel, reference, current, _config())


def test_rejects_overlap(synthetic_panel: pd.DataFrame) -> None:
    overlapping = WindowDefinition("current", "2026-06-30", "2026-09-30", "historical_test")
    with pytest.raises(MonitoringError, match="overlap"):
        prepare_windows(synthetic_panel, REFERENCE, overlapping, _config())


def test_rejects_unexpected_columns(synthetic_panel: pd.DataFrame) -> None:
    frame = synthetic_panel.assign(unreviewed_identifier="synthetic-only")
    with pytest.raises(MonitoringError, match="unexpected columns"):
        prepare_windows(frame, REFERENCE, CURRENT, _config())


def test_rejects_missing_features(synthetic_panel: pd.DataFrame) -> None:
    with pytest.raises(MonitoringError, match="missing required columns"):
        prepare_windows(
            synthetic_panel.drop(columns="feature_b"),
            REFERENCE,
            CURRENT,
            _config(),
        )


def test_rejects_invalid_timestamp(synthetic_panel: pd.DataFrame) -> None:
    frame = synthetic_panel.copy()
    frame["ts"] = frame["ts"].astype(object)
    frame.loc[0, "ts"] = "not-a-date"
    with pytest.raises(MonitoringError, match="invalid dates"):
        prepare_windows(frame, REFERENCE, CURRENT, _config())


def test_rejects_non_numeric_or_non_finite_features(synthetic_panel: pd.DataFrame) -> None:
    text_feature = synthetic_panel.copy()
    text_feature["feature_a"] = text_feature["feature_a"].astype(str)
    with pytest.raises(MonitoringError, match="numeric"):
        prepare_windows(text_feature, REFERENCE, CURRENT, _config())

    non_finite = synthetic_panel.copy()
    non_finite["feature_a"] = non_finite["feature_a"].astype(float)
    non_finite.loc[0, "feature_a"] = float("inf")
    with pytest.raises(MonitoringError, match="non-finite"):
        prepare_windows(non_finite, REFERENCE, CURRENT, _config())


def test_rejects_data_version_mismatch(synthetic_panel: pd.DataFrame) -> None:
    with pytest.raises(MonitoringError, match="does not match"):
        prepare_windows(
            synthetic_panel,
            REFERENCE,
            CURRENT,
            _config(),
            expected_data_version="another-version",
        )


def test_requires_data_version_or_explicit_expectation(synthetic_panel: pd.DataFrame) -> None:
    frame = synthetic_panel.drop(columns="data_version")
    with pytest.raises(MonitoringError, match="data_version"):
        prepare_windows(frame, REFERENCE, CURRENT, _config())

    prepared = prepare_windows(
        frame,
        REFERENCE,
        CURRENT,
        _config(),
        expected_data_version="synthetic-v1",
    )
    assert prepared.data_version == "synthetic-v1"
