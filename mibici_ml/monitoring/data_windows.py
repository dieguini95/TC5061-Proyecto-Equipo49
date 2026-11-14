"""Read-only validation and deterministic sampling of reference/current windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from mibici_ml.monitoring.errors import MonitoringError
from mibici_ml.monitoring.schemas import DriftConfig, WindowDefinition


@dataclass(frozen=True)
class PreparedWindows:
    reference: pd.DataFrame
    current: pd.DataFrame
    reference_window: WindowDefinition
    current_window: WindowDefinition
    reference_source_rows: int
    current_source_rows: int
    data_version: str | None


def _window_bounds(window: WindowDefinition) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(date.fromisoformat(window.start))
    end_exclusive = pd.Timestamp(date.fromisoformat(window.end) + timedelta(days=1))
    return start, end_exclusive


def _check_numeric_features(frame: pd.DataFrame, features: tuple[str, ...]) -> None:
    for feature in features:
        values = frame[feature]
        if not is_numeric_dtype(values.dtype) or is_bool_dtype(values.dtype):
            raise MonitoringError(f"Feature {feature!r} must have a numeric, non-boolean dtype.")
        present = values.dropna().to_numpy(dtype="float64")
        if not np.isfinite(present).all():
            raise MonitoringError(f"Feature {feature!r} contains non-finite values.")


def _sample_window(
    frame: pd.DataFrame,
    timestamp_column: str,
    features: tuple[str, ...],
    sample_limit: int,
    random_seed: int,
) -> pd.DataFrame:
    if len(frame) > sample_limit:
        frame = frame.sample(n=sample_limit, random_state=random_seed)
    return (
        frame.sort_values(timestamp_column, kind="mergesort")
        .loc[:, list(features)]
        .reset_index(drop=True)
    )


def prepare_windows(
    frame: pd.DataFrame,
    reference: WindowDefinition,
    current: WindowDefinition,
    config: DriftConfig,
    timestamp_column: str = "ts",
    expected_data_version: str | None = None,
) -> PreparedWindows:
    """Validate a P2 panel, split inclusive dates, and return feature-only samples."""
    if reference.role != "reference":
        raise MonitoringError("Reference window role must be reference.")
    if current.role not in {"historical_test", "out_of_time"}:
        raise MonitoringError("Current window role must be historical_test or out_of_time.")
    reference_start, reference_end = _window_bounds(reference)
    current_start, current_end = _window_bounds(current)
    if current_start < reference_end and reference_start < current_end:
        raise MonitoringError("Reference and historical_test windows overlap.")
    if (
        reference.expected_data_version
        and current.expected_data_version
        and reference.expected_data_version != current.expected_data_version
    ):
        raise MonitoringError("Reference and current data versions must match.")

    required = {timestamp_column, *config.feature_names}
    missing = required - set(frame.columns)
    if missing:
        raise MonitoringError(f"Input is missing required columns: {sorted(missing)}")
    permitted_non_features = set(config.excluded_columns) | {
        timestamp_column,
        "label",
        "data_version",
    }
    unexpected = set(frame.columns) - set(config.feature_names) - permitted_non_features
    if unexpected:
        raise MonitoringError(f"Input contains unexpected columns: {sorted(unexpected)}")

    parsed = pd.to_datetime(frame[timestamp_column], errors="coerce", format="mixed")
    if parsed.isna().any():
        raise MonitoringError(f"Timestamp column {timestamp_column!r} contains invalid dates.")
    if not isinstance(parsed.dtype, pd.DatetimeTZDtype) and not pd.api.types.is_datetime64_dtype(
        parsed.dtype
    ):
        raise MonitoringError("Timestamps must have a consistent, timezone-naive datetime type.")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise MonitoringError("Timestamps must be timezone-naive MiBici local clock values.")
    _check_numeric_features(frame, config.feature_names)

    if "label" in frame.columns:
        labels = frame["label"]
        if not is_numeric_dtype(labels.dtype) or is_bool_dtype(labels.dtype):
            raise MonitoringError("Optional label must be numeric binary data.")
        valid_labels = labels.dropna()
        if not valid_labels.isin([0, 1]).all():
            raise MonitoringError("Optional label must contain only 0/1 values or nulls.")

    resolved_version = expected_data_version
    if "data_version" in frame.columns:
        versions = frame["data_version"].dropna().astype(str).unique()
        if len(versions) > 1:
            raise MonitoringError("Input contains more than one data_version.")
        if len(versions) == 1:
            input_version = str(versions[0])
            if resolved_version and input_version != resolved_version:
                raise MonitoringError("Input data_version does not match the requested version.")
            resolved_version = input_version
    for window in (reference, current):
        if window.expected_data_version and resolved_version != window.expected_data_version:
            raise MonitoringError(f"{window.name} data_version does not match its expectation.")
    if not resolved_version:
        raise MonitoringError(
            "Panel must include a non-empty data_version or provide one explicitly."
        )

    reference_mask = (parsed >= reference_start) & (parsed < reference_end)
    current_mask = (parsed >= current_start) & (parsed < current_end)
    reference_rows = frame.loc[reference_mask].copy()
    current_rows = frame.loc[current_mask].copy()
    if reference_rows.empty:
        raise MonitoringError("Reference window contains no rows.")
    if current_rows.empty:
        raise MonitoringError(f"{current.role} window contains no rows.")

    reference_features = _sample_window(
        reference_rows,
        timestamp_column,
        config.feature_names,
        config.sample_limit,
        config.random_seed,
    )
    current_features = _sample_window(
        current_rows,
        timestamp_column,
        config.feature_names,
        config.sample_limit,
        config.random_seed,
    )
    return PreparedWindows(
        reference=reference_features,
        current=current_features,
        reference_window=reference,
        current_window=current,
        reference_source_rows=len(reference_rows),
        current_source_rows=len(current_rows),
        data_version=resolved_version,
    )
