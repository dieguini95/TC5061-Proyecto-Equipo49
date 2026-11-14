"""Evidently-backed P2 feature drift with a version-specific result adapter."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import math
from typing import Any

from mibici_ml.monitoring import __version__ as monitoring_version
from mibici_ml.monitoring.data_windows import PreparedWindows
from mibici_ml.monitoring.errors import MonitoringError
from mibici_ml.monitoring.schemas import (
    DriftColumnResult,
    DriftConfig,
    DriftSummary,
)

SUPPORTED_EVIDENTLY_VERSION = "0.7.23"


def _extract_summary(
    snapshot: Any, prepared: PreparedWindows, config: DriftConfig
) -> DriftSummary:
    snapshot_dict = snapshot.dict()
    metrics = snapshot_dict.get("metrics")
    tests = snapshot_dict.get("tests")
    if not isinstance(metrics, list) or not isinstance(tests, list):
        raise MonitoringError(
            "Unsupported Evidently snapshot schema: expected metrics and tests lists."
        )

    drift_count = [
        item
        for item in metrics
        if item.get("config", {}).get("type") == "evidently:metric_v2:DriftedColumnsCount"
    ]
    if len(drift_count) != 1 or not isinstance(drift_count[0].get("value"), dict):
        raise MonitoringError("Unsupported Evidently drift-count metric schema.")
    count_value = drift_count[0]["value"]
    try:
        drift_share = float(count_value["share"])
        reported_count = float(count_value["count"])
    except (KeyError, TypeError, ValueError) as error:
        raise MonitoringError("Evidently drift-count metric is missing count or share.") from error
    if (
        not math.isfinite(drift_share)
        or not 0 <= drift_share <= 1
        or not math.isfinite(reported_count)
        or not reported_count.is_integer()
        or not 0 <= reported_count <= len(config.feature_names)
    ):
        raise MonitoringError("Evidently returned an invalid drift count or share.")

    column_results: list[DriftColumnResult] = []
    warnings: list[str] = []
    for feature in config.feature_names:
        matching = [
            item
            for item in metrics
            if item.get("config", {}).get("type") == "evidently:metric_v2:ValueDrift"
            and item.get("config", {}).get("column") == feature
        ]
        if len(matching) != 1:
            raise MonitoringError(
                f"Unsupported Evidently snapshot schema for feature {feature!r}."
            )
        metric = matching[0]
        metric_config = metric["config"]
        try:
            score = float(metric["value"])
            threshold = float(
                metric_config.get(
                    "threshold", config.per_column_thresholds.get(feature, config.column_threshold)
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise MonitoringError(f"Invalid Evidently values for feature {feature!r}.") from error
        if not math.isfinite(score) or not math.isfinite(threshold):
            raise MonitoringError(f"Non-finite Evidently result for feature {feature!r}.")

        associated_tests = [
            item
            for item in tests
            if item.get("bound_test", {}).get("metric_fingerprint") == metric.get("id")
        ]
        if len(associated_tests) != 1:
            raise MonitoringError(
                f"Evidently did not return one machine-readable test for {feature!r}."
            )
        status = associated_tests[0].get("status")
        if status == "FAIL":
            drift_detected: bool | None = True
        elif status == "SUCCESS":
            drift_detected = False
        else:
            drift_detected = None
            warnings.append(f"Evidently could not determine drift status for {feature}.")

        reference_values = prepared.reference[feature]
        current_values = prepared.current[feature]
        column_results.append(
            DriftColumnResult(
                feature=feature,
                column_type="numeric",
                reference_count=len(reference_values),
                current_count=len(current_values),
                reference_missing=int(reference_values.isna().sum()),
                current_missing=int(current_values.isna().sum()),
                method=str(metric_config.get("method", "unknown")),
                score=score,
                threshold=threshold,
                drift_detected=drift_detected,
            )
        )

    drifted = [item.feature for item in column_results if item.drift_detected is True]
    unresolved = any(item.drift_detected is None for item in column_results)
    if not unresolved:
        if not math.isclose(
            len(drifted) / len(config.feature_names),
            drift_share,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise MonitoringError(
                "Evidently drift-share metric disagrees with per-column test statuses."
            )
        if int(reported_count) != len(drifted):
            raise MonitoringError(
                "Evidently drifted-column count disagrees with per-column test statuses."
            )
    dataset_drift_detected = drift_share >= config.dataset_drift_share_threshold

    if prepared.current_window.role == "out_of_time":
        window_limitation = (
            "The current window is an out-of-time period after training, reviewed offline; "
            "it is not production telemetry."
        )
    else:
        window_limitation = (
            "The current window is a retrospective historical_test, not production telemetry."
        )
    limitations = [
        window_limitation,
        "Feature drift is a review signal and does not by itself establish model degradation.",
        "Only P2 input features are passed to Evidently; identifiers and timestamps are excluded.",
        "A travel-derived inventory proxy is not observed station inventory.",
    ]

    return DriftSummary(
        evidently_version=version("evidently"),
        reference=prepared.reference_window,
        current=prepared.current_window,
        dataset_drift_detected=dataset_drift_detected,
        drifted_columns=drifted,
        total_columns=len(column_results),
        drift_share=drift_share,
        columns=column_results,
        warnings=warnings,
        limitations=limitations,
        reference_source_rows=prepared.reference_source_rows,
        current_source_rows=prepared.current_source_rows,
        reference_rows=len(prepared.reference),
        current_rows=len(prepared.current),
    )


def run_drift_report(
    prepared_windows: PreparedWindows,
    config: DriftConfig,
) -> tuple[Any, DriftSummary]:
    """Run Evidently using feature-only frames; return its snapshot and safe summary."""
    try:
        installed_version = version("evidently")
    except PackageNotFoundError as error:
        raise MonitoringError(
            "Evidently is not installed; install requirements-monitoring.txt to run reports."
        ) from error
    if installed_version != SUPPORTED_EVIDENTLY_VERSION:
        raise MonitoringError(
            f"Unsupported Evidently version {installed_version}; "
            f"expected {SUPPORTED_EVIDENTLY_VERSION}."
        )
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    if list(prepared_windows.reference.columns) != list(config.feature_names):
        raise MonitoringError("Reference columns must match the configured feature order.")
    if list(prepared_windows.current.columns) != list(config.feature_names):
        raise MonitoringError("Current columns must match the configured feature order.")

    report = Report(
        metrics=[
            DataDriftPreset(
                columns=list(config.feature_names),
                drift_share=config.dataset_drift_share_threshold,
                threshold=config.column_threshold,
                per_column_threshold=config.per_column_thresholds or None,
            ),
            DataSummaryPreset(columns=list(config.feature_names)),
        ],
        metadata={"model": "P2", "monitoring_version": monitoring_version},
        tags=list(config.report_tags),
        include_tests=True,
    )
    snapshot = report.run(
        current_data=prepared_windows.current,
        reference_data=prepared_windows.reference,
    )
    return snapshot, _extract_summary(snapshot, prepared_windows, config)
