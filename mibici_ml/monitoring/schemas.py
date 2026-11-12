"""Small typed contracts shared by the monitoring pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Literal

from mibici_ml.monitoring.errors import MonitoringError

WindowRole = Literal["reference", "historical_test", "out_of_time"]


@dataclass(frozen=True)
class WindowDefinition:
    name: str
    start: str
    end: str
    role: WindowRole
    inclusive_policy: str = "both_dates"
    expected_data_version: str | None = None

    def __post_init__(self) -> None:
        try:
            start = date.fromisoformat(self.start)
            end = date.fromisoformat(self.end)
        except ValueError as error:
            raise MonitoringError("Window dates must use YYYY-MM-DD.") from error
        if start > end:
            raise MonitoringError("Window start must not be after window end.")
        if self.role not in {"reference", "historical_test", "out_of_time"}:
            raise MonitoringError("Window role must be reference, historical_test or out_of_time.")
        if self.inclusive_policy != "both_dates":
            raise MonitoringError("Only inclusive start/end dates are supported.")
        if not self.name:
            raise MonitoringError("Window name is required.")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DriftConfig:
    feature_names: tuple[str, ...]
    excluded_columns: tuple[str, ...] = ()
    dataset_drift_share_threshold: float = 0.5
    column_threshold: float = 0.05
    per_column_thresholds: dict[str, float] = field(default_factory=dict)
    sample_limit: int = 150_000
    random_seed: int = 42
    output_directory: str = "reports/drift/p2"
    report_tags: tuple[str, ...] = ("p2", "offline", "historical_test")

    def __post_init__(self) -> None:
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise MonitoringError("Feature names must be non-empty and unique.")
        if any(not name for name in self.feature_names):
            raise MonitoringError("Feature names must be non-empty.")
        if not 0 <= self.dataset_drift_share_threshold <= 1:
            raise MonitoringError("Dataset drift share threshold must be in [0, 1].")
        if not 0 <= self.column_threshold <= 1:
            raise MonitoringError("Column threshold must be in [0, 1].")
        if self.sample_limit < 1:
            raise MonitoringError("Sample limit must be positive.")
        unknown = set(self.per_column_thresholds) - set(self.feature_names)
        if unknown:
            raise MonitoringError(f"Thresholds configured for unknown features: {sorted(unknown)}")
        if any(not 0 <= value <= 1 for value in self.per_column_thresholds.values()):
            raise MonitoringError("Per-column thresholds must be in [0, 1].")


@dataclass(frozen=True)
class DriftColumnResult:
    feature: str
    column_type: str
    reference_count: int
    current_count: int
    reference_missing: int
    current_missing: int
    method: str
    score: float | None
    threshold: float
    drift_detected: bool | None


@dataclass(frozen=True)
class DriftSummary:
    evidently_version: str
    reference: WindowDefinition
    current: WindowDefinition
    dataset_drift_detected: bool
    drifted_columns: list[str]
    total_columns: int
    drift_share: float
    columns: list[DriftColumnResult]
    warnings: list[str]
    limitations: list[str]
    reference_source_rows: int
    current_source_rows: int
    reference_rows: int
    current_rows: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReportMetadata:
    model: str
    evidently_version: str
    data_version: str | None
    generated_at_utc: str
    git_commit: str
    reference: WindowDefinition
    current: WindowDefinition
    reference_source_rows: int
    current_source_rows: int
    reference_rows_analyzed: int
    current_rows_analyzed: int
    sample_limit: int
    random_seed: int
    feature_names: tuple[str, ...]
    dataset_drift_share_threshold: float
    column_threshold: float
    per_column_thresholds: dict[str, float]
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
