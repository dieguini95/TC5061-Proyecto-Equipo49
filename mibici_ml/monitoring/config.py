"""Defaults for reproducible P2 historical drift comparisons."""

from __future__ import annotations

import json
from pathlib import Path

from mibici_ml.monitoring.errors import MonitoringError

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
P2_METADATA_PATH = REPOSITORY_ROOT / "models" / "serving" / "p2" / "model_metadata.json"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "reports" / "drift" / "p2"
DEFAULT_TIMESTAMP_COLUMN = "ts"
REFERENCE_START = "2025-10-01"
REFERENCE_END = "2026-06-30"
CURRENT_START = "2026-09-01"
CURRENT_END = "2026-09-30"
CURRENT_ROLE = "historical_test"
DATASET_DRIFT_SHARE_THRESHOLD = 0.5
COLUMN_DRIFT_THRESHOLD = 0.05
SAMPLE_LIMIT = 150_000
RANDOM_SEED = 42

EXCLUDED_COLUMNS = (
    "est",
    "station_id",
    "nombre",
    "lat",
    "lon",
    "label",
    "prediction",
    "score",
    "data_version",
    "source_file",
    "source_path",
    "sha256",
    "hash",
    "git_commit",
    "artifact_uri",
    "source_run_id",
    "mlflow_tracking_uri",
)


def load_p2_feature_names(metadata_path: Path = P2_METADATA_PATH) -> tuple[str, ...]:
    """Load the exact ordered P2 input contract from the exported model metadata."""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MonitoringError(
            f"Could not read P2 model metadata: {metadata_path.name}."
        ) from error
    features = metadata.get("feature_names")
    if metadata.get("question") != "p2" or not isinstance(features, list):
        raise MonitoringError("P2 model metadata must contain a feature_names list.")
    if not features or any(not isinstance(name, str) or not name for name in features):
        raise MonitoringError("P2 feature_names must contain non-empty strings.")
    if len(set(features)) != len(features):
        raise MonitoringError("P2 feature_names must be unique.")
    return tuple(features)
