"""Privacy-conscious, aggregate-only report serialization."""

from __future__ import annotations

import csv
from enum import Enum
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from mibici_ml.monitoring.errors import MonitoringError
from mibici_ml.monitoring.schemas import DriftSummary, ReportMetadata

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_OUTPUT_ROOT = REPOSITORY_ROOT / "reports" / "drift"
OUTPUT_NAMES = (
    "drift_report.html",
    "drift_report.json",
    "drift_columns.csv",
    "metadata.json",
)
CSV_FIELDS = (
    "model",
    "feature",
    "column_type",
    "reference_count",
    "current_count",
    "reference_missing",
    "current_missing",
    "method",
    "score",
    "threshold",
    "drift_detected",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")


def _encode_json(value: Any) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )


def _output_path(output_dir: Path) -> Path:
    try:
        resolved = output_dir.resolve()
        resolved.relative_to(ALLOWED_OUTPUT_ROOT.resolve())
    except (OSError, ValueError) as error:
        raise MonitoringError("Report output must remain under reports/drift/.") from error
    return resolved


def write_outputs(
    snapshot: Any,
    summary: DriftSummary,
    metadata: ReportMetadata,
    output_dir: Path,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write HTML, JSON, CSV, and metadata without serializing source rows."""
    destination = _output_path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_files = tuple(destination / name for name in OUTPUT_NAMES)
    if not overwrite and any(path.exists() for path in output_files):
        raise MonitoringError("Report outputs already exist; use --overwrite to replace them.")

    stage = Path(tempfile.mkdtemp(prefix=".drift-stage-", dir=destination))
    try:
        html_path = stage / OUTPUT_NAMES[0]
        if not hasattr(snapshot, "save_html"):
            raise MonitoringError("Evidently snapshot does not expose save_html.")
        snapshot.save_html(str(html_path))
        if not html_path.is_file() or html_path.stat().st_size == 0:
            raise MonitoringError("Evidently produced an empty HTML report.")

        report_payload = {
            "summary": summary.as_dict(),
            "evidently_snapshot": snapshot.dict(),
        }
        (stage / OUTPUT_NAMES[1]).write_text(
            _encode_json(report_payload), encoding="utf-8"
        )
        with (stage / OUTPUT_NAMES[2]).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for column in summary.columns:
                writer.writerow(
                    {
                        "model": metadata.model,
                        "feature": column.feature,
                        "column_type": column.column_type,
                        "reference_count": column.reference_count,
                        "current_count": column.current_count,
                        "reference_missing": column.reference_missing,
                        "current_missing": column.current_missing,
                        "method": column.method,
                        "score": column.score,
                        "threshold": column.threshold,
                        "drift_detected": column.drift_detected,
                    }
                )
        (stage / OUTPUT_NAMES[3]).write_text(
            _encode_json(metadata.as_dict()), encoding="utf-8"
        )

        for staged_file, final_file in zip(
            (stage / name for name in OUTPUT_NAMES),
            output_files,
            strict=True,
        ):
            os.replace(staged_file, final_file)
    except MonitoringError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise MonitoringError("Could not serialize the monitoring report outputs.") from error
    finally:
        shutil.rmtree(stage)

    return output_files
