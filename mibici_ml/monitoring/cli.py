"""Command-line entry point for offline P2 drift monitoring."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import subprocess
import sys

import pandas as pd

from mibici_ml.monitoring.config import (
    COLUMN_DRIFT_THRESHOLD,
    CURRENT_END,
    CURRENT_ROLE,
    CURRENT_START,
    DATASET_DRIFT_SHARE_THRESHOLD,
    DEFAULT_OUTPUT_DIR,
    EXCLUDED_COLUMNS,
    RANDOM_SEED,
    REFERENCE_END,
    REFERENCE_START,
    SAMPLE_LIMIT,
    load_p2_feature_names,
)
from mibici_ml.monitoring.data_windows import prepare_windows
from mibici_ml.monitoring.drift import SUPPORTED_EVIDENTLY_VERSION, run_drift_report
from mibici_ml.monitoring.errors import MonitoringError
from mibici_ml.monitoring.outputs import write_outputs
from mibici_ml.monitoring.schemas import DriftConfig, ReportMetadata, WindowDefinition


def _parse_feature_threshold(value: str) -> tuple[str, float]:
    name, separator, raw_threshold = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("Use FEATURE=THRESHOLD.")
    try:
        threshold = float(raw_threshold)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Threshold must be numeric.") from error
    if not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError("Threshold must be in [0, 1].")
    return name, threshold


def _read_panel(path: Path) -> pd.DataFrame:
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise MonitoringError("Could not read the input panel.") from error
    raise MonitoringError("Input panel must be a CSV or Parquet file.")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
        OSError,
    ):
        raise MonitoringError("Could not determine the current Git revision.") from None
    commit = result.stdout.strip()
    if not commit:
        raise MonitoringError("Git returned an empty revision.")
    return commit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an offline P2 feature-drift report from a prepared panel."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV or Parquet panel.")
    parser.add_argument(
        "--data-version",
        help="Expected panel data_version; if omitted, the panel value is recorded.",
    )
    parser.add_argument(
        "--dataset-drift-share",
        type=float,
        default=DATASET_DRIFT_SHARE_THRESHOLD,
        help="Feature-share threshold for the dataset-level drift flag.",
    )
    parser.add_argument(
        "--column-threshold",
        type=float,
        default=COLUMN_DRIFT_THRESHOLD,
        help="Default Evidently per-feature threshold.",
    )
    parser.add_argument(
        "--feature-threshold",
        type=_parse_feature_threshold,
        action="append",
        default=[],
        metavar="FEATURE=THRESHOLD",
        help="Per-feature threshold override; may be supplied more than once.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing reports under reports/drift/p2/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        feature_names = load_p2_feature_names()
        thresholds = dict(args.feature_threshold)
        if len(thresholds) != len(args.feature_threshold):
            raise MonitoringError("A feature threshold was supplied more than once.")
        reference = WindowDefinition(
            name="reference",
            start=REFERENCE_START,
            end=REFERENCE_END,
            role="reference",
            expected_data_version=args.data_version,
        )
        current = WindowDefinition(
            name="current",
            start=CURRENT_START,
            end=CURRENT_END,
            role=CURRENT_ROLE,
            expected_data_version=args.data_version,
        )
        config = DriftConfig(
            feature_names=feature_names,
            excluded_columns=EXCLUDED_COLUMNS,
            dataset_drift_share_threshold=args.dataset_drift_share,
            column_threshold=args.column_threshold,
            per_column_thresholds=thresholds,
            sample_limit=SAMPLE_LIMIT,
            random_seed=RANDOM_SEED,
            output_directory=str(DEFAULT_OUTPUT_DIR),
        )
        prepared = prepare_windows(
            _read_panel(args.input),
            reference,
            current,
            config,
            expected_data_version=args.data_version,
        )
        snapshot, summary = run_drift_report(prepared, config)
        metadata = ReportMetadata(
            model="P2",
            evidently_version=SUPPORTED_EVIDENTLY_VERSION,
            data_version=prepared.data_version,
            generated_at_utc=datetime.now(UTC).isoformat(),
            git_commit=_git_commit(),
            reference=reference,
            current=current,
            reference_source_rows=prepared.reference_source_rows,
            current_source_rows=prepared.current_source_rows,
            reference_rows_analyzed=len(prepared.reference),
            current_rows_analyzed=len(prepared.current),
            sample_limit=config.sample_limit,
            random_seed=config.random_seed,
            feature_names=feature_names,
            dataset_drift_share_threshold=config.dataset_drift_share_threshold,
            column_threshold=config.column_threshold,
            per_column_thresholds=dict(config.per_column_thresholds),
            limitations=tuple(summary.limitations),
        )
        outputs = write_outputs(
            snapshot,
            summary,
            metadata,
            DEFAULT_OUTPUT_DIR,
            overwrite=args.overwrite,
        )
    except MonitoringError as error:
        print(f"monitoring error: {error}", file=sys.stderr)
        return 2
    except OSError:
        print("monitoring error: a report file operation failed.", file=sys.stderr)
        return 2

    print(
        "P2 drift summary: "
        f"reference={reference.start}..{reference.end} ({summary.reference_rows} rows), "
        f"current={current.start}..{current.end} role={current.role} "
        f"({summary.current_rows} rows), "
        f"drifted_features={len(summary.drifted_columns)}/{summary.total_columns}, "
        f"drift_share={summary.drift_share:.3f}, "
        f"dataset_drift={summary.dataset_drift_detected}"
    )
    if summary.drifted_columns:
        print(f"Drifted features: {', '.join(summary.drifted_columns)}")
    if summary.warnings:
        print(f"Warnings: {'; '.join(summary.warnings)}")
    print(f"Outputs: {', '.join(path.name for path in outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
