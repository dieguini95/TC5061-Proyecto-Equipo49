"""Synthetic, explicitly non-production monitoring fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    reference_dates = pd.date_range("2025-10-01", periods=40, freq="D")
    current_dates = pd.date_range("2026-09-01", periods=40, freq="D")
    dates = reference_dates.append(current_dates)
    reference_values = list(range(40))
    current_values = [value + 100 for value in range(40)]
    return pd.DataFrame(
        {
            "ts": dates,
            "feature_a": reference_values + current_values,
            "feature_b": reference_values + reference_values,
            "label": [value % 2 for value in range(80)],
            "station_id": [f"synthetic-station-{value}" for value in range(80)],
            "nombre": [f"synthetic-name-{value}" for value in range(80)],
            "lat": [20.123456 + value / 1_000_000 for value in range(80)],
            "lon": [-103.987654 - value / 1_000_000 for value in range(80)],
            "data_version": ["synthetic-v1"] * 80,
        }
    )
