"""Pruebas del contrato temporal y cobertura de las variables de P1."""

import numpy as np
import pandas as pd
import pytest

from mibici_ml import config, features
from mibici_ml.features import (
    add_p1_features,
    aggregate_departures,
    build_station_hour_panel,
    complete_station_hour_panel,
)


def _counts():
    hours = pd.date_range("2025-10-01", "2025-11-30 23:00", freq="h")
    return pd.DataFrame(
        {"station_id": 1, "timestamp": hours, "departures": np.arange(len(hours)) + 1}
    )


def _panel():
    return complete_station_hour_panel(_counts(), "2025-10", "2025-11")


def test_weekly_baselines_and_lags_use_exact_timestamps():
    features = add_p1_features(_panel()).set_index("timestamp")
    when = pd.Timestamp("2025-11-05 13:00")
    for lag in (24, 48, 168):
        assert (
            features.loc[when, f"lag_{lag}"]
            == features.loc[when - pd.Timedelta(hours=lag), "departures"]
        )
    expected = np.mean(
        [features.loc[when - pd.Timedelta(weeks=week), "departures"] for week in range(1, 5)]
    )
    assert features.loc[when, "mean_4weeks"] == expected
    assert features.loc[when, "mean_4weeks_coverage"] == 4
    assert features.loc[when, "seasonal_168h"] == features.loc[when, "lag_168"]
    assert features.loc[when, "evaluation_eligible"]
    assert features.loc[when, "hour"] == 13
    assert features.loc[when, "day_of_week"] == 2
    assert not features.loc[when, "is_weekend"]


def test_unknown_station_days_are_missing_then_zero_after_first_observation():
    counts = pd.DataFrame(
        {
            "station_id": [1, 2],
            "timestamp": pd.to_datetime(["2025-10-01 04:00", "2025-10-03 16:00"]),
            "departures": [2, 3],
        }
    )
    panel = complete_station_hour_panel(counts, "2025-10", "2025-10")
    station = panel.loc[panel.station_id.eq(2)].set_index("timestamp")
    first_day = station.index < pd.Timestamp("2025-10-04")
    assert station.loc[first_day, "departures"].isna().all()
    assert not station.loc[first_day, "eligible"].any()
    assert station.loc["2025-10-04 00:00", "eligible"]
    assert station.loc["2025-10-04 00:00", "departures"] == 0
    assert str(station.departures.dtype) == "Int32"


def test_full_four_week_history_defines_common_evaluation_cohort():
    features = add_p1_features(_panel()).set_index("timestamp")
    assert not features.loc["2025-10-29 23:00", "evaluation_eligible"]
    assert features.loc["2025-10-29 23:00", "mean_4weeks_coverage"] == 3
    assert pd.isna(features.loc["2025-10-29 23:00", "mean_4weeks"])
    assert features.loc["2025-10-30 00:00", "evaluation_eligible"]
    assert features.loc["2025-10-30 00:00", "mean_4weeks_coverage"] == 4


def test_changing_forecast_day_labels_does_not_change_its_predictors():
    panel = _panel()
    forecast_day = panel.timestamp.dt.normalize().eq(pd.Timestamp("2025-11-05"))
    original = add_p1_features(panel)
    panel.loc[forecast_day, "departures"] = 99999
    changed = add_p1_features(panel)
    feature_columns = [
        "forecast_origin",
        "eligible",
        "hour",
        "day_of_week",
        "is_weekend",
        "lag_24",
        "lag_48",
        "lag_168",
        "seasonal_168h",
        "mean_4weeks",
        "mean_4weeks_coverage",
        "evaluation_eligible",
    ]
    pd.testing.assert_frame_equal(
        original.loc[forecast_day, feature_columns],
        changed.loc[forecast_day, feature_columns],
    )


@pytest.mark.parametrize("duplicate", [False, True])
def test_shifts_reject_missing_or_duplicate_hours(duplicate):
    panel = _panel()
    if duplicate:
        panel = pd.concat([panel, panel.iloc[[100]]], ignore_index=True)
    else:
        panel = panel.drop(index=100)
    with pytest.raises(ValueError, match="continuo"):
        add_p1_features(panel)


def test_missing_month_fails_before_reading_any_data(tmp_path):
    (tmp_path / "2025_10.parquet").write_text("not a parquet", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="2025_11.parquet"):
        build_station_hour_panel(tmp_path, "2025-10", "2025-11")


def test_monthly_aggregation_projects_columns_and_preserves_zero_hours(tmp_path):
    pd.DataFrame(
        {
            "origin_id": [2, 2, 2, 1],
            "start_time": pd.to_datetime(
                ["2025-10-01 02:01", "2025-10-01 02:59", "2025-10-02 10:01", "2025-10-03 05:00"]
            ),
            "unused_column": ["discard"] * 4,
        }
    ).to_parquet(tmp_path / "2025_10.parquet", index=False)
    counts = aggregate_departures(tmp_path, "2025-10", "2025-10")
    assert list(counts.columns) == ["station_id", "timestamp", "departures"]
    assert counts.departures.sum() == 4
    assert (
        counts.loc[counts.timestamp.eq(pd.Timestamp("2025-10-01 02:00")), "departures"].iloc[0]
        == 2
    )
    panel = build_station_hour_panel(tmp_path, "2025-10", "2025-10")
    assert len(panel) == 2 * 31 * 24
    station = panel.loc[panel.station_id.eq(2)].set_index("timestamp")
    assert station.loc["2025-10-02 03:00", "departures"] == 0
    assert station.loc["2025-10-02 10:00", "departures"] == 1


def test_aggregation_rejects_rows_from_wrong_month(tmp_path):
    pd.DataFrame({"origin_id": [1], "start_time": pd.to_datetime(["2025-11-01"])}).to_parquet(
        tmp_path / "2025_10.parquet", index=False
    )
    with pytest.raises(ValueError, match="fuera del mes"):
        aggregate_departures(tmp_path, "2025-10", "2025-10")


@pytest.mark.parametrize(
    "setting,value",
    [
        ("P1_FORECAST_HORIZON_HOURS", 12),
        ("P1_FORECAST_HORIZON_HOURS", 48),
        ("P1_FORECAST_ORIGIN_HOUR", 6),
        ("P1_LAG_HOURS", (12, 48, 168)),
    ],
)
def test_unsupported_forecast_contract_fails_before_reading_data(
    tmp_path, monkeypatch, setting, value
):
    monkeypatch.setattr(config, setting, value)
    monkeypatch.setattr(
        features,
        "aggregate_departures",
        lambda *args: pytest.fail("Must not read invalid forecast"),
    )
    with pytest.raises(ValueError, match="24 h"):
        build_station_hour_panel(tmp_path)
    with pytest.raises(ValueError, match="24 h"):
        add_p1_features(pd.DataFrame())
