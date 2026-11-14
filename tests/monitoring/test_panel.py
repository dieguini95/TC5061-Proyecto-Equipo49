"""Panel de monitoreo P2: aislamiento de meses, archivo de entrada y version de datos.

Usa viajes sinteticos; no lee ``data/`` ni ``mlflow.db``.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("sklearn")
panel = pytest.importorskip("mibici_ml.monitoring.panel")

from mibici_ml.features import P2_FEATURE_COLUMNS_V2

CAPACITIES = pd.DataFrame({"est": [1, 2, 3], "capacidad": [10, 10, 10]})


def _day_trips(days: list[str], start_id: int = 1) -> list[tuple]:
    """Ida 1->2 a las 08:00 y regreso 2->1 a las 09:00 de cada dia."""
    rows = []
    for offset, day in enumerate(days):
        base = pd.Timestamp(day)
        rows.append(
            (
                f"t{day}-a",
                base + pd.Timedelta(hours=8),
                base + pd.Timedelta(hours=8, minutes=20),
                start_id,
                start_id + 1,
            )
        )
        rows.append(
            (
                f"t{day}-b",
                base + pd.Timedelta(hours=9),
                base + pd.Timedelta(hours=9, minutes=15),
                start_id + 1,
                start_id,
            )
        )
    return rows


def _trips(rows: list[tuple]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows, columns=["trip_id", "start_time", "end_time", "origin_id", "destination_id"]
    )
    frame["start_time"] = pd.to_datetime(frame["start_time"])
    frame["end_time"] = pd.to_datetime(frame["end_time"])
    return frame


# El panel recorre todos los dias entre el primer y el ultimo viaje; septiembre completo
# evita huecos artificiales y octubre continua la serie, como en los datos reales.
SEPTEMBER = [day.strftime("%Y-%m-%d") for day in pd.date_range("2026-09-01", "2026-09-30")]
OCTOBER = ["2026-10-01", "2026-10-02", "2026-10-03"]


def test_appending_a_month_keeps_earlier_rows_unchanged():
    before = _trips(_day_trips(SEPTEMBER))
    after = _trips(_day_trips(SEPTEMBER) + _day_trips(OCTOBER))

    features_before, _ = panel.panel_from_trips(before, CAPACITIES)
    features_after, _ = panel.panel_from_trips(after, CAPACITIES)

    keys = ["est", "ts"]
    earlier_before = features_before[features_before["ts"] < "2026-10-01"]
    earlier_after = features_after[features_after["ts"] < "2026-10-01"]
    assert len(earlier_before) > 0
    cols = [*P2_FEATURE_COLUMNS_V2, "label"]
    left = earlier_before.sort_values(keys).reset_index(drop=True)
    right = earlier_after.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left[keys + cols], right[keys + cols])


def test_monitoring_input_has_only_fourteen_features_and_version_up_to_october_end(tmp_path):
    rows = _day_trips(SEPTEMBER) + _day_trips(OCTOBER)
    # Viaje que empieza el 31 de octubre y termina en noviembre: produce horas de 1 de noviembre.
    rows.append(
        (
            "late",
            pd.Timestamp("2026-10-31 23:50"),
            pd.Timestamp("2026-11-01 00:20"),
            1,
            2,
        )
    )
    full, _ = panel.panel_from_trips(_trips(rows), CAPACITIES)
    assert (full["ts"] >= "2026-11-01").any(), "el caso de prueba debe producir noviembre"

    path = panel.write_monitoring_input(full, "version-de-prueba", tmp_path / "input.parquet")
    frame = pd.read_parquet(path)

    assert list(frame.columns) == ["ts", *P2_FEATURE_COLUMNS_V2, "data_version"]
    assert frame["data_version"].unique().tolist() == ["version-de-prueba"]
    assert frame["ts"].max() < pd.Timestamp("2026-11-01")
    assert "label" not in frame.columns


def test_monitoring_version_changes_only_with_extra_month_identity():
    base = "base-sha"
    october = {"month": "2026_10", "raw_sha256": "raw-a", "processed_sha256": "proc-a"}
    october_other = {**october, "processed_sha256": "proc-b"}

    assert panel.monitoring_data_version(base, [october]) == panel.monitoring_data_version(
        base, [dict(october)]
    )
    assert panel.monitoring_data_version(base, [october]) != panel.monitoring_data_version(
        base, [october_other]
    )
    assert panel.monitoring_data_version(base, [october]) != panel.monitoring_data_version(
        base, []
    )
