"""Data quality, privacy and source-version checks without network downloads."""

import copy
import json

import pandas as pd
import pytest

from mibici_ml import dataset


def _trip(trip_id="1", **changes):
    row = {
        "Viaje Id": trip_id,
        "Usuario_Id": "personal-id",
        "Género": "F",
        "Año de nacimiento": "1980",
        "Inicio del viaje": "2025-10-01 08:00:00",
        "Fin del viaje": "2025-10-01 08:10:00",
        "Origen Id": "9999",
        "Destino Id": "2",
    }
    row.update(changes)
    return row


def test_clean_trips_privacy_exact_duplicates_and_quality_rules():
    rows = [
        _trip(),
        _trip(),  # exact duplicate before discarding personal attributes
        _trip(Usuario_Id="another-user"),  # distinct raw row, not silently deduplicated
        _trip("2", **{"Fin del viaje": "bad"}),
        _trip("3", **{"Fin del viaje": "2025-10-01 08:00:00"}),
        _trip("4", **{"Origen Id": "2.5"}),
        _trip("5", **{"Destino Id": "0"}),
        _trip("6", **{"Destino Id": "9999", "Fin del viaje": "2025-10-01 12:00:00"}),
        _trip("7", **{"Fin del viaje": "2025-10-01 08:00:30"}),
        _trip(
            "8",
            **{
                "Inicio del viaje": "2025-10-31 23:59:00",
                "Fin del viaje": "2025-11-01 00:10:00",
            },
        ),
        _trip(
            "9",
            **{
                "Inicio del viaje": "2025-11-01 08:00:00",
                "Fin del viaje": "2025-11-01 08:10:00",
            },
        ),
    ]
    clean, qa = dataset.clean_trips(pd.DataFrame(rows), "2025_10")
    assert list(clean.columns) == dataset.TRIP_COLUMNS
    assert len(clean) == 5
    assert clean["origin_id"].eq(9999).all()  # historical IDs remain, no catalogue filter
    assert pd.api.types.is_datetime64_any_dtype(clean["start_time"])
    assert clean["origin_id"].dtype == "int64"
    assert qa["exact_duplicates_removed"] == 1
    assert qa["invalid_timestamps_removed"] == 1
    assert qa["nonpositive_duration_removed"] == 1
    assert qa["invalid_station_ids_removed"] == 2
    assert qa["starts_outside_source_month_removed"] == 1
    assert qa["rows_removed_total"] == 6
    assert qa["same_station_retained"] == 1
    assert qa["over_180_minutes_retained"] == 1
    assert qa["under_1_minute_retained"] == 1
    assert qa["duplicate_trip_ids_retained"] == 1
    assert not {"Usuario_Id", "Género", "Año de nacimiento"}.intersection(clean.columns)


@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp1252"])
def test_csv_encoding_and_accented_headers(tmp_path, encoding):
    path = tmp_path / "trips.csv"
    pd.DataFrame([_trip()]).to_csv(path, index=False, encoding=encoding, sep=";")
    frame, detected = dataset.read_source_csv(path)
    clean, _ = dataset.clean_trips(frame)
    assert detected == encoding
    assert len(clean) == 1
    assert "personal-id" not in clean.to_csv(index=False)


def test_catalogue_actual_headers_and_zero_capacity():
    frame = pd.DataFrame(
        {
            "id": [1, 2, -1],
            "name": ["Estación Centro", "Otra", "Invalid"],
            "latitude": [20.6, 20.7, 20.8],
            "longitude": [-103.3, -103.4, -103.5],
            "dpcapacity": [15, 0, 10],
        }
    )
    clean, qa = dataset.clean_stations(frame)
    assert list(clean.columns) == dataset.STATION_COLUMNS
    assert clean["station_id"].tolist() == [1, 2]
    assert clean["capacity"].tolist() == [15, 0]
    assert qa["zero_capacity"] == 1
    assert qa["invalid_station_ids_removed"] == 1


def test_numbered_download_copies_are_used_once_and_hash_checked(tmp_path):
    first = tmp_path / "datos_abiertos_2025_10 (1).csv"
    first.write_text("same bytes", encoding="utf-8")
    second = tmp_path / "datos_abiertos_2025_10 (2).csv"
    second.write_bytes(first.read_bytes())
    expected = dataset.sha256_file(first)
    assert dataset._local_candidate(tmp_path, "datos_abiertos_2025_10.csv", expected) == first
    canonical = tmp_path / "datos_abiertos_2025_10.csv"
    canonical.write_text("unexpected change", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 inesperado"):
        dataset._local_candidate(tmp_path, canonical.name, expected)


def test_existing_cache_drift_fails_before_overwrite(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("original", encoding="utf-8")
    expected = dataset.sha256_file(path)
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 inesperado"):
        dataset._obtain_raw(
            None, "https://example.invalid/source.csv", path, expected, None, False
        )
    assert path.read_text(encoding="utf-8") == "changed"


def test_portal_discovers_exact_filenames_without_guessing():
    class Response:
        content = b"""<a href="/site/assets/files/77/datos_abiertos_2025_10.csv">October</a>
        <a href="/site/assets/files/99/datos_abiertos_2025_11.csv">November</a>
        <a href="/site/assets/files/42/datos_abiertos_2025_10.csv.zip">Ignore ZIP</a>"""

        def raise_for_status(self):
            pass

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    urls = dataset.discover_source_urls(Session(), ["2025_10", "2025_11"])
    assert urls["2025_10"].endswith("/77/datos_abiertos_2025_10.csv")
    with pytest.raises(ValueError, match="Meses no encontrados"):
        dataset.discover_source_urls(Session(), ["2026_01"])


def test_ingestion_is_repeatable_and_manifest_contains_only_metadata(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    csv_path = source / "datos_abiertos_2025_10 (1).csv"
    pd.DataFrame([_trip()]).to_csv(csv_path, index=False)
    catalogue = source / "nomenclatura_2026_09.csv"
    pd.DataFrame(
        {
            "id": [2],
            "name": ["Centro"],
            "latitude": [20.6],
            "longitude": [-103.3],
            "dpcapacity": [15],
        }
    ).to_csv(catalogue, index=False)
    monkeypatch.setattr(
        dataset, "_historical_hashes", lambda: {"2025_10": dataset.sha256_file(csv_path)}
    )
    monkeypatch.setattr(
        dataset,
        "discover_source_urls",
        lambda *args: {"2025_10": "https://mibici.net/actual/datos_abiertos_2025_10.csv"},
    )
    monkeypatch.setattr(
        dataset, "_download", lambda *args: pytest.fail("Cached inputs must not be downloaded")
    )
    args = {
        "raw_dir": tmp_path / "raw",
        "processed_dir": tmp_path / "processed",
        "manifest_path": tmp_path / "references" / "data_manifest.json",
        "start_month": "2025-10",
        "end_month": "2025-10",
        "source_dir": source,
    }
    first = dataset.ingest_dataset(**args)
    second = dataset.ingest_dataset(**args)
    assert first == second
    assert first["totals"] == {
        "months": 1,
        "raw_trips": 1,
        "clean_trips": 1,
        "catalogue_stations": 1,
    }
    parquet = pd.read_parquet(args["processed_dir"] / "trips" / "2025_10.parquet")
    assert list(parquet.columns) == dataset.TRIP_COLUMNS
    manifest_text = args["manifest_path"].read_text(encoding="utf-8")
    assert "personal-id" not in manifest_text
    assert str(tmp_path) not in manifest_text
    assert json.loads(manifest_text)["data_version"] == first["data_version"]
    changed = copy.deepcopy(first)
    changed["trips"][0]["processed_sha256"] = "changed-output"
    assert dataset._data_version(changed) != first["data_version"]
    csv_cache = args["raw_dir"] / "datos_abiertos_2025_10.csv"
    csv_cache.write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 inesperado"):
        dataset.ingest_dataset(**args)


@pytest.mark.parametrize("start,end", [("2026-09", "2025-10"), ("2025_10", "2026-09")])
def test_reject_invalid_month_range(start, end):
    with pytest.raises(ValueError):
        dataset._month_range(start, end)
