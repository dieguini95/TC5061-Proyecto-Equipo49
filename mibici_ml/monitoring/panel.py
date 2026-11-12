"""Panel mensual de P2 para monitoreo (oct 2025 a oct 2026) y compuerta de paridad.

Se ejecuta en el entorno principal (``.venv``): reutiliza la ingesta de ``dataset.py``
y las funciones de features de P2. Octubre de 2026 se ingiere en un manifiesto aparte
(``references/monitoring_manifest_p2.json``) para no alterar el ``data_version`` con el
que se entrenaron los modelos. Solo la fila de metadatos del manifiesto se versiona;
los Parquet y los CSV crudos viven en ``data/`` (ignorado por Git).

El panel completo (con ``est`` y ``label``) es local. El archivo que recibe Evidently
solo contiene ``ts``, las 14 variables de P2 y ``data_version``.

Uso desde la raiz del proyecto:
    .venv/bin/python -m mibici_ml.monitoring.panel build
    .venv/bin/python -m mibici_ml.monitoring.panel parity
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer

from mibici_ml import dataset
from mibici_ml.config import (
    DATA_MANIFEST_PATH,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
    REFERENCES_DIR,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    TRIPS_PROCESSED_DIR,
    VAL_END,
    VAL_START,
)
from mibici_ml.features import (
    P2_FEATURE_COLUMNS_V2,
    add_p2_v2_features,
    build_p2_features,
    build_p2_hourly_flows,
    build_p2_station_day_panel,
)
from mibici_ml.modeling.risk import load_capacities, load_trips, load_verified_manifest
from mibici_ml.monitoring.config import load_p2_feature_names
from mibici_ml.monitoring.errors import MonitoringError

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Panel de monitoreo P2.")

OCTOBER_2026 = "2026_10"
OCTOBER_START = "2026-10-01"
OCTOBER_END = "2026-10-31"
MONITORING_MANIFEST_PATH = REFERENCES_DIR / "monitoring_manifest_p2.json"
MONITORING_DIR = PROJ_ROOT / "data" / "interim" / "monitoring"
FULL_PANEL_PATH = MONITORING_DIR / "p2_panel_oct25_oct26.parquet"
MONITORING_INPUT_PATH = MONITORING_DIR / "p2_monitoring_input.parquet"
# Valores registrados por el run del champion (9dca0330...), contra los que se verifica
# el panel reconstruido: filas del panel, dias no factibles, PR-AUC de validacion y prueba.
REGISTRY_PANEL_ROWS = 3_906_456
REGISTRY_NON_FEASIBLE_DAYS = 12_911
REGISTRY_VAL_PR_AUC = 0.4029  # val_pr_auc de hist_gb en p2_metricas_validacion.csv
REGISTRY_TEST_PR_AUC = 0.4001  # test_pr_auc del champion
MANIFEST_SCHEMA_VERSION = 1


def panel_from_trips(
    trips: pd.DataFrame, capacities: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features v2 de P2 y etiqueta, a partir de viajes limpios y capacidades.

    Replica el camino de ``mibici_ml.modeling.risk.build_modeling_frame`` con el conjunto
    v2 fijado explicitamente, sin depender de la variable ``P2_FEATURE_SET``.
    """
    flows = build_p2_hourly_flows(trips)
    panel, bounds = build_p2_station_day_panel(flows, capacities)
    features = build_p2_features(panel)
    v2 = add_p2_v2_features(panel)
    extra_columns = [name for name in P2_FEATURE_COLUMNS_V2 if name not in features.columns]
    features = pd.concat([features, v2[extra_columns].reset_index(drop=True)], axis=1)
    features["label"] = panel["label"].to_numpy()
    return features, bounds


def build_full_panel(
    manifest: dict,
    extra_entries: list[dict],
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel P2 con los meses del manifiesto de entrenamiento y los meses extra."""
    combined = {**manifest, "trips": [*manifest["trips"], *extra_entries]}
    trips = load_trips(processed_dir, combined)
    capacities = load_capacities(processed_dir, combined)
    return panel_from_trips(trips, capacities)


def ingest_extra_month(
    month: str,
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    previous: dict | None = None,
) -> dict:
    """Descarga y limpia un mes adicional con la misma regla que la ingesta principal."""
    raw_path = raw_dir / f"datos_abiertos_{month}.csv"
    expected = (previous or {}).get("raw_sha256")
    with dataset._session() as session:
        url = dataset.discover_source_urls(session, [month])[month]
        raw_hash = dataset._obtain_raw(session, url, raw_path, expected, None, False)
    original, encoding = dataset.read_source_csv(raw_path)
    clean, qa = dataset.clean_trips(original, month)
    if clean.empty:
        raise MonitoringError(f"El mes {month} no contiene viajes validos.")
    processed_path = TRIPS_PROCESSED_DIR / f"{month}.parquet"
    return {
        "month": month,
        "source_url": url,
        "raw_path": raw_path.relative_to(raw_dir).as_posix(),
        "processed_path": processed_path.relative_to(processed_dir).as_posix(),
        "raw_sha256": raw_hash,
        "processed_sha256": dataset._write_parquet(clean, processed_path),
        "raw_bytes": raw_path.stat().st_size,
        "raw_rows": len(original),
        "clean_rows": len(clean),
        "encoding": encoding,
        "start_time_min": clean["start_time"].min().isoformat(),
        "start_time_max": clean["start_time"].max().isoformat(),
        "origin_stations": int(clean["origin_id"].nunique()),
        "qa": qa,
    }


def monitoring_data_version(base_data_version: str, extra_entries: list[dict]) -> str:
    """Identidad del panel de monitoreo: version base mas hashes de los meses extra."""
    identity = {
        "base_data_version": base_data_version,
        "extra_trips": [
            {key: entry[key] for key in ("month", "raw_sha256", "processed_sha256")}
            for entry in extra_entries
        ],
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_monitoring_manifest(manifest: dict, extra_entries: list[dict]) -> dict:
    """Manifiesto solo de metadatos: sin filas, sin identificadores de usuario."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "purpose": "Panel P2 para monitoreo de drift: entrenamiento mas octubre 2026.",
        "base_data_version": manifest["data_version"],
        "data_version": monitoring_data_version(manifest["data_version"], extra_entries),
        "months": [entry["month"] for entry in manifest["trips"]]
        + [entry["month"] for entry in extra_entries],
        "stations": {
            "raw_sha256": manifest["stations"]["raw_sha256"],
            "processed_sha256": manifest["stations"]["processed_sha256"],
        },
        "extra_trips": extra_entries,
        "limitations": [
            "El catalogo de estaciones es el de septiembre 2026; se usa para capacidad.",
            "La etiqueta es un proxy de inventario inferido de viajes, no inventario observado.",
        ],
    }


def write_monitoring_input(full_panel: pd.DataFrame, data_version: str, path: Path) -> Path:
    """Archivo para Evidently: ``ts``, las 14 variables y ``data_version``; nada mas.

    Solo incluye hasta ``OCTOBER_END``: las horas del 1 de noviembre tienen llegadas
    parciales (viajes que empezaron el 31 de octubre) y no forman parte de la ventana.
    """
    in_scope = inside_window(full_panel["ts"], "1900-01-01", OCTOBER_END)
    frame = full_panel.loc[in_scope, ["ts", *P2_FEATURE_COLUMNS_V2]].copy()
    frame["data_version"] = data_version
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def inside_window(ts: pd.Series, start: str, end: str) -> np.ndarray:
    """Fechas inclusivas por dia, igual que ``risk.temporal_masks``."""
    upper = pd.Timestamp(end) + pd.Timedelta(days=1)
    return ((ts >= pd.Timestamp(start)) & (ts < upper)).to_numpy()


def _split_counts(full_panel: pd.DataFrame) -> dict[str, int]:
    ts = full_panel["ts"]
    return {
        "train": int(inside_window(ts, TRAIN_START, TRAIN_END).sum()),
        "val": int(inside_window(ts, VAL_START, VAL_END).sum()),
        "test": int(inside_window(ts, TEST_START, TEST_END).sum()),
    }


def load_monitoring_panel() -> tuple[pd.DataFrame, str]:
    """Panel con octubre 2026, verificado contra el manifiesto de monitoreo."""
    manifest = load_verified_manifest(DATA_MANIFEST_PATH, PROCESSED_DATA_DIR)
    monitoring = json.loads(MONITORING_MANIFEST_PATH.read_text(encoding="utf-8"))
    extra = monitoring["extra_trips"]
    version = monitoring_data_version(manifest["data_version"], extra)
    if version != monitoring["data_version"]:
        raise MonitoringError("El manifiesto de monitoreo no coincide con los datos locales.")
    full, _ = build_full_panel(manifest, extra)
    return full.assign(data_version=version), version


def check_feature_contract() -> None:
    """Las 14 variables de P2 deben coincidir, en orden, con la metadata del bundle."""
    contract = load_p2_feature_names()
    if tuple(contract) != tuple(P2_FEATURE_COLUMNS_V2):
        raise MonitoringError("Las variables de P2 no coinciden con models/serving/p2.")


def parity_report(manifest: dict) -> dict[str, Any]:
    """Compuerta: el panel sin octubre reproduce el run del champion (version 1).

    Verifica filas del panel, dias no factibles y PR-AUC de validacion y prueba con el
    modelo ``models:/<registro>/1``. Si cualquiera difiere, no se continua.
    """
    from sklearn.metrics import average_precision_score

    from mibici_ml.modeling.registry_p2 import load_p2_version

    base, bounds = build_full_panel(manifest, [])
    rows = len(base)
    non_feasible = int((~bounds["factible"]).sum())
    if rows != REGISTRY_PANEL_ROWS or non_feasible != REGISTRY_NON_FEASIBLE_DAYS:
        raise MonitoringError(
            f"Panel distinto al del run: filas={rows}, no factibles={non_feasible}."
        )
    model = load_p2_version("1")
    matrix = base[list(P2_FEATURE_COLUMNS_V2)].to_numpy(dtype=float)
    scores = model.predict_proba(matrix)[:, 1]
    labels = base["label"].to_numpy()
    ts = base["ts"]
    val = inside_window(ts, VAL_START, VAL_END)
    test = inside_window(ts, TEST_START, TEST_END)
    val_pr_auc = float(average_precision_score(labels[val], scores[val]))
    test_pr_auc = float(average_precision_score(labels[test], scores[test]))
    if round(val_pr_auc, 4) != REGISTRY_VAL_PR_AUC:
        raise MonitoringError(f"PR-AUC de validacion {val_pr_auc:.6f} != {REGISTRY_VAL_PR_AUC}.")
    if round(test_pr_auc, 4) != REGISTRY_TEST_PR_AUC:
        raise MonitoringError(f"PR-AUC de prueba {test_pr_auc:.6f} != {REGISTRY_TEST_PR_AUC}.")
    return {
        "panel_rows": rows,
        "non_feasible_days": non_feasible,
        "val_pr_auc": round(val_pr_auc, 6),
        "test_pr_auc": round(test_pr_auc, 6),
        "split_rows": _split_counts(base),
    }


@app.command("build")
def build_command(
    month: Annotated[str, typer.Option(help="Mes adicional a ingerir (AAAA_MM).")] = OCTOBER_2026,
) -> None:
    """Ingiere el mes extra, valida la paridad y escribe panel, manifiesto y archivo de Evidently."""
    from loguru import logger

    manifest = load_verified_manifest(DATA_MANIFEST_PATH, PROCESSED_DATA_DIR)
    check_feature_contract()
    previous = {}
    if MONITORING_MANIFEST_PATH.is_file():
        previous = json.loads(MONITORING_MANIFEST_PATH.read_text(encoding="utf-8"))
    previous_extra = {e["month"]: e for e in previous.get("extra_trips", [])}
    extra = ingest_extra_month(month, previous=previous_extra.get(month))
    parity = parity_report(manifest)
    logger.info("Paridad OK: {}", parity)
    full, _ = build_full_panel(manifest, [extra])
    version = monitoring_data_version(manifest["data_version"], [extra])
    full = full.assign(data_version=version)
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    full.to_parquet(FULL_PANEL_PATH, index=False)
    write_monitoring_input(full, version, MONITORING_INPUT_PATH)
    MONITORING_MANIFEST_PATH.write_text(
        json.dumps(
            build_monitoring_manifest(manifest, [extra]),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.success("Panel de monitoreo: {:,} filas; data_version {}", len(full), version[:16])


@app.command("parity")
def parity_command() -> None:
    """Solo compuerta de paridad con los datos de entrenamiento."""
    from loguru import logger

    manifest = load_verified_manifest(DATA_MANIFEST_PATH, PROCESSED_DATA_DIR)
    logger.info("Paridad: {}", parity_report(manifest))


if __name__ == "__main__":
    app()
