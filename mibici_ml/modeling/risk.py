"""P2: riesgo de que una estacion quede vacia o llena en las siguientes H horas.

Pipeline (propuesta, seccion 2.1 y 4 de P2):

1. Lee los viajes crudos mensuales y el catalogo de estaciones (nomenclatura).
2. Construye el flujo neto por estacion-hora (llegadas menos salidas).
3. Calcula el rango factible del inventario inicial por estacion-dia y descarta los dias
   infactibles (no pueden etiquetarse con solo viajes).
4. Etiqueta `r(s, t) = 1` si el inventario estimado toca 0 o C en alguna de las
   siguientes H horas del mismo dia. El inventario usa el punto medio del rango
   factible (`s0 = (L + U) / 2`) y solo sirve para construir la etiqueta.
5. Features solo con informacion hasta `t` (sin fuga del futuro).
6. Particion temporal (entrenamiento, validacion, prueba); nunca aleatoria.

Uso (desde la raiz del proyecto):

    python -m mibici_ml.modeling.risk build-panel --raw-dir ../../01_insumos/datos
    python -m mibici_ml.modeling.risk evaluate
    python -m mibici_ml.modeling.risk run --final-model hgb --with-mlflow
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import subprocess
from typing import Annotated, Any

from loguru import logger
import mlflow
from mlflow import sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer

from mibici_ml.config import (
    EXPERIMENT_NAME,
    INTERIM_DATA_DIR,
    MLFLOW_TRACKING_URI,
    P2_HORIZON_HOURS,
    P2_MIN_PRECISION,
    P2_TOP_K_FRACTION,
    RANDOM_SEED,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)

app = typer.Typer(no_args_is_help=True, help="Clasificacion de riesgo P2 para estaciones MiBici.")

COLUMNAS_VIAJE = [
    "Viaje_Id",
    "Usuario_Id",
    "Genero",
    "Año_de_nacimiento",
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
]
FMT_FECHA = "%Y-%m-%d %H:%M:%S"
PANEL_FILE = "p2_panel_estacion_hora.parquet"
LABEL_FILE = "p2_etiquetas_estacion_hora.parquet"
MODEL_NAMES = ("mayoritaria", "persistencia", "logistica", "hist_gb")
CANDIDATE_NAMES = ("logistica", "hist_gb")
FEATURE_COLUMNS = (
    "hora",
    "dia_semana",
    "fin_de_semana",
    "capacidad",
    "flujo_acumulado",
    "flujo_acumulado_sobre_c",
    "llegadas_1h",
    "salidas_1h",
    "llegadas_2h",
    "salidas_2h",
    "salidas_misma_hora_ayer",
)
ARTIFACT_NAMES = (
    "p2_metricas_modelos.csv",
    "p2_curva_pr.png",
    "p2_calidad_datos.csv",
    "p2_importancia_features.csv",
)
N_TOP_K_MIN = 1
SKOPS_TRUSTED_TYPES = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


# ---------------------------------------------------------------------------
# Lectura de viajes crudos
# ---------------------------------------------------------------------------


def read_trip_file(path: Path) -> pd.DataFrame:
    """Lee un CSV mensual de viajes detectando UTF-8 o cp1252 (como en el EDA)."""
    if not path.exists():
        raise ValueError(f"No existe el archivo de viajes: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    try:
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError(f"No se pudo leer el CSV de viajes: {path}: {error}") from error
    if len(frame.columns) != len(COLUMNAS_VIAJE):
        raise ValueError(
            f"{path.name} tiene {len(frame.columns)} columnas; se esperaban {len(COLUMNAS_VIAJE)}."
        )
    frame.columns = COLUMNAS_VIAJE
    logger.info("Viajes leidos de {}: {:,}", path.name, len(frame))
    return frame


def load_station_catalog(catalog_path: Path) -> pd.DataFrame:
    """Lee la nomenclatura de estaciones con capacidad `dpcapacity`."""
    try:
        catalog = pd.read_csv(catalog_path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"No se pudo leer el catalogo {catalog_path}: {error}") from error
    required = {"id", "dpcapacity"}
    if not required.issubset(catalog.columns):
        raise ValueError(f"El catalogo debe contener las columnas {sorted(required)}.")
    catalog["est"] = catalog["id"].astype(str)
    catalog["capacidad"] = pd.to_numeric(catalog["dpcapacity"], errors="coerce").fillna(0)
    catalog["capacidad"] = catalog["capacidad"].astype(int)
    return catalog[["est", "capacidad"]].drop_duplicates("est")


def clean_trips(trips: pd.DataFrame, catalog_ids: set[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Elimina duplicados, fechas invalidas y viajes fuera del catalogo; devuelve conteos."""
    counts: dict[str, int] = {"viajes_entrada": len(trips)}
    trips = trips.drop_duplicates("Viaje_Id")
    counts["viajes_duplicados_eliminados"] = counts["viajes_entrada"] - len(trips)
    start = pd.to_datetime(trips["Inicio_del_viaje"], format=FMT_FECHA, errors="coerce")
    end = pd.to_datetime(trips["Fin_del_viaje"], format=FMT_FECHA, errors="coerce")
    valid_dates = start.notna() & end.notna() & (end >= start)
    counts["viajes_fechas_invalidas"] = int((~valid_dates).sum())
    trips = trips.loc[valid_dates].assign(
        inicio=start[valid_dates].to_numpy(), fin=end[valid_dates].to_numpy()
    )
    in_catalog = trips["Origen_Id"].isin(catalog_ids) & trips["Destino_Id"].isin(catalog_ids)
    counts["viajes_fuera_catalogo"] = int((~in_catalog).sum())
    trips = trips.loc[in_catalog]
    counts["viajes_validos"] = len(trips)
    return trips, counts


def build_hourly_flows(trips: pd.DataFrame) -> pd.DataFrame:
    """Salidas y llegadas por estacion-hora a partir de viajes limpios."""
    departures = (
        trips.assign(ts=trips["inicio"].dt.floor("h"), est=trips["Origen_Id"])
        .groupby(["est", "ts"])
        .size()
        .rename("salidas")
    )
    arrivals = (
        trips.assign(ts=trips["fin"].dt.floor("h"), est=trips["Destino_Id"])
        .groupby(["est", "ts"])
        .size()
        .rename("llegadas")
    )
    flows = pd.concat([departures, arrivals], axis=1).fillna(0).astype("int64")
    return flows.reset_index()


# ---------------------------------------------------------------------------
# Inventario factible, etiqueta y panel
# ---------------------------------------------------------------------------


def build_station_day_panel(
    flows: pd.DataFrame, capacities: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel estacion-hora con flujo acumulado, etiqueta y factibilidad por estacion-dia.

    Devuelve (panel_horario, resumen_por_estacion_dia). Solo se conservan los dias factibles
    y con capacidad positiva en el panel de modelado.
    """
    grid_hours = pd.date_range(
        flows["ts"].min().normalize(),
        flows["ts"].max().floor("D") + pd.Timedelta(hours=23),
        freq="h",
    )
    stations = capacities.loc[capacities["capacidad"] > 0, ["est", "capacidad"]]
    full_index = pd.MultiIndex.from_product([stations["est"], grid_hours], names=["est", "ts"])
    panel = (
        flows.set_index(["est", "ts"])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .merge(stations, on="est", how="inner")
    )
    panel["neto"] = panel["llegadas"] - panel["salidas"]
    panel["day"] = panel["ts"].dt.normalize()
    panel = panel.sort_values(["est", "ts"]).reset_index(drop=True)
    panel["flujo_acumulado"] = panel.groupby(["est", "day"])["neto"].cumsum()

    bounds = panel.groupby(["est", "day"], sort=False).agg(
        max_f=("flujo_acumulado", "max"),
        min_f=("flujo_acumulado", "min"),
        capacidad=("capacidad", "first"),
    )
    bounds["max_f"] = bounds["max_f"].clip(lower=0)
    bounds["min_f"] = bounds["min_f"].clip(upper=0)
    bounds["lower"] = np.maximum(0, -bounds["min_f"])
    bounds["upper"] = np.minimum(bounds["capacidad"], bounds["capacidad"] - bounds["max_f"])
    bounds["factible"] = bounds["lower"] <= bounds["upper"]
    bounds["s0"] = (bounds["lower"] + bounds["upper"]) / 2
    bounds = bounds.reset_index()

    panel = panel.merge(bounds[["est", "day", "factible", "s0"]], on=["est", "day"], how="left")
    panel["factible"] = panel["factible"].eq(True)
    panel = panel.loc[panel["factible"]].copy()
    panel["inventario"] = panel["s0"] + panel["flujo_acumulado"]
    panel["label"] = _risk_labels(panel, P2_HORIZON_HOURS)
    return panel, bounds


def _risk_labels(panel: pd.DataFrame, horizon: int) -> np.ndarray:
    """1 si el inventario estimado toca 0 o C en las siguientes `horizon` horas del mismo dia."""
    labels = np.zeros(len(panel), dtype="int64")
    empty = (panel["inventario"] <= 0).to_numpy()
    full = (panel["inventario"] >= panel["capacidad"]).to_numpy()
    risk = (empty | full).astype("int64")
    frame = pd.DataFrame(
        {"est": panel["est"].to_numpy(), "day": panel["day"].to_numpy(), "risk": risk}
    )
    for step in range(1, horizon + 1):
        future = frame.groupby(["est", "day"])["risk"].shift(-step)
        labels = np.maximum(labels, future.fillna(0).to_numpy().astype("int64"))
    return labels


def compute_quality_summary(
    counts: dict[str, int], bounds: pd.DataFrame, capacities: pd.DataFrame, panel: pd.DataFrame
) -> pd.DataFrame:
    """Resumen de calidad y de exclusiones para el run de MLflow."""
    total_days = len(bounds)
    summary: dict[str, Any] = dict(counts)
    summary.update(
        {
            "estacion_dia_total": total_days,
            "estacion_dia_factibles": int(bounds["factible"].sum()),
            "estacion_dia_no_factibles": int((~bounds["factible"]).sum()),
            "pct_estacion_dia_no_factibles": round(100 * (1 - bounds["factible"].mean()), 2)
            if total_days
            else 0.0,
            "estaciones_capacidad_cero": int((capacities["capacidad"] == 0).sum()),
            "filas_panel_modelado": len(panel),
            "tasa_positivos_panel": round(float(panel["label"].mean()), 4) if len(panel) else 0.0,
        }
    )
    return pd.DataFrame([summary])


def temporal_masks(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    """Mascaras booleanas por conjunto temporal; sin traslape ni particion aleatoria."""
    ts = panel["ts"]
    train = (
        (ts >= TRAIN_START) & (ts < pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1))
    ).to_numpy()
    val = ((ts >= VAL_START) & (ts < pd.Timestamp(VAL_END) + pd.Timedelta(days=1))).to_numpy()
    test = ((ts >= TEST_START) & (ts < pd.Timestamp(TEST_END) + pd.Timedelta(days=1))).to_numpy()
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Features (solo informacion hasta t)
# ---------------------------------------------------------------------------


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Features calculadas con rezagos hacia el pasado; ninguna mira a t+1 o mas."""
    frame = panel[
        ["est", "ts", "day", "salidas", "llegadas", "capacidad", "flujo_acumulado"]
    ].copy()
    grouped = frame.groupby("est", sort=False)
    frame["hora"] = frame["ts"].dt.hour
    frame["dia_semana"] = frame["ts"].dt.dayofweek
    frame["fin_de_semana"] = (frame["dia_semana"] >= 5).astype("int64")
    frame["flujo_acumulado_sobre_c"] = frame["flujo_acumulado"] / frame["capacidad"]
    frame["llegadas_1h"] = frame["llegadas"]
    frame["salidas_1h"] = frame["salidas"]
    frame["llegadas_2h"] = frame["llegadas"] + grouped["llegadas"].shift(1).fillna(0)
    frame["salidas_2h"] = frame["salidas"] + grouped["salidas"].shift(1).fillna(0)
    frame["salidas_misma_hora_ayer"] = grouped["salidas"].shift(24).fillna(0)
    return frame[["est", "ts"] + list(FEATURE_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Modelos y metricas
# ---------------------------------------------------------------------------


def persistence_scores(panel: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Persistencia: el riesgo observado en la ventana previa de H horas, sin ajuste."""
    frame = pd.DataFrame(
        {"est": panel["est"].to_numpy(), "day": panel["day"].to_numpy(), "label": labels}
    )
    shifted = frame.groupby(["est", "day"])["label"].shift(P2_HORIZON_HOURS).fillna(0)
    return shifted.to_numpy().astype(float)


def build_model(name: str, random_seed: int) -> Any:
    """Construye un modelo sin ajustarlo. Solo se usan los nombres de MODEL_NAMES."""
    if name == "mayoritaria":
        return DummyClassifier(strategy="most_frequent")
    if name == "logistica":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=random_seed
                    ),
                ),
            ]
        )
    if name == "hist_gb":
        return HistGradientBoostingClassifier(random_state=random_seed)
    raise ValueError(f"Modelo no soportado para ajuste: {name}")


def precision_at_k(
    scores: np.ndarray, labels: np.ndarray, timestamps: pd.Series, k_fraction: float
) -> float:
    """Precision de las top-k estaciones por hora segun su puntaje de riesgo."""
    frame = pd.DataFrame({"score": scores, "label": labels, "ts": timestamps.to_numpy()})
    hits, picks = 0, 0
    for _, hour in frame.groupby("ts", sort=False):
        k = max(N_TOP_K_MIN, round(k_fraction * len(hour)))
        top = hour.nlargest(min(k, len(hour)), "score")
        hits += int(top["label"].sum())
        picks += len(top)
    return float(hits / picks) if picks else float("nan")


def recall_at_precision(labels: np.ndarray, scores: np.ndarray, min_precision: float) -> float:
    """Recall maximo entre los umbrales que mantienen precision >= min_precision."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    # La ultima entrada de precision/recall no tiene umbral asociado; se descarta.
    feasible = recall[:-1][precision[:-1] >= min_precision]
    return float(feasible.max()) if feasible.size else 0.0


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Umbral que maximiza F1 en validacion. Se fija antes de mirar la prueba."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    if f1.size == 0:
        return 0.5
    return float(thresholds[int(np.nanargmax(f1))])


def score_split(
    labels: np.ndarray,
    scores: np.ndarray,
    timestamps: pd.Series,
    threshold: float,
    prefix: str,
) -> dict[str, float]:
    """Metricas de un conjunto con el prefijo indicado (val_ o test_)."""
    predictions = (scores >= threshold).astype("int64")
    if len(np.unique(labels)) < 2:
        logger.warning("El conjunto {} tiene una sola clase; ROC y PR no estan definidos.", prefix)
        roc, pr = float("nan"), float("nan")
    else:
        roc = float(roc_auc_score(labels, scores))
        pr = float(average_precision_score(labels, scores))
    return {
        f"{prefix}pr_auc": pr,
        f"{prefix}roc_auc": roc,
        f"{prefix}brier": float(brier_score_loss(labels, np.clip(scores, 0, 1))),
        f"{prefix}f1": float(f1_score(labels, predictions, zero_division=0)),
        f"{prefix}recall": float(recall_score(labels, predictions, zero_division=0)),
        f"{prefix}precision_at_k": precision_at_k(scores, labels, timestamps, P2_TOP_K_FRACTION),
        f"{prefix}recall_at_precision_50": recall_at_precision(labels, scores, P2_MIN_PRECISION),
        f"{prefix}threshold": threshold,
        f"{prefix}positive_rate": float(labels.mean()),
    }


def fit_and_score(
    name: str,
    features: pd.DataFrame,
    labels: np.ndarray,
    panel: pd.DataFrame,
    masks: dict[str, np.ndarray],
    random_seed: int,
) -> tuple[dict[str, float], dict[str, np.ndarray], Any]:
    """Ajusta un modelo en entrenamiento y regresa puntajes de validacion y prueba."""
    x_all = features[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    ts = panel["ts"]
    scores: dict[str, np.ndarray] = {}
    if name == "persistencia":
        persist = persistence_scores(panel, labels)
        scores = {"val": persist[masks["val"]], "test": persist[masks["test"]]}
        model = None
        threshold = 0.5
    else:
        model = build_model(name, random_seed)
        model.fit(x_all[masks["train"]], labels[masks["train"]])
        scores = {
            "val": _positive_proba(model, x_all[masks["val"]]),
            "test": _positive_proba(model, x_all[masks["test"]]),
        }
        threshold = best_f1_threshold(labels[masks["val"]], scores["val"])
    metrics = score_split(labels[masks["val"]], scores["val"], ts[masks["val"]], threshold, "val_")
    return metrics, scores, model


def _positive_proba(model: Any, matrix: np.ndarray) -> np.ndarray:
    """Probabilidad de la clase positiva (riesgo = 1)."""
    return model.predict_proba(matrix)[:, 1]


def feature_importances(
    model: Any, features: pd.DataFrame, labels: np.ndarray, masks: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Importancia por permutacion en validacion (no ajusta nada nuevo)."""
    from sklearn.inspection import permutation_importance

    x_val = features[list(FEATURE_COLUMNS)].to_numpy(dtype=float)[masks["val"]]
    y_val = labels[masks["val"]]
    result = permutation_importance(
        model, x_val, y_val, n_repeats=3, random_state=RANDOM_SEED, scoring="average_precision"
    )
    table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importancia_media": result.importances_mean,
            "desv_std": result.importances_std,
        }
    )
    return table.sort_values("importancia_media", ascending=False).reset_index(drop=True)


def plot_pr_curves(
    scores_by_model: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path
) -> None:
    """Curvas precision-recall de validacion por modelo."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 5))
    for name, (labels, scores) in scores_by_model.items():
        if len(np.unique(labels)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(labels, scores)
        axis.plot(recall, precision, label=name)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Curvas precision-recall en validacion (P2)")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def compute_file_sha256(input_path: Path) -> str:
    digest = hashlib.sha256()
    with input_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def build_modeling_frame(
    raw_dir: Path, catalog_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], list[Path]]:
    """Lee los 12 meses, limpia, arma el panel y devuelve (panel, features, bounds, counts, files)."""
    catalog = load_station_catalog(catalog_path)
    files = sorted(raw_dir.glob("datos_abiertos_*.csv"))
    if not files:
        raise ValueError(f"No hay CSV de viajes en {raw_dir}")
    trips = pd.concat([read_trip_file(path) for path in files], ignore_index=True)
    trips, counts = clean_trips(trips, set(catalog["est"]))
    flows = build_hourly_flows(trips)
    panel, bounds = build_station_day_panel(flows, catalog)
    features = build_features(panel)
    return panel, features, bounds, counts, files


def build_panel_artifacts(
    raw_dir: Path, catalog_path: Path, interim_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construye el panel y lo guarda en `interim_dir` para reutilizarlo sin releer los CSV."""
    catalog = load_station_catalog(catalog_path)
    panel, features, bounds, counts, _ = build_modeling_frame(raw_dir, catalog_path)
    quality = compute_quality_summary(counts, bounds, catalog, panel)
    interim_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(interim_dir / PANEL_FILE, index=False)
    quality.to_csv(interim_dir / "p2_calidad_datos.csv", index=False)
    labels = panel[["est", "ts", "label"]]
    labels.to_parquet(interim_dir / LABEL_FILE, index=False)
    return panel, features, labels, quality


def load_or_build_panel(
    raw_dir: Path, catalog_path: Path, interim_dir: Path, rebuild: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Usa el panel cacheado si existe; si no, lo construye desde los viajes crudos."""
    panel_path = interim_dir / PANEL_FILE
    quality_path = interim_dir / "p2_calidad_datos.csv"
    if rebuild or not panel_path.exists() or not quality_path.exists():
        panel, features, _labels, quality = build_panel_artifacts(
            raw_dir, catalog_path, interim_dir
        )
        return panel, features, quality
    panel = pd.read_parquet(panel_path)
    features = build_features(panel)
    quality = pd.read_csv(quality_path)
    return panel, features, quality


def evaluate_all(
    panel: pd.DataFrame, features: pd.DataFrame, random_seed: int
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Ajusta los 4 modelos y devuelve metricas de validacion, modelos y puntajes."""
    labels = panel["label"].to_numpy()
    masks = temporal_masks(panel)
    rows, models, val_scores = [], {}, {}
    for name in MODEL_NAMES:
        metrics, scores, model = fit_and_score(name, features, labels, panel, masks, random_seed)
        rows.append({"modelo": name, **metrics})
        models[name] = model
        val_scores[name] = (labels[masks["val"]], scores["val"])
    return pd.DataFrame(rows), {"models": models, "masks": masks, "labels": labels}, val_scores


def test_metrics_for(
    name: str,
    panel: pd.DataFrame,
    features: pd.DataFrame,
    context: dict[str, Any],
    threshold: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Metricas de prueba de un modelo con umbral fijado en validacion."""
    labels, masks = context["labels"], context["masks"]
    x_all = features[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    if name == "persistencia":
        test_scores = persistence_scores(panel, labels)[masks["test"]]
    else:
        test_scores = _positive_proba(context["models"][name], x_all[masks["test"]])
    test_labels = labels[masks["test"]]
    metrics = score_split(test_labels, test_scores, panel["ts"][masks["test"]], threshold, "test_")
    return metrics, test_scores


def log_p2_run(
    *,
    run_name: str,
    run_type: str,
    model_name: str,
    random_seed: int,
    params_extra: dict[str, Any],
    metrics: dict[str, float],
    quality: pd.DataFrame,
    artifacts: dict[str, Path],
    model: Any | None,
    temporal_holdout: bool,
    input_hashes: dict[str, str],
    experiment_name: str = EXPERIMENT_NAME,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> str:
    """Registra un run de P2 en MLflow y regresa su ID."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    params: dict[str, Any] = {
        "model": model_name,
        "horizon_hours": P2_HORIZON_HOURS,
        "top_k_fraction": P2_TOP_K_FRACTION,
        "min_precision": P2_MIN_PRECISION,
        "random_seed": random_seed,
        "train_window": f"{TRAIN_START}_to_{TRAIN_END}",
        "val_window": f"{VAL_START}_to_{VAL_END}",
        "test_window": f"{TEST_START}_to_{TEST_END}",
        "feature_names": ",".join(FEATURE_COLUMNS),
        "label_definition": "inventario_estimado_toca_0_o_C_en_H_horas_mismo_dia",
        "inventory_start": "midpoint_feasible_range",
        "infeasible_days_policy": "excluded_from_train_val_test",
        **params_extra,
    }
    commit = get_git_commit()
    tags = {
        "team": "equipo49",
        "question": "P2",
        "run_type": run_type,
        "data_scope": "trips_raw_full_12m",
        "temporal_holdout": str(temporal_holdout).lower(),
        "limitation": "approx_inventory_from_trips_only",
        "git_commit": commit,
    }
    quality_row = quality.iloc[0]
    for key in (
        "viajes_validos",
        "estacion_dia_no_factibles",
        "pct_estacion_dia_no_factibles",
        "filas_panel_modelado",
        "tasa_positivos_panel",
    ):
        params[f"quality_{key}"] = quality_row[key]
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})
        mlflow.set_tags(tags)
        for input_name, digest in input_hashes.items():
            mlflow.set_tag(f"input_sha256_{input_name}", digest)
        for artifact_path in artifacts.values():
            mlflow.log_artifact(str(artifact_path))
        if model is not None:
            # HistGradientBoosting usa TreePredictor, que MLflow marca como tipo no confiable
            # (skops). El modelo se entrena localmente en este repo, asi que lo permitimos
            # explicitamente solo para este tipo.
            mlflow_sklearn.log_model(
                model,
                name=f"p2_{model_name}",
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
        return run.info.run_id


def _input_hashes(files: list[Path], catalog_path: Path) -> dict[str, str]:
    """SHA-256 de cada CSV de entrada (viajes y nomenclatura)."""
    hashes = {path.stem: compute_file_sha256(path) for path in files}
    hashes["nomenclatura"] = compute_file_sha256(catalog_path)
    return hashes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command("build-panel")
def build_panel_command(
    *,
    raw_dir: Annotated[Path, typer.Option(help="Carpeta con los CSV mensuales de viajes.")],
    catalog_path: Annotated[
        Path, typer.Option(help="CSV de nomenclatura de estaciones (id, dpcapacity).")
    ],
    interim_dir: Annotated[
        Path, typer.Option(help="Destino del panel cacheado.")
    ] = INTERIM_DATA_DIR,
) -> None:
    """Limpia los viajes crudos y guarda el panel estacion-hora para P2."""
    _, _, _, quality = build_panel_artifacts(raw_dir, catalog_path, interim_dir)
    logger.success("Panel P2 guardado en {}\n{}", interim_dir, quality.T.to_string())


@app.command("evaluate")
def evaluate_command(
    *,
    raw_dir: Annotated[Path, typer.Option(help="Carpeta con los CSV mensuales de viajes.")],
    catalog_path: Annotated[Path, typer.Option(help="CSV de nomenclatura de estaciones.")],
    interim_dir: Annotated[Path, typer.Option(help="Panel cacheado.")] = INTERIM_DATA_DIR,
    output_dir: Annotated[Path, typer.Option(help="Metricas de validacion y figuras.")],
    rebuild: Annotated[bool, typer.Option(help="Reconstruir el panel desde los CSV.")] = False,
    random_seed: Annotated[int, typer.Option(help="Semilla de los modelos.")] = RANDOM_SEED,
) -> None:
    """Compara los 4 modelos solo en validacion; no abre la particion de prueba."""
    panel, features, _ = load_or_build_panel(raw_dir, catalog_path, interim_dir, rebuild)
    metrics, _, val_scores = evaluate_all(panel, features, random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "p2_metricas_validacion.csv", index=False)
    plot_pr_curves(val_scores, output_dir / "p2_curva_pr_validacion.png")
    logger.info("Metricas de validacion:\n{}", metrics.to_string())
    logger.success("Evaluacion P2 (solo validacion) guardada en {}", output_dir)


@app.command("run")
def run_command(
    *,
    raw_dir: Annotated[Path, typer.Option(help="Carpeta con los CSV mensuales de viajes.")],
    catalog_path: Annotated[Path, typer.Option(help="CSV de nomenclatura de estaciones.")],
    interim_dir: Annotated[Path, typer.Option(help="Panel cacheado.")] = INTERIM_DATA_DIR,
    output_dir: Annotated[Path, typer.Option(help="Directorio de artefactos del run final.")],
    final_model: Annotated[
        str,
        typer.Option(help="Modelo elegido tras revisar evaluate: logistica o hist_gb."),
    ],
    rebuild: Annotated[bool, typer.Option(help="Reconstruir el panel desde los CSV.")] = False,
    random_seed: Annotated[int, typer.Option(help="Semilla de los modelos.")] = RANDOM_SEED,
    with_mlflow: Annotated[bool, typer.Option(help="Registrar runs en MLflow local.")] = False,
) -> None:
    """Registra 4 runs de validacion y 1 run final con metricas de prueba del modelo elegido."""
    if final_model not in CANDIDATE_NAMES:
        raise typer.BadParameter(f"--final-model debe ser uno de: {', '.join(CANDIDATE_NAMES)}.")
    panel, features, quality = load_or_build_panel(raw_dir, catalog_path, interim_dir, rebuild)
    metrics, context, val_scores = evaluate_all(panel, features, random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "p2_metricas_validacion.csv", index=False)
    plot_pr_curves(val_scores, output_dir / "p2_curva_pr.png")
    quality.to_csv(output_dir / "p2_calidad_datos.csv", index=False)

    final_row = metrics.loc[metrics["modelo"] == final_model].iloc[0]
    threshold = float(final_row["val_threshold"])
    test_metrics, _ = test_metrics_for(final_model, panel, features, context, threshold)
    final_metrics = {**final_row.drop(labels="modelo").to_dict(), **test_metrics}

    model_obj = context["models"].get(final_model)
    importance = (
        feature_importances(model_obj, features, context["labels"], context["masks"])
        if model_obj is not None
        else pd.DataFrame()
    )
    importance.to_csv(output_dir / "p2_importancia_features.csv", index=False)
    artifacts = {
        "metricas": output_dir / "p2_metricas_validacion.csv",
        "curva_pr": output_dir / "p2_curva_pr.png",
        "calidad": output_dir / "p2_calidad_datos.csv",
        "importancia": output_dir / "p2_importancia_features.csv",
    }
    if with_mlflow:
        raw_files = sorted(raw_dir.glob("datos_abiertos_*.csv"))
        hashes = _input_hashes(raw_files, catalog_path)
        for name in MODEL_NAMES:
            row = metrics.loc[metrics["modelo"] == name].iloc[0]
            run_id = log_p2_run(
                run_name=f"P2-{name}-validacion",
                run_type="model_selection_validation",
                model_name=name,
                random_seed=random_seed,
                params_extra={"split_scope": "validation_only"},
                metrics={k: row[k] for k in row.index if k != "modelo"},
                quality=quality,
                artifacts=artifacts,
                model=context["models"].get(name),
                temporal_holdout=False,
                input_hashes=hashes,
            )
            logger.info("Run de validacion {} registrado: {}", name, run_id)
        run_id = log_p2_run(
            run_name=f"P2-{final_model}-final",
            run_type="final_test",
            model_name=final_model,
            random_seed=random_seed,
            params_extra={"split_scope": "test_once", "selected_on": "validation"},
            metrics={k: float(v) for k, v in final_metrics.items() if k != "modelo"},
            quality=quality,
            artifacts=artifacts,
            model=model_obj,
            temporal_holdout=True,
            input_hashes=hashes,
        )
        logger.success("Run final P2 ({}) registrado en MLflow: {}", final_model, run_id)
    logger.success("Artefactos P2 guardados en {}", output_dir)


if __name__ == "__main__":
    app()
