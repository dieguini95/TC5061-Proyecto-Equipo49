"""P2: riesgo de que una estacion quede vacia o llena en las siguientes H horas.

La logica de datos vive en ``mibici_ml.features`` (panel, rango factible, etiqueta y
features). Este modulo solo entrena, evalua y registra en MLflow.

Entrada: los Parquet limpios de ``dataset.py`` (``data/processed/trips`` y
``stations.parquet``), verificados contra ``references/data_manifest.json``. No lee
CSV crudos, para que P1 y P2 compartan la misma version de datos.

Uso desde la raiz del proyecto (despues de ``python -m mibici_ml.dataset``):

    python -m mibici_ml.modeling.risk evaluate --output-dir /tmp/mibici-p2-eval
    python -m mibici_ml.modeling.risk run --final-model hist_gb --with-mlflow
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Annotated, Any

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
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

from mibici_ml import config
from mibici_ml.config import (
    DATA_MANIFEST_PATH,
    EXPERIMENT_NAME,
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
from mibici_ml.dataset import _data_version
from mibici_ml.features import (
    P2_FEATURE_COLUMNS,
    build_p2_features,
    build_p2_hourly_flows,
    build_p2_station_day_panel,
)

app = typer.Typer(no_args_is_help=True, help="Clasificacion de riesgo P2 para estaciones MiBici.")

MODEL_NAMES = ("mayoritaria", "persistencia", "logistica", "hist_gb")
CANDIDATE_NAMES = ("logistica", "hist_gb")
FEATURE_COLUMNS = P2_FEATURE_COLUMNS
N_TOP_K_MIN = 1
# HistGradientBoosting usa TreePredictor, que MLflow marca como tipo no confiable (skops).
# El modelo se entrena localmente en este repo, asi que lo permitimos solo para este tipo.
SKOPS_TRUSTED_TYPES = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


# ---------------------------------------------------------------------------
# Entrada verificada: Parquet de dataset.py + manifiesto
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    """SHA-256 por bloques; no carga el archivo completo en memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_manifest(manifest_path: Path, processed_dir: Path) -> dict:
    """Carga el manifiesto y verifica identidad, meses y hashes de los Parquet usados.

    Rechaza datos ausentes o modificados: si la version no coincide, P2 no puede
    afirmar que usa la misma entrada que P1.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("data_version") or not manifest.get("trips"):
        raise ValueError("Manifiesto sin data_version o archivos de viajes.")
    if _data_version(manifest) != manifest["data_version"]:
        raise ValueError("La identidad del manifiesto fue modificada.")
    expected_months = [
        p.strftime("%Y_%m")
        for p in pd.period_range(config.START_MONTH, config.END_MONTH, freq="M")
    ]
    if [entry["month"] for entry in manifest["trips"]] != expected_months:
        raise ValueError("El manifiesto debe enumerar exactamente los meses configurados.")
    for entry in manifest["trips"]:
        path = processed_dir / entry["processed_path"]
        if not path.is_file() or file_sha256(path) != entry["processed_sha256"]:
            raise ValueError(f"Datos ausentes o modificados: {path}. Ejecute la ingesta.")
    # El catalogo de estaciones solo aporta capacidad. Su hash puede diferir entre
    # plataformas (ver README, limitacion de reproducibilidad), asi que P2 lo registra
    # como advertencia en lugar de rechazar la ejecucion.
    stations_entry = manifest["stations"]
    stations_path = processed_dir / stations_entry["processed_path"]
    if not stations_path.is_file():
        raise ValueError(f"Catalogo ausente: {stations_path}. Ejecute la ingesta.")
    stations_hash = file_sha256(stations_path)
    manifest["_stations_hash_check"] = {
        "expected_sha256": stations_entry["processed_sha256"],
        "local_sha256": stations_hash,
        "match": stations_hash == stations_entry["processed_sha256"],
    }
    if not manifest["_stations_hash_check"]["match"]:
        print(
            "ADVERTENCIA: stations.parquet no coincide con el manifiesto "
            f"(local {stations_hash[:12]}, manifiesto {stations_entry['processed_sha256'][:12]}). "
            "P2 usa solo capacidades; se registra como limitacion."
        )
    return manifest


def load_trips(processed_dir: Path, manifest: dict) -> pd.DataFrame:
    """Une los viajes limpios de los meses del manifiesto, en el orden del manifiesto."""
    frames = [
        pd.read_parquet(
            processed_dir / entry["processed_path"],
            columns=["origin_id", "destination_id", "start_time", "end_time"],
        )
        for entry in manifest["trips"]
    ]
    trips = pd.concat(frames, ignore_index=True)
    if trips.empty:
        raise ValueError("No hay viajes limpios para P2.")
    return trips


def load_capacities(processed_dir: Path, manifest: dict) -> pd.DataFrame:
    """Capacidad por estacion desde ``stations.parquet``; ceros y ausentes se excluyen."""
    stations = pd.read_parquet(processed_dir / manifest["stations"]["processed_path"])
    capacities = stations[["station_id", "capacity"]].rename(
        columns={"station_id": "est", "capacity": "capacidad"}
    )
    capacities["est"] = capacities["est"].astype("int64")
    capacities["capacidad"] = capacities["capacidad"].fillna(0).astype("int64")
    return capacities.drop_duplicates("est")


def compute_quality_summary(
    manifest: dict,
    trips: pd.DataFrame,
    bounds: pd.DataFrame,
    capacities: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Resumen de exclusiones y cobertura de P2 para el run y la tabla de calidad."""
    total_days = len(bounds)
    summary: dict[str, Any] = {
        "data_version": manifest["data_version"],
        "viajes_validos": len(trips),
        "estacion_dia_total": total_days,
        "estacion_dia_factibles": int(bounds["factible"].sum()) if total_days else 0,
        "estacion_dia_no_factibles": int((~bounds["factible"]).sum()) if total_days else 0,
        "pct_estacion_dia_no_factibles": round(100 * (1 - bounds["factible"].mean()), 2)
        if total_days
        else 0.0,
        "estaciones_capacidad_cero": int((capacities["capacidad"] == 0).sum()),
        "filas_panel_modelado": len(panel),
        "tasa_positivos_panel": round(float(panel["label"].mean()), 4) if len(panel) else 0.0,
    }
    return pd.DataFrame([summary])


def build_modeling_frame(
    manifest: dict, processed_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Panel, features, factibilidad, capacidades y viajes a partir de la entrada verificada."""
    trips = load_trips(processed_dir, manifest)
    capacities = load_capacities(processed_dir, manifest)
    flows = build_p2_hourly_flows(trips)
    panel, bounds = build_p2_station_day_panel(flows, capacities)
    features = build_p2_features(panel)
    return panel, features, bounds, capacities, trips


# ---------------------------------------------------------------------------
# Particion temporal
# ---------------------------------------------------------------------------


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
# Modelos y metricas
# ---------------------------------------------------------------------------


def persistence_scores(panel: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Persistencia: el riesgo observado H horas antes, sin ajuste."""
    frame = pd.DataFrame(
        {"est": panel["est"].to_numpy(), "day": panel["day"].to_numpy(), "label": labels}
    )
    shifted = frame.groupby(["est", "day"])["label"].shift(P2_HORIZON_HOURS).fillna(0)
    return shifted.to_numpy().astype(float)


def build_model(name: str, random_seed: int) -> Any:
    """Construye un modelo sin ajustarlo."""
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


def _positive_proba(model: Any, matrix: np.ndarray) -> np.ndarray:
    """Probabilidad de la clase positiva (riesgo = 1)."""
    return model.predict_proba(matrix)[:, 1]


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


def feature_importances(
    model: Any, features: pd.DataFrame, labels: np.ndarray, masks: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Importancia por permutacion en validacion (no ajusta nada nuevo)."""
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


# ---------------------------------------------------------------------------
# Provenance y MLflow
# ---------------------------------------------------------------------------


def get_git_commit_and_dirty() -> tuple[str, bool]:
    """Commit actual y si hay cambios sin confirmar. Un arbol sucio no es reproducible."""
    try:
        root = config.PROJ_ROOT
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True
    return commit, dirty


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
    manifest: dict,
    experiment_name: str = EXPERIMENT_NAME,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> str:
    """Registra un run de P2 en MLflow y regresa su ID.

    Etiquetas: equipo, pregunta, tipo de run, commit y estado del arbol. Parametros:
    ventanas, semilla, horizonte y ``data_version``. Sin datos a nivel viaje ni
    identificadores de usuario.
    """
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow
    from mlflow import sklearn as mlflow_sklearn

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    commit, dirty = get_git_commit_and_dirty()
    params: dict[str, Any] = {
        "model": model_name,
        "horizon_hours": P2_HORIZON_HOURS,
        "top_k_fraction": P2_TOP_K_FRACTION,
        "min_precision": P2_MIN_PRECISION,
        "random_seed": random_seed,
        "start_month": config.START_MONTH,
        "end_month": config.END_MONTH,
        "train_window": f"{TRAIN_START}_to_{TRAIN_END}",
        "val_window": f"{VAL_START}_to_{VAL_END}",
        "test_window": f"{TEST_START}_to_{TEST_END}",
        "data_version": manifest["data_version"],
        "feature_names": ",".join(FEATURE_COLUMNS),
        "label_definition": "inventario_estimado_toca_0_o_C_en_H_horas_mismo_dia",
        "inventory_start": "midpoint_feasible_range",
        "infeasible_days_policy": "excluded_from_train_val_test",
        **params_extra,
    }
    tags = {
        "team": "equipo49",
        "question": "P2",
        "run_type": run_type,
        "data_scope": "dataset_py_processed_12m",
        "temporal_holdout": str(temporal_holdout).lower(),
        "limitation": "approx_inventory_from_trips_only",
        "git_commit": commit,
        "git_dirty": str(dirty).lower(),
        "data_version": manifest["data_version"],
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
    with tempfile.TemporaryDirectory(prefix="p2-manifest-") as temporary:
        manifest_copy = Path(temporary) / "data_manifest.json"
        manifest_copy.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})
            mlflow.set_tags(tags)
            mlflow.log_artifact(str(manifest_copy), artifact_path="reproducibility")
            for artifact_path in artifacts.values():
                mlflow.log_artifact(str(artifact_path))
            if model is not None:
                mlflow_sklearn.log_model(
                    model,
                    name=f"p2_{model_name}",
                    skops_trusted_types=SKOPS_TRUSTED_TYPES,
                )
            return run.info.run_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MANIFEST_OPTION = Annotated[
    Path, typer.Option(help="Manifiesto de dataset.py con data_version y hashes.")
]
PROCESSED_OPTION = Annotated[
    Path, typer.Option(help="Carpeta con trips/ y stations.parquet de dataset.py.")
]


def _prepare(
    manifest_path: Path, processed_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Verifica la entrada y arma panel, features y calidad. Falla si los datos no coinciden."""
    manifest = load_verified_manifest(manifest_path, processed_dir)
    panel, features, bounds, capacities, trips = build_modeling_frame(manifest, processed_dir)
    quality = compute_quality_summary(manifest, trips, bounds, capacities, panel)
    logger.info(
        "P2: data_version {} | filas de modelado {:,}", manifest["data_version"][:16], len(panel)
    )
    return panel, features, quality, manifest


@app.command("evaluate")
def evaluate_command(
    *,
    manifest_path: MANIFEST_OPTION = DATA_MANIFEST_PATH,
    processed_dir: PROCESSED_OPTION = config.PROCESSED_DATA_DIR,
    output_dir: Annotated[Path, typer.Option(help="Metricas de validacion y figura.")],
    random_seed: Annotated[int, typer.Option(help="Semilla de los modelos.")] = RANDOM_SEED,
) -> None:
    """Compara los 4 modelos solo en validacion; no abre la particion de prueba."""
    panel, features, _quality, _manifest = _prepare(manifest_path, processed_dir)
    metrics, _context, val_scores = evaluate_all(panel, features, random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "p2_metricas_validacion.csv", index=False)
    plot_pr_curves(val_scores, output_dir / "p2_curva_pr_validacion.png")
    logger.info("Metricas de validacion:\n{}", metrics.to_string())
    logger.success("Evaluacion P2 (solo validacion) guardada en {}", output_dir)


@app.command("run")
def run_command(
    *,
    manifest_path: MANIFEST_OPTION = DATA_MANIFEST_PATH,
    processed_dir: PROCESSED_OPTION = config.PROCESSED_DATA_DIR,
    output_dir: Annotated[Path, typer.Option(help="Directorio de artefactos del run.")],
    final_model: Annotated[
        str, typer.Option(help="Modelo elegido tras revisar evaluate: logistica o hist_gb.")
    ],
    random_seed: Annotated[int, typer.Option(help="Semilla de los modelos.")] = RANDOM_SEED,
    with_mlflow: Annotated[bool, typer.Option(help="Registrar runs en MLflow local.")] = False,
) -> None:
    """Registra 4 runs de validacion y 1 run final con metricas de prueba del modelo elegido."""
    if final_model not in CANDIDATE_NAMES:
        raise typer.BadParameter(f"--final-model debe ser uno de: {', '.join(CANDIDATE_NAMES)}.")
    panel, features, quality, manifest = _prepare(manifest_path, processed_dir)
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
    shared = {
        "metricas": output_dir / "p2_metricas_validacion.csv",
        "curva_pr": output_dir / "p2_curva_pr.png",
        "calidad": output_dir / "p2_calidad_datos.csv",
    }
    if with_mlflow:
        for name in MODEL_NAMES:
            row = metrics.loc[metrics["modelo"] == name].iloc[0]
            model = context["models"].get(name)
            artifacts = dict(shared)
            if model is not None:
                # La importancia por permutacion solo aplica a modelos ajustados.
                importance = feature_importances(
                    model, features, context["labels"], context["masks"]
                )
                importance_path = output_dir / f"p2_importancia_{name}.csv"
                importance.to_csv(importance_path, index=False)
                artifacts["importancia"] = importance_path
            run_id = log_p2_run(
                run_name=f"P2-{name}-validacion",
                run_type="model_selection_validation",
                model_name=name,
                random_seed=random_seed,
                params_extra={"split_scope": "validation_only"},
                metrics={k: row[k] for k in row.index if k != "modelo"},
                quality=quality,
                artifacts=artifacts,
                model=model,
                temporal_holdout=False,
                manifest=manifest,
            )
            logger.info("Run de validacion {} registrado: {}", name, run_id)

        model_obj_final = model_obj
        artifacts = dict(shared)
        if model_obj_final is not None:
            importance = feature_importances(
                model_obj_final, features, context["labels"], context["masks"]
            )
            importance_path = output_dir / f"p2_importancia_{final_model}.csv"
            importance.to_csv(importance_path, index=False)
            artifacts["importancia"] = importance_path
        run_id = log_p2_run(
            run_name=f"P2-{final_model}-final",
            run_type="final_test",
            model_name=final_model,
            random_seed=random_seed,
            params_extra={"split_scope": "test_once", "selected_on": "validation"},
            metrics={k: float(v) for k, v in final_metrics.items() if k != "modelo"},
            quality=quality,
            artifacts=artifacts,
            model=model_obj_final,
            temporal_holdout=True,
            manifest=manifest,
        )
        logger.success("Run final P2 ({}) registrado en MLflow: {}", final_model, run_id)
    logger.success("Artefactos P2 guardados en {}", output_dir)


if __name__ == "__main__":
    app()
