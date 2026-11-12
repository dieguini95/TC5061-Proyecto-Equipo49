"""Version 2 de P2 para la comparacion champion-challenger (Actividad 10).

Mismo algoritmo (HistGradientBoosting) e hiperparametros que el champion (version 1),
entrenado con oct-2025 a ago-2026: incluye jul-ago 2026, que la version 1 solo uso para
validar. El umbral se elige con la misma regla que la version 1 (maximo F1), pero sobre
sep-2026, porque agosto ya es entrenamiento. La version 2 se registra como nueva version
del mismo modelo con linaje (version padre, ventanas, data_version, commit) y recibe el
alias ``challenger``. El alias ``champion`` no se mueve aqui: la promocion es una
decision aparte, tomada con el resultado de ``ab_p2``.

Entrada: el panel de ``mibici_ml.monitoring.panel`` (meses de entrenamiento mas oct-2026,
que no se usa para entrenar ni para elegir umbral). Ejecutar con el entorno principal:
    .venv/bin/python -m mibici_ml.modeling.retrain_p2 train
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici_ml.config import (
    MLFLOW_TRACKING_URI,
    P2_HORIZON_HOURS,
    P2_REGISTRY_MODEL_NAME,
    PROJ_ROOT,
    RANDOM_SEED,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)
from mibici_ml.features import P2_FEATURE_COLUMNS_V2
from mibici_ml.modeling.registry_p2 import (
    assign_p2_alias,
    ensure_activity_experiment,
    load_p2_version,
)
from mibici_ml.modeling.risk import (
    SKOPS_TRUSTED_TYPES,
    best_f1_threshold,
    build_model,
    get_git_commit_and_dirty,
    score_split,
)
from mibici_ml.monitoring.panel import (
    OCTOBER_END,
    OCTOBER_START,
    inside_window,
    load_monitoring_panel,
)

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Version 2 de P2 (challenger).")

PARENT_VERSION = "1"
CHALLENGER_ALIAS = "challenger"
CHALLENGER_RUN_NAME = "P2-hist_gb-v2-challenger"
ARTIFACT_PATH = "p2_hist_gb_v2"
TUNED_PARAMS = {
    "learning_rate": 0.03,
    "max_iter": 600,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
}
TUNED_SOURCE = "run 9dca0330ba6241dc8ccdf41d48cf4954 (tuned_hyperparams de la version 1)"
FIT_WINDOW = f"{TRAIN_START}_to_{VAL_END}"
THRESHOLD_WINDOW = f"{TEST_START}_to_{TEST_END}"
SUMMARY_PATH = PROJ_ROOT / "reports" / "tables" / "p2_v2_entrenamiento.json"
FEATURES = list(P2_FEATURE_COLUMNS_V2)


def split_masks(full: pd.DataFrame) -> dict[str, np.ndarray]:
    """Mascaras temporales inclusivas. ``oct26`` no participa en entrenamiento ni umbral."""
    ts = full["ts"]
    return {
        "train": inside_window(ts, TRAIN_START, TRAIN_END),
        "val": inside_window(ts, VAL_START, VAL_END),
        "test": inside_window(ts, TEST_START, TEST_END),
        "oct26": inside_window(ts, OCTOBER_START, OCTOBER_END),
    }


def fit_challenger(full: pd.DataFrame) -> tuple[Any, float, dict[str, float]]:
    """Ajusta la version 2 con ene-ago 2026 y elige el umbral en sep-2026."""
    masks = split_masks(full)
    matrix = full[FEATURES].to_numpy(dtype=float)
    labels = full["label"].to_numpy()
    fit = masks["train"] | masks["val"]
    model = build_model("hist_gb", RANDOM_SEED, dict(TUNED_PARAMS))
    model.fit(matrix[fit], labels[fit])
    test = masks["test"]
    scores = model.predict_proba(matrix[test])[:, 1]
    threshold = best_f1_threshold(labels[test], scores)
    metrics = score_split(labels[test], scores, full["ts"][test], threshold, "sep_")
    logger.info("Version 2 ajustada: umbral {:.6f} en {}", threshold, THRESHOLD_WINDOW)
    return model, threshold, metrics


def _example(full: pd.DataFrame) -> pd.DataFrame:
    """Cinco filas solo con las 14 variables; sirve de firma y de prueba de carga."""
    return (
        full.loc[split_masks(full)["test"], FEATURES].head(5).astype(float).reset_index(drop=True)
    )


def register_challenger(
    model: Any,
    full: pd.DataFrame,
    threshold: float,
    metrics: dict[str, float],
    data_version: str,
) -> tuple[str, str]:
    """Registra la version 2 con su linaje y regresa (version, run_id)."""
    import mlflow
    from mlflow import MlflowClient
    from mlflow import sklearn as mlflow_sklearn
    from mlflow.models import infer_signature

    ensure_activity_experiment()
    commit, dirty = get_git_commit_and_dirty()
    example = _example(full)
    proba = model.predict_proba(example.to_numpy())[:, 1]
    signature = infer_signature(example, pd.DataFrame({"riesgo_prob": proba}))
    params = {
        "model": "hist_gb",
        "variant": "challenger_v2",
        "parent_version": PARENT_VERSION,
        "fit_window": FIT_WINDOW,
        "threshold_window": THRESHOLD_WINDOW,
        "threshold": threshold,
        "tuned_hyperparams": ";".join(f"{k}={v}" for k, v in sorted(TUNED_PARAMS.items())),
        "tuned_source": TUNED_SOURCE,
        "horizon_hours": P2_HORIZON_HOURS,
        "random_seed": RANDOM_SEED,
        "feature_names": ",".join(FEATURES),
        "data_version": data_version,
    }
    tags = {
        "team": "equipo49",
        "question": "P2",
        "run_type": "challenger_retrain",
        "stage_of_work": "candidate",
        "model_family": "gradient_boosting",
        "git_commit": commit,
        "git_dirty": str(dirty).lower(),
        "data_version": data_version,
    }
    with mlflow.start_run(run_name=CHALLENGER_RUN_NAME) as run:
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics(
            {key: float(value) for key, value in metrics.items() if np.isfinite(value)}
        )
        mlflow.set_tags(tags)
        info = mlflow_sklearn.log_model(
            model,
            name=ARTIFACT_PATH,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            registered_model_name=P2_REGISTRY_MODEL_NAME,
            signature=signature,
            input_example=example,
        )
        run_id = run.info.run_id
    version = str(info.registered_model_version)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    lineage = {
        "parent_version": PARENT_VERSION,
        "source_run_id": run_id,
        "fit_window": FIT_WINDOW,
        "threshold": f"{threshold:.6f}",
        "threshold_source": f"max_f1_sobre_{THRESHOLD_WINDOW}",
        "data_version": data_version,
        "git_commit": commit,
        "role": "challenger_pending_ab",
    }
    for key, value in lineage.items():
        client.set_model_version_tag(P2_REGISTRY_MODEL_NAME, version, key, str(value))
    client.update_model_version(
        P2_REGISTRY_MODEL_NAME,
        version,
        description=(
            "Version 2 (challenger): mismo algoritmo e hiperparametros que la version 1, "
            "entrenada oct-2025 a ago-2026; umbral elegido en sep-2026. Promover solo con "
            "el resultado de ab_p2. El alias champion no cambia en este registro."
        ),
    )
    logger.success("Version {} registrada desde el run {}", version, run_id)
    return version, run_id


def validate_registered_version(version: str, model: Any, example: pd.DataFrame) -> None:
    """Recarga la version desde el Registry y compara con el modelo en memoria."""
    loaded = load_p2_version(version)
    expected = model.predict_proba(example.to_numpy())[:, 1]
    actual = loaded.predict_proba(example.to_numpy())[:, 1]
    if not np.isfinite(actual).all() or ((actual < 0) | (actual > 1)).any():
        raise ValueError("La version registrada produjo probabilidades no validas.")
    if not np.allclose(expected, actual, atol=1e-12):
        raise ValueError("La version registrada no reproduce el modelo entrenado.")


@app.command("train")
def train_command(
    register: Annotated[bool, typer.Option(help="Registrar la version en el Registry.")] = True,
) -> None:
    """Entrena la version 2, la registra y le asigna el alias challenger."""
    full, data_version = load_monitoring_panel()
    model, threshold, metrics = fit_challenger(full)
    summary = {
        "data_version": data_version,
        "fit_window": FIT_WINDOW,
        "threshold_window": THRESHOLD_WINDOW,
        "threshold": threshold,
        "metrics_threshold_window": {k: v for k, v in metrics.items() if k != "sep_threshold"},
        "tuned_hyperparams": TUNED_PARAMS,
    }
    if register:
        version, run_id = register_challenger(model, full, threshold, metrics, data_version)
        validate_registered_version(version, model, _example(full))
        assign_p2_alias(version, alias=CHALLENGER_ALIAS, tracking_uri=MLFLOW_TRACKING_URI)
        summary.update({"registry_version": version, "run_id": run_id, "alias": CHALLENGER_ALIAS})
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.success("Resumen de la version 2 en {}", Path(SUMMARY_PATH).name)


if __name__ == "__main__":
    app()
