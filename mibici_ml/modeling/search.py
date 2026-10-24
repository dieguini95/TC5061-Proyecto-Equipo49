"""Busqueda de hiperparametros con runs anidados en MLflow (P2).

Un run padre por busqueda y un run hijo por combinacion de hiperparametros. La
seleccion usa solo validacion: la particion de prueba no se abre aqui (se abre una
sola vez en ``risk.run_command``).

    python -m mibici_ml.modeling.search --question p2 --output-dir /tmp/p2-search
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Annotated, Any

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
import typer

from mibici_ml import config
from mibici_ml.config import (
    DATA_MANIFEST_PATH,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    P2_SEARCH_GRID,
    RANDOM_SEED,
)
from mibici_ml.features import P2_FEATURE_COLUMNS
from mibici_ml.modeling import risk, train

app = typer.Typer(no_args_is_help=True, help="Busqueda anidada de hiperparametros (P2).")

SEARCH_MODELS = tuple(P2_SEARCH_GRID)


def grid_combinations(model_name: str) -> list[dict[str, Any]]:
    """Todas las combinaciones del grid declarado en config para un modelo."""
    grid = P2_SEARCH_GRID[model_name]
    keys = sorted(grid)
    return [dict(zip(keys, values, strict=True)) for values in product(*(grid[k] for k in keys))]


def _fit_score_validation(
    model: Any, features: pd.DataFrame, panel: pd.DataFrame, labels: np.ndarray, masks: dict
) -> dict[str, float]:
    """Ajusta en entrenamiento y mide validacion (sin abrir prueba)."""
    x_all = features[list(P2_FEATURE_COLUMNS)].to_numpy(dtype=float)
    model.fit(x_all[masks["train"]], labels[masks["train"]])
    val_scores = risk._positive_proba(model, x_all[masks["val"]])
    threshold = risk.best_f1_threshold(labels[masks["val"]], val_scores)
    metrics = risk.score_split(
        labels[masks["val"]], val_scores, panel["ts"][masks["val"]], threshold, "val_"
    )
    return metrics


@app.command()
def search_command(
    *,
    manifest_path: Annotated[Path, typer.Option(help="Manifiesto de dataset.py.")] = (
        DATA_MANIFEST_PATH
    ),
    processed_dir: Annotated[Path, typer.Option(help="Parquet de dataset.py.")] = (
        config.PROCESSED_DATA_DIR
    ),
    random_seed: Annotated[int, typer.Option(help="Semilla de los modelos.")] = RANDOM_SEED,
    tracking_uri: Annotated[str, typer.Option(help="Backend de MLflow.")] = MLFLOW_TRACKING_URI,
) -> None:
    """Busca hiperparametros en validacion y registra un run padre con hijos anidados."""
    train._release_wrapper_run()
    panel, features, quality, manifest = risk._prepare(manifest_path, processed_dir)
    labels = panel["label"].to_numpy()
    masks = risk.temporal_masks(panel)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    commit, dirty = risk.get_git_commit_and_dirty()

    best: dict[str, Any] = {"score": -np.inf}
    with mlflow.start_run(run_name="P2-busqueda-anidada") as parent:
        mlflow.set_tags(
            {
                "team": "equipo49",
                "question": "P2",
                "run_type": "hyperparameter_search",
                "selection_split": "validation_only",
                "git_commit": commit,
                "git_dirty": str(dirty).lower(),
                "data_version": manifest["data_version"],
            }
        )
        for model_name in SEARCH_MODELS:
            for params in grid_combinations(model_name):
                model = risk.build_model(model_name, random_seed, params)
                metrics = _fit_score_validation(model, features, panel, labels, masks)
                child_name = f"P2-{model_name}-" + "-".join(
                    f"{k}={v}" for k, v in sorted(params.items())
                )
                run_id = risk.log_p2_run(
                    run_name=child_name,
                    run_type="search_candidate",
                    model_name=model_name,
                    random_seed=random_seed,
                    params_extra={
                        **params,
                        "split_scope": "validation_only",
                        "search_parent": parent.info.run_id,
                    },
                    metrics={k: float(v) for k, v in metrics.items()},
                    quality=quality,
                    artifacts={},
                    model=model,
                    temporal_holdout=False,
                    manifest=manifest,
                    tracking_uri=tracking_uri,
                    nested=True,
                    features=features.loc[masks["val"]],
                )
                logger.info(
                    "Candidato {} {} -> val_pr_auc={:.4f}",
                    model_name,
                    params,
                    metrics["val_pr_auc"],
                )
                if np.isfinite(metrics["val_pr_auc"]) and metrics["val_pr_auc"] > best["score"]:
                    best = {
                        "score": metrics["val_pr_auc"],
                        "model_name": model_name,
                        "params": params,
                        "run_id": run_id,
                    }
        mlflow.log_metric("best_val_pr_auc", float(best["score"]))
        mlflow.set_tag("best_model", best.get("model_name", "none"))
        mlflow.set_tag("best_run_id", best.get("run_id", "none"))
        logger.success(
            "Mejor candidato por validacion: {} {}", best.get("model_name"), best.get("params")
        )


if __name__ == "__main__":
    app()
