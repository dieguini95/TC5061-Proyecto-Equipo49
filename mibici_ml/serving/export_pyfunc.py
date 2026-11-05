"""Exporta el champion pyfunc de P3 del Registry a ``models/serving/p3``.

P3 se registra como ``P3ClusterModel`` (pyfunc), no como modelo sklearn, asi que
no pasa por skops. El exportador descarga el artefacto del alias en
``models/serving/p3/model``, escribe ``model_metadata.json`` sin rutas locales y
verifica que la copia prediga lo mismo que el modelo del Registry.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from loguru import logger

from mibici_ml.config import (
    MLFLOW_TRACKING_URI,
    P3_MODEL_FEATURES,
    SERVING_ALIAS,
    SERVING_METADATA_FILE,
    SERVING_REGISTRY_NAMES,
    SERVING_ROOT_DIR,
)

EXPORTED_MODEL_DIR = "model"
SCORE_KIND_CLUSTER = "cluster_label"


def _parity_check(model_dir: Path, name: str, alias: str) -> None:
    import mlflow

    from mibici_ml.modeling.p3_model import p3_input_example

    sample = p3_input_example()
    exported = mlflow.pyfunc.load_model(str(model_dir)).predict(sample)
    registry = mlflow.pyfunc.load_model(f"models:/{name}@{alias}").predict(sample)
    if not exported.equals(registry):
        raise RuntimeError("Las predicciones del bundle P3 no coinciden con el Registry.")


def export_pyfunc_champion(
    question: str = "p3",
    *,
    alias: str = SERVING_ALIAS,
    output_root: Path = SERVING_ROOT_DIR,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> Path:
    """Export the champion of ``question`` (only P3 is pyfunc) and return its folder."""
    import mlflow
    from mlflow import MlflowClient

    if question != "p3":
        raise ValueError("Solo P3 se exporta como pyfunc.")
    name = SERVING_REGISTRY_NAMES[question]
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    version = client.get_model_version_by_alias(name, alias)
    run = client.get_run(version.run_id)

    bundle_dir = output_root / question
    model_dir = bundle_dir / EXPORTED_MODEL_DIR
    shutil.rmtree(model_dir, ignore_errors=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}@{alias}",
        dst_path=str(model_dir),
    )

    metadata: dict[str, Any] = {
        "question": question,
        "registry_model_name": name,
        "alias": alias,
        "version": str(version.version),
        "source_run_id": version.run_id,
        "model_family": run.data.tags.get("model_family") or version.tags.get("model_family"),
        "model_flavor": "pyfunc",
        "model_dir": EXPORTED_MODEL_DIR,
        "score_kind": SCORE_KIND_CLUSTER,
        "feature_names": list(P3_MODEL_FEATURES),
        "threshold": None,
        "selected_k": version.tags.get("selected_k"),
        "data_version": version.tags.get("data_version"),
        "git_commit": version.tags.get("git_commit"),
    }
    (bundle_dir / SERVING_METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _parity_check(model_dir, name, alias)
    logger.info(f"Exportado {name}@{alias} (version {version.version}) en {bundle_dir.name}/")
    return bundle_dir
