"""Exporta el champion del Model Registry a un bundle para el servicio de inferencia.

Uso desde la raiz del repositorio, con el entorno del proyecto activo::

    python -m mibici_ml.serving.export --question p2

El modelo se localiza por nombre y alias en el Registry. Si la ruta de artefactos
guardada en la base de MLflow ya no existe (p. ej. el repositorio se movio), se
busca el mismo modelo en ``mlruns/`` del proyecto por su ``model_id``. Antes de
escribir el bundle se compara su prediccion con la del modelo original de MLflow.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
import yaml

from mibici_ml.config import (
    MLFLOW_TRACKING_URI,
    MLRUNS_DIR,
    PROJ_ROOT,
    SERVING_ALIAS,
    SERVING_METADATA_FILE,
    SERVING_MODEL_FILE,
    SERVING_REGISTRY_NAMES,
    SERVING_ROOT_DIR,
)
from mibici_ml.serving.bundle import (
    SCORE_PREDICT,
    SCORE_PREDICT_PROBA,
    load_bundle,
    save_bundle,
)

SCORE_KIND_BY_QUESTION = {"p1": SCORE_PREDICT, "p2": SCORE_PREDICT_PROBA}


def resolve_model_id(source: str | None) -> str:
    """Convierte ``models:/m-...`` en el ``model_id`` del modelo logueado."""
    if not source or not source.startswith("models:/"):
        raise ValueError(f"La version no apunta a un modelo logueado: {source!r}.")
    return source.removeprefix("models:/")


def find_model_artifacts(client: Any, model_id: str, mlruns_dir: Path = MLRUNS_DIR) -> Path:
    """Devuelve la carpeta de artefactos del modelo, primero la registrada y luego ``mlruns/``."""
    from mlflow.exceptions import MlflowException

    candidates: list[Path] = []
    try:
        logged = client.get_logged_model(model_id)
        if logged.artifact_location:
            candidates.append(Path(logged.artifact_location))
    except (AttributeError, MlflowException) as exc:
        logger.warning(f"No se pudo leer el modelo logueado {model_id}: {exc}")
    candidates.extend(sorted(mlruns_dir.glob(f"*/models/{model_id}/artifacts")))
    for candidate in candidates:
        if (candidate / "MLmodel").is_file():
            return candidate
    raise FileNotFoundError(
        f"No encuentro los artefactos del modelo {model_id} ni en la ruta registrada "
        f"ni en {mlruns_dir}. Verifica que el repositorio tenga su carpeta mlruns/."
    )


def read_mlmodel(artifact_dir: Path) -> dict[str, Any]:
    """Lee el flavor sklearn y el contrato de entrada de ``MLmodel``."""
    doc = yaml.safe_load((artifact_dir / "MLmodel").read_text(encoding="utf-8"))
    flavor = (doc.get("flavors") or {}).get("sklearn")
    if not flavor:
        raise ValueError("El modelo no tiene flavor sklearn; el servicio solo sirve sklearn.")
    if flavor.get("serialization_format") != "skops":
        raise ValueError(
            "El modelo no esta serializado con skops; no se carga pickle en servicio."
        )
    model_file = flavor["pickled_model"]
    if not (artifact_dir / model_file).is_file():
        raise FileNotFoundError(f"Falta {model_file} en los artefactos del modelo.")
    signature = doc.get("signature") or {}
    inputs = json.loads(signature.get("inputs") or "[]")
    if not inputs:
        raise ValueError("El modelo no declara firma de entrada; no se puede validar la API.")
    return {
        "model_file": model_file,
        "feature_names": [item["name"] for item in inputs],
        "trusted_types": list(flavor.get("skops_trusted_types") or []),
        "sklearn_version": flavor.get("sklearn_version"),
    }


def _parity_check(artifact_dir: Path, bundle_dir: Path, score_kind: str) -> None:
    """Compara el bundle exportado con el modelo original de MLflow en el input example."""
    import mlflow.sklearn

    example_path = artifact_dir / "input_example.json"
    if not example_path.is_file():
        raise FileNotFoundError("El modelo no guardo input_example.json; no se puede validar.")
    example = json.loads(example_path.read_text(encoding="utf-8"))
    columns = list(example["columns"])
    bundle = load_bundle(bundle_dir)
    matrix = np.asarray(example["data"], dtype=float)[
        :, [columns.index(f) for f in bundle.feature_names]
    ]
    reference = mlflow.sklearn.load_model(str(artifact_dir))
    if score_kind == SCORE_PREDICT_PROBA:
        expected = reference.predict_proba(matrix)[:, 1]
    else:
        expected = reference.predict(matrix)
    actual = bundle.score(matrix)
    if not np.allclose(np.ravel(expected), actual, rtol=1e-9, atol=1e-12):
        raise ValueError("El bundle exportado no reproduce las predicciones del modelo de MLflow.")


def build_bundle(
    artifact_dir: Path,
    *,
    question: str,
    registry: dict[str, Any],
    output_root: Path = SERVING_ROOT_DIR,
) -> Path:
    """Escribe ``<output_root>/<pregunta>`` a partir de los artefactos de un modelo sklearn."""
    import skops.io as sio

    mlmodel = read_mlmodel(artifact_dir)
    if question == "p2" and registry.get("threshold") is None:
        raise ValueError("P2 requiere el umbral de validacion (tag o metrica val_threshold).")
    score_kind = SCORE_KIND_BY_QUESTION[question]
    metadata = {
        "question": question,
        "registry_model_name": registry["registry_model_name"],
        "alias": registry["alias"],
        "version": str(registry["version"]),
        "model_id": registry["model_id"],
        "source_run_id": registry["source_run_id"],
        "model_family": registry.get("model_family"),
        "score_kind": score_kind,
        "feature_names": mlmodel["feature_names"],
        "threshold": registry.get("threshold"),
        "threshold_source": registry.get("threshold_source"),
        "data_version": registry.get("data_version"),
        "git_commit": registry.get("git_commit"),
        "sklearn_version": mlmodel["sklearn_version"],
        "skops_trusted_types": mlmodel["trusted_types"],
    }
    model = sio.load(artifact_dir / mlmodel["model_file"], trusted=mlmodel["trusted_types"])
    bundle_dir = output_root / question
    for stale in (SERVING_MODEL_FILE, SERVING_METADATA_FILE):
        (bundle_dir / stale).unlink(missing_ok=True)
    save_bundle(bundle_dir, model, metadata)
    _parity_check(artifact_dir, bundle_dir, score_kind)
    return bundle_dir


def export_champion(
    question: str,
    *,
    alias: str = SERVING_ALIAS,
    output_root: Path = SERVING_ROOT_DIR,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    mlruns_dir: Path = MLRUNS_DIR,
) -> Path:
    """Exporta el modelo que tiene ``alias`` en el Registry para la pregunta indicada."""
    from mlflow import MlflowClient

    if question not in SERVING_REGISTRY_NAMES:
        raise ValueError(f"Pregunta no soportada: {question}.")
    if question == "p3":
        raise NotImplementedError(
            "P3 se registra como pyfunc personalizado (P3ClusterModel); el servicio necesita "
            "un adaptador propio antes de exportarlo."
        )
    name = SERVING_REGISTRY_NAMES[question]
    client = MlflowClient(tracking_uri=tracking_uri)
    version = client.get_model_version_by_alias(name, alias)
    model_id = resolve_model_id(version.source)
    artifact_dir = find_model_artifacts(client, model_id, mlruns_dir)
    run = client.get_run(version.run_id)
    threshold, threshold_source = _threshold(version, run)
    registry = {
        "registry_model_name": name,
        "alias": alias,
        "version": version.version,
        "model_id": model_id,
        "source_run_id": version.run_id,
        "model_family": version.tags.get("model_family") or run.data.tags.get("model_family"),
        "data_version": version.tags.get("data_version"),
        "git_commit": version.tags.get("git_commit"),
        "threshold": threshold,
        "threshold_source": threshold_source,
    }
    logger.info(f"Exportando {name}@{alias} (version {version.version}) desde {artifact_dir.name}")
    return build_bundle(
        artifact_dir, question=question, registry=registry, output_root=output_root
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJ_ROOT))
    except ValueError:
        return path.name


def _threshold(version: Any, run: Any) -> tuple[float | None, str | None]:
    raw = version.tags.get("val_threshold")
    if raw is not None:
        return float(raw), "tag val_threshold de la version"
    metric = run.data.metrics.get("val_threshold")
    if metric is not None:
        return float(metric), "metrica val_threshold del run"
    return None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exporta el champion del Registry a un bundle.")
    parser.add_argument("--question", required=True, choices=sorted(SERVING_REGISTRY_NAMES))
    parser.add_argument("--alias", default=SERVING_ALIAS)
    parser.add_argument("--output-root", type=Path, default=SERVING_ROOT_DIR)
    args = parser.parse_args(argv)
    bundle_dir = export_champion(args.question, alias=args.alias, output_root=args.output_root)
    bundle = load_bundle(bundle_dir)
    print(
        f"Bundle listo: {_display_path(bundle_dir)} "
        f"({bundle.model_label} version {bundle.version}, "
        f"{len(bundle.feature_names)} variables, umbral {bundle.threshold})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
