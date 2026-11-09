"""Bundle de modelo para inferencia: modelo skops mas metadatos JSON.

El bundle es lo que el contenedor carga. No depende de MLflow ni de rutas del
host: el modelo se serializa con skops (sin pickle) y los metadatos registran el
contrato de entrada, el modo de puntuacion y la procedencia en el Registry.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from mibici_ml.config import SERVING_METADATA_FILE, SERVING_MODEL_FILE

SCORE_PREDICT_PROBA = "predict_proba_positive"
SCORE_PREDICT = "predict"
SUPPORTED_SCORE_KINDS = (SCORE_PREDICT_PROBA, SCORE_PREDICT)
REQUIRED_METADATA_KEYS = (
    "question",
    "registry_model_name",
    "alias",
    "version",
    "feature_names",
    "score_kind",
    "skops_trusted_types",
)


class BundleError(RuntimeError):
    """El bundle no existe, esta incompleto o no coincide con el contrato del servicio."""


@dataclass(frozen=True)
class ServingBundle:
    """Modelo cargado junto con sus metadatos."""

    model: Any
    metadata: dict[str, Any]
    bundle_dir: Path

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.metadata["feature_names"])

    @property
    def version(self) -> str:
        return str(self.metadata["version"])

    @property
    def model_label(self) -> str:
        return f"{self.metadata['registry_model_name']}@{self.metadata['alias']}"

    @property
    def threshold(self) -> float | None:
        value = self.metadata.get("threshold")
        return None if value is None else float(value)

    def score(self, matrix: np.ndarray) -> np.ndarray:
        """Puntua filas en el orden de ``feature_names`` y valida que sean finitas."""
        if self.metadata["score_kind"] == SCORE_PREDICT_PROBA:
            scores = self.model.predict_proba(matrix)[:, 1]
        else:
            scores = self.model.predict(matrix)
        scores = np.asarray(scores, dtype=float).ravel()
        if not np.isfinite(scores).all():
            raise ValueError("El modelo produjo valores no finitos.")
        if self.metadata["score_kind"] == SCORE_PREDICT_PROBA and (
            (scores < 0).any() or (scores > 1).any()
        ):
            raise ValueError("El modelo produjo probabilidades fuera de [0, 1].")
        return scores


def save_bundle(
    bundle_dir: Path,
    model: Any,
    metadata: dict[str, Any],
    *,
    skops_trusted_types: list[str] | None = None,
) -> Path:
    """Escribe ``model.skops`` y ``model_metadata.json`` en ``bundle_dir``."""
    import skops.io as sio

    bundle_dir.mkdir(parents=True, exist_ok=True)
    trusted = list(
        skops_trusted_types
        if skops_trusted_types is not None
        else metadata.get("skops_trusted_types", [])
    )
    full_metadata = dict(metadata)
    full_metadata["skops_trusted_types"] = trusted
    full_metadata["model_file"] = SERVING_MODEL_FILE
    sio.dump(model, bundle_dir / SERVING_MODEL_FILE)
    (bundle_dir / SERVING_METADATA_FILE).write_text(
        json.dumps(full_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return bundle_dir


def read_metadata(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / SERVING_METADATA_FILE
    if not path.is_file():
        raise BundleError(f"Falta {SERVING_METADATA_FILE} en el bundle del servicio.")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError(f"{SERVING_METADATA_FILE} no es JSON valido.") from exc
    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        raise BundleError(f"Metadatos incompletos, faltan: {', '.join(missing)}.")
    if metadata["score_kind"] not in SUPPORTED_SCORE_KINDS:
        raise BundleError(f"score_kind no soportado: {metadata['score_kind']}.")
    return metadata


def load_bundle(bundle_dir: Path) -> ServingBundle:
    """Carga el modelo skops con sus tipos de confianza declarados en los metadatos."""
    import skops.io as sio

    metadata = read_metadata(bundle_dir)
    model_path = bundle_dir / metadata.get("model_file", SERVING_MODEL_FILE)
    if not model_path.is_file():
        raise BundleError("Falta el archivo del modelo en el bundle del servicio.")
    model = sio.load(model_path, trusted=list(metadata["skops_trusted_types"]))
    n_expected = len(metadata["feature_names"])
    n_model = getattr(model, "n_features_in_", n_expected)
    if int(n_model) != n_expected:
        raise BundleError(
            f"El modelo espera {n_model} variables y los metadatos declaran {n_expected}."
        )
    return ServingBundle(model=model, metadata=metadata, bundle_dir=bundle_dir)
