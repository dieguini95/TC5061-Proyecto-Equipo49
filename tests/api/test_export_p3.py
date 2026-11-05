"""Export of the P3 pyfunc champion: metadata hygiene and parity with the Registry."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mibici_ml.config import P3_MODEL_FEATURES, SERVING_METADATA_FILE
from mibici_ml.serving.export_pyfunc import export_pyfunc_champion


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "capacidad": [30.0, 12.0, 20.0, 0.0],
            "salidas_12m": [1200.0, 80.0, 1.0, 5.0],
            "llegadas_12m": [900.0, 90.0, 1.0, 5.0],
        },
        columns=list(P3_MODEL_FEATURES),
    )


def test_exported_metadata_identifies_champion_without_local_paths(
    exported_p3_dir, registry_store, tmp_path
):
    metadata_path = exported_p3_dir.parent / SERVING_METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["registry_model_name"] == registry_store.name
    assert metadata["alias"] == "champion"
    assert metadata["version"] == "1"
    assert metadata["feature_names"] == list(P3_MODEL_FEATURES)
    assert metadata["model_flavor"] == "pyfunc"
    text = json.dumps(metadata)
    assert str(tmp_path) not in text
    assert "file:" not in text
    assert (exported_p3_dir / "MLmodel").is_file()


def test_exported_model_predicts_exactly_like_the_registry(exported_p3_dir, registry_store):
    import mlflow

    exported = mlflow.pyfunc.load_model(str(exported_p3_dir)).predict(_frame())
    registry = mlflow.pyfunc.load_model(f"models:/{registry_store.name}@champion").predict(
        _frame()
    )

    assert exported.equals(registry)


def test_export_only_supports_the_pyfunc_question(tmp_path, registry_store):
    with pytest.raises(ValueError):
        export_pyfunc_champion(
            "p2",
            output_root=tmp_path,
            tracking_uri=registry_store.tracking_uri,
        )
