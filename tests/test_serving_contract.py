"""Contrato del servicio: variables, exportacion, empaquetado y dependencias de la imagen."""

import json
from pathlib import Path
import re

import numpy as np
import pytest

from mibici_ml import config
from mibici_ml.api.p2 import P2_FEATURE_NAMES
from mibici_ml.features import P2_FEATURE_COLUMNS_V2
from mibici_ml.serving import bundle as bundle_module
from mibici_ml.serving import export

REPO_ROOT = Path(__file__).resolve().parents[1]


def _requirements(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip().lower()] = version.strip()
    return pins


def test_p2_api_features_match_training_features_in_order():
    assert tuple(P2_FEATURE_COLUMNS_V2) == P2_FEATURE_NAMES


def test_serving_config_names_registry_models_and_alias():
    assert config.SERVING_REGISTRY_NAMES["p2"] == config.P2_REGISTRY_MODEL_NAME
    assert config.SERVING_REGISTRY_NAMES["p1"] == config.P1_REGISTRY_MODEL_NAME
    assert config.SERVING_REGISTRY_NAMES["p3"] == config.P3_REGISTRY_MODEL_NAME
    assert config.SERVING_ALIAS == "champion"


def test_serving_requirements_pin_the_same_versions_as_the_project():
    project = _requirements(REPO_ROOT / "requirements.txt")
    serving = _requirements(REPO_ROOT / "requirements-serving.txt")
    assert serving, "requirements-serving.txt no declara versiones fijas"
    mismatched = {
        name: (version, project.get(name))
        for name, version in serving.items()
        if project.get(name) != version
    }
    assert not mismatched, f"Versiones distintas a requirements.txt: {mismatched}"


def test_containerfile_runs_unprivileged_with_healthcheck_and_no_secrets():
    text = (REPO_ROOT / "Containerfile").read_text(encoding="utf-8")
    assert re.search(r"^USER 10001:10001$", text, re.MULTILINE)
    assert "HEALTHCHECK" in text
    assert "/health" in text
    assert not re.search(r"^COPY \.\s", text, re.MULTILINE), "No copiar el repo completo"
    assert not re.search(r"(?i)^ENV .*(token|secret|password|key)", text, re.MULTILINE)
    assert "--reload" not in text


def test_containerfile_serves_the_shared_app_with_both_models_from_the_image():
    text = (REPO_ROOT / "Containerfile").read_text(encoding="utf-8")
    assert "mibici_ml.api.app:app" in text, "El entrypoint debe ser la API compartida"
    assert "mibici_ml.serving.apps" not in text, "La app paralela por pregunta fue retirada"
    assert "MIBICI_API_PROVIDER=mlflow" in text
    assert "MIBICI_MODEL_PATH=/app/models/serving/p3/model" in text
    assert "MIBICI_P2_BUNDLE_DIR=/app/models/serving/p2" in text
    assert "COPY models/serving ./models/serving" in text


def test_serving_requirements_cover_the_pyfunc_p3_model_imports():
    serving = _requirements(REPO_ROOT / "requirements-serving.txt")
    for package in ("mlflow-skinny", "matplotlib", "typer", "pandas", "cloudpickle", "skops"):
        assert package in serving, f"requirements-serving.txt debe incluir {package}"


def test_dockerignore_keeps_secrets_data_and_mlflow_out_of_the_image():
    lines = {
        line.strip()
        for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    }
    for pattern in (".env", "data", "mlruns", "mlflow.db", ".venv", "tests"):
        assert pattern in lines, f".dockerignore debe excluir {pattern}"
    assert "!models/serving" in lines, "El bundle del servicio debe entrar en la imagen"


def test_find_model_artifacts_falls_back_to_local_mlruns(tmp_path):
    class MissingRegisteredPath:
        def get_logged_model(self, model_id):
            class Logged:
                artifact_location = "/ruta/antigua/que/no/existe"

            return Logged()

    artifacts = tmp_path / "mlruns" / "1" / "models" / "m-abc" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")
    found = export.find_model_artifacts(MissingRegisteredPath(), "m-abc", tmp_path / "mlruns")
    assert found == artifacts


def test_find_model_artifacts_fails_clearly_when_missing(tmp_path):
    from mlflow.exceptions import MlflowException

    class Client:
        def get_logged_model(self, model_id):
            raise MlflowException("Logged model not found")

    with pytest.raises(FileNotFoundError, match="m-none"):
        export.find_model_artifacts(Client(), "m-none", tmp_path)


def test_resolve_model_id_reads_models_uri():
    assert export.resolve_model_id("models:/m-123") == "m-123"
    with pytest.raises(ValueError):
        export.resolve_model_id("runs:/abc/model")


def test_round_trip_export_from_mlflow_artifacts_matches_original(tmp_path):
    """Exporta un modelo guardado con MLflow y verifica que el bundle reproduce sus predicciones."""
    from mlflow.models import infer_signature
    import mlflow.sklearn
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(7)
    matrix = rng.random((200, len(P2_FEATURE_NAMES)))
    labels = (matrix[:, 1] > 0.6).astype(int)
    model = HistGradientBoostingClassifier(max_iter=10, random_state=7).fit(matrix, labels)
    frame = pd.DataFrame(matrix[:5], columns=list(P2_FEATURE_NAMES))
    artifact_dir = tmp_path / "artifacts"
    import skops.io as sio

    mlflow.sklearn.save_model(
        model,
        path=str(artifact_dir),
        serialization_format="skops",
        skops_trusted_types=sio.get_untrusted_types(data=sio.dumps(model)),
        signature=infer_signature(frame, model.predict_proba(matrix[:5])[:, 1]),
        input_example=frame,
    )
    registry = {
        "registry_model_name": "mibici-equipo49-p2-riesgo",
        "alias": "champion",
        "version": "3",
        "model_id": "m-test",
        "source_run_id": "run-test",
        "threshold": 0.25,
        "threshold_source": "prueba",
    }
    bundle_dir = export.build_bundle(
        artifact_dir, question="p2", registry=registry, output_root=tmp_path / "out"
    )
    loaded = bundle_module.load_bundle(bundle_dir)
    assert loaded.feature_names == P2_FEATURE_NAMES
    assert loaded.threshold == pytest.approx(0.25)
    expected = model.predict_proba(matrix[:20])[:, 1]
    np.testing.assert_allclose(loaded.score(matrix[:20]), expected)
    metadata = json.loads((bundle_dir / "model_metadata.json").read_text(encoding="utf-8"))
    assert str(tmp_path) not in json.dumps(metadata), (
        "Los metadatos no deben exponer rutas locales"
    )


def test_read_metadata_rejects_incomplete_bundle(tmp_path):
    (tmp_path / "model_metadata.json").write_text('{"question": "p2"}', encoding="utf-8")
    with pytest.raises(bundle_module.BundleError, match="incompletos"):
        bundle_module.read_metadata(tmp_path)


def test_exported_bundle_is_versioned_so_the_image_builds_from_a_clean_clone():
    for rel in (
        "models/serving/p2/model.skops",
        "models/serving/p2/model_metadata.json",
        "models/serving/p3/model/MLmodel",
        "models/serving/p3/model/python_model.pkl",
        "models/serving/p3/model_metadata.json",
    ):
        assert (REPO_ROOT / rel).is_file(), f"Falta {rel} en el repositorio"
    mlmodel = (REPO_ROOT / "models/serving/p3/model/MLmodel").read_text(encoding="utf-8")
    assert "/Users/" not in mlmodel and "/home/" not in mlmodel, "MLmodel expone una ruta local"
