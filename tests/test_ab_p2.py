"""A/B offline de P2: regla de decision, bootstrap pareado y registro en MLflow temporal.

Nunca usa ``mlflow.db`` real: el registro va a un SQLite dentro de ``tmp_path``.
"""

from __future__ import annotations

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytest

from mibici_ml.modeling import ab_p2


def test_decision_promotes_only_when_ci_is_above_zero_and_brier_not_worse():
    assert ab_p2.decide(0.01, brier_v1=0.005, brier_v2=0.004)[0] is True
    assert ab_p2.decide(-0.001, brier_v1=0.005, brier_v2=0.004)[0] is False
    assert ab_p2.decide(0.0, brier_v1=0.005, brier_v2=0.004)[0] is False
    assert ab_p2.decide(0.02, brier_v1=0.005, brier_v2=0.006)[0] is False


def test_identical_models_have_zero_delta_and_ci_containing_zero():
    rng = np.random.default_rng(0)
    labels = (rng.random(2_000) < 0.05).astype(int)
    scores = rng.random(2_000)
    clusters = np.arange(2_000) // 24

    boot = ab_p2.paired_bootstrap(labels, scores, scores, clusters, n_boot=50, seed=1)

    assert boot.delta_pr_auc == 0.0
    assert boot.delta_brier == 0.0
    assert boot.pr_auc_ci[0] <= 0.0 <= boot.pr_auc_ci[1]


def test_bootstrap_is_reproducible_with_the_same_seed():
    rng = np.random.default_rng(4)
    labels = (rng.random(3_000) < 0.05).astype(int)
    v1 = rng.random(3_000)
    v2 = np.clip(v1 + rng.normal(0, 0.1, 3_000), 0, 1)
    clusters = np.arange(3_000) // 24

    first = ab_p2.paired_bootstrap(labels, v1, v2, clusters, n_boot=30, seed=9)
    second = ab_p2.paired_bootstrap(labels, v1, v2, clusters, n_boot=30, seed=9)

    assert first == second


def test_clearly_better_challenger_gets_a_positive_interval():
    rng = np.random.default_rng(1)
    n = 6_000
    labels = (rng.random(n) < 0.05).astype(int)
    informative = np.where(labels == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))
    uninformative = rng.random(n)
    clusters = np.arange(n) // 24

    boot = ab_p2.paired_bootstrap(labels, uninformative, informative, clusters, n_boot=100, seed=2)

    assert boot.delta_pr_auc > 0
    assert boot.pr_auc_ci[0] > 0


def test_model_metrics_use_the_threshold_given_by_each_version():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    timestamps = pd.Series(pd.to_datetime(["2026-10-01 00:00"] * 4))

    metrics = ab_p2.model_metrics(labels, scores, threshold=0.5, timestamps=timestamps)

    assert metrics["precision_umbral"] == 1.0
    assert metrics["recall_umbral"] == 1.0
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["umbral"] == 0.5


def _minimal_result() -> dict:
    version = {
        "version": "1",
        "umbral": 0.2,
        "pr_auc": 0.4,
        "roc_auc": 0.9,
        "brier": 0.005,
        "recall_con_precision_min": 0.3,
        "precision_top10": 0.03,
    }
    return {
        "data_version": "synthetic",
        "motivo": "prueba",
        "bootstrap": {"remuestras": 10, "semilla": 42},
        "version_v1": version,
        "version_v2": {**version, "version": "2", "pr_auc": 0.42, "brier": 0.004},
        "delta_pr_auc": 0.02,
        "delta_pr_auc_ic": [0.01, 0.03],
        "delta_brier": -0.001,
        "promover_v2": True,
    }


def test_log_ab_run_writes_to_a_temporary_store(tmp_path):
    previous = mlflow.get_tracking_uri()
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.create_experiment("ab-prueba", artifact_location=(tmp_path / "art").as_uri())

        run_id = ab_p2.log_ab_run(_minimal_result(), experiment_name="ab-prueba", tracking_uri=uri)

        run = MlflowClient(tracking_uri=uri).get_run(run_id)
        assert run.info.status == "FINISHED"
        assert run.data.tags["run_type"] == "ab_test"
        assert run.data.metrics["delta_pr_auc"] == pytest.approx(0.02)
        assert run.data.metrics["promover_v2"] == 1.0
    finally:
        mlflow.set_tracking_uri(previous)
