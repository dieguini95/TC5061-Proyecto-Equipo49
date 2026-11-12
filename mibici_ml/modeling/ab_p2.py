"""A/B offline de P2 en octubre 2026: champion (version 1) frente a challenger (version 2).

Octubre no se uso para entrenar ni para elegir umbral en ninguna de las dos versiones.
Cada version usa su propio umbral: v1 el de validacion (tag ``val_threshold``) y v2 el de
septiembre (tag ``threshold``).

REGLA DE DECISION (fijada antes de ver el resultado):
    Promover la version 2 solo si el IC 95% de ΔPR-AUC (v2 - v1) queda por encima de cero
    (limite inferior > 0) Y el Brier de v2 no es peor que el de v1 (Brier_v2 <= Brier_v1).
    En cualquier otro caso se conserva la version 1 como champion.

El intervalo es un bootstrap pareado por estacion-dia: se remuestrean estaciones-dia (24
horas cada una) con reemplazo y la misma muestra se evalua con ambas versiones. Mover el
alias champion es una decision aparte; este modulo solo emite la recomendacion.

Ejecutar con el entorno principal:
    .venv/bin/python -m mibici_ml.modeling.ab_p2 run
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from loguru import logger
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
import typer

from mibici_ml.config import (
    MLFLOW_TRACKING_URI,
    P2_CHAMPION_ALIAS,
    P2_MIN_PRECISION,
    P2_REGISTRY_MODEL_NAME,
    P2_TOP_K_FRACTION,
    PROJ_ROOT,
    RANDOM_SEED,
)
from mibici_ml.features import P2_FEATURE_COLUMNS_V2
from mibici_ml.modeling.registry_p2 import (
    ACTIVITY10_EXPERIMENT_NAME,
    ensure_activity_experiment,
    load_p2_version,
)
from mibici_ml.modeling.risk import precision_at_k, recall_at_precision
from mibici_ml.monitoring.panel import (
    OCTOBER_END,
    OCTOBER_START,
    inside_window,
    load_monitoring_panel,
)

app = typer.Typer(add_completion=False, no_args_is_help=True, help="A/B offline de P2.")

N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
CHALLENGER_ALIAS = "challenger"
TABLES_DIR = PROJ_ROOT / "reports" / "tables"
RESULT_JSON = TABLES_DIR / "p2_ab_oct26_resultado.json"
RESULT_CSV = TABLES_DIR / "p2_ab_oct26_metricas.csv"
DECISION_RULE = (
    "Promover v2 solo si el IC 95% de delta PR-AUC (v2 - v1) tiene limite inferior > 0 "
    "y Brier_v2 <= Brier_v1; si no, conservar v1."
)


@dataclass(frozen=True)
class BootstrapResult:
    delta_pr_auc: float
    pr_auc_ci: tuple[float, float]
    delta_brier: float
    brier_ci: tuple[float, float]
    valid_resamples: int


def cluster_codes(station: pd.Series, day: pd.Series) -> np.ndarray:
    """Un codigo entero por estacion-dia, para remuestrear por bloques."""
    return pd.MultiIndex.from_arrays([station.to_numpy(), day.to_numpy()]).factorize()[0]


def model_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float, timestamps: pd.Series
) -> dict[str, float]:
    """Metricas de octubre con el umbral propio de la version."""
    predictions = scores >= threshold
    positives = labels == 1
    tp = float(np.sum(predictions & positives))
    fp = float(np.sum(predictions & ~positives))
    fn = float(np.sum(~predictions & positives))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {
        "umbral": float(threshold),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "precision_umbral": precision,
        "recall_umbral": recall,
        "f1_umbral": f1,
        "recall_con_precision_min": recall_at_precision(labels, scores, P2_MIN_PRECISION),
        "precision_top10": precision_at_k(scores, labels, timestamps, P2_TOP_K_FRACTION),
    }


def paired_bootstrap(
    labels: np.ndarray,
    scores_v1: np.ndarray,
    scores_v2: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> BootstrapResult:
    """ΔPR-AUC y ΔBrier (v2 - v1) con bootstrap pareado por estacion-dia.

    Remuestrear estaciones-dia con reemplazo equivale a ponderar cada fila por el numero de
    veces que su bloque aparece en la muestra; ``sample_weight`` lo hace sin duplicar filas.
    """
    rng = np.random.default_rng(seed)
    n_clusters = int(clusters.max()) + 1
    point_pr = average_precision_score(labels, scores_v2) - average_precision_score(
        labels, scores_v1
    )
    point_brier = brier_score_loss(labels, scores_v2) - brier_score_loss(labels, scores_v1)
    deltas_pr, deltas_brier = [], []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
        weights = counts[clusters].astype(float)
        if not np.any(weights[labels == 1]):
            continue
        pr_v1 = average_precision_score(labels, scores_v1, sample_weight=weights)
        pr_v2 = average_precision_score(labels, scores_v2, sample_weight=weights)
        deltas_pr.append(pr_v2 - pr_v1)
        deltas_brier.append(
            brier_score_loss(labels, scores_v2, sample_weight=weights)
            - brier_score_loss(labels, scores_v1, sample_weight=weights)
        )
    low_q = (1 - CI_LEVEL) / 2 * 100
    high_q = (1 + CI_LEVEL) / 2 * 100
    pr_array = np.asarray(deltas_pr)
    brier_array = np.asarray(deltas_brier)
    return BootstrapResult(
        delta_pr_auc=float(point_pr),
        pr_auc_ci=(
            float(np.percentile(pr_array, low_q)),
            float(np.percentile(pr_array, high_q)),
        ),
        delta_brier=float(point_brier),
        brier_ci=(
            float(np.percentile(brier_array, low_q)),
            float(np.percentile(brier_array, high_q)),
        ),
        valid_resamples=int(pr_array.size),
    )


def decide(pr_auc_ci_low: float, brier_v1: float, brier_v2: float) -> tuple[bool, str]:
    """Aplica la regla fija. Devuelve (promover_v2, motivo)."""
    if not pr_auc_ci_low > 0:
        return False, "IC 95% de delta PR-AUC incluye o queda por debajo de cero."
    if brier_v2 > brier_v1:
        return False, "Brier de v2 es peor que el de v1."
    return True, "IC 95% de delta PR-AUC por encima de cero y Brier no peor."


def _threshold(version: Any, tag: str) -> float:
    value = version.tags.get(tag)
    if value is None:
        raise ValueError(f"La version {version.version} no tiene el tag {tag!r}.")
    return float(value)


def challenger_version(tracking_uri: str = MLFLOW_TRACKING_URI) -> Any:
    client = MlflowClient(tracking_uri=tracking_uri)
    return client.get_model_version_by_alias(P2_REGISTRY_MODEL_NAME, CHALLENGER_ALIAS)


def run_ab(n_boot: int = N_BOOTSTRAP) -> dict[str, Any]:
    """Carga v1 y v2, puntua octubre, compara y devuelve el resultado serializable."""
    full, data_version = load_monitoring_panel()
    october = full.loc[inside_window(full["ts"], OCTOBER_START, OCTOBER_END)].reset_index(
        drop=True
    )
    labels = october["label"].to_numpy().astype(int)
    matrix = october[list(P2_FEATURE_COLUMNS_V2)].to_numpy(dtype=float)

    v1_version = "1"
    v1_model = load_p2_version(v1_version)
    v2_meta = challenger_version()
    v2_model = load_p2_version(str(v2_meta.version))
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    v1_meta = client.get_model_version(P2_REGISTRY_MODEL_NAME, v1_version)
    threshold_v1 = _threshold(v1_meta, "val_threshold")
    threshold_v2 = _threshold(v2_meta, "threshold")

    scores_v1 = v1_model.predict_proba(matrix)[:, 1]
    scores_v2 = v2_model.predict_proba(matrix)[:, 1]
    metrics_v1 = model_metrics(labels, scores_v1, threshold_v1, october["ts"])
    metrics_v2 = model_metrics(labels, scores_v2, threshold_v2, october["ts"])
    clusters = cluster_codes(october["est"], october["ts"].dt.normalize())
    boot = paired_bootstrap(labels, scores_v1, scores_v2, clusters, n_boot=n_boot)
    promote, reason = decide(boot.pr_auc_ci[0], metrics_v1["brier"], metrics_v2["brier"])
    logger.info("Decision: promover_v2={} ({})", promote, reason)

    result = {
        "ventana": {"inicio": OCTOBER_START, "fin": OCTOBER_END, "rol": "out_of_time"},
        "data_version": data_version,
        "regla": DECISION_RULE,
        "bootstrap": {
            "tipo": "pareado por estacion-dia",
            "remuestras": n_boot,
            "remuestras_validas": boot.valid_resamples,
            "semilla": RANDOM_SEED,
            "nivel_ic": CI_LEVEL,
        },
        "filas": len(october),
        "estaciones_dia": int(clusters.max() + 1),
        "positivos": int(labels.sum()),
        "version_v1": {"version": v1_version, "alias": P2_CHAMPION_ALIAS, **metrics_v1},
        "version_v2": {"version": str(v2_meta.version), "alias": CHALLENGER_ALIAS, **metrics_v2},
        "delta_pr_auc": boot.delta_pr_auc,
        "delta_pr_auc_ic": list(boot.pr_auc_ci),
        "delta_brier": boot.delta_brier,
        "delta_brier_ic": list(boot.brier_ci),
        "promover_v2": promote,
        "motivo": reason,
        "_scores": {"v1": scores_v1, "v2": scores_v2, "labels": labels},
    }
    return result


def write_results(result: dict[str, Any]) -> tuple[Path, Path]:
    """Escribe la tabla de metricas y el JSON; los puntajes crudos no salen del proceso."""
    serializable = {k: v for k, v in result.items() if not k.startswith("_")}
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"modelo": key, **result[key]} for key in ("version_v1", "version_v2")]
    pd.DataFrame(rows).to_csv(RESULT_CSV, index=False)
    RESULT_JSON.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return RESULT_JSON, RESULT_CSV


def log_ab_run(
    result: dict[str, Any],
    experiment_name: str = ACTIVITY10_EXPERIMENT_NAME,
    artifacts: tuple[Path, ...] = (),
    tracking_uri: str | None = None,
) -> str:
    """Registra el A/B como run ``ab_test`` y devuelve su run_id."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    v1, v2 = result["version_v1"], result["version_v2"]
    params = {
        "regla": DECISION_RULE,
        "ventana": f"{OCTOBER_START}_to_{OCTOBER_END}",
        "remuestras_bootstrap": result["bootstrap"]["remuestras"],
        "semilla": result["bootstrap"]["semilla"],
        "version_v1": v1["version"],
        "version_v2": v2["version"],
        "umbral_v1": f"{v1['umbral']:.6f}",
        "umbral_v2": f"{v2['umbral']:.6f}",
    }
    metrics = {
        "pr_auc_v1": v1["pr_auc"],
        "pr_auc_v2": v2["pr_auc"],
        "delta_pr_auc": result["delta_pr_auc"],
        "delta_pr_auc_ic_bajo": result["delta_pr_auc_ic"][0],
        "delta_pr_auc_ic_alto": result["delta_pr_auc_ic"][1],
        "roc_auc_v1": v1["roc_auc"],
        "roc_auc_v2": v2["roc_auc"],
        "brier_v1": v1["brier"],
        "brier_v2": v2["brier"],
        "delta_brier": result["delta_brier"],
        "recall_con_precision_min_v1": v1["recall_con_precision_min"],
        "recall_con_precision_min_v2": v2["recall_con_precision_min"],
        "promover_v2": float(result["promover_v2"]),
    }
    with mlflow.start_run(run_name="ab_test") as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})
        mlflow.set_tags(
            {
                "team": "equipo49",
                "question": "P2",
                "run_type": "ab_test",
                "data_version": result["data_version"],
                "motivo": result["motivo"],
            }
        )
        for path in artifacts:
            if path.exists():
                mlflow.log_artifact(str(path))
        return run.info.run_id


@app.callback()
def _main() -> None:
    """A/B offline de P2. Subcomando: ``run``."""


@app.command("run")
def run_command(
    n_boot: int = typer.Option(N_BOOTSTRAP, help="Remuestras bootstrap."),
) -> None:
    """Compara champion y challenger en octubre 2026 y escribe el resultado."""
    ensure_activity_experiment()
    result = run_ab(n_boot=n_boot)
    json_path, csv_path = write_results(result)
    run_id = log_ab_run(result, artifacts=(json_path, csv_path))
    logger.success("A/B escrito en {} y {}; run {}", json_path.name, csv_path.name, run_id)
    logger.success(
        "PR-AUC v1 {:.4f} v2 {:.4f} delta {:+.4f} IC95 [{:+.4f}, {:+.4f}] promover_v2={}",
        result["version_v1"]["pr_auc"],
        result["version_v2"]["pr_auc"],
        result["delta_pr_auc"],
        *result["delta_pr_auc_ic"],
        result["promover_v2"],
    )


if __name__ == "__main__":
    app()
