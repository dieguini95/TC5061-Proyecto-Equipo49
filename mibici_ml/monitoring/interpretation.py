"""Interpretacion de la Actividad 10: drift frente a importancia, desempeno y volumen.

Lee los reportes de Evidently (``reports/drift/p2/r1..r3``), la importancia por permutacion
del champion, el panel de monitoreo y los modelos del Registry. Escribe tablas en
``reports/tables/`` y figuras en ``docs/evidencias/Actividad 10 - Drift real y A-B P2/figuras``.

Ejecutar con el entorno principal, despues de ``panel build`` y de las corridas de drift:
    .venv/bin/python -m mibici_ml.monitoring.interpretation
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
import typer

from mibici_ml.config import P2_REGISTRY_MODEL_NAME, PROJ_ROOT
from mibici_ml.features import P2_FEATURE_COLUMNS_V2
from mibici_ml.modeling.ab_p2 import run_ab
from mibici_ml.modeling.registry_p2 import load_p2_version
from mibici_ml.monitoring.panel import (
    OCTOBER_END,
    inside_window,
    load_monitoring_panel,
)

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Interpretacion de P2.")

DRIFT_DIR = PROJ_ROOT / "reports" / "drift" / "p2"
RUNS = {
    "r1": "R1 oct-26 vs entrenamiento",
    "r2": "R2 sep-26 vs entrenamiento",
    "r3": "R3 control oct-25 vs oct-26",
}
CHAMPION_RUN_ID = "9dca0330ba6241dc8ccdf41d48cf4954"
IMPORTANCE_PATH = (
    PROJ_ROOT
    / "mlruns"
    / "1"
    / CHAMPION_RUN_ID
    / "artifacts"
    / "plots"
    / "p2_importancia_hist_gb.csv"
)
TABLES_DIR = PROJ_ROOT / "reports" / "tables"
FIGURES_DIR = PROJ_ROOT / "docs" / "evidencias" / "Actividad 10 - Drift real y A-B P2" / "figuras"
DRIFT_THRESHOLD = 0.05
MONTHS = {
    "2026-07": "validacion",
    "2026-08": "validacion",
    "2026-09": "prueba",
    "2026-10": "fuera_de_tiempo",
}
MANIFEST_PATH = PROJ_ROOT / "references" / "data_manifest.json"
MONITORING_MANIFEST_PATH = PROJ_ROOT / "references" / "monitoring_manifest_p2.json"


def drift_table() -> pd.DataFrame:
    """Score de drift por variable en R1-R3, con la importancia por permutacion del champion."""
    importance = pd.read_csv(IMPORTANCE_PATH).set_index("feature")
    table = importance[["importancia_media"]].rename(
        columns={"importancia_media": "importancia_permutacion"}
    )
    for run in RUNS:
        columns = pd.read_csv(DRIFT_DIR / run / "drift_columns.csv").set_index("feature")
        table[f"{run}_score"] = columns["score"]
        table[f"{run}_drift"] = columns["drift_detected"].astype(bool)
    table = table.loc[list(P2_FEATURE_COLUMNS_V2)]
    high_importance = table["importancia_permutacion"] >= 0.01
    table["lectura"] = np.select(
        [
            table["r1_drift"] & high_importance,
            table["r1_drift"] & ~high_importance,
            ~table["r1_drift"] & high_importance,
        ],
        [
            "drift y alta importancia",
            "drift con baja importancia",
            "sin drift con alta importancia",
        ],
        default="sin drift y baja importancia",
    )
    return table.sort_values("importancia_permutacion", ascending=False).reset_index(
        names="variable"
    )


def v1_threshold() -> float:
    """Umbral de validacion de la version 1, leido del Registry (tag ``val_threshold``)."""
    from mlflow import MlflowClient

    from mibici_ml.config import MLFLOW_TRACKING_URI

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    return float(client.get_model_version(P2_REGISTRY_MODEL_NAME, "1").tags["val_threshold"])


def monthly_v1(full: pd.DataFrame, model: object, threshold: float) -> pd.DataFrame:
    """Desempeno de v1 por mes con su propio umbral de validacion."""
    rows = []
    for month, split in MONTHS.items():
        start = f"{month}-01"
        end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        if month == "2026-10":
            end = OCTOBER_END
        mask = inside_window(full["ts"], start, end)
        frame = full.loc[mask]
        labels = frame["label"].to_numpy().astype(int)
        scores = model.predict_proba(frame[list(P2_FEATURE_COLUMNS_V2)].to_numpy(float))[:, 1]
        predicted = scores >= threshold
        positives = labels == 1
        tp = np.sum(predicted & positives)
        rows.append(
            {
                "mes": month,
                "division": split,
                "filas": len(frame),
                "positivos": int(labels.sum()),
                "prevalencia": labels.mean(),
                "prob_media_predicha": scores.mean(),
                "pr_auc": average_precision_score(labels, scores),
                "precision_umbral": tp / max(predicted.sum(), 1),
                "recall_umbral": tp / max(positives.sum(), 1),
            }
        )
    return pd.DataFrame(rows)


def volume_table() -> pd.DataFrame:
    """Viajes limpios por mes y llegadas medias por hora-estacion del panel."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    monitoring = json.loads(MONITORING_MANIFEST_PATH.read_text(encoding="utf-8"))
    rows = [(t["month"], t["clean_rows"]) for t in manifest["trips"]]
    rows += [(t["month"], t["clean_rows"]) for t in monitoring["extra_trips"]]
    trips = pd.DataFrame(rows, columns=["mes_portal", "viajes_limpios"])
    trips["mes"] = pd.PeriodIndex(trips["mes_portal"].str.replace("_", "-"), freq="M").astype(str)
    return trips[["mes", "viajes_limpios"]]


def _figure(name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / name


def plot_drift(table: pd.DataFrame) -> Path:
    ordered = table.sort_values("r1_score", ascending=True)
    positions = np.arange(len(ordered))
    height = 0.27
    fig, ax = plt.subplots(figsize=(9, 7))
    for offset, run in enumerate(RUNS):
        ax.barh(
            positions + (offset - 1) * height, ordered[f"{run}_score"], height, label=RUNS[run]
        )
    ax.axvline(DRIFT_THRESHOLD, color="black", linestyle="--", linewidth=1, label="umbral 0.05")
    ax.set_yticks(positions, ordered["variable"])
    ax.set_xlabel("Score de drift (Wasserstein normalizada o Jensen-Shannon)")
    ax.set_title("Drift por variable: R1, R2 y control R3")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = _figure("01_drift_por_variable.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_drift_vs_importance(table: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    importance = table["importancia_permutacion"].clip(lower=1e-4)
    colors = np.where(table["r1_drift"], "#c0392b", "#2e86c1")
    ax.scatter(importance, table["r1_score"], c=colors, s=60)
    for _, row in table.iterrows():
        ax.annotate(
            row["variable"],
            (max(row["importancia_permutacion"], 1e-4), row["r1_score"]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    ax.axhline(DRIFT_THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Importancia por permutacion (escala log; valores <= 0 se muestran en 1e-4)")
    ax.set_ylabel("Score de drift R1 (oct-26 vs entrenamiento)")
    ax.set_title("Drift frente a importancia del champion")
    fig.tight_layout()
    path = _figure("02_drift_vs_importancia.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_monthly(monthly: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(monthly["mes"], monthly["pr_auc"], marker="o", color="#1f3a5f", label="PR-AUC v1")
    ax.axhline(
        0.4001, color="gray", linestyle="--", linewidth=1, label="PR-AUC prueba sep (0.4001)"
    )
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0.3, 0.45)
    twin = ax.twinx()
    twin.bar(
        monthly["mes"],
        monthly["prevalencia"],
        alpha=0.25,
        color="#e67e22",
        label="prevalencia observada",
    )
    twin.set_ylabel("prevalencia (positivos / filas)")
    ax.set_title("Desempeno de v1 por mes (jul-oct 2026)")
    handles, labels = ax.get_legend_handles_labels()
    extra_handles, extra_labels = twin.get_legend_handles_labels()
    ax.legend(handles + extra_handles, labels + extra_labels, fontsize=8, loc="lower left")
    fig.tight_layout()
    path = _figure("03_v1_mensual.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_calibration(monthly: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(monthly))
    ax.bar(x - 0.2, monthly["prob_media_predicha"], 0.4, label="probabilidad media predicha")
    ax.bar(x + 0.2, monthly["prevalencia"], 0.4, label="prevalencia observada")
    ax.set_xticks(x, monthly["mes"])
    ax.set_ylabel("proporcion")
    ax.set_title("Calibracion agregada de v1 por mes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = _figure("04_calibracion_mensual.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_volume(volume: pd.DataFrame, arrivals: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(volume["mes"], volume["viajes_limpios"], color="#7fb3d5", label="viajes limpios")
    ax.set_ylabel("viajes limpios por mes")
    ax.tick_params(axis="x", rotation=45)
    twin = ax.twinx()
    twin.plot(
        arrivals.index,
        arrivals.values,
        color="#1f3a5f",
        marker="o",
        label="llegadas medias por hora-estacion",
    )
    twin.set_ylabel("llegadas medias por hora-estacion")
    ax.set_title("Volumen de viajes y llegadas por mes (oct-25 a oct-26)")
    handles, labels = ax.get_legend_handles_labels()
    extra_handles, extra_labels = twin.get_legend_handles_labels()
    ax.legend(handles + extra_handles, labels + extra_labels, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = _figure("05_volumen_mensual.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_pr_curves(labels: np.ndarray, scores_v1: np.ndarray, scores_v2: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, scores, color in (
        ("v1 champion", scores_v1, "#1f3a5f"),
        ("v2 challenger", scores_v2, "#e67e22"),
    ):
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        ax.plot(recall, precision, color=color, label=f"{name} (PR-AUC {ap:.4f})")
    ax.axhline(
        labels.mean(),
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"azar (prevalencia {labels.mean():.4f})",
    )
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Curvas precision-recall en octubre 2026")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = _figure("06_pr_curvas_oct26.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_bootstrap(deltas: np.ndarray, point: float, ci: tuple[float, float]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(deltas, bins=40, color="#7fb3d5", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1, label="cero")
    ax.axvline(point, color="#1f3a5f", linewidth=2, label=f"delta puntual {point:+.4f}")
    ax.axvline(
        ci[0], color="#e67e22", linestyle="--", label=f"IC 95% [{ci[0]:+.4f}, {ci[1]:+.4f}]"
    )
    ax.axvline(ci[1], color="#e67e22", linestyle="--")
    ax.set_xlabel("delta PR-AUC (v2 - v1)")
    ax.set_ylabel("remuestras bootstrap")
    ax.set_title("Distribucion bootstrap pareada por estacion-dia")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = _figure("07_bootstrap_delta_pr_auc.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


@app.callback()
def _main() -> None:
    """Interpretacion de P2. Subcomando: ``run``."""


@app.command("run")
def run_command() -> None:
    """Escribe tablas y figuras de interpretacion de la Actividad 10."""
    full, _ = load_monitoring_panel()
    model_v1 = load_p2_version("1")
    table = drift_table()
    monthly = monthly_v1(full, model_v1, v1_threshold())
    volume = volume_table()
    arrivals = (
        full.assign(mes=full["ts"].dt.to_period("M").astype(str))
        .groupby("mes")["llegadas_1h"]
        .mean()
    )
    arrivals = arrivals.loc[arrivals.index.isin(volume["mes"])]
    volume = volume.merge(arrivals.rename("llegadas_1h_media").reset_index(), on="mes", how="left")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "p2_drift_importancia.csv", index=False)
    monthly.to_csv(TABLES_DIR / "p2_v1_mensual.csv", index=False)
    volume.to_csv(TABLES_DIR / "p2_volumen_mensual.csv", index=False)

    ab = run_ab()
    scores = ab["_scores"]
    boot = ab["_bootstrap"]
    figures = [
        plot_drift(table),
        plot_drift_vs_importance(table),
        plot_monthly(monthly),
        plot_calibration(monthly),
        plot_volume(volume, arrivals.loc[volume["mes"]]),
        plot_pr_curves(scores["labels"], scores["v1"], scores["v2"]),
        plot_bootstrap(boot.pr_deltas, boot.delta_pr_auc, boot.pr_auc_ci),
    ]
    for path in figures:
        logger.success("Figura: {}", path.name)
    logger.success("Tablas escritas en {}", TABLES_DIR)


if __name__ == "__main__":
    app()
