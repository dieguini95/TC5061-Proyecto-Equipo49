"""Punto de entrada unico de entrenamiento y registro en MLflow.

Despacha la pregunta elegida a su modulo (``demand`` para P1, ``risk`` para P2 y
``cluster`` para P3). Los valores por defecto (rutas, ventana, semilla, experimento y
URI de tracking) viven solo en ``mibici_ml.config``.

Los parametros llegan como texto desde ``mlflow run`` (que solo sustituye ``{nombre}``),
por eso un valor vacio significa "no definido" y los booleanos se parsean aqui.

Uso desde la raiz del proyecto:

    python -m mibici_ml.modeling.train --question p2 --model hist_gb --output-dir /tmp/p2
    python -m mibici_ml.modeling.train --question p1 --output-dir /tmp/p1
    python -m mibici_ml.modeling.train --question p3 --final-k 5 --output-dir /tmp/p3
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from loguru import logger
import mlflow
import typer

from mibici_ml import config
from mibici_ml.modeling import cluster, demand, risk

app = typer.Typer(no_args_is_help=True, help="Entrenamiento y registro por pregunta (P1, P2, P3).")

QUESTIONS = ("p1", "p2", "p3")
P2_MODELS = risk.CANDIDATE_NAMES
TRUE_VALUES = {"1", "true", "yes", "si", "sí"}


def _as_bool(value: str) -> bool:
    return value.strip().lower() in TRUE_VALUES


def _release_wrapper_run() -> None:
    """Libera el run que crea ``mlflow run`` para que cada modulo abra sus propios runs.

    ``mlflow run`` exporta ``MLFLOW_RUN_ID`` y deja un run activo. Si no se libera, el
    primer ``start_run`` reutiliza ese run y los parametros de otra corrida chocan
    (``Changing param values is not allowed``).
    """
    if mlflow.active_run() is not None:
        mlflow.end_run()
    os.environ.pop("MLFLOW_RUN_ID", None)


@app.command()
def main(
    question: Annotated[str, typer.Option(help="Pregunta del proyecto: p1, p2 o p3.")],
    model: Annotated[
        str, typer.Option(help="Modelo final de P2: logistica o hist_gb. Vacio en p1 y p3.")
    ] = "",
    output_dir: Annotated[
        Path, typer.Option(help="Directorio de artefactos del run (no usar reports/tables).")
    ] = Path("/tmp/mibici-runs"),
    final_k: Annotated[
        str, typer.Option(help="k final de P3 (obligatorio para p3). Vacio si no aplica.")
    ] = "",
    with_mlflow: Annotated[str, typer.Option(help="true o false: registrar en MLflow.")] = "true",
    tuned_json: Annotated[
        str, typer.Option(help="P2: hiperparametros del ganador de la busqueda (JSON).")
    ] = "",
) -> None:
    """Entrena o evalua la pregunta indicada y registra los runs en MLflow."""
    question = question.strip().lower()
    if question not in QUESTIONS:
        raise typer.BadParameter(f"--question debe ser uno de: {', '.join(QUESTIONS)}.")
    _release_wrapper_run()
    mlflow_on = _as_bool(with_mlflow)
    logger.info("Pregunta {} | experimento {}", question.upper(), config.EXPERIMENT_NAME)

    if question == "p1":
        demand.main(
            manifest_path=config.DATA_MANIFEST_PATH,
            with_mlflow=mlflow_on,
            tracking_uri=config.MLFLOW_TRACKING_URI,
        )
        return

    if question == "p2":
        model = model.strip()
        if model not in P2_MODELS:
            raise typer.BadParameter(f"--model debe ser uno de: {', '.join(P2_MODELS)}.")
        risk.run_command(
            output_dir=output_dir,
            final_model=model,
            with_mlflow=mlflow_on,
            tuned_json=tuned_json,
        )
        return

    # question == "p3"
    k_text = final_k.strip()
    if not k_text:
        raise typer.BadParameter("--final-k es obligatorio para p3 (elegido tras evaluar).")
    if not k_text.isdigit():
        raise typer.BadParameter("--final-k debe ser un entero.")
    cluster.run_command(
        output_dir=output_dir,
        final_k=int(k_text),
        with_mlflow=mlflow_on,
    )


if __name__ == "__main__":
    app()
