"""Punto de entrada unico de entrenamiento y registro en MLflow.

Despacha la pregunta elegida a su modulo (``search_p1`` para P1, ``risk`` para P2 y
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
from mibici_ml.modeling import cluster, risk

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
        str,
        typer.Option(help="P1: ridge, hist_poisson, hist_squared; vacio = grid. P2: candidato."),
    ] = "",
    output_dir: Annotated[
        str, typer.Option(help="Directorio de artefactos; vacio usa config.py por pregunta.")
    ] = "",
    final_k: Annotated[
        str, typer.Option(help="k final de P3 (obligatorio para p3). Vacio si no aplica.")
    ] = "",
    with_mlflow: Annotated[str, typer.Option(help="true o false: registrar en MLflow.")] = "true",
    tuned_json: Annotated[
        str,
        typer.Option(help="P1/P2: hiperparametros (clave=valor;clave=valor)."),
    ] = "",
    register_best: Annotated[
        str, typer.Option(help="Solo P1: validar y registrar champion local; true o false.")
    ] = "false",
) -> None:
    """Entrena o evalua la pregunta indicada y registra los runs en MLflow."""
    question = question.strip().lower()
    if question not in QUESTIONS:
        raise typer.BadParameter(f"--question debe ser uno de: {', '.join(QUESTIONS)}.")
    if register_best.lower() not in {"true", "false"}:
        raise typer.BadParameter("--register-best debe ser true o false.")
    mlflow_on = _as_bool(with_mlflow)
    logger.info("Pregunta {} | experimento {}", question.upper(), config.EXPERIMENT_NAME)

    if question == "p1":
        from mibici_ml.modeling.search_p1 import run_search

        run_search(
            output_dir=Path(output_dir) if output_dir else None,
            model=model.strip(),
            tuned_json=tuned_json,
            register_best=_as_bool(register_best),
            with_mlflow=mlflow_on,
        )
        return

    if _as_bool(register_best):
        raise typer.BadParameter("--register-best en train solo aplica a P1.")
    _release_wrapper_run()
    run_output = Path(output_dir) if output_dir else config.MLFLOW_TRAIN_OUTPUT_DIR

    if question == "p2":
        model = model.strip()
        if model not in P2_MODELS:
            raise typer.BadParameter(f"--model debe ser uno de: {', '.join(P2_MODELS)}.")
        risk.run_command(
            output_dir=run_output,
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
        output_dir=run_output,
        final_k=int(k_text),
        with_mlflow=mlflow_on,
    )


if __name__ == "__main__":
    app()
