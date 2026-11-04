# Guía: agregar la API de una pregunta al servicio de inferencia

Esta guía es para P1 (demanda) y P3 (grupos). La base la aporta P2 y ya funciona para
los tres: solo cambian el contrato de entrada, la salida y el archivo de la app.

## Qué ya existe

| Archivo | Qué hace |
|---|---|
| `mibici_ml/config.py` (sección *Servicio de inferencia*) | Nombres del Registry por pregunta, alias `champion`, raíz de bundles, tamaño máximo de lote. |
| `mibici_ml/serving/export.py` | Exporta el champion a `models/serving/<pregunta>/`. |
| `mibici_ml/serving/bundle.py` | Carga el bundle (`model.skops` + `model_metadata.json`) y puntúa filas. |
| `mibici_ml/serving/service.py` | Fábrica `create_service_app(ServiceSpec)`: `/health`, `/ready`, `/v1/<q>/model`, `/predict`, `/predict/batch`. |
| `mibici_ml/serving/apps/p2.py` | Ejemplo completo a seguir. |
| `Containerfile` | Imagen genérica: `--build-arg QUESTION=<pregunta>`. |

## Pasos

### 1. Exportar el champion

Desde la raíz, con el entorno del proyecto activo:

```bash
make serving-export QUESTION=p1
```

Esto escribe `models/serving/p1/` (ignorado por Git). El exportador:

- localiza el modelo por `mibici-equipo49-p1-demanda@champion`;
- si la ruta guardada en `mlflow.db` ya no existe, usa `mlruns/` del proyecto;
- acepta solo modelos **sklearn serializados con skops** (no pickle);
- compara sus predicciones con las del modelo original de MLflow antes de terminar.

P3 se registra como `pyfunc` personalizado (`P3ClusterModel`). El exportador lo rechaza
a propósito: hay que escribir un adaptador que exponga el mismo contrato.

### 2. Crear `mibici_ml/serving/apps/<pregunta>.py`

Define cuatro cosas: las variables de entrada (con sus rangos), el modelo de salida, la
función que convierte la puntuación en respuesta y el `ServiceSpec`.

```python
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field

from mibici_ml.config import SERVING_ROOT_DIR
from mibici_ml.serving.bundle import ServingBundle
from mibici_ml.serving.service import ServiceSpec, create_service_app

P1_FEATURE_NAMES = (...)  # mismo orden que el modelo; un test debe verificarlo


class RegistroP1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # un campo por variable, con ge/le/gt y allow_inf_nan=False


class PrediccionP1(BaseModel):
    demanda_viajes: float = Field(ge=0)
    modelo: str
    version: str


def _to_output(score: float, bundle: ServingBundle) -> PrediccionP1:
    return PrediccionP1(demanda_viajes=score, modelo=bundle.model_label, version=bundle.version)


P1_SPEC = ServiceSpec(
    question="p1",
    title="MiBici - Demanda por estacion-hora (P1)",
    description="...",
    feature_names=P1_FEATURE_NAMES,
    input_model=RegistroP1,
    output_model=PrediccionP1,
    to_output=_to_output,
    bundle_dir=SERVING_ROOT_DIR / "p1",
)

app = create_service_app(P1_SPEC)
```

Reglas:

- Los nombres de los campos de entrada **deben** coincidir con `feature_names`. Si no
  coinciden, `/ready` responde 503 con un mensaje claro.
- Las variables que no pueden ser negativas van con `ge=0`; las que sí pueden (saldos,
  tendencias) van solo con `allow_inf_nan=False`.
- Las reglas entre campos (p. ej. `fin_de_semana` según `dia_semana`) van en un
  `model_validator(mode="after")` que lance `ValueError`. No usar `__post_init__`
  ni validar dentro del endpoint: un error ahí se convierte en 500, no en 422.
- `score_kind` del bundle decide si se usa `predict_proba` (clasificación) o `predict`
  (regresión). P1 usa `predict`.

### 3. Probar

Copiar `tests/test_serving_p2.py` a `tests/test_serving_p1.py` y cambiar el spec. Como
mínimo: `/ready` 200, predicción válida, cada variable fuera de rango con 422, lote y
503 sin bundle. Ejecutar con `make serving-test`.

### 4. Construir y ejecutar el contenedor

```bash
make serving-build QUESTION=p1 ENGINE=podman   # o ENGINE=docker
make serving-run   QUESTION=p1 ENGINE=podman PORT=8001
curl -s http://127.0.0.1:8001/health
```

`PORT` evita chocar con otra pregunta corriendo en 8000.

### 5. Documentar

Agregar en el README una subsección con el comando de exportación, el de construcción,
un ejemplo real de petición y respuesta, y el umbral o la interpretación de la salida.

## Lo que no hacer

- No copiar `.env`, `mlruns/`, `mlflow.db` ni datos a la imagen (ya lo excluye `.dockerignore`).
- No cargar modelos con `pickle` ni `cloudpickle` en el servicio: el bundle es solo skops.
- No cambiar `P2_FEATURE_NAMES` sin reexportar el modelo: la validación de contrato lo detecta.
- No hardcodear rutas del host ni la URI de MLflow dentro del contenedor.
