# MiBici: demanda, riesgo de estación y perfiles de uso (TC5061, Equipo 49)

Proyecto de la materia **TC5061 · Construye | Proyecto MiBici**, Etapa 1: propuesta y base reproducible.
Sistema de bicicletas públicas del Área Metropolitana de Guadalajara (MiBici).

> **Estado (18 de octubre de 2026):** `main` contiene P1, P2 y P3 integradas sobre la misma ingesta
> (`dataset.py`) y el mismo manifiesto (`data_version`). P1 y P2 tienen runs en MLflow; P3 tiene su baseline
> exploratorio. Ver la tabla de [estado de la actividad](#estado-de-la-actividad).

## Equipo

- Daniel Ruán Aguilar
- Saúl Salgado Ortega — P1: ingesta, variables y baseline de demanda.
- Diego Sánchez Hernández

## Problema y propuesta

Tres problemas sobre los viajes de MiBici. P1 usa una partición temporal (sin barajar);
el alcance exploratorio de P3 se describe más adelante:

| Problema | Tipo | Qué predice o describe |
|---|---|---|
| P1 | Regresión | Salidas por estación-hora (demanda) |
| P2 | Clasificación binaria | Riesgo de que una estación quede vacía o llena en las siguientes 2 horas |
| P3 | Clustering | Grupos de estaciones por patrón de uso |

Decisiones y su justificación: ver el documento de la Etapa 1 (`references/`) y el ML Canvas.

## Datos

- **Fuente:** [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/) (CSV mensuales) y el
  catálogo de estaciones (nomenclatura).
- **Ventana:** octubre de 2025 a septiembre de 2026 (12 meses, 4,497,877 viajes antes de limpieza).
- **Partición temporal:** entrenamiento oct-25 a jun-26, validación jul-ago-26, prueba sep-26.
- **Sin datos en el repositorio:** `data/` está en `.gitignore`. La ingesta reproducible está implementada en
  `mibici_ml/dataset.py`; descarga y prepara los 12 meses y la nomenclatura sin pasos manuales.
- **Preparación P1:** 4,493,328 viajes limpios; 471 IDs de origen observados y 484 estaciones en el catálogo.
  Son universos distintos: no se eliminan viajes por faltar su estación en la nomenclatura actual.
- **Versionado:** `references/data_manifest.json` registra URLs, conteos y SHA-256 de CSV y Parquet.
  La versión ejecutada es `0f000bd5e6563196d7c1133e119024e4e864f6325752e4f6b48e40011895d1a3`.
- **Privacidad P1:** los Parquet conservan solo viaje, fechas y estaciones; los campos del usuario se excluyen.
  Datos crudos, modelos, `.env`, `mlruns/` y `mlartifacts/` permanecen fuera de Git.
- **EDA inicial:** [`notebooks/1.0-dsh-eda-inicial.ipynb`](notebooks/1.0-dsh-eda-inicial.ipynb). Solo muestra
  agregados (sin datos a nivel viaje ni usuario).

## Estructura del proyecto

```
├── data/           # raw, interim, processed, external (no versionados)
├── docs/           # documentación adicional
├── models/         # modelos serializados (no versionados)
├── notebooks/      # EDA inicial y 1.0-sso-eda-p1.ipynb
├── references/     # ML Canvas, arquitectura, propuesta P1 y manifiesto
├── reports/        # figures/ y tables/ agregadas del EDA y P1
├── mibici_ml/      # código del proyecto
│   ├── config.py   # rutas, ventana temporal, semillas y parámetros de P1-P3
│   ├── dataset.py  # ingesta, limpieza, Parquet y manifiesto
│   ├── features.py # panel estación-hora y variables históricas P1
│   ├── plots.py
│   └── modeling/   # demand.py (P1), cluster.py (P3) y scaffold compartido
└── tests/          # pruebas (pytest)
```

Estructura generada con [cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science)
(`ccds`) y ajustada para el proyecto.

## Cómo reproducir el entorno

Requiere Python 3.12. Las dependencias están fijadas en `requirements.txt`.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "from mibici_ml.config import START_MONTH, END_MONTH; print(START_MONTH, END_MONTH)"
pytest
ruff check .
```

Para **P1 en Windows**, usar Python 3.12.14 y el lock validado; `requirements.txt` original se conserva.
El lock es específico de Windows: elegir un archivo de dependencias por entorno. Desde la raíz:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-p1.lock.txt
.\.venv\Scripts\python.exe -m pip check
```

En VS Code, seleccionar `.venv\Scripts\python.exe` como intérprete y kernel del notebook.
Verificar cambios con `python -m pytest -q` y `python -m ruff check .` usando ese entorno.

Los parámetros (rutas, fechas, semilla, nombre del experimento) viven en `mibici_ml/config.py`. Los valores
marcados como **PROVISIONAL** se ajustarán con los resultados de MLflow.

## Pipeline y modelos

Pipeline integrado sobre una sola ingesta (`dataset.py`). El pipeline contempla:

- Ingesta reproducible y manifiesto con hash SHA-256 por archivo.
- Panel estación-hora y features sin fuga del futuro (rezagos de 24 h o más en P1).
- Baselines, modelos candidatos y métricas de prueba por problema.

### P1: salidas por estación-hora

A las 00:00 se emiten 24 predicciones por estación para el día que empieza (hora local de Guadalajara).
El operador usaría el perfil para planear la distribución de bicicletas. El objetivo cuenta salidas
observadas; no mide demanda insatisfecha ni calcula rutas o inventario óptimo.

```powershell
.\.venv\Scripts\python.exe -m mibici_ml.dataset
.\.venv\Scripts\python.exe -m mibici_ml.modeling.demand
```

La ingesta normaliza codificación, columnas y fechas, elimina duplicados exactos y registros inválidos,
y guarda Parquet por mes. Añadir `--source-dir "$env:USERPROFILE\Downloads"` al comando de ingesta
reutiliza CSV locales y descarga faltantes. Los hashes esperados detectan cambios en las fuentes;
`--refresh` acepta una versión nueva y no debe usarse para reproducir esta ejecución.

`features.py` agrega el panel regular y construye calendario, rezagos de 24, 48 y 168 h y media semanal.
Una estación se considera conocida desde el día posterior a su primera salida; antes no se imputan ceros.
Una fuente mensual ausente produce error. Se comparan `seasonal_168h` y `mean_4weeks` sobre las mismas
filas con cuatro semanas disponibles. La elección usa WAPE de julio-agosto; septiembre evalúa reglas
ya fijadas. El backtest diario usa días anteriores ya observados, nunca salidas del día pronosticado.
El módulo rechaza horizontes distintos de 24 h, emisiones distintas de las 00:00 y rezagos menores de 24 h.

MAE mide viajes de error por estación-hora; RMSE penaliza errores grandes; WAPE divide error absoluto
total entre viajes observados; el sesgo identifica sobre/subestimación. WAPE se registra como razón,
no como porcentaje de aciertos. PoissonRegressor y HistGradientBoostingRegressor son candidatos
propuestos para la siguiente iteración; no se les atribuyen resultados de estos baselines.

El catálogo actual no demuestra aperturas o cierres históricos. La cohorte excluye estaciones sin
cuatro semanas de historia y reporta cobertura. Los CSV mensuales permiten evaluación retrospectiva;
una operación diaria requeriría una fuente actualizada. Formulación, condiciones de uso y resultados:
[`P1_propuesta_y_resultados.md`](references/P1_propuesta_y_resultados.md).
Exploración: [`1.0-sso-eda-p1.ipynb`](notebooks/1.0-sso-eda-p1.ipynb).
El documento del equipo actualizado está en [`TC5061_Propuesta_Proyecto_Equipo49.docx`](references/TC5061_Propuesta_Proyecto_Equipo49.docx).

### Baseline P3 exploratorio

Este baseline segmenta estaciones por **volumen agregado de salidas/llegadas y capacidad**. No estima
patrones horarios por estación y no sustituye el pipeline reproducible desde datos crudos.

- **Fuente:** `reports/tables/estaciones_resumen.csv`, agregado de los 12 meses completos.
- **Features:** `log1p(salidas_12m)`, `log1p(llegadas_12m)` y `capacidad_modelo`, escaladas con
  `StandardScaler` antes de `KMeans`. `est`, `nombre`, `lat`, `lon` y `capacidad_imputada` no entran
  al modelo; coordenadas solo sirven para el mapa.
- **Capacidad cero:** cuatro estaciones tienen `capacidad == 0` en la tabla agregada. Para este
  baseline, cero es un valor no utilizable para modelado, no una capacidad física real confirmada
  por MiBici. Se conserva `capacidad` original y se calcula `capacidad_modelo` sustituyendo los
  ceros por la mediana de capacidades estrictamente positivas del archivo leído. La columna
  `capacidad_imputada` identifica esas filas. Es una decisión metodológica provisional; la ingesta
  reproducible deberá investigar la causa de esos ceros.
- **Evaluación histórica:** comparar `k=3..8` con inertia, silhouette, Calinski-Harabasz,
  Davies-Bouldin y tamaños de cluster. `cluster evaluate` no selecciona `final_k`.

Evaluar candidatos sin MLflow:

```bash
python -m mibici_ml.modeling.cluster evaluate --input-path reports/tables/estaciones_resumen.csv --output-dir /tmp/mibici-p3-evaluation --k-min 3 --k-max 8 --random-seed 42
```

Después de revisar la tabla de métricas y decidir `FINAL_K`, ejecutar el baseline y registrar MLflow:

```bash
python -m mibici_ml.modeling.cluster run --input-path reports/tables/estaciones_resumen.csv --output-dir /tmp/mibici-p3-baseline-artifacts --k-min 3 --k-max 8 --final-k FINAL_K --random-seed 42 --with-mlflow
```

El registro usa el backend SQLite `mlflow.db` definido en `mibici_ml/config.py` (no requiere variables de entorno). La corrida usa el experimento `mibici-equipo49`.
No ejecutes el segundo comando hasta que el equipo acuerde `FINAL_K`.

El baseline produce `p3_metricas_por_k.csv`, `p3_comparacion_k.png`, `p3_asignaciones_cluster.csv`,
`p3_perfiles_cluster.csv`, `p3_perfiles_cluster.png`, `p3_mapa_clusters.png` y
`p3_calidad_datos.csv`. El alcance es exploratorio: no hay perfiles horarios por estación, holdout
temporal ni validación de producción. Deberá integrarse posteriormente con la ingesta reproducible.

### Tracking avanzado y Registry P3 (pendiente de ejecución manual)

El flujo `search_p3` mantiene **k=5 como baseline histórico** y compara los candidatos **k=3, 4, 6, 7 y 8**.
Todos usan el mismo CSV, SHA-256, tres features, mediana positiva para capacidad cero, `StandardScaler`,
semilla base y `n_init=10`. Crea un run padre coordinador `P3-kmeans-nested-search` y seis hijos nested:
`P3-kmeans-k5-baseline` más `P3-kmeans-k<K>-candidate` para los cinco candidatos. Baseline e hijos
comparten parámetros, `log_input`, firma, artefactos y métricas sin prefijo; el padre registra la
comparación y el resultado de selección. Los hijos comparten los tags `team`, `question`, `model_family`,
`stage_of_work`, `git_commit`, `parent_search`, `data_scope`, `temporal_holdout`, `input_sha256` y
`data_version`; `candidate_k` identifica candidatos y `baseline_k` identifica el baseline.

La selección de recomendación compara el baseline k=5 y los cinco candidatos; exige al menos 2 estaciones
por cluster y ordena por silhouette descendente, Davies-Bouldin ascendente y menor k. El resultado separa
el mejor candidato, la recomendación global y la selección efectiva. Antes de registrar, Daniel debe revisar
la comparación y decidir explícitamente `selected_k`; el valor de la opción manual prevalece sobre la
recomendación. La recomendación no asigna alias automáticamente. `reports/tables/estaciones_resumen.csv` tiene
versión propia `sha256:<hash completo del CSV>`; no se presenta el manifiesto P1/P2 como versión directa.
El manifiesto P3 incluye conteo, columnas, periodo, hash y limitaciones, nunca filas de estaciones.
`mlflow.log_input` registra solo las tres features crudas (sin nombre, coordenadas ni IDs). MLflow limita
el digest del dataset a 36 caracteres, por lo que se usa el prefijo de ese SHA-256; el hash completo queda
en el manifiesto, tags y parámetros.

Cada hijo guarda bajo `plots/` mapa y perfiles agregados; bajo `data/`, manifiesto, métricas, perfiles
y calidad; bajo `config/`, configuración reproducible; y bajo `model`, un pyfunc firmado. El contrato
de inferencia recibe únicamente `capacidad`, `salidas_12m` y `llegadas_12m`; el modelo valida valores,
imputa capacidad cero usando la mediana aprendida y retorna el cluster. El input example es sintético
y contiene las mismas tres columnas. El CSV fuente completo y las asignaciones por estación no se
registran como artefactos.

Desde la raíz y con el servidor de tracking disponible, Daniel ejecutará manualmente (aún pendiente;
estos comandos no fueron ejecutados al preparar el código):

```bash
# Comparar k=5 con los cinco candidatos; no registra ni asigna alias.
mlflow run . -e search_p3

# Solo después de revisar la comparación y decidir K, repetir la búsqueda para
# registrar esa selección, validar su inferencia y asignar champion.
# register_best=true registra la versión, valida su inferencia y solo entonces asigna champion.
mlflow run . -e search_p3 -P register_best=true -P selected_k=5
```

En el segundo comando, sustituir `5` por el K elegido tras revisar la primera búsqueda. El código
rechaza `register_best=true` si no se especifica explícitamente `selected_k`.

El nombre fijo es `mibici-equipo49-p3-grupos`; el alias de producción es `champion`.
El flujo asigna únicamente `champion`. No crea ni asigna `challenger`; si el equipo lo requiere, puede
asignarse manualmente de forma deliberada con
`assign_champion_alias(champion_alias="challenger", version="VERSION")`, sustituyendo `VERSION`
por la versión revisada.
Para cargar y probar el alias
después de una ejecución real:

```python
from mibici_ml.modeling.p3_model import p3_input_example
from mibici_ml.modeling.registry_p3 import validate_champion_inference

predictions = validate_champion_inference(p3_input_example(), selected_k=5)
```

Sustituir `5` por el K que fue registrado como champion.

El código carga la versión por URI, prueba predicción finita con etiquetas enteras en el rango esperado
y solo después mueve el alias. Para rollback, reasignar `champion` a la versión anterior con
`assign_champion_alias(version="VERSION_ANTERIOR")`, sustituyendo el texto por la versión concreta;
no se borran versiones. No hay holdout temporal,
los agregados no son perfiles horarios y la imputación de capacidad cero sigue siendo provisional.
Los runs, versiones, métricas y alias reales quedan pendientes de la ejecución y revisión de Daniel.

### Modelo P2 de riesgo de estación

Predice si una estación quedará **vacía o llena** en las siguientes 2 horas (`P2_HORIZON_HOURS`).
Se construye desde los viajes crudos de los 12 meses, sin inventario real por estación.

- **Etiqueta:** el inventario estimado `s0 + F(t)` toca 0 o `C` dentro de las siguientes `H` horas del
  mismo día. `s0 = (L + U) / 2` es el punto medio del rango factible y solo sirve para construir la
  etiqueta. Un estación-día infactible (`L > U`, con solo viajes no hay inventario consistente) se
  excluye de entrenamiento, validación y prueba.
- **Features:** solo información hasta `t`: hora, día de la semana, fin de semana, capacidad, flujo
  acumulado y su proporción de la capacidad, llegadas y salidas de 1 y 2 horas, y salidas de la misma
  hora del día anterior. Ninguna mira al futuro.
- **Partición temporal:** entrenamiento oct-25 a jun-26, validación jul-ago-26, prueba sep-26 (se abre
  una sola vez, con el modelo elegido en validación). Sin partición aleatoria.
- **Modelos:** mayoritaria, persistencia (`r(s, t-H)`), regresión logística balanceada y
  `HistGradientBoostingClassifier`. Métrica principal: PR-AUC; además precision@k, recall con precisión
  ≥ 0.5, F1 con umbral de validación, ROC-AUC y Brier.

Comandos (desde la raíz del proyecto, después de la ingesta de `dataset.py`):

```bash
# 1. Compara los 4 modelos solo en validación (no abre la prueba)
python -m mibici_ml.modeling.risk evaluate --output-dir /tmp/mibici-p2-evaluate

# 2. Registra 4 runs de validación y 1 run final con prueba del modelo elegido
python -m mibici_ml.modeling.risk run --output-dir /tmp/mibici-p2-run \
  --final-model hist_gb --with-mlflow
```

El modelo final se elige en validación (PR-AUC). La prueba se usa una sola vez, para el modelo elegido.

Resultados (run final `P2-hist_gb-final`, commit `b8651e1`, `data_version` `0f000bd5…`, `git_dirty=false`):

| Modelo | PR-AUC validación | ROC-AUC validación |
|---|---:|---:|
| Persistencia `r(s, t-H)` | **0.1228** | 0.6654 |
| HistGradientBoosting (final) | 0.0624 | **0.8701** |
| Regresión logística | 0.0123 | 0.7722 |
| Mayoritaria | 0.0038 | 0.5000 |

El modelo final `hist_gb` obtiene en **prueba** PR-AUC 0.0733, ROC-AUC 0.8561, Brier 0.0051 y precisión en el
top 10 % por hora de 0.0279. **Lectura honesta:** en PR-AUC de validación, la persistencia supera a los modelos
aprendidos; el modelo se eligió con la regla de selección pre-declarada (mejor modelo aprendido en PR-AUC), no
porque supere a la persistencia. Las cifras completas están en `reports/tables/p2_metrics.csv`.

Limitaciones: el inventario es una aproximación desde viajes (no hay rebalanceo ni bicicletas no registradas);
la clase positiva es rara (0.46 % de las filas del panel), lo que limita la precisión absoluta; y las 4 estaciones
con capacidad 0 se excluyen del modelado. El porcentaje de estación-día no factibles es 7.35 % sobre la malla
completa (estación × día, 175,680 filas) y 9.07 % sobre solo los estación-día con actividad (142,406); el
denominador distinto explica la diferencia.

Limitación de reproducibilidad: el hash de `stations.parquet` generado localmente (`d58c6e3e…`) no coincide con el
del manifiesto publicado (`de449228…`), aunque los 12 Parquet de viajes sí coinciden. P2 verifica los hashes de los
viajes y registra la diferencia en el artefacto `reproducibility/data_manifest.json` de cada run, en lugar de
detener la ejecución, porque su única dependencia del catálogo es la capacidad por estación.

**Qué esperar al reproducir la ingesta en otra máquina.** Con `--source-dir` y los mismos CSV, los 12 Parquet de
viajes reproducen byte a byte el manifiesto. Pero `data_version` también depende del hash de `stations.parquet`, que
en una prueba en clon limpio salió distinto (`d58c6e3e…` frente a `de449228…`). Por eso la ingesta puede imprimir una
versión distinta de `0f000bd5…`. Eso no indica un error de los viajes: es la misma limitación de arriba. No use
`--refresh` para "arreglarlo", porque reescribiría el manifiesto versionado; la versión de referencia es la del
manifiesto en Git.

## Proyecto MLflow

El proyecto es un [MLflow Project](https://mlflow.org/docs/latest/ml/projects/) (`MLproject`)
con entorno en `python_env.yaml` (Python 3.12, dependencias fijas de `requirements.txt`). Los
valores por defecto (rutas, ventana, semilla, experimento y URI de tracking) están en
`mibici_ml/config.py`; el `MLproject` no los repite.

### 1. Servidor de tracking (SQLite)

El backend es `sqlite:///mlflow.db` en la raíz del repositorio (ignorado por Git). Desde la raíz,
con el entorno del proyecto activo:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

En PowerShell (la ruta debe ser absoluta para que `mlflow run` y el servidor usen el mismo archivo):

```powershell
$db = (Resolve-Path .\mlflow.db).Path -replace '\\', '/'
mlflow server --backend-store-uri "sqlite:///$db" --host 127.0.0.1 --port 5000
```

Abrir [http://127.0.0.1:5000](http://127.0.0.1:5000). Para que `mlflow run` escriba en el mismo
servidor, exportar `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` antes de ejecutar los comandos
(si no se exporta, escribe directamente en `mlflow.db` mediante `config.py`).

### 2. Ejecutar los entry points

Desde la raíz del repositorio:

```bash
# Ingesta reproducible (descarga o reutiliza los CSV, escribe Parquet y el manifiesto)
mlflow run . -e ingest

# Entrenamiento por pregunta: question = p1 | p2 | p3
mlflow run . -e train -P question=p2 -P model=hist_gb
mlflow run . -e train -P question=p1
mlflow run . -e train -P question=p3 -P final_k=FINAL_K

# Busqueda anidada de hiperparametros de P2 (run padre + un run hijo por combinacion)
mlflow run . -e search

# Busqueda anidada P3; register_best es false por defecto
mlflow run . -e search_p3
```

Los parámetros vacíos o no declarados usan el valor de `config.py`. `search` y `train` de P1/P2 usan
la partición de validación para elegir; P3 es clustering exploratorio, sin holdout temporal.

### 3. Esquema del Model Registry

Cada modelo final se registra con el nombre `mibici-equipo49-<pregunta>-<tema>` y se asigna un
**alias** (no se usan las etapas legacy `Staging`/`Production`):

| Modelo registrado | Pregunta | Estado | Alias de producción | Alias de comparación |
|---|---|---|---|---|
| `mibici-equipo49-p2-riesgo-hist_gb` | P2 | Registrado por `train -P question=p2` con `--final-model hist_gb` | `champion` | `challenger` (opcional) |
| `mibici-equipo49-p2-riesgo-logistica` | P2 | Registrado si se entrena con `--final-model logistica` | `champion` si gana en validación | `challenger` |
| `mibici-equipo49-p1-demanda` | P1 | Pendiente: P1 es solo baselines fijados, sin modelo registrado | n/a | n/a |
| `mibici-equipo49-p3-grupos` | P3 | Código listo; registro real pendiente de ejecución manual | `champion` (pendiente) | `challenger` (opcional) |

Asignar el alias después de revisar la validación (ejemplo P2):

```python
from mlflow import MlflowClient
client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
client.set_registered_model_alias("mibici-equipo49-p2-riesgo-hist_gb", "champion", version="1")
```

Cargar el modelo por alias, sin depender de un número de versión:

```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/mibici-equipo49-p2-riesgo-hist_gb@champion")
```

Cada versión conserva su **linaje**: el run de origen (parámetros, métricas, etiquetas
`git_commit`, `git_dirty`, `data_version`) y los artefactos (manifiesto, curva PR, importancia).

## Experimentos (MLflow)

Experimento base integrado (paso 6). Configuración base en `mibici_ml/config.py`:

- Nombre del experimento: `mibici-equipo49`.
- Seguimiento local en SQLite (`mlflow.db`, no versionado). Ver la sección *Proyecto MLflow* más abajo.
- Cada run registra parámetros (ventanas, semilla, horizonte, `data_version`), métricas `val_*` y `test_*`
  (`test_*` solo en el run final), etiquetas (`team`, `question`, `git_commit`, `git_dirty`, `data_version`)
  y artefactos. Ningún dato a nivel de viaje o de usuario se registra.

P1 crea `P1-estacional-168h` y `P1-media-4semanas`; IDs en `reports/tables/p1_runs.json`.
Se registra `data_version`, semilla, horizonte y versión del código. El snapshot
`reproducibility/code_snapshot.zip` y `code_provenance.json` conservan módulos y dependencias ejecutados.
El ZIP incluye también README y licencia, cuyos hashes de contexto se registran por separado de la versión
del código. Para reproducir, clonar el repositorio, hacer checkout del commit registrado e instalar el lock.
Los runs finales corresponden al commit `52af95c13416705906f0630f058a4c78645a2fbe`, con `git_dirty=false`.
Las primeras corridas locales se conservan en MLflow; las capturas adjuntas muestran los runs finales.

Para abrir la interfaz de MLflow sobre el mismo backend, ver *Proyecto MLflow* (servidor en el puerto 5000).

## Flujo de trabajo (GitHub Flow)

Reglas acordadas:

- Ramas cortas desde `main`, un tema por rama.
- Commits con [Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `docs`, `build`, `chore`).
- Cambios a `main` solo por Pull Request, con revisión de un integrante distinto de quien lo abre.
- Plantilla de PR en `.github/pull_request_template.md`.
- La protección de `main` exige una aprobación y conversaciones resueltas. Los administradores no están obligados
  por la regla (decisión provisional, ver el documento de la Etapa 1).

Pull requests del proyecto:

| PR | Rama | Autor (GitHub) | Revisión aprobada por |
|---|---|---|---|
| [#1](https://github.com/dieguini95/TC5061-Proyecto-Equipo49/pull/1) | `feat/p3-clustering-mlflow` | DaniRuan (Daniel Ruán) | dieguini95 (Diego Sánchez) |
| [#2](https://github.com/dieguini95/TC5061-Proyecto-Equipo49/pull/2) | `feat/p2-risk-mlflow` | dieguini95 (Diego Sánchez) | DaniRuan (Daniel Ruán) |
| [#3](https://github.com/dieguini95/TC5061-Proyecto-Equipo49/pull/3) | `Saul` | Drezneck-RM (Saúl Salgado) | dieguini95 (Diego Sánchez) |
| [#4](https://github.com/dieguini95/TC5061-Proyecto-Equipo49/pull/4) | `feat/p2-integracion-cierre` | dieguini95 (Diego Sánchez) | Drezneck-RM (Saúl Salgado) |

Los cuatro PR están fusionados en `main`. Cada integrante abrió al menos uno y revisó al menos uno de otro integrante.

Roles del equipo: Daniel Ruán (P3 y revisión de PRs); Saúl Salgado (P1: ingesta, variables y baseline de demanda);
Diego Sánchez (P2 y cierre del repositorio).

## Resultados

P1 seleccionó la media de cuatro semanas por validación. MAE/RMSE en viajes por estación-hora:

| Baseline | Partición | MAE | RMSE | WAPE |
|---|---|---:|---:|---:|
| Semana anterior | Validación | 0.9287 | 1.8594 | 76.12% |
| Media de cuatro semanas | Validación | **0.8011** | **1.5052** | **65.67%** |
| Semana anterior | Prueba | 1.0297 | 2.1127 | 71.01% |
| Media de cuatro semanas | Prueba | **0.8806** | **1.7412** | **60.73%** |

Prueba: 336,480 estaciones-hora de 470 estaciones; 487,915 salidas y cobertura del 99.30% de horas conocidas.
El sesgo de la media es −0.2236: subestima el volumen. Son referencias iniciales, sin validación operativa.
Tablas agregadas: `reports/tables/p1_metrics.csv`, `p1_station_metrics.csv` y `p1_quality.json`.
Evidencia P1:

- [Lista de runs](reports/figures/P1_mlflow_runs.png).
- [Detalle y etiquetas](reports/figures/P1_mlflow_detail.png).
- [Métricas y parámetros](reports/figures/P1_mlflow_metrics.png).
- [Artefacto](reports/figures/P1_mlflow_artifact.png).
- [Comparación](reports/figures/P1_mlflow_comparison.png).

## Estado de la actividad

| Paso | Descripción | Estado |
|---|---|---|
| 1 | Formulación (ML Canvas) | Hecho (`references/`) |
| 2 | Datos y EDA | Hecho (notebook, tablas agregadas y manifiesto) |
| 3 | Pipeline | Hecho: P1, P2 y P3 (P3 no usa `dataset.py`, ver limitaciones) |
| 4 | Modelos y métricas | Hecho: baselines y candidatos por pregunta |
| 5 | Arquitectura | Hecho (`references/arquitectura_mibici_equipo49.png`) |
| 6 | Experimentos en MLflow | Hecho: runs de P1, P2 y P3 en `mibici-equipo49` |
| 7 | Capturas de MLflow | Hecho para P1, P2 y P3 (las de P1 también en `reports/figures/P1_mlflow_*.png`) |
| 8 | Estructura del repositorio (ccds) | Hecho |
| 9 | Flujo Git y PRs | Hecho: PR #1 a #4 fusionados, cada uno con revisión de otro integrante |
| 10 | Ingesta (`dataset.py`) | Hecho: 12 meses, catálogo, Parquet y manifiesto |
| 11 | Features (`features.py`) | Hecho: P1 y P2 documentados y probados |
| 12 | README final | Hecho en este PR |

Limitaciones abiertas: P3 aún no usa `dataset.py` ni registra `data_version`; la regla de 3 frente a 4 integrantes
se confirma con el profesor; y el hash de `stations.parquet` difiere del manifiesto (ver P2).

## Licencia

Código bajo licencia [MIT](LICENSE). Los datos son del portal de MiBici; su uso es académico y se atribuye a MiBici.
