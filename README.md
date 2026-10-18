# MiBici: demanda, riesgo de estación y perfiles de uso (TC5061, Equipo 49)

Proyecto de la materia **TC5061 · Construye | Proyecto MiBici**, Etapa 1: propuesta y base reproducible.
Sistema de bicicletas públicas del Área Metropolitana de Guadalajara (MiBici).

> **Estado (18 de octubre de 2026):** P1 tiene ingesta, variables, dos baselines y evidencia en MLflow.
> La integración de las tres preguntas y los apartados marcados como **Pendiente** siguen en curso.

## Equipo

- Daniel Ruán Aguilar
- Saúl Salgado Ortega — P1: ingesta, variables y baseline de demanda.
- Diego Sánchez Hernández

## Problema y propuesta

Tres problemas sobre los viajes de MiBici, todos con una partición temporal (sin barajar):

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

**P1 implementado** (pasos 10, 11 y 6); la integración de P2/P3 sigue pendiente. El pipeline contempla:

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
- **Evaluación:** comparar `k=3..8` con inertia, silhouette, Calinski-Harabasz,
  Davies-Bouldin y tamaños de cluster. La herramienta no selecciona `final_k`.

Evaluar candidatos sin MLflow:

```bash
python -m mibici_ml.modeling.cluster evaluate --input-path reports/tables/estaciones_resumen.csv --output-dir /tmp/mibici-p3-evaluation --k-min 3 --k-max 8 --random-seed 42
```

Después de revisar la tabla de métricas y decidir `FINAL_K`, ejecutar el baseline y registrar MLflow:

```bash
MLFLOW_ALLOW_FILE_STORE=true python -m mibici_ml.modeling.cluster run --input-path reports/tables/estaciones_resumen.csv --output-dir /tmp/mibici-p3-baseline-artifacts --k-min 3 --k-max 8 --final-k FINAL_K --random-seed 42 --with-mlflow
```

`MLFLOW_ALLOW_FILE_STORE=true` habilita el backend local `file://` configurado en este proyecto con
MLflow 3.17.1; solo aplica al proceso de ese comando. La corrida usa el experimento `mibici-equipo49`.
No ejecutes el segundo comando hasta que el equipo acuerde `FINAL_K`.

El baseline produce `p3_metricas_por_k.csv`, `p3_comparacion_k.png`, `p3_asignaciones_cluster.csv`,
`p3_perfiles_cluster.csv`, `p3_perfiles_cluster.png`, `p3_mapa_clusters.png` y
`p3_calidad_datos.csv`. El alcance es exploratorio: no hay perfiles horarios por estación, holdout
temporal ni validación de producción. Deberá integrarse posteriormente con la ingesta reproducible.

## Experimentos (MLflow)

P1 ejecutado; integración conjunta pendiente (paso 6). Configuración base en `mibici_ml/config.py`:

- Nombre del experimento: `mibici-equipo49`.
- Seguimiento local en `mlruns/` (no versionado).
- P1 registra parámetros, métricas `val_*`/`test_*`, etiquetas, artefactos y el commit de Git.

P1 crea `P1-estacional-168h` y `P1-media-4semanas`; IDs en `reports/tables/p1_runs.json`.
Se registra `data_version`, semilla, horizonte y versión del código. El snapshot
`reproducibility/code_snapshot.zip` y `code_provenance.json` conservan módulos y dependencias ejecutados.
El ZIP complementa el repositorio: para restaurarlo, primero clonar y hacer checkout del commit indicado,
luego extraer el snapshot sobre ese checkout e instalar el lock. No contiene el repositorio completo.
Las primeras corridas locales declaran commit base y `git_dirty=true`; no se atribuyen a ese commit
los cambios pendientes. La evidencia final debe corresponder al commit que incluya P1.

Para abrir MLflow desde otra terminal PowerShell, en la raíz del repositorio:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
$trackingUri = ([System.Uri]((Resolve-Path .\mlruns).Path)).AbsoluteUri
.\.venv\Scripts\mlflow.exe ui --backend-store-uri $trackingUri --host 127.0.0.1 --port 5000
```

Abrir [MLflow local](http://127.0.0.1:5000). El comando P1 habilita el backend `file://` automáticamente.

## Flujo de trabajo (GitHub Flow)

**Pendiente** de evidencia (paso 9). Reglas acordadas:

- Ramas cortas desde `main`, un tema por rama.
- Commits con [Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `docs`, `build`, `chore`).
- Cambios a `main` solo por Pull Request, con revisión de un integrante.
- Plantilla de PR en `.github/pull_request_template.md`.

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
Evidencia P1 (la integración conjunta continúa pendiente):

- [Lista de runs](reports/figures/P1_mlflow_runs.png).
- [Detalle y etiquetas](reports/figures/P1_mlflow_detail.png).
- [Métricas y parámetros](reports/figures/P1_mlflow_metrics.png).
- [Artefacto](reports/figures/P1_mlflow_artifact.png).
- [Comparación](reports/figures/P1_mlflow_comparison.png).

## Estado de la actividad

| Paso | Descripción | Estado |
|---|---|---|
| 1 | Formulación (ML Canvas) | Hecho (`references/`) |
| 2 | Datos y EDA | Hecho (notebook y tablas agregadas) |
| 3 | Pipeline | P1 implementado; integración de P2/P3 pendiente |
| 4 | Modelos y métricas | Hecho en la propuesta |
| 5 | Arquitectura | Hecho (`references/arquitectura_mibici_equipo49.png`) |
| 6 | Experimentos en MLflow | Dos runs P1; integración conjunta pendiente |
| 7 | Capturas de MLflow | P1 en reports/figures/; evidencia conjunta pendiente |
| 8 | Estructura del repositorio (ccds) | Hecho |
| 9 | Flujo Git y evidencia de PRs | Pendiente |
| 10 | Ingesta (`dataset.py`) | Ejecutada: 12 meses, catálogo y manifiesto |
| 11 | Features (`features.py`) | P1 implementado y probado; integración pendiente |
| 12 | README final | P1 documentado; cierre conjunto pendiente |

## Licencia

Código bajo licencia [MIT](LICENSE). Los datos son del portal de MiBici; su uso es académico y se atribuye a MiBici.
