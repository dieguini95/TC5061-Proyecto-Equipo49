# MiBici: demanda, riesgo de estación y perfiles de uso (TC5061, Equipo 49)

Proyecto de la materia **TC5061 · Construye | Proyecto MiBici**, Etapa 1: propuesta y base reproducible.
Sistema de bicicletas públicas del Área Metropolitana de Guadalajara (MiBici).

> **Estado:** Etapa 1 en curso. Este README es un esqueleto: los apartados marcados como
> **Pendiente** se completan en las siguientes etapas.

## Equipo

- Daniel Ruán Aguilar
- Saúl Salgado Ortega
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
- **Sin datos en el repositorio:** `data/` está en `.gitignore`. La ingesta reproducible se implementará en
  `mibici_ml/dataset.py` (**Pendiente**, paso 10).
- **EDA inicial:** [`notebooks/1.0-dsh-eda-inicial.ipynb`](notebooks/1.0-dsh-eda-inicial.ipynb). Solo muestra
  agregados (sin datos a nivel viaje ni usuario).

## Estructura del proyecto

```
├── data/           # raw, interim, processed, external (no versionados)
├── docs/           # documentación adicional
├── models/         # modelos serializados (no versionados)
├── notebooks/      # análisis exploratorio (1.0-dsh-eda-inicial.ipynb)
├── references/     # ML Canvas (DOCX y PDF) y diagrama de arquitectura
├── reports/        # figures/ y tables/ agregadas del EDA
├── mibici_ml/      # código del proyecto
│   ├── config.py   # rutas, ventana temporal, semillas y parámetros de P1-P3
│   ├── dataset.py  # ingesta (Pendiente)
│   ├── features.py # features (Pendiente)
│   ├── plots.py
│   └── modeling/   # entrenamiento y predicción (Pendiente)
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

Los parámetros (rutas, fechas, semilla, nombre del experimento) viven en `mibici_ml/config.py`. Los valores
marcados como **PROVISIONAL** se ajustarán con los resultados de MLflow.

## Pipeline y modelos

**Pendiente** (pasos 10, 11 y 5/6 de la actividad). Se documentará:

- Ingesta reproducible y manifiesto con hash SHA-256 por archivo.
- Panel estación-hora y features sin fuga del futuro (rezagos de 24 h o más en P1).
- Baselines, modelos candidatos y métricas de prueba por problema.

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

Comandos (desde la raíz del proyecto, con los CSV en `01_insumos/datos/`):

```bash
# 1. Construye el panel estación-hora (cache en data/interim/)
python -m mibici_ml.modeling.risk build-panel --raw-dir ../../01_insumos/datos --catalog-path ../../01_insumos/datos/nomenclatura_2026_09.csv

# 2. Compara los 4 modelos solo en validación (no abre la prueba)
python -m mibici_ml.modeling.risk evaluate --raw-dir ../../01_insumos/datos --catalog-path ../../01_insumos/datos/nomenclatura_2026_09.csv --output-dir /tmp/mibici-p2-evaluation

# 3. Registra 4 runs de validación y 1 run final con prueba del modelo elegido
MLFLOW_ALLOW_FILE_STORE=true python -m mibici_ml.modeling.risk run --raw-dir ../../01_insumos/datos --catalog-path ../../01_insumos/datos/nomenclatura_2026_09.csv --output-dir /tmp/mibici-p2-artifacts --final-model hist_gb --with-mlflow
```

Limitaciones: el inventario es una aproximación desde viajes (no hay rebalanceo ni bicicletas no
registradas); la clase positiva es rara (~0.45 % de las filas del panel), lo que limita la precisión
absoluta; y 4 estaciones con capacidad 0 se excluyen del modelado.

Cifras del panel de 12 meses (`data/interim/p2_calidad_datos.csv`): el porcentaje de estación-día no
factibles es 7.35 % sobre la malla completa (estación × día, 175,680). El EDA reporta 9.07 % sobre
solo los estación-día con actividad (142,406); el denominador distinto explica la diferencia.

## Experimentos (MLflow)

**Pendiente** (paso 6). Configuración base en `mibici_ml/config.py`:

- Nombre del experimento: `mibici-equipo49`.
- Seguimiento local en `mlruns/` (no versionado).
- Cada corrida registrará parámetros, métricas, etiquetas, artefactos y el commit de Git.

## Flujo de trabajo (GitHub Flow)

**Pendiente** de evidencia (paso 9). Reglas acordadas:

- Ramas cortas desde `main`, un tema por rama.
- Commits con [Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `docs`, `build`, `chore`).
- Cambios a `main` solo por Pull Request, con revisión de un integrante.
- Plantilla de PR en `.github/pull_request_template.md`.

## Resultados

**Pendiente** (pasos 6 y 7): tabla comparativa de modelos, métricas de prueba y capturas de MLflow.

## Estado de la actividad

| Paso | Descripción | Estado |
|---|---|---|
| 1 | Formulación (ML Canvas) | Hecho (`references/`) |
| 2 | Datos y EDA | Hecho (notebook y tablas agregadas) |
| 3 | Pipeline | Hecho en la propuesta, pendiente de implementar |
| 4 | Modelos y métricas | Hecho en la propuesta |
| 5 | Arquitectura | Hecho (`references/arquitectura_mibici_equipo49.png`) |
| 6 | Experimentos en MLflow | Pendiente |
| 7 | Capturas de MLflow | Pendiente |
| 8 | Estructura del repositorio (ccds) | Hecho |
| 9 | Flujo Git y evidencia de PRs | Pendiente |
| 10 | Ingesta (`dataset.py`) | Pendiente |
| 11 | Features (`features.py`) | Pendiente |
| 12 | README final | Pendiente (esqueleto) |

## Licencia

Código bajo licencia [MIT](LICENSE). Los datos son del portal de MiBici; su uso es académico y se atribuye a MiBici.
