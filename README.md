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
