# Actividad 9: operación, carga y FinOps (evidencias)

Esta carpeta guarda las evidencias de la operación del stack local con Docker Compose: reportes HTML de Locust y las gráficas usadas en el documento final. Las tablas de datos crudos y procesados están en [`reports/tables/`](../../../reports/tables/) con prefijo `a9_`.

## Entorno de medición

| Dato | Valor (de `reports/tables/a9_entorno.json`) |
|---|---|
| Equipo | Apple M3 Pro (arm64), macOS 26.7.1 |
| Docker | Docker Desktop, servidor 29.8.0, Compose 5.5.1 |
| VM de Docker | 11 CPU, 8.3 GB de memoria |
| Límites de contenedor | `mibici-api` 2 vCPU y 1 GiB (S5: 1 vCPU); `mibici-mlflow` 1 vCPU y 1 GiB |
| Generador de carga | Locust 2.47.0 y Python 3.12.13, en el host (no en el contenedor) |
| Semilla de payloads | 42 |
| Commit base | `a843407` (merge del PR #22) |

## Metodología

- **Escenarios** (`scripts/run_a9_scenarios.py`): S1 (10 usuarios, mezcla operativa, 60 s), S2 (40 usuarios, 90 s), S3 (80 usuarios, 90 s, estrés), S4a (20 usuarios, P2 unitario sin espera, 60 s), S4b (20 usuarios, P2 por lotes de 20 estaciones sin espera, 60 s). S5 (1 vCPU con lote) es la réplica de right-sizing y requiere reiniciar la API con `API_CPU_LIMIT=1.0`.
- **Ventana de medición:** Locust corre con `--reset-stats`, así que las estadísticas excluyen la rampa. La ventana de `docker stats` empieza y termina en las mismas marcas UTC (`a9_ventanas_<escenario>.json`).
- **Réplicas:** S4b y S5 se ejecutaron dos veces (`r1` y `r2`) para medir la dispersión entre corridas. El resto son corridas únicas.
- **Recursos:** `scripts/monitor_docker_stats.py` muestrea `docker stats` cada segundo sobre `mibici-api` y `mibici-mlflow`. CPU se normaliza al límite de cada contenedor. Red y disco son deltas dentro de la ventana.

## Resultados de Locust

| Escenario | Usuarios | Solicitudes | Fallas | req/s | Predicciones/s | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 10 | 546 | 0 | 9.95 | 37.5 | 16 | 29 | 33 |
| S2 | 40 | 3,229 | 0 | 39.27 | 151.8 | 10 | 39 | 81 |
| S3 | 80 | 6,283 | 0 | 76.44 | 285.2 | 32 | 150 | 220 |
| S4a | 20 | 4,208 | 0 | 71.12 | 71.1 | 280 | 310 | 380 |
| S4b r1 | 20 | 3,822 | 0 | 64.61 | 1,292 | 310 | 350 | 400 |
| S4b r2 | 20 | 4,054 | 0 | 68.53 | 1,371 | 290 | 330 | 400 |
| S5 r1 (1 vCPU) | 20 | 5,314 | 0 | 89.89 | 1,798 | 220 | 280 | 360 |
| S5 r2 (1 vCPU) | 20 | 5,111 | 0 | 86.52 | 1,731 | 220 | 300 | 390 |

Predicciones por segundo = req/s × predicciones por solicitud (1 en unitario, 20 en lote, 0 en health/ready). Detalle en [`a9_finops_rendimiento.csv`](../../../reports/tables/a9_finops_rendimiento.csv).

Reportes HTML de Locust: [S1](locust/S1.html), [S2](locust/S2.html), [S3](locust/S3.html), [S4a](locust/S4a.html), [S4b réplica 1](locust/S4b_r1.html), [S5 réplica 1](locust/S5_r1.html).

## Recursos dentro de la ventana de medición

| Escenario | CPU API (% del límite) | Núcleos medios API | CPU-ms por predicción | Memoria pico API (MiB) |
|---|---:|---:|---:|---:|
| S1 | 9.6 | 0.19 | 5.1 | 215 |
| S2 | 28.0 | 0.56 | 3.7 | 227 |
| S3 | 67.9 | 1.36 | 4.8 | 228 |
| S4a | 102.9 | 2.06 | 28.9 | 230 |
| S4b r1 / r2 | 103.8 / 102.2 | 2.08 / 2.04 | 1.6 / 1.5 | 218 / 223 |
| S5 r1 / r2 (1 vCPU) | 99.1 / 100.1 | 0.99 / 1.00 | 0.55 / 0.58 | 225 / 226 |

MLflow se mantuvo en ~352 MiB (~34 a 36 % de su límite) con una CPU media de ~1 % de su límite y un máximo de 17 % en cualquier muestra. Tabla completa: [`a9_finops_recursos.csv`](../../../reports/tables/a9_finops_recursos.csv).

**Hallazgo:** con 2 vCPU, la API consume el doble de CPU por predicción que con 1 vCPU (1.5 frente a 0.56 ms). El segundo núcleo se gasta sin producir throughput. Una hipótesis plausible es la contención de hilos internos (BLAS, OpenMP o el GIL de Python) con un solo worker de uvicorn. **Esta hipótesis no está verificada** y no se afirma como causa en el documento. Una prueba directa sería repetir S4b con `OMP_NUM_THREADS=1` y `OPENBLAS_NUM_THREADS=1` en la API.

## Análisis FinOps

Script: [`scripts/finops_analysis.py`](../../../scripts/finops_analysis.py). Resumen con supuestos y limitaciones: [`a9_finops_resumen.json`](../../../reports/tables/a9_finops_resumen.json).

Capacidad medida en lote de 20 (promedio de réplicas): 1,331 predicciones/s con 2 vCPU y 1,764 con 1 vCPU. Costo al 70 % de utilización:

| Instancia | Régimen | USD por 1M inferencias |
|---|---|---:|
| EC2 t3.small (2 vCPU, 2 GiB) | capacidad al 70 %, lote | 0.0062 |
| EC2 c6i.large (2 vCPU, 4 GiB) | capacidad al 70 %, lote | 0.0253 |
| Fargate 1 vCPU / 2 GB | capacidad al 70 %, lote | 0.0111 |
| EC2 t3.small | demanda horaria MiBici (484 consultas/h), siempre encendida | 43.18 |
| EC2 c6i.large | demanda horaria, siempre encendida | 175.62 |
| Fargate 1 vCPU / 2 GB | demanda horaria, siempre encendida | 102.00 |
| Fargate 1 vCPU / 2 GB | programado, 1 tarea por hora (mínimo facturable de 60 s) | 1.70 |
| AWS Lambda 1 GB | por solicitud (16 ms supuestos) | 0.47 |
| Cloud Run 1 vCPU / 512 MiB | por solicitud (16 ms supuestos) | 0.80 |

El costo por inferencia depende casi por completo de la utilización. Entre capacidad máxima y demanda real hay un factor de unos 7,000 veces: la instancia ociosa domina el costo. Por eso las optimizaciones que reducen tiempo ocioso pesan más que cambiar de tipo de instancia.

Optimizaciones cuantificadas ([`a9_finops_optimizaciones.csv`](../../../reports/tables/a9_finops_optimizaciones.csv)):

| ID | Optimización | Ahorro en costo por inferencia |
|---|---|---:|
| O1 | Lotes de 20 estaciones en lugar de unitario (S4a → S4b) | 94.7 % (x18.7) |
| O2 | Right-sizing de CPU de 2 a 1 vCPU con lote (S4b → S5) | 62.3 % (x2.6) |
| O3 | Ejecución programada por hora en lugar de siempre encendido | 98.3 % (x60) |
| O4 | Right-sizing de memoria: c6i.large (4 GiB) a t3.small (2 GiB) | 75.4 % (x4.1) |

**Sensibilidad:** con carga sostenida al 70 %, t3.small paga el excedente sobre su línea base de 20 % por vCPU (modo unlimited). Así cuesta unos 0.021 USD por millón, apenas por debajo de c6i.large (0.025). La ventaja de t3 solo existe si la CPU se queda por debajo de la línea base. Tabla completa en [`a9_finops_sensibilidad.csv`](../../../reports/tables/a9_finops_sensibilidad.csv) (utilización de 10 % a 100 % y factor de capacidad de 0.5 a 1.5).

## Figuras

1. [Latencia y rendimiento por escenario](figuras/01_latencia_y_rendimiento.png)
2. [Serie temporal de S3: punto de saturación](figuras/02_serie_temporal_S3.png)
3. [Recursos de API y MLflow con ventanas de medición](figuras/03_recursos_api_mlflow.png)
4. [Costo por inferencia, programado/serverless y optimizaciones](figuras/04_finops_costo_y_optimizaciones.png)

## Limitaciones

- La capacidad se midió en Apple M3 Pro (arm64) con Docker Desktop. Las instancias de nube son x86 y no se midieron. Los números de FinOps son estimaciones de referencia para comparar regímenes, no predicciones de factura.
- Locust corre en la misma máquina que la API y compite por CPU. Cada escenario tiene una corrida, salvo S4b y S5 (dos réplicas).
- Las tarifas (EC2, Fargate, Lambda, Cloud Run y el modo unlimited de t3) son públicas de referencia us-east-1 y deben verificarse en la página oficial de cada proveedor antes de publicar.
- Fargate 0.5 vCPU / 1 GB es extrapolación lineal de la medición con 1 vCPU, no una medición.
- Lambda y Cloud Run usan 16 ms por solicitud (p50 de S1 con carga baja), sin arranque en frío ni carga del modelo.
- No se incluye el costo de almacenamiento, de transferencia de red ni del contenedor MLflow en el costo por inferencia.
- El plan inicial estimaba ~360 estaciones. Se usan 484, que es el número del catálogo de referencia `references/nomenclatura_2026_09.csv`.

## Cómo reproducir

```bash
# Entorno de la Actividad 9 (Python 3.12). Las dependencias del proyecto y Locust.
uv venv .venv-a9 --python 3.12
uv pip install --python .venv-a9/bin/python -r requirements.txt -r requirements-loadtest.txt

# Stack local
cp .env.example .env            # opcional
docker compose up -d --build
docker compose ps               # api y mlflow deben estar "healthy"

# Escenarios S1 a S4b con monitoreo sincronizado (desde la raíz del repositorio)
.venv-a9/bin/python scripts/run_a9_scenarios.py --host http://127.0.0.1:8000

# Análisis FinOps: tablas en reports/tables/a9_finops_*, figuras en figuras/
.venv-a9/bin/python scripts/finops_analysis.py

# Pruebas de la actividad
.venv-a9/bin/python -m pytest -q tests/test_compose_and_loadtest.py
```
