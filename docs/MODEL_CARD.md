# Model Card — MiBici Riesgo Operativo de Estación (P2)

**Proyecto:** TC5061 · Proyecto Integrador MLOps — Equipo 49  
**Responsables:** Diego Sánchez Hernández (`dieguini95`), Daniel Ruán Aguilar (`DaniRuan`), Saúl Salgado Ortega (`Drezneck-RM`)  
**Fecha de revisión:** 26 de noviembre de 2026  
**Estado de gobernanza:** Aprobado para entrega final (v1 servida en API; v2 promovida en Registry local, pendiente re-exportación tras validar noviembre de 2026)

---

## 1. Detalles del modelo y linaje

| Campo | Valor |
|---|---|
| Nombre en MLflow Registry | `mibici-equipo49-p2-riesgo` |
| Tipo de tarea | Clasificación binaria (probabilidad de evento raro) |
| Familia / algoritmo | `HistGradientBoostingClassifier` (`scikit-learn 1.9.1`) |
| Hiperparámetros congelados | `learning_rate=0.03`, `max_iter=600`, `max_leaf_nodes=15`, `min_samples_leaf=100`, `random_seed=42` |
| Contrato de puntuación | `predict_proba` (clase positiva) + umbral de validación. **No usar `predict()` con corte 0.5.** |
| Versión servida en API | **v1** (`models/serving/p2/model.skops`, umbral `0.200205`) |
| Versión con alias `@champion` en Registry local | **v2** (promovida el 12-nov-2026 tras A/B offline; umbral `0.273049`) |
| Formato de serialización en serving | `skops` con lista explícita de tipos de confianza (sin `pickle`) |
| Licencia del código | MIT (`LICENSE`) |

### Linaje en MLflow Tracking y Model Registry

| Versión | Run ID | Commit Git | Ventana de entrenamiento | Ventana de umbral / prueba | `data_version` (SHA-256 prefijo) | Estado |
|---|---|---|---|---|---|---|
| **v1** | `9dca0330ba6241dc8ccdf41d48cf4954` | `74d5e51c` | `2025-10-01` a `2026-06-30` | Val `2026-07`–`08`, Test `2026-09` | `0f000bd5…` | **Servida en `models/serving/p2`** (respaldo en Registry) |
| **v2** | `2a0fa001974c4a6180d29033a695842e` (`parent_version=1`) | `28ceb7ad` | `2025-10-01` a `2026-08-31` | Umbral en `2026-09` (max F1); A/B en `2026-10` | `220569d9…` | **`@champion` en Registry local** (pendiente re-exportar) |

> **Nota sobre P3 en el mismo contenedor:** la API FastAPI sirve también el modelo de perfiles de estación `mibici-equipo49-p3-grupos@champion` (`v1`, `KMeans` con `k=8`, `silhouette_score=0.5174`, run `8c5ecfac5218433883a30c0b222bcf28`, `data_version=sha256:ee581cfe...`) en `/predict` y `/predict/batch` como apoyo exploratorio de segmentación.

---

## 2. Uso previsto y alcance

### Uso previsto
- Estimar, en el instante horario `t`, la probabilidad de que una estación MiBici quede **vacía (0 bicicletas) o llena (capacidad `C`)** dentro de las siguientes **2 horas** (`H = 2`).
- Apoyar la priorización operativa de monitoreo y rebalanceo preventivo mediante:
  - Inferencia individual en tiempo real (`POST /v1/p2/predict`).
  - Evaluación horaria por lotes de hasta 500 estaciones (`POST /v1/p2/predict/batch`).

### Usuarios previstos
- Equipo operativo y de analítica del sistema de bicicletas públicas (escenario académico sobre datos abiertos de MiBici Guadalajara).

### Fuera de alcance
- **No calcula rutas de rebalanceo** ni cantidades óptimas de traslado entre estaciones.
- **No mide demanda latente** (usuarios que no pudieron iniciar un viaje por falta de bicicletas) ni reemplaza sensores físicos de inventario en los puertos.
- **No debe operarse con el umbral por defecto de 0.5**, ya que la prevalencia de eventos positivos es baja (`0.38%`–`0.69%`) y un corte de 0.5 colapsa el recall.

---

## 3. Datos y variables de entrada

- **Fuente:** portal de [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/) (viajes mensuales y catálogo de nomenclatura de septiembre de 2026, 484 estaciones).
- **Volumen base (oct-2025 a sep-2026):** `4,497,877` viajes crudos → `4,493,328` viajes válidos → `3,906,456` filas en el panel modelado (`12,911` días-estación no factibles excluidos, `7.35%`).
- **Ventana de monitoreo fuera de tiempo (oct-2026):** `635,341` viajes limpios → `328,176` filas (`13,674` pares estación-día, `2,265` positivos, prevalencia `0.69%`).
- **Partición temporal estricta (sin fuga futura):** entrenamiento `2025-10` a `2026-06`, validación `2026-07` a `2026-08`, prueba cerrada `2026-09`, evaluación out-of-time `2026-10`.
- **14 variables de entrada (`P2_FEATURE_COLUMNS_V2`):** `hora`, `dia_semana`, `fin_de_semana`, `capacidad`, `flujo_acumulado`, `flujo_acumulado_sobre_c`, `llegadas_1h`, `salidas_1h`, `llegadas_2h`, `salidas_2h`, `salidas_misma_hora_ayer`, `riesgo_observado_h`, `riesgo_observado_previo_h`, `tendencia_flujo_3h`.
- **Privacidad y seguridad de datos:** se excluyen todos los campos de usuario (género, año de nacimiento, identificadores personales). Datos crudos, `mlflow.db` y `mlruns/` permanecen fuera de Git (verificado por `tests/test_repository_security.py`).

---

## 4. Evaluación y métricas

### Desempeño histórico de P2 v1 (bundle servido)

| Partición | PR-AUC | ROC-AUC | Brier | F1 (umbral `0.2002`) | Recall (umbral) | Recall @ Precisión ≥ 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Validación (`jul–ago 2026`) | 0.4029 | 0.9484 | 0.002778 | 0.4461 | 0.3623 | 0.3868 |
| Prueba (`sep 2026`) | 0.4001 | 0.9397 | 0.003878 | 0.4307 | 0.3529 | 0.3699 |

### Evaluación A/B offline: v1 vs. v2 (octubre de 2026, `328,176` filas)

- **Regla de decisión fijada antes de ver resultados:** promover `v2` solo si el límite inferior del IC 95% de `ΔPR-AUC (v2 − v1)` (bootstrap pareado por estación-día, 1,000 remuestras, semilla 42) es `> 0` **y** `Brier_v2 <= Brier_v1`.

| Métrica (octubre 2026) | v1 (`umbral 0.2002`) | v2 (`umbral 0.2730`) | Diferencia / IC 95% |
|---|---:|---:|---|
| **PR-AUC (principal)** | 0.3748 | **0.3777** | **+0.002920** · IC 95% `[+0.000213, +0.005485]` |
| ROC-AUC | 0.9292 | 0.9304 | +0.0012 |
| **Brier** | 0.005200 | **0.005187** | −0.000013 · IC 95% `[−0.000030, +0.000005]` |
| Precisión al umbral | 0.5510 | 0.5932 | +0.0422 |
| Recall al umbral | **0.3364** | 0.3148 | −0.0216 |
| F1 al umbral | 0.4178 | 0.4113 | −0.0065 |
| Precisión en top 10% por hora | 0.0479 | 0.0487 | +0.0007 |

- **Decisión:** la regla se cumplió (sensibilidad confirmada en 5 semillas con límite inferior entre `+0.00016` y `+0.00047`), por lo que el alias `@champion` se movió a `v2` en el Registry local el 12-nov-2026. Dado que el margen es pequeño y el recall al umbral disminuye, el bundle versionado en `models/serving/p2` se mantiene en `v1` hasta confirmar el desempeño con noviembre de 2026.

---

## 5. Operación, rendimiento y FinOps

- **Despliegue:** API FastAPI contenerizada (`Containerfile`, usuario sin privilegios `UID 10001`, `HEALTHCHECK` en `/health`), orquestada con MLflow y Locust en `docker-compose.yml`.
- **Validación automática (CI/CD):** `.github/workflows/ci-serving.yml` ejecuta 6 jobs (`compose-config`, `quality`, `tests`, `deepchecks-tests`, `security`, `image-smoke` con build OCI real y smoke P2/P3 HTTP 200).
- **Rendimiento medido (Locust, 0 fallas):**
  - Mezcla operativa: `9.95` req/s (S1), `39.27` req/s (S2), `76.44` req/s con `p95 = 150 ms` (S3).
  - Lotes de 20 estaciones: `~1,331` pred/s con 2 vCPU (S4b) y `~1,764` pred/s con 1 vCPU (S5).
  - Memoria pico: `215–230 MiB` (API) y `~352 MiB` (MLflow), dentro del límite de `1 GiB`.
- **Estimaciones FinOps (referencia `us-east-1`, no facturas reales):**
  - Al 70% de capacidad en lote: `~USD 0.0062` / 1M inferencias (`EC2 t3.small`) y `~USD 0.0111` / 1M (`Fargate 1 vCPU / 2 GB`).
  - Optimizaciones clave: batching de 20 (`−94.7%`), right-sizing a 1 vCPU (`−62.3%`) y ejecución horaria programada frente a instancia ociosa 24/7 (`−98.3%`, de `~USD 102` a `~USD 1.70` / 1M).

---

## 6. Riesgos, limitaciones y consideraciones éticas

- **Proxy de inventario:** la etiqueta de riesgo se infiere a partir del flujo acumulado de viajes y la capacidad del catálogo de septiembre de 2026, no de telemetría física de los anclajes.
- **Días-estación no factibles:** el `7.35%` de los días-estación (`12,911`) no admiten un inventario inicial factible sin rebalanceo no observado y se excluyen del entrenamiento y evaluación.
- **Horizonte de validación out-of-time:** el A/B cubre un solo mes (`oct-2026`); el intervalo de confianza captura variabilidad muestral dentro de ese mes, no variabilidad entre estaciones del año.
- **Discrepancia de hash en catálogo procesado:** `data_version` del panel de monitoreo (`220569d9…`) difiere del run original (`0f000bd5…`) por la serialización de `stations.parquet`, aunque el catálogo crudo es idéntico y la paridad de métricas de `v1` fue verificada (`3,906,456` filas, PR-AUC val `0.4029`, test `0.4001`).
- **Compromiso precisión/recall en v2:** el umbral más alto de `v2` (`0.2730`) eleva la precisión (`0.551 → 0.593`) pero reduce el recall (`0.336 → 0.315`), implicando más eventos de saturación no alertados si no se recalibra el umbral.
- **Sin datos personales (PII) ni riesgo discriminatorio:** el modelo opera exclusivamente sobre conteos agregados por estación y hora; no perfila personas ni utiliza variables demográficas.

---

## 7. Monitoreo, Continuous Training y Rollback

### Monitoreo de drift (Evidently)
- Corridas reproducibles (`make monitoring-drift`, 150,000 filas/ventana, semilla 42):
  - **R1 (`oct-26` vs entrenamiento), R2 (`sep-26` vs entrenamiento) y R3 (control estacional `oct-25` vs `oct-26`):** las 3 corridas detectan drift en las mismas **8 de 14 variables de flujo** (`share = 0.571`).
  - Las variables más importantes por permutación (`riesgo_observado_h` y `hora`) **no** presentan drift; `flujo_acumulado_sobre_c` y `llegadas_2h` sí cruzan drift con importancia relevante, reflejando un aumento de `+51%` en volumen interanual (`419,839` → `635,341` viajes en octubre).

### Disparadores de Continuous Training (CT)

| ID | Disparador | Regla | Estado en octubre 2026 |
|---|---|---|---|
| **D1** | Desempeño | PR-AUC mensual `< 0.95 × 0.4001` (`≈ 0.380`) durante 2 meses consecutivos | Primer mes por debajo (`0.3748`); vigilar noviembre |
| **D2** | Drift en variables clave | Drift en variable con importancia por permutación `≥ 0.01` | **Activo** (`llegadas_2h`, `flujo_acumulado_sobre_c`) |
| **D3** | Volumen y prevalencia | Volumen mensual `> 25%` sobre media de entrenamiento o cambio `> 20%` en prevalencia | **Activo** (`+77%` en volumen); recalibrar umbral |
| **D4** | Calibración | Diferencia `> 20%` entre probabilidad media predicha y prevalencia observada | Inactivo (`−7%`) |

### Política de promoción y Rollback
- **Promoción:** reentrenamiento mensual como `@challenger`, evaluación A/B offline en el último mes completo y promoción manual de `@champion` únicamente si se cumple la regla fija; luego `make serving-export QUESTION=p2` y validación por PR + `CI Serving`.
- **Rollback inmediato:**
  1. **En producción/API:** revertir el commit de `models/serving/p2/` al bundle anterior verificado (actualmente `v1`) y reconstruir/desplegar la imagen validada en CI.
  2. **En MLflow Registry:** reasignar el alias `champion` a la versión de respaldo (`v1`) mediante `mibici_ml.modeling.registry_p2.assign_p2_alias`.

---

## 8. Referencias dentro del repositorio

- Evidencia de monitoreo, drift y A/B: [`docs/evidencias/Actividad 10 - Drift real y A-B P2/README.md`](evidencias/Actividad%2010%20-%20Drift%20real%20y%20A-B%20P2/README.md)
- Evidencia de CI/CD del serving: [`docs/evidencias/Actividad 9 - CI Serving/README.md`](evidencias/Actividad%209%20-%20CI%20Serving/README.md)
- Evidencia de operación, carga y FinOps: [`docs/evidencias/Actividad 9 - Operacion y FinOps/README.md`](evidencias/Actividad%209%20-%20Operacion%20y%20FinOps/README.md)
- Metadatos del bundle servido: [`models/serving/p2/model_metadata.json`](../models/serving/p2/model_metadata.json) y [`models/serving/p3/model_metadata.json`](../models/serving/p3/model_metadata.json)
- Manifiesto de monitoreo: [`references/monitoring_manifest_p2.json`](../references/monitoring_manifest_p2.json)
