# Actividad 10: drift real, A/B y Continuous Training (P2)

Evidencia de la Actividad 10 sobre el modelo P2 de riesgo de estación (probabilidad de que una
estación MiBici quede vacía o llena en las siguientes 2 horas). Complementa el monitoreo offline
de la [sección de la Actividad 10 en el README](../../../README.md#monitoreo-offline-de-drift-p2-actividad-10).

## Qué demuestra esta evidencia

| Parte de la rúbrica | Dónde está |
|---|---|
| Panel de monitoreo y paridad con la versión 1 | `transcripts/00_panel_paridad.txt` |
| Drift con Evidently sobre datos reales | Corridas R1, R2 y R3 (`transcripts/01_*`), capturas `capturas/08` a `10` |
| Interpretación del drift | `figuras/01` a `05`, `transcripts/05_*`, tablas en `reports/tables/p2_drift_importancia.csv` |
| A/B champion frente a challenger | `transcripts/03_ab_oct26.txt`, `figuras/06`, `figuras/07`, `reports/tables/p2_ab_oct26_*` |
| Registro de la versión 2 y linaje en MLflow | `capturas/11_mlflow_registro_p2.png` (versiones 1 y 2, alias) y `capturas/12_mlflow_runs_actividad10.png` (runs `P2-hist_gb-v2-challenger` y `ab_test`) |
| Continuous Training | Sección "Recomendación de Continuous Training" de este README |

## Datos y ventanas

- Panel de monitoreo: `references/monitoring_manifest_p2.json`. Entrenamiento de la versión 1 más
  octubre 2026 (ingesta aislada, misma regla de limpieza que la ingesta principal).
- `data_version` del panel de monitoreo: `220569d9969f3ce2...` (completo en el manifiesto).
- Referencia de drift: 2025-10-01 a 2026-06-30 (los datos con los que se entrenó la versión 1).
- Ventanas actuales:
  - **R1** octubre 2026 (fuera de tiempo, no vista por ningún modelo). Ventana principal.
  - **R2** septiembre 2026 (prueba abierta de la versión 1). Segunda corrida para ver la tendencia.
  - **R3** control estacional: octubre 2025 frente a octubre 2026. Separa la estacionalidad del cambio.
- Cada corrida de Evidently usa 150,000 filas por ventana (muestreo con semilla 42).

## Paridad del panel con la versión 1

Antes de usar el panel, se verificó que reconstruye el run del champion:

| Verificación | Registrado en el run | Reconstruido |
|---|---|---|
| Filas del panel (sin octubre) | 3,906,456 | 3,906,456 |
| Días estación no factibles | 12,911 | 12,911 |
| PR-AUC de validación (hist_gb) | 0.4029 | 0.402895 |
| PR-AUC de prueba (sep 2026) | 0.4001 | 0.400067 |

El `data_version` no coincide con el del run (`0f000bd5…` frente a `99e366…` en el manifiesto de
entrenamiento). La causa es la serialización de `stations.parquet`: el catálogo bruto es idéntico,
pero los bytes procesados difieren. Las métricas sí coinciden, así que se documenta como limitación.

## Drift: R1, R2 y R3

Resumen completo en `transcripts/01_drift_r1_r2_r3_resumen.txt`. Se usa el umbral por defecto de
Evidently (Wasserstein normalizada para variables numéricas, Jensen-Shannon para categóricas) y el
umbral de dataset de 0.5 de la fracción de columnas con drift.

| Corrida | Columnas con drift | Share | Dataset drift |
|---|---|---|---|
| R1 oct-26 vs entrenamiento | 8 de 14 | 0.571 | sí |
| R2 sep-26 vs entrenamiento | 8 de 14 | 0.571 | sí |
| R3 control oct-25 vs oct-26 | 8 de 14 | 0.571 | sí |

Las mismas 8 variables se marcan en las tres corridas: `flujo_acumulado`, `flujo_acumulado_sobre_c`,
`llegadas_1h`, `salidas_1h`, `llegadas_2h`, `salidas_2h`, `salidas_misma_hora_ayer` y
`tendencia_flujo_3h`. Que el control R3 (octubre contra octubre, un año de diferencia) dé el mismo
resultado indica que el cambio no es propio de octubre 2026: es un cambio de volumen entre años.

## Interpretación

**1. El drift es de volumen, no de estación.** Las variables que se mueven son de flujo (llegadas,
salidas y acumulados). Las variables de la estación (capacidad, hora, día y riesgo observado) casi
no cambian. Los viajes limpios por mes pasaron de 419,839 en octubre 2025 a 635,341 en octubre 2026
(+51%), y la llegada media por hora-estación de 0.93 a 1.52 (`figuras/05_volumen_mensual.png`).

**2. El drift no es donde está el modelo.** Por importancia por permutación en validación, las
variables más usadas son `riesgo_observado_h` (0.347) y `hora` (0.175). Ninguna de las dos muestra
drift. Las variables con drift tienen importancia baja, salvo `flujo_acumulado_sobre_c` (0.035) y
`llegadas_2h` (0.012), que sí cruzan drift con importancia alta (`figuras/02_drift_vs_importancia.png`).

**3. El desempeño de la versión 1 baja en octubre, aunque el drift se marque en todos los meses.**
El PR-AUC mensual de la versión 1 es 0.405 (jul), 0.401 (ago), 0.400 (sep) y 0.375 (oct). La
prevalencia observada sube de 0.0036 a 0.0069 y la probabilidad media predicha la sigue de cerca
(`figuras/03_v1_mensual.png`, `figuras/04_calibracion_mensual.png`, `reports/tables/p2_v1_mensual.csv`).
La caída es real en el orden de magnitud (−6% relativo frente a septiembre), pero con un solo mes
fuera de tiempo no se puede afirmar una tendencia.

## A/B offline en octubre 2026

Diseño: la versión 1 (champion) y la versión 2 (challenger) se evalúan sobre las mismas 328,176 filas
de octubre, con el umbral propio de cada una. La versión 2 usa los mismos hiperparámetros que la 1
y se entrenó con octubre 2025 a agosto 2026. Su umbral (0.273) se eligió en septiembre con la regla
de máximo F1. Octubre no participó en el ajuste ni en la elección de umbral de ninguna de las dos.

**Regla de decisión, fijada antes de ver el resultado:** promover la versión 2 solo si el intervalo
de confianza al 95% de ΔPR-AUC (v2 − v1), obtenido con bootstrap pareado por estación-día, tiene
límite inferior mayor que cero y el Brier de la versión 2 no es peor que el de la versión 1.

| Métrica (octubre 2026) | v1 champion (umbral 0.2002) | v2 challenger (umbral 0.2730) |
|---|---|---|
| PR-AUC (principal) | 0.3748 | 0.3777 |
| ROC-AUC | 0.9292 | 0.9304 |
| Brier | 0.005200 | 0.005187 |
| Precisión al umbral | 0.551 | 0.593 |
| Recall al umbral | 0.336 | 0.315 |
| F1 al umbral | 0.418 | 0.411 |
| Recall con precisión ≥ 0.5 | 0.357 | 0.350 |
| Precisión en top 10% por hora | 0.0479 | 0.0487 |

- ΔPR-AUC = **+0.0029**, IC 95% **[+0.0002, +0.0055]**. Límite inferior mayor que cero.
- ΔBrier = −1.3e-5, IC 95% [−3.0e-5, +5.0e-6]. El intervalo incluye cero: no hay diferencia clara.
- Sensibilidad a la semilla del bootstrap (cinco semillas, `transcripts/04_sensibilidad_semillas.txt`):
  el límite inferior del ΔPR-AUC va de 0.00016 a 0.00047, siempre por encima de cero.
- Septiembre, ventana que la versión 2 no vio al entrenar: PR-AUC 0.4027 frente a 0.4001 (+0.0026).

**Resultado de la regla:** se cumple (promover v2). El margen es mínimo: el efecto es menor a
0.3 puntos de PR-AUC, el Brier no mejora de forma significativa y el recall al umbral operativo baja.
El alias `champion` **no se movió**: la promoción es una decisión del equipo, separada del análisis.

Resultados en `reports/tables/p2_ab_oct26_resultado.json`, `p2_ab_oct26_metricas.csv` y
`p2_ab_oct26_sensibilidad.csv`. Registro MLflow: run `ab_test` en el experimento
`mibici-equipo49-actividad10`.

## Recomendación de Continuous Training

**Política propuesta:** reentrenamiento mensual con los meses más recientes con etiqueta completa, y
evaluación del candidato en el último mes completo antes de cualquier promoción. El candidato se
registra como `challenger`; el alias `champion` solo cambia si la regla fija del A/B se cumple, y
el cambio lo hace una persona del equipo.

**Disparadores para reentrenar o revisar fuera de calendario** (umbrales propuestos a partir de
esta única evidencia; deben recalibrarse con más meses):

| Disparador | Regla | Estado con octubre 2026 |
|---|---|---|
| D1. Desempeño | PR-AUC mensual < 0.95 × 0.4001 (≈ 0.380) durante 2 meses seguidos | 0.3748 (0.937×): **primer mes por debajo**; vigilar noviembre |
| D2. Drift en variables importantes | Drift en una variable con importancia ≥ 0.01 | `llegadas_2h` y `flujo_acumulado_sobre_c`: **activo**; revisar |
| D3. Volumen y umbral | Volumen mensual de viajes > 25% por encima del promedio de entrenamiento, o cambio de > 20% en la prevalencia observada frente a la media predicha | Volumen de octubre **+77%** frente al promedio mensual de entrenamiento (358,065 viajes, oct-25 a jun-26): **activo**; recalibrar umbral |
| D4. Calibración | Probabilidad media predicha frente a prevalencia observada con diferencia > 20% en un mes | −7%: no activo |

**Lectura:** octubre activa D2 y D3. D1 solo llegará a su condición si noviembre también queda por
debajo. La acción inmediata propuesta es recalibrar el umbral con el volumen reciente y volver a
correr el A/B en noviembre con la versión 2 antes de decidir la promoción.

## Limitaciones

- El drift se mide con variables derivadas de viajes, no con inventario observado de la estación.
  La etiqueta de riesgo también es un proxy del inventario.
- Octubre es un solo mes fuera de tiempo. El IC del A/B refleja la variabilidad de muestreo de
  ese mes, no la variación entre meses.
- El cambio de volumen puede venir de crecimiento de la demanda, de cambios en el portal o de
  cambios en la forma de registrar viajes. Los datos no permiten distinguirlos.
- Los umbrales de los disparadores son propuestas, no valores validados históricamente.
- El `data_version` reconstruido no coincide con el del run por diferencias de serialización
  (ver "Paridad del panel"). Las métricas sí coinciden.
- La captura de Evidently es el HTML renderizado en Chrome sin interacción; las tablas completas
  están en `drift_columns.csv` de cada corrida.
- Calidad: la suite completa pasa (350 passed, 13 skipped; `transcripts/06_calidad_pruebas.txt`).
  Una prueba de la suite de Daniel/equipo, `tests/test_risk.py::test_evaluate_all_metrics_are_identical_across_runs_with_same_seed`,
  falla de forma intermitente (≈4e-7 relativo en `val_pr_auc`). No la modifica esta rama; queda como
  tarea aparte (la tolerancia de 1e-12 es demasiado estricta para el paralelismo de HistGradientBoosting).

## Cómo reproducir

```bash
# 1. Panel de monitoreo (ingesta aislada de octubre 2026 y compuerta de paridad)
.venv/bin/python -m mibici_ml.monitoring.panel build --month 2026_10

# 2. Corridas de drift (entorno aislado con Evidently)
.venv-monitoring/bin/python -m mibici_ml.monitoring.cli --input data/interim/monitoring/p2_monitoring_input.parquet --data-version <data_version> --run-name r2
.venv-monitoring/bin/python -m mibici_ml.monitoring.cli --input data/interim/monitoring/p2_monitoring_input.parquet --data-version <data_version> --current-start 2026-10-01 --current-end 2026-10-31 --current-role out_of_time --run-name r1
.venv-monitoring/bin/python -m mibici_ml.monitoring.cli --input data/interim/monitoring/p2_monitoring_input.parquet --data-version <data_version> --reference-start 2025-10-01 --reference-end 2025-10-31 --current-start 2026-10-01 --current-end 2026-10-31 --current-role out_of_time --run-name r3

# 3. Challenger (versión 2) y A/B
.venv/bin/python -m mibici_ml.modeling.retrain_p2 train
.venv/bin/python -m mibici_ml.modeling.ab_p2 run
.venv/bin/python -m mibici_ml.modeling.ab_p2 sensibilidad

# 4. Interpretación (tablas y figuras de esta carpeta)
.venv/bin/python -m mibici_ml.monitoring.interpretation run
```

Los reportes de Evidently (`reports/drift/`) están fuera de git por diseño: contienen agregados, pero
no se versionan junto con el código.
