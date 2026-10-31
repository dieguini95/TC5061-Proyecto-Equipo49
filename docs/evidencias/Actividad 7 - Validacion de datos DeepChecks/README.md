# Actividad 7 - Evidencia de validación de datos con DeepChecks

Esta carpeta reúne la evidencia de la validación de datos con DeepChecks. La evidencia de seguridad y
pruebas de higiene está en la carpeta `Actividad 7 - Chequeos de seguridad`, que no se modifica aquí.

## 1. Alcance

- Suites de DeepChecks 0.19.1 sobre viajes limpios y sobre el panel de riesgo de P2, con la partición
  temporal de `mibici_ml/config.py` (entrenamiento oct-25 a jun-26; prueba sep-26).
- Chequeo de nulos con `PercentOfNulls` (umbral 0%), que `data_integrity` no incluye.
- Pruebas deterministas con datos sintéticos (`tests/test_data_validation.py`) para casos normales,
  límite y de error.
- Entorno aislado `.venv-quality` (ver `requirements-quality.txt`): DeepChecks no funciona con
  numpy 2.x ni scikit-learn 1.9, que usa el proyecto.

## 2. Comandos reproducibles

```bash
uv venv --python 3.12 .venv-quality
uv pip install --python .venv-quality/bin/python -r requirements-quality.txt
.venv-quality/bin/python -m pytest -q tests/test_data_validation.py
.venv-quality/bin/python -m mibici_ml.data_validation
```

Con GNU Make: `make data-validation-tests` y `make data-validation`.

## 3. Resultados verificados

Versión de datos `0f000bd5e656`, semilla 42, muestra de hasta 150,000 filas por conjunto.

| Suite | Chequeos | Aprobados | Fallidos | No aplicables | Errores |
|---|---:|---:|---:|---:|---:|
| `integridad_viajes` | 13 | 9 | 0 | 3 | 0 |
| `integridad_riesgo_p2` | 13 | 10 | 2 | 0 | 0 |
| `train_test_viajes` | 12 | 7 | 1 | 4 | 0 |
| `train_test_riesgo_p2` | 12 | 10 | 1 | 1 | 0 |

Pruebas: ver la salida de `pytest -q tests/test_data_validation.py` en las capturas (sección 5).

Los 3 chequeos de etiqueta de `integridad_viajes`, los 4 de `train_test_viajes` y `Index Train Test Leakage`
son `no_aplicable`: los viajes no tienen etiqueta ni índice. Se reportan con su motivo.

## 4. Interpretación de hallazgos

| Chequeo | Resultado | Lectura |
|---|---|---|
| Valores faltantes (`PercentOfNulls`) | 0% en viajes y en el panel | Sin faltantes en las variables usadas |
| Duplicados en viajes (`Data Duplicates`) | 71,495 de 4,493,328 (1.6%) en variables derivadas | `trip_id` es único; son viajes distintos con el mismo perfil (duración, hora, día, origen, destino). Aprobado: bajo el umbral de 5% en la muestra |
| Duplicados en el panel (`Data Duplicates`) | 1,699,371 de 3,906,456 (43.5%) | El panel no tiene columna de tiempo absoluto; misma estación, hora, día de la semana y flujo cero producen filas iguales. Fallido: esperado por diseño |
| Conflictos de etiqueta (`Conflicting Labels`) | 0.11% de las muestras | Variables idénticas con etiquetas distintas: el riesgo no es determinista con estas variables. Fallido |
| Correlación alta (`Feature-Feature Correlation`) | `capacidad` con `est`; `flujo_acumulado` con `flujo_acumulado_sobre_c` | Redundancias conocidas. Fallido; no invalida los datos |
| Categorías nuevas (`New Category Train Test`) | 3.84% de origen y 3.74% de destino en prueba | Estaciones que no aparecen en entrenamiento. Fallido; las variables de P2 no incluyen la estación |
| Mezcla train/test (`Train Test Samples Mix`) | 18.81% de las filas de prueba del panel repiten un vector de entrenamiento | Consecuencia de los duplicados del panel. Fallido; no hay fuga temporal (ver fechas) |
| Fuga por fechas (`Date Train-Test Leakage`) | Aprobado en viajes y en el panel | La partición temporal no se traslapa |
| Drift (`Feature Drift`, `Multivariate Drift`, `Label Drift`) | Aprobado | Sin drift de variables ni de etiqueta entre entrenamiento y prueba |

Los fallos son hallazgos documentados. No detienen la ejecución: el comando termina con código 0 y deja el
detalle en el resumen JSON y en los HTML.

## 5. Capturas

Las capturas muestran el comando y su resultado. Nombres exactos de archivo:

| Archivo | Comando | Qué debe verse |
|---|---|---|
| `01_pytest_data_validation.png` | `.venv-quality/bin/python -m pytest -q tests/test_data_validation.py` | Pruebas en verde, sin fallos |
| `02_data_validation_summary.png` | `.venv-quality/bin/python -m mibici_ml.data_validation` | Resumen de las cuatro suites y ruta del JSON |
| `03_deepchecks_report.png` | Reporte HTML `reports/data_validation/integridad_riesgo_p2.html` en el navegador | Vista de los chequeos de DeepChecks |
| `04_pytest_risk_reproducibility.png` | `pytest -q tests/test_risk.py -k "same_seed or different_seed or identical_across"` | Pruebas de reproducibilidad en verde |
| `05_pytest_full_p2.png` | `pytest -q` | Total de pruebas en verde |

Estado: pendientes de captura. Cuando estén en esta carpeta, se incrustan así:

```markdown
![Pruebas de validación de datos](./01_pytest_data_validation.png)
```

## 6. Limitaciones

- La validación usa muestras de 150,000 filas por conjunto (semilla 42), no el total. Los conteos de nulos y
  duplicados del resumen sí se calculan sobre todas las filas.
- El panel de riesgo se reconstruye con `features.py` a partir de los viajes limpios. El panel de
  `data/interim` no lo produce el código del repositorio, así que no se usa.
- DeepChecks 0.19.1 no es compatible con la pila principal; por eso las pruebas de DeepChecks corren en
  `.venv-quality` y se omiten en el entorno principal.
- Los HTML de DeepChecks no se versionan porque pueden contener filas de ejemplo.
