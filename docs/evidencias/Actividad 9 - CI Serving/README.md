# Actividad 9 - CI del serving

## Alcance implementado

El workflow `.github/workflows/ci-serving.yml` valida el repositorio en pull requests dirigidos a `main`, pushes a `main` y ejecuciones manuales. Usa permisos de solo lectura y no requiere secretos.

Los jobs `quality`, `tests`, `deepchecks-tests` y `security` ejecutan Ruff, la suite PyTest, los casos DeepChecks deterministas en un entorno aislado y `bash scripts/run_security_checks.sh`. El job de seguridad cubre `pip check`, Bandit con severidad/confianza medium y las pruebas de higiene. La auditoría `pip-audit` no forma parte del gate; no se afirma ausencia de vulnerabilidades de dependencias.

`image-smoke` espera a esos gates, construye la imagen desde `Containerfile`, la ejecuta con el puerto publicado sólo en loopback y comprueba `/health`, `/ready`, `/v1/p2/ready`, `/predict` y `/v1/p2/predict`. El contenedor se registra en logs y se limpia incluso si falla el smoke test. La imagen no se publica y no hay despliegue.

## Validación local

La validación local de esta contribución usa:

```bash
.venv/bin/python -m pip check
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/ruff format --check scripts/ci_smoke_serving.py tests/test_ci_smoke_serving.py tests/test_ci_workflow.py
.venv/bin/ruff check scripts/ci_smoke_serving.py tests/test_ci_smoke_serving.py tests/test_ci_workflow.py
.venv/bin/pytest -q tests/test_ci_smoke_serving.py tests/test_ci_workflow.py
.venv/bin/pytest -q
bash scripts/run_security_checks.sh
```

La validación YAML se realiza con PyYAML del entorno existente. No se construye ni ejecuta Docker localmente; el build real de la imagen y el smoke test en contenedor se verifican en el runner de GitHub Actions.

Resultados locales de esta rama:

- `pip check`: sin dependencias rotas.
- `ruff check .`: aprobado.
- `ruff format --check .`: aprobado después de normalizar únicamente cinco archivos preexistentes con Ruff 0.17.1: `mibici_ml/api/errors.py`, `mibici_ml/modeling/cluster.py`, `mibici_ml/modeling/p3_model.py`, `mibici_ml/modeling/search_p3.py` y `mibici_ml/serving/export_pyfunc.py`. La comparación AST contra HEAD fue idéntica en los cinco y la suite completa pasó.
- Ruff format y lint focalizados en `scripts/ci_smoke_serving.py`, `tests/test_ci_smoke_serving.py` y `tests/test_ci_workflow.py`: aprobados.
- Pruebas focalizadas del script y el workflow: 10 aprobadas.
- Suite principal `pytest -q`: 360 aprobadas, 13 omitidas y 44 warnings.
- `bash scripts/run_security_checks.sh`: código 0; `pip check`, Bandit y 11 pruebas de higiene aprobados.
- Los casos DeepChecks están declarados en su job aislado; `.venv-quality` no está disponible en este entorno local y no se reporta aquí un resultado de su ejecución.

El YAML del workflow se valida con PyYAML en `tests/test_ci_workflow.py`; Ruff interpreta YAML como sintaxis Python y por eso sus comprobaciones focalizadas se limitan a los tres archivos Python nuevos.

## Evidencia remota

La evidencia remota se incorpora únicamente a partir de una ejecución real del workflow asociada a un commit.

| Evidencia | Registro esperado |
| --- | --- |
| Workflow completo verde | Vista del run real con todos los jobs completados. |
| Job `quality` | Resultado de Ruff format y lint. |
| Job `tests` | Resultado de la suite PyTest principal. |
| Job `security` | Salida del gate offline, sin ejecutar `pip-audit`. |
| Job `image-smoke` | Build desde `Containerfile` y salida del smoke test. |
| Readiness P2/P3 | Logs con respuestas satisfactorias de `/ready` y `/v1/p2/ready`. |

No se incluyen capturas ni resultados remotos antes de que exista una ejecución real.

Después del push se actualizará el README en un commit documental separado.
