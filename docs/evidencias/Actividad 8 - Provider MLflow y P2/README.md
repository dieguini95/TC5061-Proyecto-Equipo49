# Actividad 8: evidencia del contenedor (P3 desde el Registry y P2)

Imagen `mibici-serving:latest` construida con Docker 29.8 desde el `Containerfile` del repositorio y ejecutada con `docker run -p 8000:8000`. El contenedor queda en estado `healthy` y corre como UID 10001.

## Archivos

| Archivo | Contenido |
|---|---|
| `01_api_contenedor_p3_p2.png` | Captura de la terminal con los comandos y las respuestas reales del contenedor. |
| `01_transcript_contenedor.txt` | Texto plano de esa misma sesión, para revisar sin abrir la imagen. |
| `02_pytest_provider_p2_export.png` | Salida de `pytest -v` de las pruebas del provider MLflow, del router de P2 y de la exportación de P3 (28 pasan). |
| `02_transcript_pytest.txt` | Texto plano de esa ejecución. |

## Qué muestra

1. `docker ps` con el contenedor `healthy` y el puerto 8000 publicado.
2. `docker exec ... id`: usuario `app` con UID 10001.
3. `GET /ready` y `GET /model` de P3: modelo `mibici-equipo49-p3-grupos`, alias `champion`, versión 1.
4. `POST /predict` y `POST /predict/batch` de P3 con respuestas `cluster`.
5. `POST /predict` de P3 con `capacidad` negativa: HTTP 422.
6. `GET /v1/p2/model` y `POST /v1/p2/predict` de P2: `riesgo_prob` 0.0005925946619037167, igual a la predicción del modelo local.
7. `POST /v1/p2/predict` con `hora` 25: HTTP 422.

## Nota sobre el modelo servido

Las predicciones de P3 provienen del champion real `models:/mibici-equipo49-p3-grupos@champion` (KMeans con k=8, seleccionado por silhouette). El modelo se exporta a `models/serving/p3/model` con `make serving-export QUESTION=p3`, que verifica que las predicciones coincidan con las del Registry antes de empaquetar.
