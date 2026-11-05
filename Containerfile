# Containerfile de la API de inferencia de MiBici (Actividad 8).
# Una sola imagen sirve P3 (pyfunc exportado del Registry) y P2 (bundle skops)
# con la misma aplicacion FastAPI: mibici_ml.api.app:app.
# Compatible con Podman y Docker (estandar OCI).
#
#   make serving-export-all                       # genera models/serving/{p2,p3}
#   podman build -f Containerfile -t mibici-serving .
#   podman run --rm -p 8000:8000 mibici-serving
#
# Con Docker: docker build -f Containerfile -t mibici-serving .

FROM python:3.12-slim

LABEL org.opencontainers.image.title="mibici-serving" \
      org.opencontainers.image.description="API de inferencia MiBici (FastAPI): P3 desde el Registry y P2 como router"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    MIBICI_API_PROVIDER=mlflow \
    MIBICI_MODEL_PATH=/app/models/serving/p3/model \
    MIBICI_P2_BUNDLE_DIR=/app/models/serving/p2

WORKDIR /app

# Usuario sin privilegios para ejecutar el servicio.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

# Dependencias primero: esta capa se reutiliza mientras no cambie requirements-serving.txt.
COPY requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY mibici_ml ./mibici_ml
COPY models/serving ./models/serving

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["uvicorn", "mibici_ml.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
