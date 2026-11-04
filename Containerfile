# Containerfile del servicio de inferencia de MiBici (Actividad 8).
# Compatible con Podman y Docker (estandar OCI). El modelo se empaqueta desde
# models/serving/<pregunta>, que genera `make serving-export`.
#
#   podman build -f Containerfile --build-arg QUESTION=p2 -t mibici-p2 .
#   podman run --rm -p 8000:8000 mibici-p2
#
# Con Docker: docker build -f Containerfile --build-arg QUESTION=p2 -t mibici-p2 .

FROM python:3.12-slim

ARG QUESTION=p2

LABEL org.opencontainers.image.title="mibici-serving" \
      org.opencontainers.image.description="API de inferencia MiBici (FastAPI) con modelo del Registry"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    QUESTION=${QUESTION} \
    MIBICI_SERVING_ROOT=/app/models/serving

WORKDIR /app

# Usuario sin privilegios para ejecutar el servicio.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

# Dependencias primero: esta capa se reutiliza mientras no cambie requirements-serving.txt.
COPY requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY mibici_ml ./mibici_ml
COPY models/serving/${QUESTION} ./models/serving/${QUESTION}

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["sh", "-c", "exec uvicorn mibici_ml.serving.apps.${QUESTION}:app --host 0.0.0.0 --port 8000"]
