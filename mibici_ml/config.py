import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
MLRUNS_DIR = PROJ_ROOT / "mlruns"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# ---------------------------------------------------------------------------
# Fuentes de datos (portal MiBici). Los enlaces de los CSV mensuales cambian
# de ID, por eso dataset.py los lee de la pagina de datos abiertos.
# ---------------------------------------------------------------------------
PORTAL_URL = "https://mibici.net/es/datos-abiertos/"
NOMENCLATURA_URL = "https://mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv"
# El portal responde 406 a clientes sin User-Agent de navegador.
HTTP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Ventana de datos y particion temporal (sin particion aleatoria)
# Valores fijados en la propuesta (02_investigacion/03_propuestas_solucion).
# ---------------------------------------------------------------------------
START_MONTH = "2025-10"  # primer mes descargado (inclusive)
END_MONTH = "2026-09"  # ultimo mes descargado (inclusive)

TRAIN_START = "2025-10-01"
TRAIN_END = "2026-06-30"  # entrenamiento: oct-25 a jun-26
VAL_START = "2026-07-01"
VAL_END = "2026-08-31"  # validacion: jul-ago-26
TEST_START = "2026-09-01"
TEST_END = "2026-09-30"  # prueba: sep-26, se usa una sola vez

# ---------------------------------------------------------------------------
# Reproducibilidad y seguimiento de experimentos
# ---------------------------------------------------------------------------
# PROVISIONAL: ajustar tras los primeros runs de MLflow (paso 6).
RANDOM_SEED = 42
EXPERIMENT_NAME = "mibici-equipo49"
# Backend SQLite local. Se puede sobrescribir con la variable de entorno
# MLFLOW_TRACKING_URI (p. ej. apuntando a un servidor `mlflow server`).
MLFLOW_DB_PATH = PROJ_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)

# Validacion cruzada dentro de entrenamiento (sin shuffle).
CV_N_SPLITS = 5

# ---------------------------------------------------------------------------
# P1: regresion de salidas por estacion-hora
# ---------------------------------------------------------------------------
# PROVISIONAL: solo rezagos de 24 h o mas (no hay fuga del dia actual).
P1_LAG_HOURS = (24, 48, 168)
P1_PURGE_GAP_HOURS = 24  # hueco entre pliegues de validacion cruzada
P1_BASELINE_SEASONAL_LAG_HOURS = 168  # estacional-ingenuo: misma hora de la semana pasada
P1_BASELINE_MEAN_WEEKS = 4  # media de las ultimas 4 semanas, mismo dia y hora

# P1: artefactos y contrato de la entrega local. Los datos completos siguen fuera de Git.
REFERENCES_DIR = PROJ_ROOT / "references"
DATA_MANIFEST_PATH = REFERENCES_DIR / "data_manifest.json"
TRIPS_PROCESSED_DIR = PROCESSED_DATA_DIR / "trips"
P1_PANEL_PATH = PROCESSED_DATA_DIR / "p1_station_hour.parquet"
P1_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "p1_predictions.parquet"
P1_METRICS_PATH = TABLES_DIR / "p1_metrics.csv"
P1_QUALITY_PATH = TABLES_DIR / "p1_quality.json"
P1_RUNS_PATH = TABLES_DIR / "p1_runs.json"
P1_STATION_METRICS_PATH = TABLES_DIR / "p1_station_metrics.csv"
P1_FIGURES = {
    "comparison": FIGURES_DIR / "P1_error_comparison.png",
    "daily": FIGURES_DIR / "P1_daily_totals.png",
    "station": FIGURES_DIR / "P1_station_example.png",
}
P1_MODEL_NAMES = ("seasonal_168h", "mean_4weeks")
P1_FORECAST_HORIZON_HOURS = 24
P1_FORECAST_ORIGIN_HOUR = 0
P1_TIMEZONE = "America/Mexico_City"
P1_PRIMARY_METRIC = "wape"
P1_SOURCE_VERSION = "p1-station-hour-v1"

# ---------------------------------------------------------------------------
# P2: clasificacion de riesgo (estacion vacia o llena en las siguientes H horas)
# ---------------------------------------------------------------------------
P2_HORIZON_HOURS = 2  # H
# Inventario aproximado solo con viajes: el inicio se ubica en el punto medio
# de su rango factible y solo se usa para construir etiquetas.
P2_INVENTORY_START = "midpoint"
P2_MIN_PRECISION = 0.5  # metrica: recall cuando la precision es al menos este valor
# PROVISIONAL: fraccion de estaciones-hora mas riesgosas para precision@k.
P2_TOP_K_FRACTION = 0.10
# P2 consume los Parquet limpios de dataset.py (mismo contrato que P1).
P2_STATIONS_PATH = PROCESSED_DATA_DIR / "stations.parquet"
P2_PANEL_PATH = PROCESSED_DATA_DIR / "p2_station_hour.parquet"
P2_METRICS_PATH = TABLES_DIR / "p2_metrics.csv"
P2_RUNS_PATH = TABLES_DIR / "p2_runs.json"
P2_FIGURES = {"pr_curve": FIGURES_DIR / "P2_pr_curve.png"}
# Grid de busqueda anidada (solo validacion). PROVISIONAL: ajustar con el equipo.
P2_SEARCH_GRID = {
    # C casi no cambia el PR-AUC en Etapa 1 (0.0123 en los tres valores); se mantiene
    # solo para documentar que se probo y se reemplaza el eje por penalizacion L1/L2.
    "logistica": {"C": (0.1, 1.0), "penalty": ("l2", "l1")},
    "hist_gb": {
        "learning_rate": (0.03, 0.05),
        "max_iter": (300, 600),
        "max_leaf_nodes": (15, 31),
        "min_samples_leaf": (20, 100),
    },
    "extra_trees": {
        "n_estimators": (300,),
        "max_depth": (None, 12),
        "min_samples_leaf": (20, 100),
    },
}
P2_REGISTERED_MODEL_PREFIX = "mibici-equipo49-p2-riesgo"

# ---------------------------------------------------------------------------
# P3: clustering de estaciones por perfil de uso
# ---------------------------------------------------------------------------
# PROVISIONAL: rango de K a comparar con silueta, Davies-Bouldin y Calinski-Harabasz.
P3_K_RANGE = tuple(range(2, 9))
P3_PCA_VARIANCE = 0.90  # varianza explicada minima antes de K-Means
P3_STABILITY_RUNS = 10  # corridas con semillas distintas para medir ARI

# ---------------------------------------------------------------------------
# Logging: si tqdm esta instalado, loguru escribe con tqdm.write
# https://github.com/Delgan/loguru/issues/135
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except (ModuleNotFoundError, ValueError):
    pass
