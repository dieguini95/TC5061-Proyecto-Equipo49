import os
from pathlib import Path
import tempfile

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
def get_mlflow_train_output_base_dir() -> Path:
    """Return the configured parent directory for isolated training outputs."""
    configured_dir = os.environ.get("MLFLOW_TRAIN_OUTPUT_DIR")
    return Path(configured_dir) if configured_dir else Path(tempfile.gettempdir())


def create_mlflow_train_output_dir() -> Path:
    """Create a private, unique output directory for one P2/P3 invocation."""
    base_dir = get_mlflow_train_output_base_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="mibici-runs-", dir=base_dir))


# P1 uses its own output path; P2/P3 create a unique child at runtime.
MLFLOW_TRAIN_OUTPUT_DIR = get_mlflow_train_output_base_dir()

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

# Etapa 2: misma cohorte y ventanas que los baselines; sin parada aleatoria.
P1_FEATURE_COLUMNS = (
    "lag_24",
    "lag_48",
    "lag_168",
    "mean_4weeks",
    "hour",
    "day_of_week",
    "is_weekend",
)
P1_MODEL_THREADS = 4
P1_MODEL_DEFAULTS = {
    "ridge": {"alpha": 1.0},
    "hist_poisson": {
        "learning_rate": 0.08,
        "max_iter": 120,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 40,
    },
    "hist_squared": {
        "learning_rate": 0.08,
        "max_iter": 120,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 40,
    },
}
P1_CANDIDATE_GRID = {
    "ridge": [{"alpha": 1.0}],
    "hist_poisson": [{"max_leaf_nodes": 15}, {"max_leaf_nodes": 31}],
    "hist_squared": [{"max_leaf_nodes": 15}],
}
P1_REGISTRY_MODEL_NAME = "mibici-equipo49-p1-demanda"
P1_CHAMPION_ALIAS = "champion"
P1_STAGE2_OUTPUT_DIR = PROJ_ROOT / "work" / "p1-etapa2"
P1_STAGE2_SOURCE_VERSION = "p1-supervised-daily-origin-v2"

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
P2_REGISTRY_MODEL_NAME = "mibici-equipo49-p2-riesgo"  # nombre unico, como P3
P2_CHAMPION_ALIAS = "champion"
P2_CHALLENGER_ALIAS = "challenger"

# ---------------------------------------------------------------------------
# P3: clustering de estaciones por perfil de uso
# ---------------------------------------------------------------------------
P3_INPUT_PATH = TABLES_DIR / "estaciones_resumen.csv"
P3_BASELINE_K = 5
P3_CANDIDATE_K = (3, 4, 6, 7, 8)
P3_MODEL_FEATURES = ("capacidad", "salidas_12m", "llegadas_12m")
P3_PERIOD = "2025-10_to_2026-09"
P3_REGISTRY_MODEL_NAME = "mibici-equipo49-p3-grupos"
P3_CHAMPION_ALIAS = "champion"
P3_SEARCH_EXPERIMENT_NAME = EXPERIMENT_NAME
P3_N_INIT = 10

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
