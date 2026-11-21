"""Pruebas de la Actividad 9: contrato de Compose, carga de Locust, monitoreo y FinOps.

Son pruebas offline y deterministas: no levantan contenedores ni llaman a la API.
La validación de `docker compose config` se omite si Docker no está instalado.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shutil
import subprocess

from load_tests.payloads import (
    DEFAULT_SEED,
    P2_BATCH_SIZE,
    P3_STATION_COUNT,
    make_rng,
    p2_batch,
    p2_record,
    p3_station,
)
import pytest
from scripts import finops_analysis as finops
from scripts import monitor_docker_stats as monitor
from scripts import run_a9_scenarios as escenarios
import yaml

from mibici_ml.api.p2 import LoteP2, RegistroP2
from mibici_ml.api.schemas import StationInput

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE = REPO_ROOT / "docker-compose.yml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
CONTAINERFILE_MLFLOW = REPO_ROOT / "Containerfile.mlflow"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
REQUIREMENTS_LOADTEST = REPO_ROOT / "requirements-loadtest.txt"
TABLES = REPO_ROOT / "reports" / "tables"
FIGURAS = REPO_ROOT / "docs" / "evidencias" / "Actividad 9 - Operacion y FinOps" / "figuras"


def _compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- Compose y contenedores


def test_compose_declares_api_mlflow_and_optional_loadtest_profile():
    servicios = _compose()["services"]

    assert set(servicios) == {"api", "mlflow", "locust"}
    assert servicios["locust"]["profiles"] == ["loadtest"]
    assert "profiles" not in servicios["api"]
    assert "profiles" not in servicios["mlflow"]
    assert servicios["locust"]["depends_on"]["api"]["condition"] == "service_healthy"


def test_compose_services_define_healthchecks():
    servicios = _compose()["services"]

    for nombre in ("api", "mlflow"):
        salud = servicios[nombre]["healthcheck"]
        assert salud["test"][0] == "CMD", nombre
        assert int(salud["retries"]) >= 1, nombre


def test_compose_publishes_ports_only_on_loopback():
    servicios = _compose()["services"]

    for nombre in ("api", "mlflow"):
        for puerto in servicios[nombre]["ports"]:
            assert puerto.startswith("127.0.0.1:"), f"{nombre} publica {puerto} fuera de loopback"


def test_compose_sets_cpu_and_memory_limits_for_long_running_services():
    servicios = _compose()["services"]

    for nombre in ("api", "mlflow"):
        limites = servicios[nombre]["deploy"]["resources"]["limits"]
        assert "cpus" in limites, nombre
        assert "memory" in limites, nombre


def test_compose_does_not_request_privileges_or_host_network():
    texto = COMPOSE.read_text(encoding="utf-8")

    assert "privileged" not in texto
    assert "network_mode: host" not in texto


def test_compose_and_env_example_contain_no_credentials():
    for ruta in (COMPOSE, ENV_EXAMPLE):
        texto = ruta.read_text(encoding="utf-8")
        assert "GH_TOKEN" not in texto, ruta.name
        assert not re.search(r"ghp_[A-Za-z0-9]{20,}|github_pat_", texto), ruta.name


def test_env_example_declares_every_variable_used_by_compose():
    usadas = set(re.findall(r"\$\{([A-Z_]+)(?::-[^}]*)?\}", COMPOSE.read_text(encoding="utf-8")))
    definidas = {
        linea.split("=", 1)[0]
        for linea in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
        if "=" in linea and not linea.lstrip().startswith("#")
    }

    assert usadas <= definidas, f"Faltan en .env.example: {sorted(usadas - definidas)}"


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker no está instalado")
def test_compose_file_is_valid_for_docker_compose():
    resultado = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "config", "--quiet"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert resultado.returncode == 0, resultado.stderr


def test_mlflow_image_pins_the_version_used_by_requirements():
    requisitos = re.search(
        r"^mlflow==([\w.]+)$", REQUIREMENTS.read_text(encoding="utf-8"), re.MULTILINE
    )
    imagen = re.search(r'"mlflow==([\w.]+)"', CONTAINERFILE_MLFLOW.read_text(encoding="utf-8"))

    assert requisitos is not None
    assert imagen is not None
    assert requisitos.group(1) == imagen.group(1) == "3.17.1"


def test_mlflow_image_runs_single_worker_without_job_execution_as_non_root():
    texto = CONTAINERFILE_MLFLOW.read_text(encoding="utf-8")

    assert "MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false" in texto
    assert '"--workers", "1"' in texto
    assert "USER 10001:10001" in texto


def test_loadtest_requirements_pin_locust_version():
    lineas = [
        linea.strip()
        for linea in REQUIREMENTS_LOADTEST.read_text(encoding="utf-8").splitlines()
        if linea.strip() and not linea.lstrip().startswith("#")
    ]

    assert lineas == ["locust==2.47.0"]


# --------------------------------------------------------------------------- payloads de Locust


def test_p2_records_satisfy_the_serving_contract():
    rng = make_rng(DEFAULT_SEED, 0)

    for _ in range(200):
        RegistroP2.model_validate(p2_record(rng))


def test_p2_batches_have_the_configured_size_and_validate():
    lote = p2_batch(make_rng(DEFAULT_SEED, 1), P2_BATCH_SIZE)

    assert len(lote["items"]) == P2_BATCH_SIZE
    LoteP2.model_validate(lote)


def test_p3_stations_validate_and_cover_the_catalog():
    rng = make_rng(DEFAULT_SEED, 2)
    estaciones = set()

    for _ in range(2000):
        estacion = p3_station(rng)
        StationInput.model_validate(estacion)
        estaciones.add(estacion["station_id"])

    assert min(estaciones) >= 1
    assert max(estaciones) <= P3_STATION_COUNT
    # 2,000 muestras uniformes sobre 484 estaciones cubren casi todo el catálogo.
    assert len(estaciones) > 400


def test_payload_generation_is_deterministic_per_virtual_user():
    assert p2_record(make_rng(DEFAULT_SEED, 3)) == p2_record(make_rng(DEFAULT_SEED, 3))
    assert p2_record(make_rng(DEFAULT_SEED, 3)) != p2_record(make_rng(DEFAULT_SEED, 4))


# --------------------------------------------------------------------------- monitoreo


def test_monitor_converts_docker_stats_units_to_numbers():
    registro = {
        "Name": "mibici-api",
        "CPUPerc": "172.40%",
        "MemUsage": "214.7MiB / 1GiB",
        "MemPerc": "20.97%",
        "NetIO": "1.5kB / 2.3MB",
        "BlockIO": "1.93MB / 29kB",
        "PIDs": "23",
    }

    fila = monitor.parse_stats_record(registro)

    assert fila["contenedor"] == "mibici-api"
    assert fila["cpu_pct"] == pytest.approx(172.4)
    assert fila["mem_mib"] == pytest.approx(214.7, abs=0.01)
    assert fila["mem_limite_mib"] == pytest.approx(1024.0)
    assert fila["mem_pct"] == pytest.approx(20.97)
    # El monitor redondea a 3 decimales en MB: 1.5 kB se guarda como 0.002 MB.
    assert fila["red_rx_mb"] == pytest.approx(0.0015, abs=0.001)
    assert fila["red_tx_mb"] == pytest.approx(2.3)
    assert fila["disco_lectura_mb"] == pytest.approx(1.93)
    assert fila["disco_escritura_mb"] == pytest.approx(0.029)
    assert fila["pids"] == 23


def test_monitor_extracts_json_records_from_ansi_redraw_stream():
    # Sin TTY, docker stats redibuja con secuencias ANSI; los registros deben extraerse igual.
    flujo = (
        '\x1b[H\x1b[2J{"Name":"mibici-api","CPUPerc":"1.00%"}\x1b[K\n'
        '{"Name":"mibici-mlflow","CPUPerc":"0.50%"}\x1b[K'
    )

    nombres = [json.loads(objeto)["Name"] for objeto in monitor._JSON_OBJECT_RE.findall(flujo)]

    assert nombres == ["mibici-api", "mibici-mlflow"]


def test_monitor_command_streams_json_for_both_containers():
    comando = monitor.build_docker_stats_command(["mibici-api", "mibici-mlflow"])

    assert comando == [
        "docker",
        "stats",
        "--format",
        "{{json .}}",
        "mibici-api",
        "mibici-mlflow",
    ]


# --------------------------------------------------------------------------- escenarios de Locust


def test_scenarios_match_the_experiment_plan():
    assert [e.nombre for e in escenarios.ESCENARIOS] == ["S1", "S2", "S3", "S4a", "S4b"]
    assert [e.nombre for e in escenarios.ESCENARIOS_OPCIONALES] == ["S5"]

    s4b = escenarios.seleccionar("S4b")[0]

    assert (s4b.user_class, s4b.usuarios, s4b.duracion_s) == ("P2BatchUser", 20, 60)


def test_measurement_window_starts_after_the_ramp_up():
    inicio = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)

    ventana = escenarios.ventana_medicion(escenarios.seleccionar("S2")[0], inicio, 2.0)

    assert ventana["rampa_s"] == 8  # 40 usuarios a 5 por segundo
    assert ventana["medicion_s"] == 82
    assert ventana["inicio_medicion_utc"] == "2026-11-21T12:00:08+00:00"
    assert ventana["fin_medicion_utc"] == "2026-11-21T12:01:30+00:00"


def test_locust_command_runs_headless_with_reset_and_csv_outputs():
    comando = escenarios.comando_locust(escenarios.seleccionar("S4b")[0], "http://127.0.0.1:8000")

    assert comando[comando.index("-m") + 1] == "locust"
    for bandera in ("--headless", "--reset-stats", "--csv", "--html", "--only-summary"):
        assert bandera in comando, bandera
    assert "P2BatchUser" in comando
    assert comando[comando.index("-u") + 1] == "20"
    assert comando[comando.index("-t") + 1] == "60s"


def test_replica_suffix_separates_output_files():
    comando = escenarios.comando_monitor(escenarios.seleccionar("S5")[0], "_r2")

    assert comando[comando.index("--output") + 1].endswith("a9_docker_stats_S5_r2.csv")
    assert comando[comando.index("--duration") + 1] == str(60 + escenarios.MARGEN_MONITOR_S)


# --------------------------------------------------------------------------- FinOps


def test_finops_constants_track_the_load_test_and_the_catalog():
    assert finops.PREDICCIONES_POR_LOTE == P2_BATCH_SIZE
    assert finops.ESTACIONES == P3_STATION_COUNT == 484


def test_capacity_cost_is_hourly_price_per_prediction_scaled_to_millions():
    # 0.0209 USD/h entre 3,600 s * 1,000 predicciones/s * 0.70 de utilización.
    costo = finops.costo_capacidad_por_millon(0.0209, 1000, 0.70)

    assert costo == pytest.approx(0.0209 / (3600 * 1000 * 0.70) * 1_000_000)
    assert costo == pytest.approx(0.0082937, rel=1e-4)


def test_always_on_cost_spreads_the_hourly_price_over_daily_demand():
    assert finops.demanda_diaria("horario") == 11_616  # 484 estaciones x 24 cortes
    assert finops.demanda_diaria("cada_5_min") == 139_392  # 484 estaciones x 288 cortes
    assert finops.costo_siempre_encendido_por_millon(0.0850, 11_616) == pytest.approx(
        0.0850 * 24 / 11_616 * 1_000_000
    )


def test_scheduled_fargate_bills_the_one_minute_minimum_per_run():
    por_corrida = 60 * (1 * finops.FARGATE_USD_VCPU_S + 2 * finops.FARGATE_USD_GB_S)

    assert finops.costo_corrida_programada(1, 2, 0.27) == pytest.approx(por_corrida)
    assert finops.costo_programado_por_millon(1, 2, 24, 0.27) == pytest.approx(
        por_corrida / 484 * 1_000_000
    )
    assert finops.costo_programado_por_millon(1, 2, 24, 0.27) == pytest.approx(1.70, abs=0.01)


def test_fargate_hourly_prices_follow_the_per_second_rates():
    assert finops.precio_fargate_h(1, 2) == pytest.approx(0.049370, abs=1e-6)
    assert finops.precio_fargate_h(2, 4) == pytest.approx(0.098741, abs=1e-6)
    assert finops.precio_fargate_h(0.5, 1) == pytest.approx(0.024685, abs=1e-6)


def test_t3_unlimited_charges_only_the_excess_over_its_baseline():
    # 70% de 2 vCPU = 1.4 vCPU; la línea base de t3.small es 20% por vCPU = 0.4 vCPU.
    assert finops.costo_t3_unlimited_h(0.0209, 0.70, 2) == pytest.approx(
        0.0209 + (1.4 - 0.4) * 0.05
    )
    assert finops.costo_t3_unlimited_h(0.0209, 0.10, 2) == pytest.approx(0.0209)


def test_percentile_uses_linear_interpolation():
    assert finops.percentil([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert finops.percentil([10], 95) == 10


def test_predictions_per_request_follow_the_endpoint_name():
    assert finops.predicciones_por_solicitud("POST /v1/p2/predict/batch (lote de 20)") == 20
    assert finops.predicciones_por_solicitud("POST /v1/p2/predict") == 1
    assert finops.predicciones_por_solicitud("POST /predict") == 1
    assert finops.predicciones_por_solicitud("GET /health") == 0


def test_serverless_cost_per_million_requests_matches_the_formula():
    # 16 ms x 1 GB x tarifa por GB-s, más la tarifa por solicitud.
    assert finops.costo_lambda_por_solicitud() * 1_000_000 == pytest.approx(0.4667, abs=1e-3)


def test_generated_finops_tables_and_figures_exist():
    for nombre in (
        "a9_finops_rendimiento",
        "a9_finops_recursos",
        "a9_finops_capacidad",
        "a9_finops_costos",
        "a9_finops_sensibilidad",
        "a9_finops_optimizaciones",
    ):
        assert (TABLES / f"{nombre}.csv").is_file(), nombre
        assert (TABLES / f"{nombre}.json").is_file(), nombre
    for figura in (
        "01_latencia_y_rendimiento.png",
        "02_serie_temporal_S3.png",
        "03_recursos_api_mlflow.png",
        "04_finops_costo_y_optimizaciones.png",
    ):
        assert (FIGURAS / figura).stat().st_size > 50_000, figura


def test_measured_runs_have_no_failures_and_batching_multiplies_throughput():
    filas = json.loads((TABLES / "a9_finops_rendimiento.json").read_text(encoding="utf-8"))
    por_etiqueta = {fila["etiqueta"]: fila for fila in filas}

    assert all(fila["fallas"] == 0 for fila in filas)
    assert por_etiqueta["S4b_r1"]["predicciones_s"] > 10 * por_etiqueta["S4a"]["predicciones_s"]


def test_optimizations_report_savings_consistent_with_their_costs():
    optimizaciones = json.loads(
        (TABLES / "a9_finops_optimizaciones.json").read_text(encoding="utf-8")
    )

    assert [o["id"] for o in optimizaciones] == ["O1", "O2", "O3", "O4"]
    for optimizacion in optimizaciones:
        base = optimizacion["usd_por_1M_base"]
        optimizada = optimizacion["usd_por_1M_optimizada"]
        assert optimizada < base, optimizacion["id"]
        assert optimizacion["factor_mejora"] == pytest.approx(base / optimizada, rel=1e-3)
        assert optimizacion["ahorro_pct"] == pytest.approx(100 * (1 - optimizada / base), rel=1e-3)
