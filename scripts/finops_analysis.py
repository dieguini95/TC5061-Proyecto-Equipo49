"""Análisis FinOps de la Actividad 9 a partir de mediciones reales de Compose y Locust.

Lee reports/tables/a9_* (Locust, docker stats y ventanas de medición) y calcula:
  - rendimiento por escenario y predicciones por segundo;
  - CPU, memoria, red y disco por contenedor dentro de cada ventana de medición;
  - costo por inferencia, por 1,000 y por 1,000,000 en instancias de referencia;
  - sensibilidad a la utilización y al factor de capacidad;
  - cuatro optimizaciones cuantificadas con los datos medidos.

Escribe tablas CSV/JSON en reports/tables/a9_finops_*.csv|json y cuatro gráficas PNG en
docs/evidencias/Actividad 9 - Operacion y FinOps/figuras/.

Uso (desde la raíz del repositorio):
    .venv-a9/bin/python scripts/finops_analysis.py
"""

from __future__ import annotations

from collections.abc import Sequence
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURAS = ROOT / "docs" / "evidencias" / "Actividad 9 - Operacion y FinOps" / "figuras"
CONTENEDORES = ("mibici-api", "mibici-mlflow")
ETIQUETAS = ("S1", "S2", "S3", "S4a", "S4b_r1", "S4b_r2", "S5_r1", "S5_r2")
LIMITE_API_DEFECTO = 2.0  # corridas anteriores a S5 no registran limite_cpus_api

# Constantes del experimento. PREDICCIONES_POR_LOTE y ESTACIONES se verifican en tests.
PREDICCIONES_POR_LOTE = 20  # igual a P2_BATCH_SIZE en load_tests/payloads.py
ESTACIONES = 484  # igual a P3_STATION_COUNT; references/nomenclatura_2026_09.csv
UTILIZACION_CAPACIDAD = 0.70
UTILIZACIONES_SENSIBILIDAD = (0.10, 0.30, 0.50, 0.70, 1.00)
FACTORES_CAPACIDAD = (0.5, 1.0, 1.5)
SEGUNDOS_POR_HORA = 3600
MINUTO_MINIMO_FACTURABLE_S = 60  # Fargate factura al menos 1 minuto por tarea
CORTES_POR_DIA = {"horario": 24, "cada_5_min": 288}
DURACION_INFERENCIA_S = 0.016  # p50 de S1 con carga baja; supuesto para cómputo serverless

# Tarifas públicas de referencia (us-east-1, Linux). Verificar en la página oficial antes de publicar.
PRECIO_EC2_USD_H = {"t3.small": 0.0209, "c6i.large": 0.0850}
T3_BASELINE_POR_VCPU = 0.20  # línea base de CPU sostenida por vCPU en t3.small
T3_UNLIMITED_USD_POR_VCPU_H = 0.05  # cargo por vCPU-hora sobre la línea base (modo unlimited)
FARGATE_USD_VCPU_S = 0.000011244
FARGATE_USD_GB_S = 0.000001235
LAMBDA_USD_GB_S = 0.0000166667
LAMBDA_USD_POR_SOLICITUD = 0.20 / 1_000_000
CLOUD_RUN_USD_VCPU_S = 0.000024
CLOUD_RUN_USD_GIB_S = 0.0000025
CLOUD_RUN_USD_POR_SOLICITUD = 0.40 / 1_000_000

NOMBRE_T3 = "EC2 t3.small (2 vCPU, 2 GiB)"
NOMBRE_C6I = "EC2 c6i.large (2 vCPU, 4 GiB)"
NOMBRE_F_2_4 = "Fargate 2 vCPU / 4 GB"
NOMBRE_F_1_2 = "Fargate 1 vCPU / 2 GB"
NOMBRE_F_05_1 = "Fargate 0.5 vCPU / 1 GB (extrapolada)"
NOMBRE_LAMBDA = "AWS Lambda 1 GB"
NOMBRE_CLOUD_RUN = "Cloud Run 1 vCPU / 512 MiB"

MODO_LOTE = "capacidad 70% (lote de 20)"
MODO_UNITARIO = "capacidad 70% (unitaria)"
MODO_HORARIO = "demanda horaria, siempre encendida"
MODO_5MIN = "demanda cada 5 min, siempre encendida"
MODO_UNLIMITED = "capacidad 70% (lote), t3 modo unlimited"
MODO_PROG_HORA = "programado, 1 corrida por hora"
MODO_PROG_5MIN = "programado, cada 5 min"
MODO_SERVERLESS = "por solicitud, duración supuesta 16 ms"

LIMITACIONES = [
    (
        "Capacidad medida en Apple M3 Pro (arm64) con Docker Desktop; las instancias cloud son "
        "x86 y no se midieron."
    ),
    "Locust corre en la misma máquina que la API y compite por CPU.",
    (
        "t3.small tiene línea base de 20% por vCPU; la carga sostenida consume créditos o paga "
        "modo unlimited (fila específica en a9_finops_costos)."
    ),
    "Fargate 0.5 vCPU / 1 GB es extrapolación lineal; no se midió.",
    (
        "Lambda y Cloud Run usan 16 ms por solicitud (p50 de S1 con carga baja); no se midió en "
        "la plataforma ni incluye arranque en frío."
    ),
    (
        "No se incluye costo de almacenamiento, transferencia de red ni del contenedor MLflow en "
        "el costo por inferencia."
    ),
    "El plan inicial estimaba ~360 estaciones; se usan 484 (catálogo 2026-09).",
]


@dataclass(frozen=True)
class Instancia:
    nombre: str
    vcpu: float
    gib: float
    precio_h: float
    pred_s_lote: float | None
    pred_s_unitaria: float | None
    origen_capacidad: str


# --------------------------------------------------------------------------- utilidades


def predicciones_por_solicitud(nombre: str) -> int:
    """Predicciones que produce una petición de Locust según su nombre de endpoint."""
    if nombre.startswith("POST /v1/p2/predict/batch"):
        return PREDICCIONES_POR_LOTE
    if nombre in ("POST /v1/p2/predict", "POST /predict"):
        return 1
    return 0


def percentil(valores: Sequence[float], p: float) -> float:
    """Percentil por interpolación lineal (igual al método por defecto de numpy)."""
    if not valores:
        return math.nan
    ordenados = sorted(valores)
    k = (len(ordenados) - 1) * p / 100
    inferior = math.floor(k)
    superior = math.ceil(k)
    return ordenados[inferior] + (ordenados[superior] - ordenados[inferior]) * (k - inferior)


def dispersion_pct(valores: Sequence[float]) -> float:
    """Rango relativo entre réplicas: (máx - mín) / media, en porcentaje."""
    return 100 * (max(valores) - min(valores)) / mean(valores) if len(valores) > 1 else 0.0


def _float(valor: str) -> float:
    return 0.0 if valor in ("", "N/A") else float(valor)


def _leer_csv(nombre: str) -> list[dict[str, str]]:
    with (TABLES / nombre).open(encoding="utf-8", newline="") as archivo:
        return list(csv.DictReader(archivo))


def _leer_json(nombre: str) -> dict[str, Any]:
    return json.loads((TABLES / nombre).read_text(encoding="utf-8"))


def _limpiar(valor: Any) -> Any:
    if isinstance(valor, float):
        if math.isnan(valor) or math.isinf(valor):
            return None
        return round(valor, 6)
    return valor


def escribir_tabla(nombre: str, filas: list[dict[str, Any]]) -> None:
    """Escribe la misma tabla en CSV y JSON dentro de reports/tables/."""
    limpias = [{clave: _limpiar(valor) for clave, valor in fila.items()} for fila in filas]
    with (TABLES / f"{nombre}.csv").open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(limpias[0].keys()))
        escritor.writeheader()
        escritor.writerows(limpias)
    texto = json.dumps(limpias, indent=2, ensure_ascii=False) + "\n"
    (TABLES / f"{nombre}.json").write_text(texto, encoding="utf-8")


# --------------------------------------------------------------------------- mediciones


def resumen_locust(etiqueta: str) -> dict[str, Any]:
    """Métricas agregadas de Locust y predicciones por segundo de un escenario."""
    filas = _leer_csv(f"a9_locust_{etiqueta}_stats.csv")
    agregado = next(fila for fila in filas if fila["Name"] == "Aggregated")
    solicitudes = int(agregado["Request Count"])
    fallas = int(agregado["Failure Count"])
    exitosas = solicitudes - fallas
    predicciones = sum(
        (int(fila["Request Count"]) - int(fila["Failure Count"]))
        * predicciones_por_solicitud(fila["Name"])
        for fila in filas
        if fila["Name"] != "Aggregated"
    )
    por_solicitud = predicciones / exitosas if exitosas else 0.0
    rps = _float(agregado["Requests/s"])
    ventana = _leer_json(f"a9_ventanas_{etiqueta}.json")
    return {
        "etiqueta": etiqueta,
        "escenario": etiqueta.split("_")[0],
        "replica": etiqueta.split("_")[-1] if "_" in etiqueta else "única",
        "descripcion": ventana["descripcion"],
        "usuarios": ventana["usuarios"],
        "solicitudes": solicitudes,
        "fallas": fallas,
        "tasa_error_pct": 100 * fallas / solicitudes if solicitudes else 0.0,
        "rps": rps,
        "predicciones_por_solicitud": por_solicitud,
        "predicciones_s": rps * por_solicitud,
        "p50_ms": _float(agregado["50%"]),
        "p95_ms": _float(agregado["95%"]),
        "p99_ms": _float(agregado["99%"]),
        "promedio_ms": _float(agregado["Average Response Time"]),
    }


def _limites_cpu(ventana: dict[str, Any]) -> dict[str, float]:
    contenedores = _leer_json("a9_entorno.json")["contenedores"]
    limites = {nombre: float(datos["limite_cpus"]) for nombre, datos in contenedores.items()}
    if "limite_cpus_api" in ventana:
        limites["mibici-api"] = float(ventana["limite_cpus_api"])
    return limites


def recursos_ventana(etiqueta: str) -> list[dict[str, Any]]:
    """Estadísticas de cada contenedor dentro de la ventana de medición de Locust."""
    ventana = _leer_json(f"a9_ventanas_{etiqueta}.json")
    inicio = datetime.fromisoformat(ventana["inicio_medicion_utc"]).timestamp()
    fin = datetime.fromisoformat(ventana["fin_medicion_utc"]).timestamp()
    limites = _limites_cpu(ventana)
    filas = _leer_csv(f"a9_docker_stats_{etiqueta}.csv")
    resultado: list[dict[str, Any]] = []
    for contenedor in CONTENEDORES:
        muestras = [
            fila
            for fila in filas
            if fila["contenedor"] == contenedor and inicio <= _float(fila["epoch_s"]) <= fin
        ]
        limite = limites[contenedor]
        # docker stats reporta CPU como % de un núcleo; se normaliza al límite del contenedor.
        cpu = [_float(fila["cpu_pct"]) / limite for fila in muestras]
        memoria = [_float(fila["mem_mib"]) for fila in muestras]

        def delta(columna: str, datos: list[dict[str, str]] = muestras) -> float:
            # Red y disco son contadores acumulados desde el arranque del contenedor.
            if not datos:
                return 0.0
            return _float(datos[-1][columna]) - _float(datos[0][columna])

        resultado.append(
            {
                "etiqueta": etiqueta,
                "contenedor": contenedor,
                "limite_cpus": limite,
                "muestras": len(muestras),
                "cpu_media_pct_limite": mean(cpu) if cpu else math.nan,
                "cpu_p95_pct_limite": percentil(cpu, 95),
                "cpu_max_pct_limite": max(cpu, default=math.nan),
                "mem_media_mib": mean(memoria) if memoria else math.nan,
                "mem_max_mib": max(memoria, default=math.nan),
                "mem_max_pct_limite": max(
                    (_float(fila["mem_pct"]) for fila in muestras), default=math.nan
                ),
                "red_rx_mb": delta("red_rx_mb"),
                "red_tx_mb": delta("red_tx_mb"),
                "disco_lectura_mb": delta("disco_lectura_mb"),
                "disco_escritura_mb": delta("disco_escritura_mb"),
            }
        )
    return resultado


def anotar_cpu_por_prediccion(
    recursos: list[dict[str, Any]], rendimiento: list[dict[str, Any]]
) -> None:
    """Agrega núcleos medios y CPU-ms por predicción a las filas de la API (in place)."""
    predicciones = {fila["etiqueta"]: fila["predicciones_s"] for fila in rendimiento}
    for fila in recursos:
        nucleos = fila["cpu_media_pct_limite"] * fila["limite_cpus"] / 100
        fila["nucleos_medios"] = nucleos
        fila["cpu_ms_por_prediccion"] = (
            nucleos * 1000 / predicciones[fila["etiqueta"]]
            if fila["contenedor"] == "mibici-api" and predicciones[fila["etiqueta"]]
            else None
        )


def capacidad_medida() -> dict[str, dict[str, Any]]:
    """Predicciones por segundo medidas por configuración de la API (promedio de réplicas)."""
    unitaria = resumen_locust("S4a")
    lote_2 = [resumen_locust(e)["predicciones_s"] for e in ("S4b_r1", "S4b_r2")]
    lote_1 = [resumen_locust(e)["predicciones_s"] for e in ("S5_r1", "S5_r2")]
    return {
        "unitaria_2vcpu": {
            "predicciones_s": unitaria["predicciones_s"],
            "replicas": 1,
            "dispersion_pct": 0.0,
            "fuente": "S4a (API 2 vCPU)",
        },
        "lote_2vcpu": {
            "predicciones_s": mean(lote_2),
            "replicas": len(lote_2),
            "dispersion_pct": dispersion_pct(lote_2),
            "fuente": "S4b_r1, S4b_r2 (API 2 vCPU, lote de 20)",
        },
        "lote_1vcpu": {
            "predicciones_s": mean(lote_1),
            "replicas": len(lote_1),
            "dispersion_pct": dispersion_pct(lote_1),
            "fuente": "S5_r1, S5_r2 (API 1 vCPU, lote de 20)",
        },
    }


# --------------------------------------------------------------------------- modelo de costos


def precio_fargate_h(vcpu: float, gib: float) -> float:
    return SEGUNDOS_POR_HORA * (vcpu * FARGATE_USD_VCPU_S + gib * FARGATE_USD_GB_S)


def demanda_diaria(regimen: str) -> int:
    """Predicciones por día de MiBici: 484 estaciones por corte."""
    return ESTACIONES * CORTES_POR_DIA[regimen]


def costo_capacidad_por_millon(
    precio_h: float, pred_s: float, utilizacion: float = UTILIZACION_CAPACIDAD
) -> float:
    """USD por 1M predicciones con la instancia ocupada al `utilizacion` de su capacidad."""
    return precio_h / (SEGUNDOS_POR_HORA * pred_s * utilizacion) * 1_000_000


def costo_siempre_encendido_por_millon(precio_h: float, predicciones_dia: int) -> float:
    """USD por 1M predicciones cuando la instancia corre 24 h con la demanda dada."""
    return precio_h * 24 / predicciones_dia * 1_000_000


def costo_t3_unlimited_h(precio_h: float, utilizacion: float, vcpu: float) -> float:
    """Precio por hora de t3.small en carga sostenida con modo unlimited."""
    uso_vcpu = utilizacion * vcpu
    exceso_vcpu = max(0.0, uso_vcpu - T3_BASELINE_POR_VCPU * vcpu)
    return precio_h + exceso_vcpu * T3_UNLIMITED_USD_POR_VCPU_H


def costo_corrida_programada(vcpu: float, gib: float, duracion_activa_s: float) -> float:
    """USD de una tarea Fargate efímera, facturada con el mínimo de 1 minuto."""
    facturable = max(MINUTO_MINIMO_FACTURABLE_S, duracion_activa_s)
    return facturable * (vcpu * FARGATE_USD_VCPU_S + gib * FARGATE_USD_GB_S)


def costo_programado_por_millon(
    vcpu: float, gib: float, corridas_dia: int, duracion_activa_s: float
) -> float:
    """USD por 1M predicciones si cada corrida procesa las 484 estaciones."""
    costo_dia = corridas_dia * costo_corrida_programada(vcpu, gib, duracion_activa_s)
    return costo_dia / (corridas_dia * ESTACIONES) * 1_000_000


def costo_lambda_por_solicitud(
    memoria_gib: float = 1.0, duracion_s: float = DURACION_INFERENCIA_S
) -> float:
    return duracion_s * memoria_gib * LAMBDA_USD_GB_S + LAMBDA_USD_POR_SOLICITUD


def costo_cloud_run_por_solicitud(
    vcpu: float = 1.0, memoria_gib: float = 0.5, duracion_s: float = DURACION_INFERENCIA_S
) -> float:
    cómputo = duracion_s * (vcpu * CLOUD_RUN_USD_VCPU_S + memoria_gib * CLOUD_RUN_USD_GIB_S)
    return cómputo + CLOUD_RUN_USD_POR_SOLICITUD


def instancias(cap: dict[str, dict[str, Any]]) -> dict[str, Instancia]:
    unitaria_2 = cap["unitaria_2vcpu"]["predicciones_s"]
    lote_2 = cap["lote_2vcpu"]["predicciones_s"]
    lote_1 = cap["lote_1vcpu"]["predicciones_s"]
    return {
        "t3": Instancia(
            NOMBRE_T3, 2, 2, PRECIO_EC2_USD_H["t3.small"], lote_2, unitaria_2, "medida (2 vCPU)"
        ),
        "c6i": Instancia(
            NOMBRE_C6I, 2, 4, PRECIO_EC2_USD_H["c6i.large"], lote_2, unitaria_2, "medida (2 vCPU)"
        ),
        "f_2_4": Instancia(
            NOMBRE_F_2_4, 2, 4, precio_fargate_h(2, 4), lote_2, unitaria_2, "medida (2 vCPU)"
        ),
        "f_1_2": Instancia(
            NOMBRE_F_1_2, 1, 2, precio_fargate_h(1, 2), lote_1, None, "medida (1 vCPU, lote)"
        ),
        "f_05_1": Instancia(
            NOMBRE_F_05_1,
            0.5,
            1,
            precio_fargate_h(0.5, 1),
            lote_1 * 0.5,
            None,
            "extrapolada: 0.5 x medición con 1 vCPU (no medida)",
        ),
    }


def tabla_costos(inst: dict[str, Instancia]) -> list[dict[str, Any]]:
    """Costo por inferencia para cada instancia y régimen de uso."""
    filas: list[dict[str, Any]] = []

    def agregar(
        nombre: str,
        modo: str,
        precio_h: float | None,
        capacidad: float | None,
        usd_por_1m: float,
        notas: str,
    ) -> None:
        filas.append(
            {
                "instancia": nombre,
                "modo": modo,
                "precio_h_usd": precio_h,
                "predicciones_s_capacidad": capacidad,
                "usd_por_1000": usd_por_1m / 1000,
                "usd_por_1M": usd_por_1m,
                "notas": notas,
            }
        )

    horario = demanda_diaria("horario")
    cada_5 = demanda_diaria("cada_5_min")
    for clave in ("t3", "c6i", "f_2_4", "f_1_2", "f_05_1"):
        i = inst[clave]
        if i.pred_s_lote is not None:
            agregar(
                i.nombre,
                MODO_LOTE,
                i.precio_h,
                i.pred_s_lote,
                costo_capacidad_por_millon(i.precio_h, i.pred_s_lote),
                i.origen_capacidad,
            )
        if i.pred_s_unitaria is not None:
            agregar(
                i.nombre,
                MODO_UNITARIO,
                i.precio_h,
                i.pred_s_unitaria,
                costo_capacidad_por_millon(i.precio_h, i.pred_s_unitaria),
                i.origen_capacidad,
            )
        agregar(
            i.nombre,
            MODO_HORARIO,
            i.precio_h,
            None,
            costo_siempre_encendido_por_millon(i.precio_h, horario),
            "484 consultas por hora",
        )
        agregar(
            i.nombre,
            MODO_5MIN,
            i.precio_h,
            None,
            costo_siempre_encendido_por_millon(i.precio_h, cada_5),
            "484 consultas cada 5 min",
        )

    t3 = inst["t3"]
    precio_unlimited = costo_t3_unlimited_h(t3.precio_h, UTILIZACION_CAPACIDAD, t3.vcpu)
    agregar(
        t3.nombre,
        MODO_UNLIMITED,
        precio_unlimited,
        t3.pred_s_lote,
        costo_capacidad_por_millon(precio_unlimited, t3.pred_s_lote),
        "carga sostenida al 70%: consume línea base de 20% por vCPU y paga excedente",
    )

    f1 = inst["f_1_2"]
    duracion_activa = ESTACIONES / f1.pred_s_lote
    for modo, corridas in ((MODO_PROG_HORA, 24), (MODO_PROG_5MIN, 288)):
        agregar(
            f1.nombre,
            modo,
            None,
            None,
            costo_programado_por_millon(1, 2, corridas, duracion_activa),
            f"{corridas} tareas/día; activo {duracion_activa:.2f} s; mínimo facturable 60 s",
        )

    agregar(
        NOMBRE_LAMBDA,
        MODO_SERVERLESS,
        None,
        None,
        costo_lambda_por_solicitud() * 1_000_000,
        "tarifa x86 de referencia; sin arranque en frío ni carga del modelo",
    )
    agregar(
        NOMBRE_CLOUD_RUN,
        MODO_SERVERLESS,
        None,
        None,
        costo_cloud_run_por_solicitud() * 1_000_000,
        "tarifa request-based de referencia; duración supuesta",
    )
    return filas


def tabla_sensibilidad(inst: dict[str, Instancia]) -> list[dict[str, Any]]:
    """USD por 1M con lote de 20 según utilización y factor de capacidad medida."""
    filas: list[dict[str, Any]] = []
    for clave in ("t3", "c6i", "f_2_4", "f_1_2", "f_05_1"):
        i = inst[clave]
        for utilizacion in UTILIZACIONES_SENSIBILIDAD:
            for factor in FACTORES_CAPACIDAD:
                filas.append(
                    {
                        "instancia": i.nombre,
                        "utilizacion": utilizacion,
                        "factor_capacidad": factor,
                        "predicciones_s": i.pred_s_lote * factor,
                        "usd_por_1M": costo_capacidad_por_millon(
                            i.precio_h, i.pred_s_lote * factor, utilizacion
                        ),
                    }
                )
    return filas


def optimizaciones(
    inst: dict[str, Instancia], recursos: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Cuatro optimizaciones cuantificadas con datos medidos (antes y después)."""
    t3, c6i = inst["t3"], inst["c6i"]
    f2, f1 = inst["f_2_4"], inst["f_1_2"]
    demanda_horaria = demanda_diaria("horario")
    pico_api_mib = max(r["mem_max_mib"] for r in recursos if r["contenedor"] == "mibici-api")
    casos = [
        (
            "O1",
            "Inferencia por lotes de 20 estaciones",
            "EC2 t3.small, unitaria (S4a)",
            costo_capacidad_por_millon(t3.precio_h, t3.pred_s_unitaria),
            "EC2 t3.small, lote de 20 (S4b)",
            costo_capacidad_por_millon(t3.precio_h, t3.pred_s_lote),
            "S4a vs S4b_r1/r2 con 2 vCPU; mismo tipo de instancia",
            "La capacidad se midió en Apple M3 Pro; la nube puede diferir",
        ),
        (
            "O2",
            "Right-sizing de CPU: 2 a 1 vCPU con lote",
            "Fargate 2 vCPU / 4 GB, lote (S4b)",
            costo_capacidad_por_millon(f2.precio_h, f2.pred_s_lote),
            "Fargate 1 vCPU / 2 GB, lote (S5)",
            costo_capacidad_por_millon(f1.precio_h, f1.pred_s_lote),
            "S5 con 1 vCPU rindió igual o más que S4b con 2 vCPU",
            "Hipótesis no verificada: contención de hilos o GIL con 2 vCPU",
        ),
        (
            "O3",
            "Ejecución programada en lugar de siempre encendido",
            "Fargate 1 vCPU / 2 GB, 24 h encendido (demanda horaria)",
            costo_siempre_encendido_por_millon(f1.precio_h, demanda_horaria),
            "Fargate 1 vCPU / 2 GB, 1 tarea por hora",
            costo_programado_por_millon(1, 2, 24, ESTACIONES / f1.pred_s_lote),
            "Capacidad de 1 vCPU medida; mínimo facturable de 60 s",
            "No mide arranque en frío ni carga del modelo; supone 1 corrida por hora",
        ),
        (
            "O4",
            "Right-sizing de memoria: 4 GiB a 2 GiB",
            f"EC2 c6i.large (4 GiB), 24 h (demanda horaria); pico API {pico_api_mib:.0f} MiB",
            costo_siempre_encendido_por_millon(c6i.precio_h, demanda_horaria),
            "EC2 t3.small (2 GiB), 24 h (demanda horaria)",
            costo_siempre_encendido_por_millon(t3.precio_h, demanda_horaria),
            f"Pico de memoria medido de la API: {pico_api_mib:.0f} MiB (límite de compose 1 GiB)",
            "t3 tiene línea base de 20% por vCPU; con carga sostenida aplica modo unlimited",
        ),
    ]
    filas: list[dict[str, Any]] = []
    for oid, nombre, base_desc, base, opt_desc, opt, evidencia, limitacion in casos:
        filas.append(
            {
                "id": oid,
                "optimizacion": nombre,
                "base": base_desc,
                "usd_por_1M_base": base,
                "optimizada": opt_desc,
                "usd_por_1M_optimizada": opt,
                "ahorro_pct": 100 * (1 - opt / base),
                "factor_mejora": base / opt,
                "evidencia": evidencia,
                "limitacion": limitacion,
            }
        )
    return filas


# --------------------------------------------------------------------------- figuras


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False}
    )
    return plt


def figura_latencia(rendimiento: list[dict[str, Any]]) -> Path:
    plt = _pyplot()
    etiquetas = [fila["etiqueta"].replace("_", " ") for fila in rendimiento]
    x = list(range(len(etiquetas)))
    fig, (ax_lat, ax_rps) = plt.subplots(1, 2, figsize=(11, 4.4))
    ancho = 0.27
    series = (("p50_ms", "p50", -ancho), ("p95_ms", "p95", 0.0), ("p99_ms", "p99", ancho))
    for clave, nombre, desplazamiento in series:
        ax_lat.bar(
            [i + desplazamiento for i in x], [f[clave] for f in rendimiento], ancho, label=nombre
        )
    ax_lat.set_xticks(x, etiquetas, rotation=30, ha="right")
    ax_lat.set_ylabel("Latencia (ms)")
    ax_lat.set_title("Latencia por escenario")
    ax_lat.legend(frameon=False)

    barras = ax_rps.bar(x, [f["predicciones_s"] for f in rendimiento], color="#4C72B0")
    for barra, fila in zip(barras, rendimiento, strict=True):
        ax_rps.annotate(
            f"{fila['rps']:.1f} req/s\nerror {fila['tasa_error_pct']:.1f}%",
            (barra.get_x() + barra.get_width() / 2, barra.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    ax_rps.set_xticks(x, etiquetas, rotation=30, ha="right")
    ax_rps.set_ylabel("Predicciones por segundo")
    ax_rps.set_title("Rendimiento por escenario")
    ax_rps.set_ylim(0, max(f["predicciones_s"] for f in rendimiento) * 1.3)
    fig.tight_layout()
    ruta = FIGURAS / "01_latencia_y_rendimiento.png"
    fig.savefig(ruta, dpi=300)
    plt.close(fig)
    return ruta


def figura_serie_s3() -> Path:
    plt = _pyplot()
    filas = [f for f in _leer_csv("a9_locust_S3_stats_history.csv") if f["Name"] == "Aggregated"]
    t0 = float(filas[0]["Timestamp"])
    t = [float(f["Timestamp"]) - t0 for f in filas]
    fig, ejes = plt.subplots(3, 1, sharex=True, figsize=(9, 6.5))
    ejes[0].plot(t, [int(f["User Count"]) for f in filas], color="#55A868")
    ejes[0].set_ylabel("Usuarios")
    ejes[1].plot(t, [_float(f["Requests/s"]) for f in filas], color="#4C72B0")
    ejes[1].set_ylabel("Solicitudes/s")
    # "N/A" significa que no hubo peticiones en ese segundo: se dibuja como hueco, no como 0.
    ejes[2].plot(
        t,
        [_float(f["95%"]) if f["95%"] != "N/A" else math.nan for f in filas],
        color="#C44E52",
    )
    ejes[2].set_ylabel("p95 (ms)")
    ejes[2].set_xlabel("Tiempo desde el inicio de S3 (s)")
    ejes[0].set_title("S3 (estrés, 80 usuarios): punto de saturación de la API")
    for eje in ejes:
        eje.grid(alpha=0.3)
    fig.tight_layout()
    ruta = FIGURAS / "02_serie_temporal_S3.png"
    fig.savefig(ruta, dpi=300)
    plt.close(fig)
    return ruta


def figura_recursos() -> Path:
    plt = _pyplot()
    colores = {"mibici-api": "#4C72B0", "mibici-mlflow": "#DD8452"}
    datos = {etiqueta: _leer_csv(f"a9_docker_stats_{etiqueta}.csv") for etiqueta in ETIQUETAS}
    t0 = min(_float(f["epoch_s"]) for filas in datos.values() for f in filas)
    fig, (ax_cpu, ax_mem) = plt.subplots(2, 1, sharex=True, figsize=(11, 6.8))
    for etiqueta in ETIQUETAS:
        ventana = _leer_json(f"a9_ventanas_{etiqueta}.json")
        limites = _limites_cpu(ventana)
        inicio = (datetime.fromisoformat(ventana["inicio_medicion_utc"]).timestamp() - t0) / 60
        fin = (datetime.fromisoformat(ventana["fin_medicion_utc"]).timestamp() - t0) / 60
        for contenedor in CONTENEDORES:
            muestras = [f for f in datos[etiqueta] if f["contenedor"] == contenedor]
            t = [(_float(m["epoch_s"]) - t0) / 60 for m in muestras]
            ax_cpu.plot(
                t,
                [_float(m["cpu_pct"]) / limites[contenedor] for m in muestras],
                color=colores[contenedor],
                linewidth=0.8,
                label=contenedor if etiqueta == ETIQUETAS[0] else None,
            )
            ax_mem.plot(
                t,
                [_float(m["mem_mib"]) for m in muestras],
                color=colores[contenedor],
                linewidth=0.8,
                label=contenedor if etiqueta == ETIQUETAS[0] else None,
            )
        for eje in (ax_cpu, ax_mem):
            eje.axvspan(inicio, fin, color="grey", alpha=0.12)
        ax_cpu.text(
            (inicio + fin) / 2,
            0.97,
            etiqueta.replace("_", " "),
            transform=ax_cpu.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
        )
    ax_cpu.set_ylabel("CPU (% del límite)")
    ax_cpu.set_title("Recursos por contenedor; las franjas son las ventanas de medición de Locust")
    ax_cpu.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax_mem.set_ylabel("Memoria (MiB)")
    ax_mem.set_xlabel("Minutos desde la primera muestra de docker stats")
    ax_mem.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    for eje in (ax_cpu, ax_mem):
        eje.grid(alpha=0.3)
    fig.tight_layout()
    ruta = FIGURAS / "03_recursos_api_mlflow.png"
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ruta


def _usd(valor: float) -> str:
    """Formato de etiqueta: más decimales para costos menores a un dólar."""
    return f"${valor:.4f}" if valor < 1 else f"${valor:,.2f}"


def _costo(costos: list[dict[str, Any]], instancia: str, modo: str) -> float:
    return next(
        fila["usd_por_1M"]
        for fila in costos
        if fila["instancia"] == instancia and fila["modo"] == modo
    )


def figura_finops(costos: list[dict[str, Any]], optim: list[dict[str, Any]]) -> Path:
    plt = _pyplot()
    fig = plt.figure(figsize=(11, 10))
    ax_a = fig.add_subplot(2, 1, 1)
    ax_b = fig.add_subplot(2, 2, 3)
    ax_c = fig.add_subplot(2, 2, 4)

    instancias_a = [NOMBRE_T3, NOMBRE_C6I, NOMBRE_F_2_4, NOMBRE_F_1_2, NOMBRE_F_05_1]
    x = list(range(len(instancias_a)))
    ancho = 0.38
    capacidad = [_costo(costos, n, MODO_LOTE) for n in instancias_a]
    horaria = [_costo(costos, n, MODO_HORARIO) for n in instancias_a]
    barras_cap = ax_a.bar(
        [i - ancho / 2 for i in x], capacidad, ancho, label="Capacidad máx. al 70% (lote de 20)"
    )
    barras_hor = ax_a.bar(
        [i + ancho / 2 for i in x],
        horaria,
        ancho,
        label="Demanda horaria MiBici, siempre encendida",
    )
    barras_cap[-1].set_hatch("//")
    barras_cap[-1].set_alpha(0.6)
    for barras in (barras_cap, barras_hor):
        for barra in barras:
            ax_a.annotate(
                _usd(barra.get_height()),
                (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    ax_a.set_yscale("log")
    ax_a.set_xticks(x, [n.replace(" (", "\n(") for n in instancias_a], fontsize=8)
    ax_a.set_ylabel("USD por 1M inferencias (log)")
    ax_a.set_title("A. Costo por inferencia por instancia y régimen (tarifas de referencia)")
    ax_a.legend(frameon=False, fontsize=8)

    escenarios_b = [
        (NOMBRE_F_1_2, MODO_HORARIO, "Fargate\nsiempre"),
        (NOMBRE_F_1_2, MODO_PROG_HORA, "Fargate\nprogramado"),
        (NOMBRE_LAMBDA, MODO_SERVERLESS, "Lambda\nsolicitud"),
        (NOMBRE_CLOUD_RUN, MODO_SERVERLESS, "Cloud Run\nsolicitud"),
    ]
    valores_b = [_costo(costos, n, m) for n, m, _ in escenarios_b]
    colores_b = ["#C44E52", "#55A868", "#8172B2", "#8172B2"]
    barras_b = ax_b.bar(range(len(escenarios_b)), valores_b, color=colores_b)
    for barra in barras_b:
        ax_b.annotate(
            _usd(barra.get_height()),
            (barra.get_x() + barra.get_width() / 2, barra.get_height()),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    ax_b.set_yscale("log")
    ax_b.set_xticks(range(len(escenarios_b)), [e[2] for e in escenarios_b], fontsize=7)
    ax_b.set_ylabel("USD por 1M inferencias (log)")
    ax_b.set_title("B. Demanda horaria: programado y serverless", fontsize=9)

    ax_c.barh(
        [o["id"] + " " + o["optimizacion"] for o in optim],
        [o["ahorro_pct"] for o in optim],
        color="#4C72B0",
    )
    for indice, optimizacion in enumerate(optim):
        ax_c.annotate(
            f"{optimizacion['ahorro_pct']:.1f}% (x{optimizacion['factor_mejora']:.1f})",
            (optimizacion["ahorro_pct"], indice),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
        )
    ax_c.set_xlim(0, 115)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Ahorro en costo por inferencia (%)")
    ax_c.set_title("C. Optimizaciones medidas", fontsize=9)
    ax_c.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    ruta = FIGURAS / "04_finops_costo_y_optimizaciones.png"
    fig.savefig(ruta, dpi=300)
    plt.close(fig)
    return ruta


# --------------------------------------------------------------------------- orquestación


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)

    rendimiento = [resumen_locust(etiqueta) for etiqueta in ETIQUETAS]
    recursos = [fila for etiqueta in ETIQUETAS for fila in recursos_ventana(etiqueta)]
    anotar_cpu_por_prediccion(recursos, rendimiento)
    cap = capacidad_medida()
    inst = instancias(cap)
    costos = tabla_costos(inst)
    sensibilidad = tabla_sensibilidad(inst)
    optim = optimizaciones(inst, recursos)

    escribir_tabla("a9_finops_rendimiento", rendimiento)
    escribir_tabla("a9_finops_recursos", recursos)
    escribir_tabla(
        "a9_finops_capacidad",
        [{"configuracion": clave, **valores} for clave, valores in cap.items()],
    )
    escribir_tabla("a9_finops_costos", costos)
    escribir_tabla("a9_finops_sensibilidad", sensibilidad)
    escribir_tabla("a9_finops_optimizaciones", optim)

    resumen = {
        "generado_con": "scripts/finops_analysis.py",
        "supuestos": {
            "estaciones": ESTACIONES,
            "predicciones_por_lote": PREDICCIONES_POR_LOTE,
            "utilizacion_capacidad": UTILIZACION_CAPACIDAD,
            "demanda_predicciones_dia": {k: demanda_diaria(k) for k in CORTES_POR_DIA},
            "duracion_inferencia_serverless_s": DURACION_INFERENCIA_S,
            "minimo_facturable_fargate_s": MINUTO_MINIMO_FACTURABLE_S,
        },
        "tarifas_referencia": {
            "ec2_usd_h": PRECIO_EC2_USD_H,
            "t3_linea_base_por_vcpu": T3_BASELINE_POR_VCPU,
            "t3_unlimited_usd_vcpu_h": T3_UNLIMITED_USD_POR_VCPU_H,
            "fargate_usd_vcpu_s": FARGATE_USD_VCPU_S,
            "fargate_usd_gb_s": FARGATE_USD_GB_S,
            "lambda_usd_gb_s": LAMBDA_USD_GB_S,
            "lambda_usd_por_solicitud": LAMBDA_USD_POR_SOLICITUD,
            "cloud_run_usd_vcpu_s": CLOUD_RUN_USD_VCPU_S,
            "cloud_run_usd_gib_s": CLOUD_RUN_USD_GIB_S,
            "cloud_run_usd_por_solicitud": CLOUD_RUN_USD_POR_SOLICITUD,
            "nota": "Tarifas públicas de referencia us-east-1; verificar en AWS y Google "
            "antes de la entrega.",
        },
        "capacidad_medida": cap,
        "limitaciones": LIMITACIONES,
    }
    (TABLES / "a9_finops_resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )

    figuras = [
        figura_latencia(rendimiento),
        figura_serie_s3(),
        figura_recursos(),
        figura_finops(costos, optim),
    ]
    for fila in optim:
        print(f"{fila['id']} {fila['optimizacion']}: ahorro {fila['ahorro_pct']:.1f}%")
    for ruta in figuras:
        print(f"figura: {ruta.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
