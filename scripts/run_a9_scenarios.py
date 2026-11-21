"""Ejecuta los escenarios de carga S1 a S4 de la Actividad 9 con monitoreo sincronizado.

Por cada escenario:
  1. Verifica que la API responde (/health, /ready y /v1/p2/ready).
  2. Inicia scripts/monitor_docker_stats.py sobre mibici-api y mibici-mlflow.
  3. Ejecuta Locust en modo headless con --reset-stats: las estadísticas excluyen la rampa.
  4. Escribe reports/tables/a9_ventanas_<escenario>.json con la ventana de medición en UTC.

Al final escribe reports/tables/a9_entorno.json (hardware, Docker, commit y configuración).

Escenarios:
    S1   10 usuarios, mezcla operativa, 60 s
    S2   40 usuarios, mezcla operativa, 90 s (pico operativo)
    S3   80 usuarios, mezcla operativa, 90 s (estrés)
    S4a  20 usuarios, solo P2 unitario, sin espera, 60 s (línea base por lotes)
    S4b  20 usuarios, solo P2 por lotes de 20 estaciones, sin espera, 60 s

Uso (desde la raíz del repositorio):
    .venv-a9/bin/python scripts/run_a9_scenarios.py --host http://127.0.0.1:8000
    .venv-a9/bin/python scripts/run_a9_scenarios.py --only S1 --dry-run
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
EVIDENCIA = ROOT / "docs" / "evidencias" / "Actividad 9 - Operacion y FinOps"
LOCUSTFILE = "load_tests/locustfile.py"
CONTENEDORES = ("mibici-api", "mibici-mlflow")
MARGEN_MONITOR_S = 15  # el monitor sigue unos segundos después de Locust para cubrir la cola
PREFLIGHT_PATHS = ("/health", "/ready", "/v1/p2/ready")


@dataclass(frozen=True)
class Escenario:
    nombre: str
    descripcion: str
    user_class: str
    usuarios: int
    tasa_spawn: int
    duracion_s: int


ESCENARIOS: tuple[Escenario, ...] = (
    Escenario("S1", "Nominal: operación habitual", "MiBiciMixedUser", 10, 2, 60),
    Escenario("S2", "Pico operativo", "MiBiciMixedUser", 40, 5, 90),
    Escenario("S3", "Estrés: búsqueda del punto de saturación", "MiBiciMixedUser", 80, 10, 90),
    Escenario("S4a", "Línea base: P2 unitario sin espera", "P2UnitUser", 20, 20, 60),
    Escenario("S4b", "Optimización: P2 por lotes de 20 sin espera", "P2BatchUser", 20, 20, 60),
)
# Opcional: requiere reiniciar el API con API_CPU_LIMIT=1.0 antes de ejecutarlo (--only S5).
ESCENARIOS_OPCIONALES: tuple[Escenario, ...] = (
    Escenario("S5", "Right-sizing: P2 por lotes con 1 vCPU", "P2BatchUser", 20, 20, 60),
)


def seleccionar(nombres: str | None) -> tuple[Escenario, ...]:
    """Filtra escenarios por nombre separado por comas; None devuelve los obligatorios."""
    if not nombres:
        return ESCENARIOS
    pedidos = [n.strip() for n in nombres.split(",") if n.strip()]
    por_nombre = {e.nombre: e for e in (*ESCENARIOS, *ESCENARIOS_OPCIONALES)}
    desconocidos = [n for n in pedidos if n not in por_nombre]
    if desconocidos:
        raise SystemExit(f"Escenarios desconocidos: {', '.join(desconocidos)}")
    return tuple(por_nombre[n] for n in pedidos)


def duracion_rampa_s(escenario: Escenario) -> int:
    """Segundos hasta que todos los usuarios están activos."""
    return math.ceil(escenario.usuarios / escenario.tasa_spawn)


def ventana_medicion(
    escenario: Escenario, inicio_locust: datetime, limite_cpus: float | None
) -> dict[str, object]:
    """Ventana de medición en UTC: desde que termina la rampa hasta el final de Locust."""
    rampa = duracion_rampa_s(escenario)
    inicio = inicio_locust + timedelta(seconds=rampa)
    fin = inicio_locust + timedelta(seconds=escenario.duracion_s)
    return {
        "escenario": escenario.nombre,
        "descripcion": escenario.descripcion,
        "user_class": escenario.user_class,
        "usuarios": escenario.usuarios,
        "tasa_spawn": escenario.tasa_spawn,
        "duracion_total_s": escenario.duracion_s,
        "rampa_s": rampa,
        "medicion_s": escenario.duracion_s - rampa,
        "inicio_locust_utc": inicio_locust.isoformat(timespec="seconds"),
        "inicio_medicion_utc": inicio.isoformat(timespec="seconds"),
        "fin_medicion_utc": fin.isoformat(timespec="seconds"),
        "limite_cpus_api": limite_cpus,
    }


def limite_cpus(contenedor: str) -> float | None:
    """CPUs asignadas al contenedor (HostConfig.NanoCpus), o None si no tiene límite."""
    valor = _salida(["docker", "inspect", "--format", "{{.HostConfig.NanoCpus}}", contenedor])
    return int(valor) / 1e9 if valor.isdigit() and int(valor) > 0 else None


def comando_locust(escenario: Escenario, host: str, sufijo: str = "") -> list[str]:
    """Línea de comando de Locust para el escenario (sin ejecutarla)."""
    prefijo = TABLES / f"a9_locust_{escenario.nombre}{sufijo}"
    html = EVIDENCIA / "locust" / f"{escenario.nombre}{sufijo}.html"
    return [
        sys.executable,
        "-m",
        "locust",
        "-f",
        LOCUSTFILE,
        escenario.user_class,
        "--headless",
        "--only-summary",
        "--host",
        host,
        "-u",
        str(escenario.usuarios),
        "-r",
        str(escenario.tasa_spawn),
        "-t",
        f"{escenario.duracion_s}s",
        "--reset-stats",
        "--csv",
        str(prefijo.relative_to(ROOT)),
        "--html",
        str(html.relative_to(ROOT)),
    ]


def comando_monitor(escenario: Escenario, sufijo: str = "") -> list[str]:
    """Monitor de docker stats que cubre la rampa, la medición y la cola de Locust."""
    return [
        sys.executable,
        "scripts/monitor_docker_stats.py",
        "--containers",
        *CONTENEDORES,
        "--duration",
        str(escenario.duracion_s + MARGEN_MONITOR_S),
        "--output",
        str((TABLES / f"a9_docker_stats_{escenario.nombre}{sufijo}.csv").relative_to(ROOT)),
    ]


def verificar_api(host: str) -> None:
    """Falla antes de medir si la API no responde en las rutas de readiness."""
    for ruta in PREFLIGHT_PATHS:
        try:
            with urlopen(f"{host.rstrip('/')}{ruta}", timeout=5) as respuesta:
                if respuesta.status != 200:
                    raise SystemExit(f"{ruta} respondió HTTP {respuesta.status}")
        except (HTTPError, URLError, TimeoutError) as error:
            raise SystemExit(f"La API no está lista en {ruta}: {error}") from None
        print(f"preflight OK {ruta}")


def _salida(comando: list[str]) -> str:
    try:
        resultado = subprocess.run(comando, capture_output=True, text=True, check=True, cwd=ROOT)
    except (OSError, subprocess.CalledProcessError):
        return "no disponible"
    return resultado.stdout.strip()


def metadatos_entorno(host: str) -> dict[str, object]:
    """Hardware, versiones y configuración que acompañan a las mediciones."""
    info_docker = _salida(
        [
            "docker",
            "info",
            "--format",
            "{{.ServerVersion}}|{{.NCPU}}|{{.MemTotal}}|{{.Architecture}}",
        ]
    )
    version, ncpu, mem_total, arquitectura = (info_docker.split("|") + ["", "", "", ""])[:4]
    contenedores = {}
    for nombre in CONTENEDORES:
        formato = "{{.Config.Image}}|{{.Image}}|{{.HostConfig.NanoCpus}}|{{.HostConfig.Memory}}"
        datos = _salida(["docker", "inspect", "--format", formato, nombre]).split("|")
        if len(datos) == 4:
            imagen, imagen_id, nano_cpus, memoria = datos
            contenedores[nombre] = {
                "imagen": imagen,
                "imagen_id": imagen_id,
                "limite_cpus": int(nano_cpus) / 1e9 if nano_cpus.isdigit() else None,
                "limite_memoria_bytes": int(memoria) if memoria.isdigit() else None,
            }
    return {
        "generado_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "host_url": host,
        "plataforma": platform.platform(),
        "procesador": _salida(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "docker_server": version,
        "docker_cpus_vm": ncpu,
        "docker_memoria_vm_bytes": int(mem_total) if mem_total.isdigit() else None,
        "docker_arquitectura": arquitectura,
        "docker_compose": _salida(["docker", "compose", "version", "--short"]),
        "locust": _salida([sys.executable, "-m", "locust", "--version"]),
        "python": platform.python_version(),
        "commit": _salida(["git", "rev-parse", "--short", "HEAD"]),
        "rama": _salida(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "semilla_payloads": 42,
        "contenedores": contenedores,
    }


def ejecutar(
    escenarios: tuple[Escenario, ...],
    host: str,
    dry_run: bool,
    entorno_completo: bool,
    sufijo: str = "",
) -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    (EVIDENCIA / "locust").mkdir(parents=True, exist_ok=True)
    if dry_run:
        for escenario in escenarios:
            print(f"# {escenario.nombre}: {' '.join(comando_monitor(escenario, sufijo))}")
            print(f"# {escenario.nombre}: {' '.join(comando_locust(escenario, host, sufijo))}")
        return 0

    verificar_api(host)
    if entorno_completo:
        (TABLES / "a9_entorno.json").write_text(
            json.dumps(metadatos_entorno(host), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    codigos = {}
    for escenario in escenarios:
        etiqueta = f"{escenario.nombre}{sufijo}"
        print(f"\n=== {etiqueta}: {escenario.descripcion} ===")
        cpus = limite_cpus("mibici-api")
        monitor = subprocess.Popen(comando_monitor(escenario, sufijo), cwd=ROOT)
        time.sleep(3)  # deja que docker stats entregue su primera muestra antes de la carga
        inicio = datetime.now(UTC)
        locust = subprocess.run(comando_locust(escenario, host, sufijo), cwd=ROOT, check=False)
        monitor.wait()
        codigos[etiqueta] = locust.returncode
        ventana = ventana_medicion(escenario, inicio, cpus)
        ventana["codigo_salida_locust"] = locust.returncode
        ventana["replica"] = sufijo.lstrip("_") or "r1"
        (TABLES / f"a9_ventanas_{etiqueta}.json").write_text(
            json.dumps(ventana, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps({"escenario": etiqueta, **asdict(escenario)}, ensure_ascii=False))
    print("\nCódigos de salida de Locust:", codigos)
    return 0 if all(code == 0 for code in codigos.values()) else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--only", help="Escenarios a ejecutar, separados por comas (p. ej. S1,S4b)."
    )
    parser.add_argument(
        "--sufijo", default="", help="Sufijo de archivos, p. ej. _r2 para réplicas."
    )
    parser.add_argument("--dry-run", action="store_true", help="Imprime comandos sin ejecutar.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    entorno = TABLES / "a9_entorno.json"
    # Una corrida parcial (--only) no sobrescribe el entorno de la corrida completa.
    entorno_completo = not args.only or not entorno.exists()
    return ejecutar(seleccionar(args.only), args.host, args.dry_run, entorno_completo, args.sufijo)


if __name__ == "__main__":
    sys.exit(main())
