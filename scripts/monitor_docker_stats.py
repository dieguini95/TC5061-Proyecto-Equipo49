"""Muestreo de `docker stats` para la Actividad 9: CPU, memoria, red y disco por contenedor.

Ejecuta `docker stats` en modo continuo sobre los contenedores indicados y escribe una fila por
contenedor y por muestra en un CSV. Cada fila lleva la marca de tiempo UTC de la lectura, para
cruzarla con la ventana de medición de Locust (ver scripts/run_a9_scenarios.py).

Las unidades de Docker mezclan decimales (kB, MB, las de red y disco) y binarias (MiB, GiB, las de
memoria). El análisis parte de bytes y expone MiB para memoria y MB decimales para red y disco.

Uso:
    python scripts/monitor_docker_stats.py --containers mibici-api mibici-mlflow \\
        --duration 115 --output reports/tables/a9_docker_stats_S1.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

FIELDS = (
    "timestamp_utc",
    "epoch_s",
    "contenedor",
    "cpu_pct",
    "mem_mib",
    "mem_limite_mib",
    "mem_pct",
    "red_rx_mb",
    "red_tx_mb",
    "disco_lectura_mb",
    "disco_escritura_mb",
    "pids",
)

_UNITS = {
    "B": 1.0,
    "kB": 1e3,
    "KB": 1e3,
    "MB": 1e6,
    "GB": 1e9,
    "TB": 1e12,
    "KiB": 2.0**10,
    "MiB": 2.0**20,
    "GiB": 2.0**30,
    "TiB": 2.0**40,
}
_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*$")
# Sin TTY, docker stats redibuja la tabla con secuencias ANSI (ESC[H, ESC[K) antes de cada JSON.
# Los valores del JSON no contienen llaves, así que basta con extraer el primer objeto de la línea.
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")


def parse_size_bytes(text: str) -> float:
    """Convierte '1.5GiB' o '350kB' a bytes."""
    match = _SIZE_RE.match(text)
    if match is None or match.group(2) not in _UNITS:
        raise ValueError(f"tamaño no reconocido: {text!r}")
    return float(match.group(1)) * _UNITS[match.group(2)]


def parse_pair_bytes(text: str) -> tuple[float, float]:
    """Separa un par 'usado / límite' (MemUsage, NetIO, BlockIO) y lo convierte a bytes."""
    left, sep, right = text.partition("/")
    if not sep:
        raise ValueError(f"par no reconocido: {text!r}")
    return parse_size_bytes(left), parse_size_bytes(right)


def parse_percent(text: str) -> float:
    """Convierte '12.34%' a 12.34."""
    value = text.strip()
    if not value.endswith("%"):
        raise ValueError(f"porcentaje no reconocido: {text!r}")
    return float(value[:-1])


def parse_stats_record(record: dict[str, str]) -> dict[str, object]:
    """Transforma una línea JSON de `docker stats --format '{{json .}}'` en valores numéricos."""
    mem_used, mem_limit = parse_pair_bytes(record["MemUsage"])
    rx_bytes, tx_bytes = parse_pair_bytes(record["NetIO"])
    read_bytes, write_bytes = parse_pair_bytes(record["BlockIO"])
    pids = record.get("PIDs", "").strip()
    return {
        "contenedor": record["Name"],
        "cpu_pct": round(parse_percent(record["CPUPerc"]), 3),
        "mem_mib": round(mem_used / 2.0**20, 3),
        "mem_limite_mib": round(mem_limit / 2.0**20, 3),
        "mem_pct": round(parse_percent(record["MemPerc"]), 3),
        "red_rx_mb": round(rx_bytes / 1e6, 3),
        "red_tx_mb": round(tx_bytes / 1e6, 3),
        "disco_lectura_mb": round(read_bytes / 1e6, 3),
        "disco_escritura_mb": round(write_bytes / 1e6, 3),
        "pids": int(pids) if pids.isdigit() else "",
    }


def build_docker_stats_command(containers: list[str]) -> list[str]:
    """Comando de streaming: una línea JSON por contenedor y por muestra (cada ~1 s)."""
    return ["docker", "stats", "--format", "{{json .}}", *containers]


def _utc_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat(timespec="milliseconds")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--containers", nargs="+", required=True, help="Nombres de contenedor.")
    parser.add_argument("--duration", type=float, required=True, help="Segundos de muestreo.")
    parser.add_argument("--output", type=Path, required=True, help="CSV de salida.")
    args = parser.parse_args(argv)
    if args.duration <= 0:
        parser.error("--duration debe ser positivo")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        build_docker_stats_command(args.containers),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    # El temporizador detiene docker stats al terminar la ventana; así no hay procesos huérfanos
    # aunque un contenedor deje de emitir líneas.
    timer = threading.Timer(args.duration, process.terminate)
    timer.start()
    rows = 0
    try:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            assert process.stdout is not None
            for line in process.stdout:
                match = _JSON_OBJECT_RE.search(line)
                if match is None:
                    continue
                epoch = time.time()
                try:
                    parsed = parse_stats_record(json.loads(match.group(0)))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
                writer.writerow(
                    {"timestamp_utc": _utc_iso(epoch), "epoch_s": round(epoch, 3), **parsed}
                )
                handle.flush()
                rows += 1
    finally:
        timer.cancel()
        process.terminate()
        process.wait(timeout=10)

    stderr = process.stderr.read() if process.stderr is not None else ""
    if rows == 0:
        print(f"Sin muestras de docker stats. Revisa que los contenedores existan: {stderr}")
        return 1
    print(f"{rows} muestras escritas en {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
