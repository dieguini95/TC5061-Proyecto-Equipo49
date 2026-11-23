"""Contrato de los comandos de operación, validación y FinOps en GNU Make."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = REPO_ROOT / "Makefile"
MAKE = shutil.which("make")

pytestmark = pytest.mark.skipif(
    os.name == "nt" and MAKE is None,
    reason="GNU Make requiere WSL o una instalación de Make en Windows.",
)

COMMANDS = [
    ("compose-up", ["docker", "compose", "up", "-d", "--build"]),
    ("compose-down", ["docker", "compose", "down"]),
    ("compose-ps", ["docker", "compose", "ps"]),
    ("compose-logs", ["docker", "compose", "logs"]),
    ("compose-validate", ["docker", "compose", "config", "--quiet"]),
    ("loadtest-smoke", ["docker", "compose", "--profile", "loadtest", "run", "--rm", "locust"]),
    ("finops-report", [".venv-a9/bin/python", "scripts/finops_analysis.py"]),
]


@pytest.fixture
def isolated_makefile(tmp_path):
    """Usa una copia privada cuya ruta contiene espacios."""
    makefile = tmp_path / "Makefile de prueba"
    shutil.copyfile(MAKEFILE, makefile)
    return makefile


@pytest.mark.parametrize(("target", "command"), COMMANDS)
@pytest.mark.parametrize("same_name_file", [False, True], ids=["normal", "archivo-homonimo"])
def test_targets_preserve_documented_command(target, command, same_name_file, isolated_makefile):
    """Las recetas siguen activas incluso si existe un archivo con el nombre del target."""
    assert MAKE is not None, "Las pruebas de contrato requieren GNU Make."
    if same_name_file:
        (isolated_makefile.parent / target).touch()

    result = subprocess.run(
        [MAKE, "--no-print-directory", "-n", "-f", str(isolated_makefile), target],
        cwd=isolated_makefile.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert shlex.split(result.stdout) == command


def test_finops_python_can_be_overridden_with_a_path_containing_spaces(isolated_makefile):
    """Permite usar un entorno alternativo sin dividir la ruta del intérprete."""
    assert MAKE is not None, "Las pruebas de contrato requieren GNU Make."
    interpreter = "/path with spaces/python"
    result = subprocess.run(
        [
            MAKE,
            "--no-print-directory",
            "-n",
            "-f",
            str(isolated_makefile),
            "finops-report",
            f"FINOPS_PYTHON={interpreter}",
        ],
        cwd=isolated_makefile.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert shlex.split(result.stdout) == [interpreter, "scripts/finops_analysis.py"]
