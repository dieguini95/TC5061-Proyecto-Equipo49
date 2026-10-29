from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile

import pytest

from mibici_ml import config

REPO_ROOT = Path(__file__).resolve().parents[1]
SECURITY_SCRIPT = REPO_ROOT / "scripts" / "run_security_checks.sh"
OFFICIAL_EVIDENCE_RELATIVE_DIR = (
    PurePosixPath("docs") / "evidencias" / "Actividad 7 - Chequeos de seguridad"
)
FICTIONAL_EXAMPLES = {".env.example", ".env.sample", ".env.template"}
SENSITIVE_SUFFIXES = {
    ".db",
    ".db-shm",
    ".db-wal",
    ".h5",
    ".joblib",
    ".key",
    ".onnx",
    ".p12",
    ".pem",
    ".pfx",
    ".pickle",
    ".pkl",
    ".pt",
    ".skops",
    ".sqlite",
    ".sqlite3",
}
BINARY_SUFFIXES = {
    ".bmp",
    ".docx",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".parquet",
    ".webp",
    ".zip",
}
IGNORED_DIRECTORIES = {".git", ".venv", "mlartifacts", "mlruns"}
IGNORED_PATHS = (
    ".env",
    ".env.local",
    ".venv/bin/python",
    "mlflow.db",
    "mlruns/example/meta.yaml",
    "mlartifacts/1/model",
    "model.pkl",
    "model.pickle",
    "model.joblib",
    "coverage.xml",
    "htmlcov/",
    "bandit-report.json",
    "bandit-final.json",
    "pip-audit-report.json",
    "pip-audit-final.json",
    "private.pem",
    "tls.key",
    "bundle.p12",
    "bundle.pfx",
    "state.sqlite",
    "state.sqlite3",
)
TRACKABLE_SECURITY_FILES = (
    "requirements-security.txt",
    ".bandit",
    "scripts/run_security_checks.sh",
    "tests/test_repository_security.py",
    str(OFFICIAL_EVIDENCE_RELATIVE_DIR / "README.md"),
)
SECRET_PATTERNS = (
    (
        "GitHub personal access token",
        re.compile(
            r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github"
            r"_pat_[A-Za-z0-9_]{20,})\b"
        ),
    ),
    (
        "AWS access key",
        re.compile("A" + "KIA" + r"[0-9A-Z]{16}"),
    ),
    (
        "Google API key",
        re.compile("A" + "I" + "za" + r"[0-9A-Za-z_-]{35}"),
    ),
    (
        "private key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ),
    (
        "HTTP URL with embedded credentials",
        re.compile(r"\bhttps?://[^\s/:@]+:[^\s/@]+@"),
    ),
)


def _tracked_paths() -> list[PurePosixPath]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        PurePosixPath(path.decode("utf-8", errors="surrogateescape"))
        for path in result.stdout.split(b"\0")
        if path
    ]


def _is_fictional_example(path: PurePosixPath) -> bool:
    name = path.name.lower()
    return name in FICTIONAL_EXAMPLES or any(
        marker in name for marker in ("example", "sample", "template")
    )


def _is_sensitive_path(path: PurePosixPath) -> bool:
    parts = tuple(part.lower() for part in path.parts)
    name = path.name.lower()
    if _is_fictional_example(path):
        return False
    if name == ".env" or name.startswith(".env."):
        return True
    if IGNORED_DIRECTORIES.intersection(parts):
        return True
    if path.suffix.lower() in SENSITIVE_SUFFIXES:
        return True
    if re.search(r"(?:^|[._-])(credentials?|tokens?|secrets?)(?:[._-]|$)", name):
        return True
    return False


def _secret_findings(text: str, path: PurePosixPath) -> list[str]:
    findings = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for secret_type, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(
                    f"{path}:{line_number}: {secret_type}; value=[REDACTED]"
                )
    return findings


def _script_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    virtual_env = tmp_path / "fake-venv"
    bin_dir = virtual_env / "bin"
    bin_dir.mkdir(parents=True)
    command_log = tmp_path / "commands.txt"
    fake_python = bin_dir / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >> \"$COMMAND_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(virtual_env)
    env["COMMAND_LOG"] = str(command_log)
    return env, command_log


def _git_status() -> str:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_no_sensitive_files_are_tracked():
    sensitive = [str(path) for path in _tracked_paths() if _is_sensitive_path(path)]
    assert not sensitive, "Git rastrea archivos sensibles: " + ", ".join(sensitive)


def test_tracked_text_files_have_no_obvious_secrets():
    findings = []
    paths = _tracked_paths()
    official_readme = OFFICIAL_EVIDENCE_RELATIVE_DIR / "README.md"
    if REPO_ROOT.joinpath(*official_readme.parts).is_file():
        paths.append(official_readme)
    for path in paths:
        if IGNORED_DIRECTORIES.intersection(part.lower() for part in path.parts):
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        absolute_path = REPO_ROOT.joinpath(*path.parts)
        if not absolute_path.is_file():
            continue
        content = absolute_path.read_bytes()
        if b"\0" in content[:8192]:
            continue
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            continue
        findings.extend(_secret_findings(text, path))
    assert not findings, "Posibles secretos encontrados (valores redactados):\n" + "\n".join(
        findings
    )


def test_gitignore_excludes_sensitive_and_temporary_paths():
    for path in IGNORED_PATHS:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", "--", path],
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0, f".gitignore no excluye {path}"


def test_security_sources_remain_trackable():
    for path in TRACKABLE_SECURITY_FILES:
        absolute_path = REPO_ROOT.joinpath(*PurePosixPath(path).parts)
        assert absolute_path.is_file(), f"Falta el archivo versionable {path}"
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", "--", path],
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 1, f"{path} no debe estar ignorado"
    evidence_dir = REPO_ROOT.joinpath(*OFFICIAL_EVIDENCE_RELATIVE_DIR.parts)
    screenshots = sorted(evidence_dir.glob("*.png"))
    assert screenshots, "La evidencia oficial debe contener al menos un PNG"
    for screenshot in screenshots:
        relative_path = screenshot.relative_to(REPO_ROOT).as_posix()
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", "--", relative_path],
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 1, f"{relative_path} no debe estar ignorado"


def test_train_output_directories_use_portable_temp_and_are_unique(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.delenv("MLFLOW_TRAIN_OUTPUT_DIR", raising=False)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    assert config.get_mlflow_train_output_base_dir() == tmp_path
    first = config.create_mlflow_train_output_dir()
    second = config.create_mlflow_train_output_dir()

    assert first.parent == tmp_path
    assert second.parent == tmp_path
    assert first != second
    assert first.name.startswith("mibici-runs-")
    assert second.name.startswith("mibici-runs-")
    assert stat.S_IMODE(first.stat().st_mode) == 0o700
    assert stat.S_IMODE(second.stat().st_mode) == 0o700


def test_train_output_directory_can_be_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    configured_dir = tmp_path / "configured-output"
    monkeypatch.setenv("MLFLOW_TRAIN_OUTPUT_DIR", str(configured_dir))

    output_dir = config.create_mlflow_train_output_dir()

    assert config.get_mlflow_train_output_base_dir() == configured_dir
    assert output_dir.parent == configured_dir
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700


def test_train_output_config_has_no_hardcoded_tmp_directory():
    source = (REPO_ROOT / "mibici_ml" / "config.py").read_text(encoding="utf-8")
    assert "/tmp/mibici-runs" not in source


def test_importing_config_does_not_create_train_output_directory(tmp_path: Path):
    output_base = tmp_path / "not-created-on-import"
    env = os.environ.copy()
    env["MLFLOW_TRAIN_OUTPUT_DIR"] = str(output_base)

    result = subprocess.run(
        [sys.executable, "-c", "import mibici_ml.config"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert not output_base.exists()


def test_security_script_help_and_invalid_arguments_do_not_run_tools(tmp_path):
    assert SECURITY_SCRIPT.is_file(), "No existe scripts/run_security_checks.sh"
    syntax = subprocess.run(
        ["bash", "-n", str(SECURITY_SCRIPT)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, f"El script no es válido en Bash: {syntax.stderr}"

    env, command_log = _script_environment(tmp_path)
    help_result = subprocess.run(
        ["bash", str(SECURITY_SCRIPT), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--with-audit" in help_result.stdout
    assert not command_log.exists(), "--help ejecutó el Python de herramientas"

    invalid_result = subprocess.run(
        ["bash", str(SECURITY_SCRIPT), "--unknown"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert invalid_result.returncode != 0
    assert not command_log.exists(), "Un argumento inválido ejecutó herramientas"


def test_security_script_default_mode_is_offline_and_writes_only_to_tmp(tmp_path):
    env, command_log = _script_environment(tmp_path)
    status_before = _git_status()
    repo_outputs = (
        REPO_ROOT / "security-checks.txt",
        REPO_ROOT / "bandit-gate.txt",
        REPO_ROOT / "pytest-security.txt",
        REPO_ROOT / "pip-audit.txt",
        REPO_ROOT.joinpath(*OFFICIAL_EVIDENCE_RELATIVE_DIR.parts),
    )
    outputs_before = {path: path.exists() for path in repo_outputs}

    result = subprocess.run(
        ["bash", str(SECURITY_SCRIPT)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = command_log.read_text(encoding="utf-8").splitlines()
    assert "-m pip check" in calls
    assert any(
        call.startswith("-m bandit -r mibici_ml -c .bandit")
        and "--severity-level medium" in call
        and "--confidence-level medium" in call
        for call in calls
    )
    assert "-m pytest -q tests/test_repository_security.py" in calls
    assert not any("pip_audit" in call for call in calls)
    assert "/tmp/actividad7-security" in SECURITY_SCRIPT.read_text(encoding="utf-8")
    assert _git_status() == status_before, "El script modificó el estado de Git"
    assert {path: path.exists() for path in repo_outputs} == outputs_before


def test_security_script_reports_pip_audit_findings_without_mislabelling_them(tmp_path):
    env, command_log = _script_environment(tmp_path)
    fake_python = Path(env["VIRTUAL_ENV"]) / "bin" / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >> \"$COMMAND_LOG\"\n"
        "if [[ \"$2\" == \"pip_audit\" ]]; then\n"
        "  printf 'Found 6 known vulnerabilities in 3 packages\\n'\n"
        "  exit 1\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        ["bash", str(SECURITY_SCRIPT), "--with-audit"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 1
    assert "Found 6 known vulnerabilities in 3 packages" in output
    assert "AUDITORÍA EJECUTADA CON HALLAZGOS" in output
    assert "código final 1" in output
    assert "ERROR: pip-audit" not in output
    assert "-m pip_audit -r requirements.txt" in command_log.read_text(encoding="utf-8")
