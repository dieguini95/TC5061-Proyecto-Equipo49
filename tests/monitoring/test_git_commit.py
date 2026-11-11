import subprocess
from pathlib import Path

import pytest

from mibici_ml.monitoring import cli
from mibici_ml.monitoring.errors import MonitoringError


def test_git_commit_uses_bounded_git_invocation(monkeypatch) -> None:
    call: dict[str, object] = {}

    def fake_run(command, **kwargs):
        call["command"] = command
        call.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="synthetic-revision\n")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli._git_commit() == "synthetic-revision"
    assert call["command"] == ["git", "rev-parse", "HEAD"]
    assert call["cwd"] == Path(cli.__file__).resolve().parents[2]
    assert call["check"] is True
    assert call["capture_output"] is True
    assert call["text"] is True
    assert call["timeout"] == 10


def test_git_commit_timeout_raises_safe_error(monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["git", "rev-parse", "HEAD"],
            timeout=10,
            stderr="SYNTHETIC-SECRET-CANARY",
        )

    monkeypatch.setattr(cli.subprocess, "run", timeout)

    with pytest.raises(MonitoringError) as error:
        cli._git_commit()

    assert str(error.value) == "Could not determine the current Git revision."
    assert "SYNTHETIC-SECRET-CANARY" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__
