from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci-serving.yml"


def _workflow() -> dict:
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_workflow_triggers_and_permissions_are_minimal():
    workflow = _workflow()

    assert set(workflow["on"]) == {"pull_request", "push", "workflow_dispatch"}
    assert workflow["on"]["pull_request"]["branches"] == ["main"]
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] == "true"


def test_jobs_run_required_quality_security_tests_and_image_smoke_gates():
    workflow = _workflow()
    jobs = workflow["jobs"]

    assert {"quality", "tests", "deepchecks-tests", "security", "image-smoke"} <= set(jobs)
    assert set(jobs["image-smoke"]["needs"]) == {
        "quality",
        "tests",
        "deepchecks-tests",
        "security",
    }
    assert jobs["quality"]["timeout-minutes"] == "10"
    assert jobs["tests"]["timeout-minutes"] == "20"
    assert jobs["deepchecks-tests"]["timeout-minutes"] == "20"
    assert jobs["security"]["timeout-minutes"] == "15"
    assert jobs["image-smoke"]["timeout-minutes"] == "25"
    quality_commands = "\n".join(step.get("run", "") for step in jobs["quality"]["steps"])
    assert "python -m ruff format --check ." in quality_commands
    assert "python -m ruff check ." in quality_commands
    assert "tests/test_ci_smoke_serving.py" in quality_commands


def test_workflow_builds_and_smokes_locally_without_publishing_or_secrets():
    text = WORKFLOW_PATH.read_text(encoding="utf-8").lower()
    cleanup = next(
        step
        for step in _workflow()["jobs"]["image-smoke"]["steps"]
        if step.get("name") == "Cleanup container"
    )

    assert "containerfile" in text
    assert "docker build" in text
    assert "scripts/ci_smoke_serving.py" in text
    assert cleanup["if"] == "always()"
    assert "docker logs" in cleanup["run"]
    assert "docker stop" in cleanup["run"]
    assert "docker rm" in cleanup["run"]
    assert "docker login" not in text
    assert "docker push" not in text
    assert "ghcr.io" not in text
    assert "secrets." not in text
    assert "pull_request_target" not in text
    assert "continue-on-error" not in text
