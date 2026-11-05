"""Explicit local-development entrypoint using the deterministic fake provider."""

from mibici_ml.api.app import create_app
from mibici_ml.api.providers import FakeModelProvider

app = create_app(provider=FakeModelProvider())
