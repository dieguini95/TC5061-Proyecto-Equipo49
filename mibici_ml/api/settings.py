"""Environment-backed settings for the P3 serving API."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re

DEFAULT_MODEL_NAME = "mibici-equipo49-p3-grupos"
DEFAULT_MODEL_ALIAS = "champion"
DEFAULT_MAX_BATCH_SIZE = 100


@dataclass(frozen=True)
class APISettings:
    provider: str | None
    model_uri: str
    model_name: str
    model_alias: str
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE

    def __post_init__(self) -> None:
        safe_name = re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.model_name)
        safe_alias = re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.model_alias)
        if not safe_name or not safe_alias:
            raise ValueError("Model name and alias must be safe identifiers.")
        if self.model_uri != f"models:/{self.model_name}@{self.model_alias}":
            raise ValueError("MIBICI_MODEL_URI must identify the configured model alias.")
        if not 1 <= self.max_batch_size <= DEFAULT_MAX_BATCH_SIZE:
            raise ValueError(
                f"MIBICI_MAX_BATCH_SIZE must be between 1 and {DEFAULT_MAX_BATCH_SIZE}."
            )
        if self.provider not in {None, "fake", "mlflow"}:
            raise ValueError("MIBICI_API_PROVIDER must be 'fake' or 'mlflow'.")

    @classmethod
    def from_env(cls) -> APISettings:
        model_name = os.environ.get("MIBICI_MODEL_NAME", DEFAULT_MODEL_NAME)
        model_alias = os.environ.get("MIBICI_MODEL_ALIAS", DEFAULT_MODEL_ALIAS)
        model_uri = os.environ.get(
            "MIBICI_MODEL_URI",
            f"models:/{model_name}@{model_alias}",
        )
        try:
            max_batch_size = int(
                os.environ.get("MIBICI_MAX_BATCH_SIZE", str(DEFAULT_MAX_BATCH_SIZE))
            )
        except ValueError as error:
            raise ValueError("MIBICI_MAX_BATCH_SIZE must be an integer.") from error
        provider = os.environ.get("MIBICI_API_PROVIDER")

        return cls(
            provider=provider,
            model_uri=model_uri,
            model_name=model_name,
            model_alias=model_alias,
            max_batch_size=max_batch_size,
        )
