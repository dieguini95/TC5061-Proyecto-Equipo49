"""Version 2 de P2: octubre 2026 no entra al ajuste ni al umbral.

Datos sinteticos; no entrena ni registra nada en el Registry real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mibici_ml.features import P2_FEATURE_COLUMNS_V2
from mibici_ml.modeling import retrain_p2


class _RecordingModel:
    """Sustituto de HistGradientBoosting: registra cuantas filas recibio al ajustar."""

    def __init__(self) -> None:
        self.fit_rows: int | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> _RecordingModel:
        self.fit_rows = len(features)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        score = np.linspace(0.01, 0.99, len(features))
        return np.column_stack([1 - score, score])


def _synthetic_full() -> pd.DataFrame:
    dates = pd.date_range("2025-10-01", "2026-10-31", freq="D")
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        rng.normal(size=(len(dates), len(P2_FEATURE_COLUMNS_V2))),
        columns=list(P2_FEATURE_COLUMNS_V2),
    )
    frame["ts"] = dates
    frame["label"] = (rng.random(len(dates)) < 0.2).astype(int)
    return frame


def test_october_is_outside_fit_and_threshold_windows():
    full = _synthetic_full()
    masks = retrain_p2.split_masks(full)

    assert masks["oct26"].sum() == 31
    assert not (masks["oct26"] & (masks["train"] | masks["val"] | masks["test"])).any()


def test_challenger_fits_only_on_train_and_validation_windows(monkeypatch):
    full = _synthetic_full()
    recorder = _RecordingModel()
    monkeypatch.setattr(retrain_p2, "build_model", lambda *args, **kwargs: recorder)

    retrain_p2.fit_challenger(full)

    masks = retrain_p2.split_masks(full)
    expected = int((masks["train"] | masks["val"]).sum())
    assert recorder.fit_rows == expected
    assert recorder.fit_rows < int((masks["train"] | masks["val"] | masks["oct26"]).sum())
