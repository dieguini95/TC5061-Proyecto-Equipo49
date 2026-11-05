from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mibici_ml.api.app import _provider_predictions
from mibici_ml.api.errors import PredictionShapeError


@pytest.mark.parametrize("values", [[1, 2], (1, 2)])
def test_provider_contract_accepts_python_sequences(values: list[int] | tuple[int, ...]) -> None:
    assert _provider_predictions(values, 2) == [1, 2]


@pytest.mark.parametrize(
    "values",
    [
        np.array([1, 2]),
        pd.Series([1, 2]),
        pd.DataFrame({"cluster": [1, 2]}),
        pd.DataFrame({"cluster": [1, 2], "extra": [3, 4]}),
    ],
)
def test_provider_contract_requires_normalized_sequence(values: object) -> None:
    with pytest.raises(PredictionShapeError):
        _provider_predictions(values, 2)
