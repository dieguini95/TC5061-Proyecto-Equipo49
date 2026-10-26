"""Contrato de inferencia, serialización y causalidad de los modelos P1."""

from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from mibici_ml import config
from mibici_ml.features import add_p1_features, complete_station_hour_panel, p1_model_features
from mibici_ml.modeling.p1_model import build_p1_model, get_effective_params


def _input(rows=160):
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(
        {
            "lag_24": rng.integers(0, 11, rows),
            "lag_48": rng.integers(0, 11, rows),
            "lag_168": rng.integers(0, 11, rows),
            "mean_4weeks": rng.uniform(0, 10, rows),
            "hour": np.arange(rows) % 24,
            "day_of_week": np.arange(rows) % 7,
        }
    )
    frame["is_weekend"] = frame.day_of_week >= 5
    return frame


def test_feature_contract_selects_numeric_predictors_without_target_or_station():
    frame = _input()
    frame["departures"] = 99999
    frame["station_id"] = 423
    before = frame.copy(deep=True)
    selected = p1_model_features(frame)
    assert list(selected) == list(config.P1_FEATURE_COLUMNS)
    assert selected.dtypes.eq(np.dtype("float64")).all()
    pd.testing.assert_frame_equal(frame, before)
    assert "departures" not in selected and "station_id" not in selected


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_feature_contract_rejects_nonfinite_history_instead_of_imputing(value):
    frame = _input(3).astype("float64")
    frame.loc[0, "lag_24"] = value
    with pytest.raises(ValueError, match="no finitos"):
        p1_model_features(frame)


def test_feature_contract_reports_missing_columns():
    with pytest.raises(ValueError, match="lag_48"):
        p1_model_features(_input(3).drop(columns="lag_48"))


@pytest.mark.parametrize(
    "model_name,column", [("seasonal_168h", "lag_168"), ("mean_4weeks", "mean_4weeks")]
)
def test_serialized_baseline_reproduces_reference_exactly(model_name, column):
    features = _input()
    model = build_p1_model(model_name).fit(features)
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    np.testing.assert_array_equal(restored.predict(features), features[column].to_numpy())


@pytest.mark.parametrize("model_name", ["ridge", "hist_poisson", "hist_squared"])
def test_candidates_produce_finite_nonnegative_predictions_after_serialization(model_name):
    features = _input()
    target = np.maximum(0.0, 0.7 * features.lag_24 + 0.1 * features.hour)
    params = {"alpha": 2.0} if model_name == "ridge" else {"max_iter": 8}
    model = build_p1_model(model_name, params)
    fresh = clone(model)
    assert fresh.get_params()["params"] == params
    with pytest.raises(NotFittedError):
        fresh.predict(features)
    model.fit(features, target)
    expected = model.predict(features)
    assert np.isfinite(expected).all() and (expected >= 0).all()
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    np.testing.assert_allclose(restored.predict(features), expected, rtol=0, atol=0)
    if model_name == "ridge":
        transformed = model.estimator_.named_steps["features"].transform(
            p1_model_features(features)
        )
        assert issparse(transformed)
    else:
        assert model.estimator_.early_stopping is False
        assert model.estimator_.random_state == config.RANDOM_SEED
        assert model.estimator_.n_iter_ == 8


def test_clipping_is_part_of_saved_ridge_inference():
    features = _input(200)
    target = 10 - features.lag_24
    model = build_p1_model("ridge", {"alpha": 1.0}).fit(features, target)
    extrapolation = features.iloc[[0]].copy()
    extrapolation["lag_24"] = 100000
    assert model.estimator_.predict(p1_model_features(extrapolation))[0] < 0
    assert model.predict(extrapolation)[0] == 0.0


@pytest.mark.parametrize(
    "model_name,params",
    [
        ("unknown", {}),
        ("ridge", {"learning_rate": 0.1}),
        ("hist_poisson", {"early_stopping": True}),
        ("hist_squared", {"loss": "poisson"}),
        ("mean_4weeks", {"alpha": 1}),
    ],
)
def test_constructor_rejects_unsupported_models_or_overrides(model_name, params):
    with pytest.raises(ValueError):
        build_p1_model(model_name, params)


def test_effective_params_include_locked_temporal_protocol_and_overrides():
    params = get_effective_params("hist_poisson", {"max_iter": 9})
    assert params["loss"] == "poisson"
    assert params["max_iter"] == 9
    assert params["early_stopping"] is False
    assert params["random_state"] == config.RANDOM_SEED
    assert get_effective_params("mean_4weeks") == {"source_column": "mean_4weeks"}


@pytest.mark.parametrize("model_name", ["ridge", "hist_poisson", "hist_squared"])
def test_future_trip_changes_do_not_change_day_ahead_features_or_predictions(model_name):
    hours = pd.date_range("2025-10-01", "2025-11-30 23:00", freq="h")
    counts = pd.DataFrame(
        {"station_id": 1, "timestamp": hours, "departures": 1 + np.arange(len(hours)) % 9}
    )
    origin = pd.Timestamp("2025-11-06")
    modified = counts.copy()
    modified.loc[modified.timestamp >= origin, "departures"] = 123456
    original = add_p1_features(complete_station_hour_panel(counts, "2025-10", "2025-11"))
    changed = add_p1_features(complete_station_hour_panel(modified, "2025-10", "2025-11"))
    train = original.evaluation_eligible & original.timestamp.lt(origin)
    day = original.evaluation_eligible & original.forecast_origin.eq(origin)
    assert day.sum() == 24
    training = p1_model_features(original.loc[train])
    pd.testing.assert_frame_equal(training, p1_model_features(changed.loc[train]))
    inputs = p1_model_features(original.loc[day])
    changed_inputs = p1_model_features(changed.loc[day])
    pd.testing.assert_frame_equal(inputs, changed_inputs)
    overrides = {"alpha": 1.0} if model_name == "ridge" else {"max_iter": 5}
    original_model = build_p1_model(model_name, overrides).fit(
        training, original.loc[train, "departures"].astype(float)
    )
    changed_model = build_p1_model(model_name, overrides).fit(
        p1_model_features(changed.loc[train]), changed.loc[train, "departures"].astype(float)
    )
    np.testing.assert_array_equal(
        original_model.predict(inputs), changed_model.predict(changed_inputs)
    )
