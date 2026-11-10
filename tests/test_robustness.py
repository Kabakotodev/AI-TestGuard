"""Tests de robustesse et de sensibilité."""

import numpy as np
import pandas as pd
import pytest

from app.prediction import predict_fraud
from app.preprocessing import FEATURE_COLUMNS


def _make_features(n=20, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)


def test_extreme_amount_is_handled(trained_pipeline):
    """Un montant extrême ne doit pas casser le pipeline."""
    X = _make_features()
    X["Amount"] = 1e12
    out = predict_fraud(trained_pipeline, X)
    assert np.isfinite(out["probabilities"]).all()


def test_noise_sensitivity_is_measured(trained_pipeline):
    """Mesure la sensibilité à un bruit modéré (pas d'invariance imposée)."""
    X = _make_features(n=200)
    rng = np.random.default_rng(7)
    v_cols = [c for c in FEATURE_COLUMNS if c.startswith("V")]
    X_noisy = X.copy()
    X_noisy[v_cols] += rng.normal(0, 0.1, size=(len(X), len(v_cols)))

    p_orig = predict_fraud(trained_pipeline, X)["probabilities"]
    p_noisy = predict_fraud(trained_pipeline, X_noisy)["probabilities"]

    delta = np.abs(p_orig - p_noisy)
    assert np.isfinite(delta).all()
    assert delta.mean() < 0.5  # pas d'effondrement complet


def test_rejects_nan_input(trained_pipeline):
    X = _make_features()
    X.loc[0, "V1"] = np.nan
    with pytest.raises(ValueError):
        predict_fraud(trained_pipeline, X)


def test_rejects_inf_input(trained_pipeline):
    X = _make_features()
    X.loc[0, "Amount"] = np.inf
    with pytest.raises(ValueError):
        predict_fraud(trained_pipeline, X)


def test_single_row_is_handled(trained_pipeline):
    X = _make_features(n=1)
    out = predict_fraud(trained_pipeline, X)
    assert len(out["probabilities"]) == 1
