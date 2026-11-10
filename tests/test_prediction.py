"""Tests unitaires pour app.prediction."""

import numpy as np
import pandas as pd
import pytest

from app.prediction import predict_fraud
from app.preprocessing import FEATURE_COLUMNS


def _make_features(n=5, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)


def test_predict_returns_valid_probas(trained_pipeline):
    X = _make_features()
    out = predict_fraud(trained_pipeline, X)
    assert np.isfinite(out["probabilities"]).all()
    assert (out["probabilities"] >= 0).all()
    assert (out["probabilities"] <= 1).all()
    assert set(np.unique(out["predictions"])) <= {0, 1}


def test_predict_rejects_missing_columns(trained_pipeline):
    X = _make_features().drop(columns=["Amount"])
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        predict_fraud(trained_pipeline, X)


def test_predict_rejects_extra_columns(trained_pipeline):
    X = _make_features()
    X["Unexpected"] = 0.0
    with pytest.raises(ValueError, match="non autorisées"):
        predict_fraud(trained_pipeline, X)


def test_predict_rejects_nan(trained_pipeline):
    X = _make_features()
    X.loc[0, "V1"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        predict_fraud(trained_pipeline, X)


def test_predict_rejects_inf(trained_pipeline):
    X = _make_features()
    X.loc[0, "Amount"] = np.inf
    with pytest.raises(ValueError, match="infinies"):
        predict_fraud(trained_pipeline, X)


def test_predict_rejects_invalid_threshold(trained_pipeline):
    X = _make_features()
    with pytest.raises(ValueError, match="Seuil invalide"):
        predict_fraud(trained_pipeline, X, threshold=1.5)


def test_predict_rejects_empty(trained_pipeline):
    with pytest.raises(ValueError, match="vide"):
        predict_fraud(trained_pipeline, pd.DataFrame(columns=FEATURE_COLUMNS))
