"""Tests d'intégration : pipeline de bout en bout."""

import joblib
import numpy as np

from app.evaluation import evaluate_model
from app.prediction import predict_fraud
from app.preprocessing import preprocess_data, validate_data
from app.training import compute_metrics, save_model


def test_end_to_end_pipeline(synthetic_df, tmp_path):
    """Charge → valide → split → train → save → load → predict."""
    validate_data(synthetic_df)
    X, y = preprocess_data(synthetic_df)

    from app.training import split_data, train_model

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    model = train_model(X_train, y_train, model_type="logistic")

    # Sauvegarde et rechargement
    path = tmp_path / "pipeline.joblib"
    save_model(model, path)
    loaded = joblib.load(path)

    # Cohérence des prédictions
    p1 = predict_fraud(model, X_test)["probabilities"]
    p2 = predict_fraud(loaded, X_test)["probabilities"]
    assert np.max(np.abs(p1 - p2)) < 1e-6


def test_no_scaler_leakage(trained_pipeline, splits):
    """Le scaler ne doit être ajusté QUE sur le train."""
    X_train, _, X_test, _, _, _ = splits
    scaler = trained_pipeline.named_steps["scaler"]
    mean_train = scaler.mean_
    mean_test = X_test.mean().values
    assert np.abs(mean_train - mean_test).mean() > 0


def test_metrics_are_deterministic(trained_pipeline, splits):
    """Deux évaluations successives donnent les mêmes résultats."""
    _, X_val, _, _, y_val, _ = splits
    m1 = compute_metrics(trained_pipeline, X_val, y_val)
    m2 = compute_metrics(trained_pipeline, X_val, y_val)
    assert m1 == m2


def test_full_evaluation_report(trained_pipeline, splits):
    _, X_val, _, _, y_val, _ = splits
    report = evaluate_model(trained_pipeline, X_val, y_val)
    for key in ["precision", "recall", "f1", "roc_auc", "pr_auc", "TP", "FP", "TN", "FN"]:
        assert key in report
    assert 0.0 <= report["recall"] <= 1.0
