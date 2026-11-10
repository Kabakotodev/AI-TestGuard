"""
Évaluation du modèle : métriques, matrices de confusion, courbes.
"""


import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def compute_confusion(y_true, y_pred) -> dict:
    """Retourne TP, FP, TN, FN et la matrice."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "FPR": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "FNR": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "matrix": cm.tolist(),
    }


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict:
    """
    Évalue un modèle sur un jeu donné.

    Returns:
        dict avec Precision, Recall, F1, ROC-AUC, PR-AUC, matrice de confusion.
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    conf = compute_confusion(y, y_pred)
    return {
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "pr_auc": float(average_precision_score(y, y_proba)),
        **conf,
    }


def print_report(model: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5):
    """Affiche un rapport lisible (matrice + classification report)."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print("Matrice de confusion :")
    print(confusion_matrix(y, y_pred, labels=[0, 1]))
    print(
        classification_report(
            y,
            y_pred,
            labels=[0, 1],
            target_names=["Normale", "Fraude"],
            zero_division=0,
        )
    )
    print(f"ROC-AUC : {roc_auc_score(y, y_proba):.4f}")
    print(f"PR-AUC  : {average_precision_score(y, y_proba):.4f}")


def choose_threshold(
    model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, min_recall: float = 0.80
) -> float:
    """
    Choisit le seuil de décision sur la VALIDATION (jamais sur le test).

    Stratégie : plus petit seuil garantissant recall >= min_recall.
    """
    probas = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5

    for t in thresholds:
        y_pred = (probas >= t).astype(int)
        r = recall_score(y_val, y_pred, zero_division=0)
        if r >= min_recall:
            best_threshold = t

    print(f" Seuil choisi (recall ≥ {min_recall}) : {best_threshold:.2f}")
    return float(best_threshold)
