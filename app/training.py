"""
Entraînement du pipeline ML et gestion de la baseline.

Utilise scikit-learn Pipeline pour éviter les fuites de données.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.preprocessing import preprocess_data

RANDOM_STATE = 42
MODEL_DIR = Path("models")


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: float = 0.2):
    """
    Découpe stratifiée train / val / test.

    Args:
        X: features
        y: cible
        test_size: proportion du test final (ex: 0.2)
        val_size: proportion de validation (parmi le train restant)

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_logistic_pipeline() -> Pipeline:
    """Construit le pipeline de régression logistique."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_tree_pipeline() -> Pipeline:
    """Construit le pipeline d'arbre de décision."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=5,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_type: str = "logistic") -> Pipeline:
    """
    Entraîne un pipeline ML sur les données d'entraînement.

    Args:
        X_train: features d'entraînement
        y_train: cible d'entraînement
        model_type: 'logistic' ou 'tree'

    Returns:
        Pipeline entraîné.
    """
    if model_type == "logistic":
        pipeline = build_logistic_pipeline()
    elif model_type == "tree":
        pipeline = build_tree_pipeline()
    else:
        raise ValueError(f"model_type inconnu : {model_type}")

    pipeline.fit(X_train, y_train)
    return pipeline


def compute_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Calcule les métriques principales sur un jeu donné."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, y_proba)),
    }


def save_baseline(model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, path: Path = None):
    """
    Sauvegarde la baseline de référence (métriques + métadonnées).

    ⚠️ Ne doit JAMAIS être recalculée automatiquement en CI,
    sinon les régressions deviennent invisibles.
    """
    path = path or MODEL_DIR / "baseline.json"
    path.parent.mkdir(exist_ok=True)

    metrics = compute_metrics(model, X_val, y_val)
    baseline = {
        "version": "1.0.0",
        "random_state": RANDOM_STATE,
        "model_type": type(model.named_steps["model"]).__name__,
        "metrics": metrics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    print(f"📝 Baseline sauvegardée : {metrics}")
    return baseline


def save_model(model: Pipeline, path: Path = None):
    """Sérialise le pipeline entraîné."""
    path = path or MODEL_DIR / "pipeline.joblib"
    path.parent.mkdir(exist_ok=True)
    joblib.dump(model, path)
    print(f"💾 Modèle sauvegardé : {path}")


def run_full_training(csv_path: str = "data/creditcard.csv"):
    """
    Pipeline complet : chargement → split → entraînement → sauvegarde.

    Retourne le modèle et les splits pour réutilisation.
    """
    import pandas as pd

    from app.preprocessing import validate_data

    print(f"📂 Chargement : {csv_path}")
    df = pd.read_csv(csv_path)
    validate_data(df)

    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print(f"📊 Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    model = train_model(X_train, y_train, model_type="logistic")
    save_model(model)
    save_baseline(model, X_val, y_val)

    print("✅ Entraînement terminé.")
    return model, (X_train, X_val, X_test, y_train, y_val, y_test)
