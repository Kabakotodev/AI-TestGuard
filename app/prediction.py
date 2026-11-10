"""
Module de prédiction et contrat de sortie.

Garantit que :
- les entrées sont validées ;
- les probabilités sont dans [0, 1] et finies.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.preprocessing import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = Path("models/pipeline.joblib")
DEFAULT_THRESHOLD = 0.5


def load_model(path: Path = None) -> Pipeline:
    """
    Charge un pipeline sérialisé.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
    """
    path = path or DEFAULT_MODEL_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return joblib.load(path)


def _validate_input_features(X: pd.DataFrame) -> None:
    """Vérifie le contrat d'entrée (colonnes, NaN, inf)."""
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X doit être un DataFrame")

    if X.empty:
        raise ValueError("X est vide")

    missing = set(FEATURE_COLUMNS) - set(X.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")

    extra = set(X.columns) - set(FEATURE_COLUMNS)
    if extra:
        raise ValueError(f"Colonnes non autorisées : {sorted(extra)}")

    if X.isna().any().any():
        raise ValueError("X contient des NaN")

    if np.isinf(X.values).any():
        raise ValueError("X contient des valeurs infinies")


def predict_fraud(
    model: Pipeline, X: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
) -> dict:
    """
    Produit les prédictions et métadonnées.

    Args:
        model: pipeline entraîné
        X: features (doit contenir FEATURE_COLUMNS)
        threshold: seuil de décision ∈ [0, 1]

    Returns:
        dict {probabilities, predictions, threshold}

    Raises:
        ValueError: si l'entrée est invalide.
    """
    _validate_input_features(X)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Seuil invalide : {threshold}")

    probas = model.predict_proba(X)[:, 1]

    if not np.isfinite(probas).all():
        raise RuntimeError("Probabilités non finies produites par le modèle")
    if (probas < 0).any() or (probas > 1).any():
        raise RuntimeError("Probabilités hors de [0, 1]")

    return {
        "probabilities": probas,
        "predictions": (probas >= threshold).astype(int),
        "threshold": threshold,
    }
