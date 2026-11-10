"""
Module de prétraitement et validation des données.

Responsabilités :
- Valider le schéma du DataFrame (colonnes, types, cible).
- Nettoyer les données invalides (NaN, inf).
- Séparer features / cible.
"""

import numpy as np
import pandas as pd

# Colonnes attendues (Kaggle Credit Card Fraud)
V_COLUMNS = [f"V{i}" for i in range(1, 29)]
REQUIRED_COLUMNS = ["Time"] + V_COLUMNS + ["Amount", "Class"]
FEATURE_COLUMNS = V_COLUMNS + ["Amount"]


def validate_data(df: pd.DataFrame) -> bool:
    """
    Valide le schéma et la cohérence du DataFrame.

    Lève ValueError si le contrat n'est pas respecté.

    Returns:
        True si valide.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("L'entrée doit être un pandas DataFrame")

    if df.empty:
        raise ValueError("DataFrame vide")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")

    if df.columns.duplicated().any():
        raise ValueError("Colonnes dupliquées détectées")

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Cible invalide : 'Class' doit contenir 0 ou 1 uniquement")

    feature_cols = [c for c in REQUIRED_COLUMNS if c != "Class"]
    if df[feature_cols].isna().any().any():
        raise ValueError("Valeurs manquantes dans les features")

    if np.isinf(df[feature_cols].values).any():
        raise ValueError("Valeurs infinies détectées dans les features")

    return True


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Sépare les features et la cible après validation.

    Args:
        df: DataFrame brut contenant 'Class'.

    Returns:
        (X, y) avec X de dimensions (n, 29) et y ∈ {0, 1}.
    """
    validate_data(df)
    X = df[FEATURE_COLUMNS].copy()
    y = df["Class"].astype(int).copy()
    return X, y


def handle_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stratégie documentée : rejette les lignes avec NaN ou inf.

    Utilisé dans un contexte de production où l'on préfère
    rejeter plutôt que d'imputer arbitrairement.
    """
    feature_cols = [c for c in REQUIRED_COLUMNS if c != "Class"]
    mask = (
        df[feature_cols].notna().all(axis=1)
        & ~np.isinf(df[feature_cols].values).any(axis=1)
    )
    n_rejected = (~mask).sum()
    if n_rejected > 0:
        print(f"⚠️ {n_rejected} lignes rejetées (NaN ou inf)")
    return df[mask].reset_index(drop=True)
