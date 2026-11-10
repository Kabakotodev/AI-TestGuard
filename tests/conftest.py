"""Fixtures pytest partagées."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.preprocessing import FEATURE_COLUMNS, REQUIRED_COLUMNS
from app.training import split_data, train_model


@pytest.fixture(scope="session")
def synthetic_df():
    """Génère un dataset synthétique cohérent pour les tests."""
    rng = np.random.default_rng(42)
    n, n_frauds = 2000, 40
    n_norm = n - n_frauds

    cols = [f"V{i}" for i in range(1, 29)]
    data = {c: rng.standard_normal(n) for c in cols}
    data["Time"] = rng.uniform(0, 172792, n)
    data["Amount"] = rng.exponential(100, n)
    data["Class"] = [1] * n_frauds + [0] * n_norm
    df = pd.DataFrame(data)
    df.loc[df["Class"] == 1, "V1"] += 3.0
    df.loc[df["Class"] == 1, "V2"] -= 2.5
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


@pytest.fixture(scope="session")
def trained_pipeline(synthetic_df):
    """Pipeline entraîné sur le dataset synthétique."""
    X = synthetic_df[FEATURE_COLUMNS]
    y = synthetic_df["Class"]
    X_train, _, _, y_train, _, _ = split_data(X, y)
    return train_model(X_train, y_train, model_type="logistic")


@pytest.fixture(scope="session")
def splits(synthetic_df):
    """Retourne les splits train/val/test prêts à l'emploi."""
    X = synthetic_df[FEATURE_COLUMNS]
    y = synthetic_df["Class"]
    return split_data(X, y)


@pytest.fixture
def valid_df():
    """Petit DataFrame valide."""
    return pd.DataFrame({**{c: [1.0] for c in REQUIRED_COLUMNS[:-1]}, "Class": [0]})
