"""Tests unitaires pour app.preprocessing."""

import numpy as np
import pandas as pd
import pytest

from app.preprocessing import (
    FEATURE_COLUMNS,
    handle_invalid_rows,
    preprocess_data,
    validate_data,
)


def test_validate_ok(valid_df):
    assert validate_data(valid_df) is True


def test_validate_missing_column(valid_df):
    df = valid_df.drop(columns=["Amount"])
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        validate_data(df)


def test_validate_duplicated_columns(valid_df):
    df = pd.concat([valid_df, valid_df[["V1"]]], axis=1)
    with pytest.raises(ValueError, match="dupliquées"):
        validate_data(df)


def test_validate_invalid_target(valid_df):
    df = valid_df.copy()
    df.loc[0, "Class"] = 2
    with pytest.raises(ValueError, match="Cible invalide"):
        validate_data(df)


def test_validate_nan(valid_df):
    df = valid_df.copy()
    df.loc[0, "V1"] = np.nan
    with pytest.raises(ValueError, match="Valeurs manquantes"):
        validate_data(df)


def test_validate_inf(valid_df):
    df = valid_df.copy()
    df.loc[0, "Amount"] = np.inf
    with pytest.raises(ValueError, match="Valeurs infinies"):
        validate_data(df)


def test_validate_empty():
    with pytest.raises(ValueError, match="vide"):
        validate_data(pd.DataFrame())


def test_validate_not_dataframe():
    with pytest.raises(ValueError, match="DataFrame"):
        validate_data([1, 2, 3])


def test_preprocess_shapes(synthetic_df):
    X, y = preprocess_data(synthetic_df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == len(synthetic_df)
    assert set(y.unique()) <= {0, 1}


def test_handle_invalid_rows(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "V1"] = np.nan
    df.loc[1, "Amount"] = np.inf
    cleaned = handle_invalid_rows(df)
    assert len(cleaned) == len(df) - 2
