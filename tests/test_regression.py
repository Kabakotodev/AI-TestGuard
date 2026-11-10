"""Tests de non-régression par rapport à une baseline figée."""

import json
from pathlib import Path

import pytest

from app.training import compute_metrics

BASELINE_PATH = Path("models/baseline.json")
TOLERANCE = 0.05  # 5% de dégradation tolérée


@pytest.mark.skipif(not BASELINE_PATH.exists(), reason="Baseline absente")
def test_no_regression(trained_pipeline, splits):
    """Compare les métriques actuelles à la baseline figée."""
    _, X_val, _, _, y_val, _ = splits
    with open(BASELINE_PATH, encoding="utf-8") as f:
        baseline = json.load(f)["metrics"]

    current = compute_metrics(trained_pipeline, X_val, y_val)

    for metric, ref_value in baseline.items():
        delta = current[metric] - ref_value
        assert delta >= -TOLERANCE, (
            f"Régression détectée sur '{metric}': "
            f"{current[metric]:.4f} vs baseline {ref_value:.4f} (Δ={delta:+.4f})"
        )
