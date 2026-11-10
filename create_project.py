"""
Script de création automatique de la structure du projet
credit_card_fraud_detection.

Usage:
    python create_project.py
    python create_project.py --path /chemin/vers/parent
"""

import argparse
import os
from pathlib import Path
from textwrap import dedent


# ============================================================
# CONTENU DES FICHIERS
# ============================================================

FILES = {
    # -------------------- ROOT --------------------
    "requirements.txt": """pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
pytest>=7.0
pytest-cov>=4.0
ruff>=0.1
joblib>=1.3
matplotlib>=3.7
scipy>=1.10
""",

    "pyproject.toml": """[project]
name = "credit-card-fraud-detection"
version = "0.1.0"
description = "QA pipeline pour modèle de détection de fraude bancaire"
requires-python = ">=3.9"

[tool.ruff]
line-length = 100
target-version = "py39"
select = ["E", "F", "W", "I", "N"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=app --cov-report=term-missing --cov-report=xml"

[tool.coverage.run]
source = ["app"]
omit = ["tests/*"]
""",

    ".gitignore": """# Données
data/*.csv
data/*.zip

# Modèles
models/*.joblib
models/*.pkl
models/*.json

# Rapports
reports/*
!reports/.gitkeep

# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.coverage
coverage.xml
htmlcov/

# Environnements
.env
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp
.DS_Store
""",

    "README.md": """# 🛡️ credit_card_fraud_detection

Pipeline QA complet pour un modèle de détection de fraude bancaire.

## 📋 Structure

- `app/` — Code source du pipeline (preprocessing, training, prediction, evaluation)
- `tests/` — Suite de tests pytest (unitaire, intégration, robustesse, non-régression)
- `data/` — Données (Kaggle, non versionnées)
- `models/` — Modèles et baselines sérialisés (générés)
- `reports/` — Rapports d'exécution (générés)
- `.github/workflows/` — CI GitHub Actions

## 🚀 Installation


```bash
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\\\\Scripts\\\\activate
pip install -r requirements.txt
"""
}