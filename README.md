# 🛡️ AI-TestGuard

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
git clone <ton-repo>
cd AI-TestGuard
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt