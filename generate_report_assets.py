"""
Script de génération des preuves pour le rapport de qualification.
Exécuter : python generate_report_assets.py
"""
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

from app.training import run_full_training, compute_metrics, split_data, train_model
from app.evaluation import evaluate_model, print_report, choose_threshold
from app.prediction import predict_fraud, load_model
from app.preprocessing import preprocess_data, validate_data

# Création du dossier de rapports
os.makedirs("reports", exist_ok=True)
os.makedirs("reports/screenshots", exist_ok=True)

print("="*70)
print(" GÉNÉRATION DES PREUVES POUR LE RAPPORT DE QUALIFICATION")
print("="*70)

# --- Préparation des données (Synthétiques pour éviter de dépendre de Kaggle) ---
print("\n[SETUP] Génération d'un dataset synthétique pour la démo...")
rng = np.random.default_rng(42)
n, n_frauds = 5000, 100
cols = [f"V{i}" for i in range(1, 29)]
data = {c: rng.standard_normal(n) for c in cols}
data["Time"] = rng.uniform(0, 172792, n)
data["Amount"] = rng.exponential(100, n)
data["Class"] = [1]*n_frauds + [0]*(n - n_frauds)
df = pd.DataFrame(data)
df.loc[df["Class"]==1, "V1"] += 3.0
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X, y = preprocess_data(df)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
model = train_model(X_train, y_train, model_type="logistic")

# ============================================================
# PARTIE II : Tests unitaires (Déjà couvert par `pytest`, on simule ici)
# ============================================================
print("\n" + "="*70)
print(" PARTIE II : VALIDATION DES DONNÉES (Tests Unitaires)")
print("="*70)
print(" Validation d'un DataFrame correct :", validate_data(df))

# Test de rejet volontaire
bad_df = df.copy()
bad_df.loc[0, 'V1'] = np.nan
try:
    validate_data(bad_df)
except ValueError as e:
    print(f" Rejet attendu (NaN) : {e}")

# ============================================================
# PARTIE III : Évaluation des performances
# ============================================================
print("\n" + "="*70)
print(" PARTIE III : ÉVALUATION DES PERFORMANCES")
print("="*70)

print("\n--- Rapport détaillé sur la Validation ---")
print_report(model, X_val, y_val)

# Graphiques ROC et PR
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
y_proba = model.predict_proba(X_val)[:, 1]

# ROC
fpr, tpr, _ = roc_curve(y_val, y_proba)
axes[0].plot(fpr, tpr, color='blue', label='Régression Logistique')
axes[0].plot([0,1], [0,1], 'k--')
axes[0].set_title('Courbe ROC')
axes[0].set_xlabel('Taux de Faux Positifs (FPR)')
axes[0].set_ylabel('Taux de Vrais Positifs (TPR)')
axes[0].legend()

# PR
precision, recall, _ = precision_recall_curve(y_val, y_proba)
axes[1].plot(recall, precision, color='green')
axes[1].set_title('Courbe Précision-Rappel')
axes[1].set_xlabel('Rappel (Recall)')
axes[1].set_ylabel('Précision')
plt.tight_layout()
plt.savefig("reports/screenshots/partie3_courbes_roc_pr.png")
print(" Graphique sauvegardé : reports/screenshots/partie3_courbes_roc_pr.png")
plt.show()

# ============================================================
# PARTIE IV : Intégration (Save / Load)
# ============================================================
print("\n" + "="*70)
print(" PARTIE IV : TEST D'INTÉGRATION (Save / Load)")
print("="*70)
from app.training import save_model
save_model(model, Path("models/pipeline_demo.joblib"))
loaded_model = load_model(Path("models/pipeline_demo.joblib"))

probas_orig = predict_fraud(model, X_test)["probabilities"]
probas_load = predict_fraud(loaded_model, X_test)["probabilities"]
diff = np.max(np.abs(probas_orig - probas_load))
print(f" Différence max après rechargement : {diff:.2e}")
print(" Test d'intégration : OK" if diff < 1e-6 else " Échec")

# ============================================================
# PARTIE V : Robustesse
# ============================================================
print("\n" + "="*70)
print(" PARTIE V : TEST DE ROBUSTESSE")
print("="*70)

# Test 1 : Amount extrême
X_extreme = X_test.copy()
X_extreme["Amount"] = 1e12
probas_extreme = predict_fraud(model, X_extreme)["probabilities"]
print(f" Amount extrême (1e12) : prédictions valides ? {np.isfinite(probas_extreme).all()}")

# Test 2 : Bruit sur les variables
X_noisy = X_test.copy()
v_cols = [c for c in X.columns if c.startswith("V")]
X_noisy[v_cols] += rng.normal(0, 0.1, size=(len(X_noisy), len(v_cols)))
probas_noisy = predict_fraud(model, X_noisy)["probabilities"]
flips = ((probas_orig >= 0.5) != (probas_noisy >= 0.5)).sum()
print(f" Bruit léger sur V1-V28 : {flips} changements de classe sur {len(X_test)}")

# ============================================================
# PARTIE VI : Non-régression
# ============================================================
print("\n" + "="*70)
print(" PARTIE VI : NON-RÉGRESSION (BASELINE)")
print("="*70)
# On simule une baseline
baseline_metrics = compute_metrics(model, X_val, y_val)
with open("models/baseline.json", "w") as f:
    json.dump({"metrics": baseline_metrics}, f, indent=2)
print(f" Baseline sauvegardée : {baseline_metrics}")

# ============================================================
# PARTIE VIII : Performance & Latence
# ============================================================
print("\n" + "="*70)
print(" PARTIE VIII : PERFORMANCE & LATENCE")
print("="*70)
latencies = []
for _ in range(100):
    start = time.perf_counter()
    model.predict_proba(X_test.iloc[:1])
    latencies.append(time.perf_counter() - start)
latencies = np.array(latencies)
print(f" Latence moyenne : {latencies.mean()*1000:.2f} ms")
print(f" Latence P95 : {np.percentile(latencies, 95)*1000:.2f} ms")
print(f" Taille du modèle : {os.path.getsize('models/pipeline_demo.joblib')/1024:.1f} Ko")

print("\n" + "="*70)
print(" TOUTES LES PREUVES ONT ÉTÉ GÉNÉRÉES DANS reports/")
print("="*70)