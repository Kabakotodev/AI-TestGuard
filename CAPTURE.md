 Comment faire tes captures d'écran pour le rapport
Une fois le script créé, exécute-le dans ton terminal :

bash
python generate_report_assets.py
Voici ce que tu dois capturer pour chaque partie de ton rapport :

📌 Partie II : Tests unitaires
Action : Dans ton terminal, lance pytest tests/test_preprocessing.py -v

Capture : Prends une capture de la liste des tests avec les "PASSED" en vert.

📌 Partie III : Évaluation des performances
Action : Le script génère un fichier image et affiche du texte.

Captures :

Ouvre l'image générée : reports/screenshots/partie3_courbes_roc_pr.png (Montre les courbes ROC et PR).
Capture le terminal où s'affiche la Matrice de confusion et le classification_report (généré par print_report).
📌 Partie IV : Intégration
Action : Le script affiche le test Save/Load.

Capture : Capture la ligne du terminal : 🔄 Différence max après rechargement : ... suivie de ✅ Test d'intégration : OK.

📌 Partie V : Robustesse
Action : Le script simule des perturbations.

Capture : Capture les lignes du terminal : 💰 Amount extrême... et 🔊 Bruit léger....

📌 Partie VI : Non-régression
Action : Le script génère la baseline.

Capture : Capture la ligne du terminal ✅ Baseline sauvegardée : ....

Ensuite, lance pytest tests/test_regression.py -v. Cette fois, le test ne sera plus "SKIPPED" mais "PASSED" (car le fichier baseline.json existe maintenant). Capture ce résultat !

📌 Partie VIII : Performance
Action : Le script mesure la latence.

Capture : Capture les lignes ⏱️ Latence moyenne... et 💾 Taille du modèle....

