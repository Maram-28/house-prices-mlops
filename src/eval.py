# src/eval.py

# ==============================================================
# Chargement des bibliothèques (relevant pour l'évaluation)
# ==============================================================
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib # Pour charger le modèle et les données de test
import os     # Pour la gestion des chemins de fichiers

# ==============================================================
# Fonction d'évaluation du modèle
# ==============================================================
def evaluate_model(data_dir="data/processed", model_path="models/model.pkl"):
    """
    Charge les données de test et le modèle entraîné, puis évalue la performance.
    """
    print("🔍 Début de l'évaluation du modèle...")
    
    # Chargement des données de test et du modèle
    try:
        X_test = joblib.load(os.path.join(data_dir, 'X_test.pkl'))
        y_test = joblib.load(os.path.join(data_dir, 'y_test.pkl'))
        model = joblib.load(model_path)
        print("✅ Données de test et modèle chargés avec succès.")
    except FileNotFoundError as e:
        print(f"❌ Erreur: Fichiers manquants. Assurez-vous que data_process.py et train.py ont été exécutés. Détail: {e}")
        return

    # Étape 7 : Évaluation du modèle
    y_test_pred = model.predict(X_test)

    # Calcul des métriques
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test = r2_score(y_test, y_test_pred)

    # Affichage des résultats
    print("\n" + "="*50)
    print("📊 RÉSULTATS DE TEST")
    print("="*50)
    print(f"RMSE Test: {rmse_test:,.2f}")
    print(f"R² Test: {r2_test:.4f}")
    print("\n✅ Évaluation terminée.")


# ==============================================================
# Exécution du script
# ==============================================================
if __name__ == "__main__":
    evaluate_model()