# src/train.py (Nouvelle version allégée)

# ==============================================================
# Chargement des bibliothèques (relevant pour l'entraînement)
# ==============================================================
from sklearn.linear_model import LinearRegression
import joblib # Pour charger les données et sauvegarder le modèle
import os     # Pour la gestion des chemins de fichiers

# ==============================================================
# Fonction d'entraînement du modèle
# ==============================================================
def train_model(data_dir="data/processed", output_dir="models"):
    """
    Charge les données d'entraînement, entraîne le modèle, et le sauvegarde.
    """
    print("🏋️ Début de l'entraînement du modèle...")
    
    # Chargement des données d'entraînement (Nouveau)
    try:
        X_train = joblib.load(os.path.join(data_dir, 'X_train.pkl'))
        y_train = joblib.load(os.path.join(data_dir, 'y_train.pkl'))
        print(f"✅ Données d'entraînement chargées depuis '{data_dir}'.")
    except FileNotFoundError:
        print(f"❌ Erreur: Fichiers de données introuvables. Assurez-vous que data_process.py a été exécuté.")
        return

    # Étape 6 : Entraînement du modèle
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\n✅ Modèle LinearRegression entraîné avec succès.")

    # Étape 8 : Sauvegarde du modèle
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, 'model.pkl'))
    print(f"\n✅ Modèle sauvegardé dans '{os.path.join(output_dir, 'model.pkl')}'")


# ==============================================================
# Exécution du script
# ==============================================================
if __name__ == "__main__":
    train_model()