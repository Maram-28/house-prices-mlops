# src/data_process.py

# ==============================================================
# Étape 1 : Chargement des bibliothèques (relevant pour le traitement des données)
# ==============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib # Pour la sauvegarde des données prétraitées
import os     # Pour la gestion des chemins de fichiers

# ==============================================================
# Fonction de préparation des données
# ==============================================================
def prepare_data(input_path="data/train.csv", output_dir="data/processed"):
    """
    Charge les données brutes, les nettoie, les divise en jeux
    d'entraînement et de test, puis les sauvegarde.
    """
    print("🚀 Début de la préparation des données...")

    # Étape 2 : Chargement des données
    df = pd.read_csv(input_path)
    print(f"Chargement de {input_path} réussi. Shape: {df.shape}")

    # Étape 3 : Sélection des features numériques et suppression des NaN
    df_numeric = df.select_dtypes(include=[np.number])
    df_numeric = df_numeric.dropna()
    print(f"Nombre de lignes après suppression des NaN : {df_numeric.shape[0]}")

    # Étape 4 : Séparation features (X) / target (y)
    X = df_numeric.drop("SalePrice", axis=1)
    y = df_numeric["SalePrice"]
    y_binned = pd.cut(y, bins=10, labels=False) # Pour la stratification

    # Étape 5 : Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_binned
    )

    print(f"Taille du jeu d'entraînement : {X_train.shape}")
    print(f"Taille du jeu de test : {X_test.shape}")
    
    # Sauvegarde des jeux de données traités (Nouveau)
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(X_train, os.path.join(output_dir, 'X_train.pkl'))
    joblib.dump(X_test, os.path.join(output_dir, 'X_test.pkl'))
    joblib.dump(y_train, os.path.join(output_dir, 'y_train.pkl'))
    joblib.dump(y_test, os.path.join(output_dir, 'y_test.pkl'))
    
    print(f"\n✅ Données traitées et sauvegardées dans '{output_dir}'.")


# ==============================================================
# Exécution du script
# ==============================================================
if __name__ == "__main__":
    prepare_data()