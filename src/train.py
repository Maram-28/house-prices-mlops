# ==============================================================
# Étape 1 : Chargement des bibliothèques
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ==============================================================
# Étape 2 : Chargement des données
# ==============================================================

df = pd.read_csv('data/train.csv')

print("Aperçu du dataset :")
print(df.head())

print(f"\nNombre de lignes : {df.shape[0]}, Nombre de colonnes : {df.shape[1]}")

# ==============================================================
# Étape 3 : Sélection des features numériques et suppression des NaN
# ==============================================================

df_numeric = df.select_dtypes(include=[np.number])
df_numeric = df_numeric.dropna()

print(f"\nNombre de lignes après suppression des NaN : {df_numeric.shape[0]}")
print(df_numeric.describe().T.head())

# ==============================================================
# Étape 4 : Séparation features (X) / target (y)
# ==============================================================

X = df_numeric.drop("SalePrice", axis=1)
y = df_numeric["SalePrice"]

# Stratification par binning
y_binned = pd.cut(y, bins=10, labels=False)

# ==============================================================
# Étape 5 : Split train / test
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_binned
)

print(f"\nTaille du jeu d'entraînement : {X_train.shape}")
print(f"Taille du jeu de test : {X_test.shape}")

# ==============================================================
# Étape 6 : Entraînement du modèle
# ==============================================================

model = LinearRegression()
model.fit(X_train, y_train)

print("\n✅ Modèle LinearRegression entraîné avec succès.")

# ==============================================================
# Étape 7 : Évaluation du modèle
# ==============================================================

# Prédictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calcul des métriques
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Affichage des résultats
print("\n" + "="*50)
print("📊 RÉSULTATS D'ENTRAÎNEMENT")
print("="*50)
print(f"RMSE Train: {rmse_train:,.2f}")
print(f"R² Train: {r2_train:.4f}")

print("\n" + "="*50)
print("📊 RÉSULTATS DE TEST")
print("="*50)
print(f"RMSE Test: {rmse_test:,.2f}")
print(f"R² Test: {r2_test:.4f}")

# ==============================================================
# Étape 8 : Sauvegarde du modèle
# ==============================================================

joblib.dump(model, 'models/model.pkl')
print("\n✅ Modèle sauvegardé dans 'models/model.pkl'")