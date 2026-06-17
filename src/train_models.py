import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os

# 1. Caricamento dati
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(base_dir, "data", "processed", "dataset_final_tabular.csv")
df = pd.read_csv(data_path)

# Gestione dei valori nulli nelle feature semantiche
df['Roles_A'] = df['Roles_A'].fillna('None').str.replace('|', ' ')
df['Roles_B'] = df['Roles_B'].fillna('None').str.replace('|', ' ')

# Unione delle feature semantiche della coppia in un unico blocco di testo per la vettorizzazione
X_semantic_text = df['Roles_A'] + " " + df['Roles_B']

# 2. Vettorizzazione (Estrazione Feature Numeriche dalla KB)
vectorizer = CountVectorizer(binary=True)
X_semantic = vectorizer.fit_transform(X_semantic_text)

# Codifica del Target (Minor, Moderate, Major -> 0, 1, 2)
le = LabelEncoder()
y = le.fit_transform(df['Level'])

print(f"Matrice delle feature semantiche generata: {X_semantic.shape}")
print(f"Classi target codificate: {list(le.classes_)}")

# 3. Fase di Tuning degli Iperparametri (Grid Search)
print("\nInizio fase di Tuning degli Iperparametri (Grid Search su Random Forest)...")
rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

# Griglia ristretta per garantire tempi di computazione rapidi (< 2 minuti)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 15]
}

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X_semantic, y)

print(f"-> Migliori parametri trovati: {grid_search.best_params_}")
print(f"-> Miglior Macro F1 Score in CV: {grid_search.best_score_:.4f}")

# 4. Definizione dei Modelli per il Benchmark Comparativo
best_rf = grid_search.best_estimator_

models = {
    "Baseline (Logistic Regression - KB features)": LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
    "Advanced (Tuned Random Forest - KB features)": best_rf
}

# 5. Cross-Validation e Calcolo Metriche con Deviazione Standard
scoring = ['accuracy', 'f1_macro']

print("\nInizio fase di Cross-Validation di confronto (5-Fold)...")
for name, model in models.items():
    print(f"\nValutazione modello: {name}...")
    scores = cross_validate(model, X_semantic, y, cv=5, scoring=scoring, n_jobs=-1)

    # Calcolo medie e deviazioni standard
    acc_mean = np.mean(scores['test_accuracy'])
    acc_std = np.std(scores['test_accuracy'])
    f1_mean = np.mean(scores['test_f1_macro'])
    f1_std = np.std(scores['test_f1_macro'])

    print(f"-> Accuracy: {acc_mean:.4f} (+/- {acc_std:.4f})")
    print(f"-> F1-Score (Macro): {f1_mean:.4f} (+/- {f1_std:.4f})")

print("\n[SUCCESSO] Addestramento, tuning e valutazione completati.")
