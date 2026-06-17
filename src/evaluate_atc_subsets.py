import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Caricamento feature dei farmaci
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
features_path = os.path.join(base_dir, "data", "processed", "drug_features_chebi.csv")
df_features = pd.read_csv(features_path)

valid_features = df_features[df_features['ChEBI_ID'] != 'NOT_FOUND']
valid_drugs = set(valid_features['Drug_Name'])

subsets = {
    "P (Antiparassitari)": "ddinter_downloads_code_P.csv",
    "H (Ormonali Sistemici)": "ddinter_downloads_code_H.csv",
    "V (Vari)": "ddinter_downloads_code_V.csv",
    "B (Sangue)": "ddinter_downloads_code_B.csv",
    "D (Dermatologici)": "ddinter_downloads_code_D.csv",
    "R (Sistema Respiratorio)": "ddinter_downloads_code_R.csv",
    "A (Apparato Digerente)": "ddinter_downloads_code_A.csv",
    "L (Agenti Antineoplastici)": "ddinter_downloads_code_L.csv"
}

models = {
    "Logistic Regression (Baseline)": LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
    "Random Forest (Advanced)": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
}

print("=== VALUTAZIONE COMPARATIVA SUI SOTTO-DATASET ATC (5-Fold Cross Validation) ===\n")

# Tabella riassuntiva in formato Markdown
markdown_table = [
    "| Sotto-dataset ATC | Modello | Accuracy Media | Dev. Std Accuracy | Macro F1 Media | Dev. Std F1 |",
    "| :--- | :--- | :---: | :---: | :---: | :---: |"
]

for name, filename in subsets.items():
    filepath = os.path.join(base_dir, "data", "raw", filename)
    if not os.path.exists(filepath):
        print(f"File {filename} non trovato. Salto...")
        continue
        
    df_raw = pd.read_csv(filepath)
    # Rimuovi record non validi
    df_clean = df_raw[df_raw['Level'] != 'Unknown'].copy()
    
    # Mantieni solo se entrambi i farmaci sono mappati in ChEBI
    df_final = df_clean[
        df_clean['Drug_A'].isin(valid_drugs) &
        df_clean['Drug_B'].isin(valid_drugs)
    ].copy()
    
    if len(df_final) < 10:
        print(f"Sotto-dataset {name} ha troppi pochi dati ({len(df_final)} record). Salto...")
        continue

    # Join Feature
    df_final = df_final.merge(
        valid_features[['Drug_Name', 'Roles']],
        left_on='Drug_A',
        right_on='Drug_Name',
        how='left'
    ).rename(columns={'Roles': 'Roles_A'}).drop(columns=['Drug_Name'])

    df_final = df_final.merge(
        valid_features[['Drug_Name', 'Roles']],
        left_on='Drug_B',
        right_on='Drug_Name',
        how='left'
    ).rename(columns={'Roles': 'Roles_B'}).drop(columns=['Drug_Name'])

    df_final['Roles_A'] = df_final['Roles_A'].fillna('None').str.replace('|', ' ')
    df_final['Roles_B'] = df_final['Roles_B'].fillna('None').str.replace('|', ' ')
    X_semantic_text = df_final['Roles_A'] + " " + df_final['Roles_B']

    # Vectorizzazione
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(X_semantic_text)

    # Label encoding del target
    le = LabelEncoder()
    y = le.fit_transform(df_final['Level'])
    
    # Verifica che ci sia più di una classe nel target per fare CV
    if len(np.unique(y)) < 2:
        print(f"Sotto-dataset {name} ha meno di 2 classi target. Salto...")
        continue

    print(f"Sotto-dataset: {name}")
    print(f"  Record utilizzabili: {len(df_final)} / {len(df_raw)}")
    print(f"  Classi target: {list(le.classes_)}")
    print(f"  Feature semantiche (Vocabolario): {X.shape[1]}")

    scoring = ['accuracy', 'f1_macro']
    
    for m_name, model in models.items():
        # Esegui 5-fold CV
        scores = cross_validate(model, X, y, cv=5, scoring=scoring, n_jobs=-1)
        
        acc_mean = np.mean(scores['test_accuracy'])
        acc_std = np.std(scores['test_accuracy'])
        f1_mean = np.mean(scores['test_f1_macro'])
        f1_std = np.std(scores['test_f1_macro'])
        
        print(f"  -> {m_name}:")
        print(f"     Accuracy = {acc_mean:.4f} (+/- {acc_std:.4f})")
        print(f"     Macro F1 = {f1_mean:.4f} (+/- {f1_std:.4f})")
        
        # Aggiungi alla tabella markdown
        m_short = "LR" if "Logistic" in m_name else "RF"
        markdown_table.append(
            f"| {name} | {m_short} | {acc_mean:.4f} | &plusmn; {acc_std:.4f} | {f1_mean:.4f} | &plusmn; {f1_std:.4f} |"
        )
    print()

print("\n--- TABELLA RISULTATI IN FORMATO MARKDOWN ---")
print("\n".join(markdown_table))
