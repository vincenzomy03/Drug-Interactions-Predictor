import pandas as pd
import os

# 1. Caricamento dei dati elaborati
df_interactions = pd.read_csv("data/processed/ddinter_clean.csv")
df_features = pd.read_csv("data/processed/drug_features_chebi.csv")

print(f"Interazioni iniziali: {len(df_interactions)}")

# Isola i farmaci validi (escludi NOT_FOUND)
valid_features = df_features[df_features['ChEBI_ID'] != 'NOT_FOUND']
valid_drugs = set(valid_features['Drug_Name'])

# 2. Filtro delle interazioni bilaterali
# Mantieni la riga solo se sia Drug_A che Drug_B hanno una controparte in ChEBI
df_final = df_interactions[
    df_interactions['Drug_A'].isin(valid_drugs) &
    df_interactions['Drug_B'].isin(valid_drugs)
].copy()

print(
    f"Interazioni utili (entrambi i farmaci mappati in ChEBI): {len(df_final)}")

# 3. Join delle Feature per Drug_A
df_final = df_final.merge(
    valid_features[['Drug_Name', 'Parent_Classes', 'Roles']],
    left_on='Drug_A',
    right_on='Drug_Name',
    how='left'
).rename(columns={'Parent_Classes': 'Parent_Classes_A', 'Roles': 'Roles_A'}).drop(columns=['Drug_Name'])

# 4. Join delle Feature per Drug_B
df_final = df_final.merge(
    valid_features[['Drug_Name', 'Parent_Classes', 'Roles']],
    left_on='Drug_B',
    right_on='Drug_Name',
    how='left'
).rename(columns={'Parent_Classes': 'Parent_Classes_B', 'Roles': 'Roles_B'}).drop(columns=['Drug_Name'])

# 5. Salvataggio della matrice semantica tabulare
output_path = "data/processed/dataset_final_tabular.csv"
df_final.to_csv(output_path, index=False)
print(f"Dataset finale salvato correttamente in: {output_path}")

print("\nEsempio di record generato:")
print(df_final[['Drug_A', 'Drug_B', 'Level', 'Roles_A', 'Roles_B']].head(
    1).to_dict(orient='records'))
