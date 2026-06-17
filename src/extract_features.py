import pandas as pd
# pyrefly: ignore [missing-import]
from owlready2 import get_ontology, ThingClass, Restriction
import os

# 1. Caricamento e pulizia dataset
df_full = pd.read_csv("data/processed/ddinter_full.csv")
df_clean = df_full[df_full['Level'] != 'Unknown'].copy()
df_clean.to_csv("data/processed/ddinter_clean.csv", index=False)

unique_drugs = pd.concat([df_clean['Drug_A'], df_clean['Drug_B']]).unique()
print(f"Dataset pulito. Record rimanenti: {len(df_clean)}")
print(f"Farmaci unici da mappare: {len(unique_drugs)}")

# 2. Caricamento Ontologia
onto_path = "data/ontologies/chebi.owl"
print("\nCaricamento ontologia ChEBI...")
onto = get_ontology(f"file://{onto_path}").load()

# Identificazione della proprietà dei sinonimi in ChEBI
synonym_prop = onto.search(iri="*has_exact_synonym")
has_synonym = synonym_prop[0] if synonym_prop else None

# 3. Costruzione dell'indice dei nomi e sinonimi in memoria per velocizzare l'allineamento (O(1))
print("\nCostruzione dell'indice dei nomi e sinonimi in memoria...")
name_to_node = {}

for c in onto.classes():
    try:
        for lbl in c.label:
            name_to_node[str(lbl).strip().lower()] = c
    except AttributeError:
        pass
    if has_synonym:
        try:
            syns = getattr(c, has_synonym.name, [])
            for syn in syns:
                name_to_node[str(syn).strip().lower()] = c
        except AttributeError:
            pass

print(f"Indice costruito con successo. Voci indicizzate: {len(name_to_node)}")

# Pipeline di estrazione arricchita con chiusura transitiva (DL Reasoning)
drug_features = []
print("\nInizio estrazione feature semantiche...")

for idx, drug_name in enumerate(unique_drugs):
    drug_lower = drug_name.strip().lower()

    node = None
    # Strategia 1: Match esatto su label o sinonimi (O(1))
    if drug_lower in name_to_node:
        node = name_to_node[drug_lower]
    else:
        # Strategia 2: Ricerca parziale fuzzy (substring match) in memoria
        for key in name_to_node:
            if drug_lower in key:
                node = name_to_node[key]
                break

    parent_classes = []
    roles = []
    chebi_id = "NOT_FOUND"

    if node:
        chebi_id = node.name
        parent_set = set()
        roles_set = set()
        try:
            # Ragionamento DL: chiusura transitiva sulla tassonomia ed ereditarietà dei ruoli
            for ancestor in node.ancestors():
                if isinstance(ancestor, ThingClass):
                    if ancestor != node:
                        parent_set.add(ancestor.name)
                    # Verifica restrizioni di ruolo dell'antenato
                    for parent in ancestor.is_a:
                        if isinstance(parent, Restriction):
                            if parent.property and hasattr(parent.property, 'name') and "RO_0000087" in parent.property.name:
                                if isinstance(parent.value, ThingClass):
                                    roles_set.add(parent.value.name)
        except Exception as e:
            print(f"Errore durante l'estrazione ricorsiva per {drug_name}: {e}")

        parent_classes = sorted(list(parent_set))
        roles = sorted(list(roles_set))

    drug_features.append({
        "Drug_Name": drug_name,
        "ChEBI_ID": chebi_id,
        "Parent_Classes": "|".join(parent_classes) if parent_classes else "None",
        "Roles": "|".join(roles) if roles else "None"
    })

    if (idx + 1) % 200 == 0:
        print(f"Elaborati {idx + 1}/{len(unique_drugs)} farmaci...")

# 4. Salvataggio e metriche di copertura
df_features = pd.DataFrame(drug_features)
df_features.to_csv("data/processed/drug_features_chebi.csv", index=False)
print("\nEstrazione completata. File salvato.")

matched = df_features[df_features['ChEBI_ID'] != 'NOT_FOUND'].shape[0]
coverage = (matched / len(unique_drugs)) * 100
print(
    f"Nuovo tasso di allineamento con ChEBI: {matched}/{len(unique_drugs)} ({round(coverage, 2)}%)")
