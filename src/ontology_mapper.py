# pyrefly: ignore [missing-import]
from owlready2 import get_ontology
import time

onto_path = "data/ontologies/chebi.owl"

print("Caricamento ontologia ChEBI in corso (potrebbe richiedere diversi minuti)...")
start_time = time.time()

# Caricamento locale dell'ontologia
onto = get_ontology(f"file://{onto_path}").load()

print(
    f"Caricamento completato in {round(time.time() - start_time, 1)} secondi.")
print(f"Classi totali disponibili: {len(list(onto.classes()))}")

# Test di ricerca per un farmaco presente nel tuo dataset
# Owlready2 esegue ricerche case-insensitive di default sulle label
test_drug = "abacavir"
print(f"\nRicerca del nodo per '{test_drug}'...")
results = onto.search(label=test_drug)

if results:
    node = results[0]
    print(f"Trovato: {node.name} (IRI: {node.iri})")
    print(f"Classi genitrici (Superclassi dirette): {node.is_a}")
else:
    print(f"Nessun risultato trovato per '{test_drug}'.")
