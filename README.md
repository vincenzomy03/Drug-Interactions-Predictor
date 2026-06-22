# 💊 Predittore di Interazioni Farmacologiche (KBS & ML)

Un sistema integrato basato su Machine Learning e Semantic Web per la predizione e la spiegazione delle interazioni farmaco-farmaco (Drug-Drug Interactions). 

Questo progetto è stato realizzato per il corso di **Ingegneria della Conoscenza**, tenuto dal **Prof. Nicola Fanizzi**, all'interno del **corso di laurea triennale in Informatica** presso l'**Università degli Studi di Bari Aldo Moro**.

## 📌 Descrizione del Progetto

L'architettura fonde due fonti di conoscenza:
1. **Dati Clinici (Kaggle):** Un dataset storico di interazioni farmacologiche classificate per severità (*Minor*, *Moderate*, *Major*).
2. **Background Knowledge (Web Semantico):** L'ontologia biomedica **ChEBI** (Chemical Entities of Biological Interest), utilizzata per estrarre le proprietà strutturali e i ruoli biologici dei farmaci tramite ragionamento automatico (`owlready2`).

Iniettando le feature semantiche estratte dall'ontologia (classi chimiche e ruoli) in un modello ad albero (Random Forest), il sistema non solo apprende a classificare interazioni inedite, ma fornisce anche la spiegazione causale della predizione evidenziando la sovrapposizione logica dei meccanismi d'azione.

## 📂 Struttura del Repository

```text
📦 Drug-Interactions-Predictor
 ┣ 📂 data
 ┃ ┣ 📂 ontologies          # Contiene chebi.owl (File ignorato da git per dimensioni)
 ┃ ┣ 📂 processed           # Dataset puliti e feature semantiche estratte
 ┃ ┗ 📂 raw                 # Dati grezzi originali (inclusi i sotto-dataset ATC)
 ┣ 📂 models                # Modelli addestrati e serializzati (.pkl)
 ┣ 📂 src
 ┃ ┣ 📜 extract_features.py     # Modulo di mapping testuale e interrogazione ChEBI
 ┃ ┣ 📜 build_dataset.py        # Modulo di join relazionale tra clinica e KB
 ┃ ┣ 📜 train_models.py         # Benchmark sperimentale (Cross-Validation)
 ┃ ┣ 📜 save_final_model.py     # Serializzazione del modello ottimizzato
 ┃ ┣ 📜 evaluate_atc_subsets.py # Benchmark sperimentale specifico sui sotto-dataset ATC
 ┃ ┣ 📜 ontology_mapper.py      # Utility di test caricamento ontologia e query
 ┃ ┣ 📜 prolog_interpreter.py   # Ragionatore simbolico (Prolog Interpreter) custom
 ┃ ┗ 📜 app.py                  # Interfaccia UI Streamlit (Dashboard)
 ┣ 📜 .gitignore
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```
## ⚙️ Installazione e Setup

### 1. Clonazione e Ambiente Virtuale

Clona il repository e posizionati nella cartella di progetto:
```Bash
git clone <URL_DEL_REPO>
cd Drug-Interactions-Predictor
```

Crea e attiva un ambiente virtuale:
```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
.\venv\Scripts\Activate.ps1      # Windows
```
## 2. Installazione Dipendenze

Installa le librerie necessarie:
```bash
pip install -r requirements.txt
```
## 3. Download della Knowledge Base (ChEBI)

L'ontologia ChEBI pesa oltre 800MB e non è tracciata nel repository. È necessario scaricarla direttamente dall'European Bioinformatics Institute.

Via terminale (Windows):
 ```bash
Invoke-WebRequest -Uri "[https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.owl](https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.owl)" -OutFile "data\ontologies\chebi.owl"
```
Manualmente: 
```text
Scarica il file chebi.owl da EBI Downloads e posizionalo in data/ontologies/.
```

## 🚀 Utilizzo e Riproducibilità Sperimentale

Il progetto è progettato per essere completamente riproducibile eseguendo gli script in sequenza.
### 1. Pipeline di Estrazione e Addestramento (CLI)

Esegui questi comandi per ricreare la matrice dati e i modelli da zero:

Estrazione Semantica: Estrae classi e ruoli dall'ontologia ChEBI.

```bash
python src/extract_features.py
```
Costruzione Dataset: Fonde i dati clinici con la Background Knowledge.
 ```Bash
python src/build_dataset.py
```
Valutazione Sperimentale (Benchmark): Esegue una k-Fold Cross-Validation confrontando una Baseline puramente lineare con la Random Forest arricchita semanticamente. Stampa metriche e deviazioni standard.
```Bash
python src/train_models.py
```
Valutazione Sperimentale su Sotto-dataset ATC: Valuta le performance predittive del modello su singoli sotto-dataset divisi per categoria ATC.
```Bash
python src/evaluate_atc_subsets.py
```
Salvataggio Artefatti: Genera i file .pkl per l'interfaccia di inferenza.
```Bash
python src/save_final_model.py
```

### 2. Interfaccia Utente (KBS Dashboard)
Per testare il sistema di raccomandazione ed esaminare le deduzioni logiche del reasoner, avvia l'interfaccia Streamlit:
```Bash
streamlit run src/app.py
```

Il browser si aprirà automaticamente all'indirizzo http://localhost:8501.


## 🔬 Valutazione e Metriche
L'introduzione della Background Knowledge (KB) ha generato un incremento significativo delle performance predittive. Come verificabile dallo script di training e riportato nella tesi (Tabella 7.1):

*   **Baseline (Logistic Regression):** Accuracy Media del **56.17%** (&plusmn; 5.29%) e Macro F1-Score Medio del **46.99%** (&plusmn; 3.71%).
*   **Modello Semanticamente Arricchito (Tuned Random Forest):** Accuracy Media del **77.02%** (&plusmn; 3.50%) e Macro F1-Score Medio del **65.18%** (&plusmn; 2.90%).

L'arricchimento semantico ha comportato un incremento prestazionale netto rispetto alla baseline lineare, aumentando l'Accuracy di circa il **+20.85%** e il Macro F1-Score del **+18.19%**.

## ⚠️ Limiti Clinici (Disclaimer)
Questo sistema è un prototipo accademico. Le predizioni non tengono conto di variabili cliniche fondamentali quali il dosaggio (farmacocinetica), le co-morbilità del paziente, l'età e la funzionalità epatica/renale. Non deve in alcun caso essere utilizzato come supporto decisionale per terapie su pazienti reali al di fuori di un ambiente di simulazione biomedica.

## 👨‍💻 Autore
Vincenzo My  
Matricola: 774189

Corso di Laurea Triennale in Informatica  
Università degli Studi di Bari "Aldo Moro"  
Anno Accademico 2025-2026
