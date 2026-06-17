import streamlit as st
import pandas as pd
import joblib
import os

# --- Configurazione Pagina ---
st.set_page_config(
    page_title="Predittore Interazioni Farmacologiche", layout="centered")

# --- Percorsi Assoluti ---
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURES_PATH = os.path.join(
    BASE_DIR, "data", "processed", "drug_features_chebi.csv")

# --- Caricamento Dati e Modelli ---


@st.cache_resource
def load_system():
    model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    df_features = pd.read_csv(FEATURES_PATH)
    valid_features = df_features[df_features['ChEBI_ID'] != 'NOT_FOUND'].copy()

    return model, vectorizer, le, valid_features


st.title("💊 Analisi Interazioni Farmaci")
st.markdown(
    "Sistema integrato di Machine Learning e Semantic Web (Ontologia ChEBI).")

try:
    model, vectorizer, le, df_features = load_system()
    drug_list = sorted(df_features['Drug_Name'].unique())
except Exception as e:
    st.error(f"Errore nel caricamento dei modelli o dei dati: {e}")
    st.stop()

# --- Interfaccia Utente ---
st.header("Seleziona i Farmaci")
col1, col2 = st.columns(2)

with col1:
    drug_a = st.selectbox("Farmaco A", drug_list, index=0)
with col2:
    drug_b = st.selectbox("Farmaco B", drug_list,
                          index=1 if len(drug_list) > 1 else 0)

if st.button("Analizza Interazione", use_container_width=True):
    if drug_a == drug_b:
        st.warning("Seleziona due farmaci diversi.")
    else:
        # Estrazione Feature
        feat_a = df_features[df_features['Drug_Name'] == drug_a].iloc[0]
        feat_b = df_features[df_features['Drug_Name'] == drug_b].iloc[0]

        roles_a_str = str(feat_a['Roles']).replace(
            '|', ' ') if pd.notna(feat_a['Roles']) else 'None'
        roles_b_str = str(feat_b['Roles']).replace(
            '|', ' ') if pd.notna(feat_b['Roles']) else 'None'

        # Preparazione Input
        combined_text = roles_a_str + " " + roles_b_str
        X_input = vectorizer.transform([combined_text])

        # Predizione
        prediction_idx = model.predict(X_input)[0]
        confidence = model.predict_proba(X_input)[0][prediction_idx]
        predicted_level = le.inverse_transform([prediction_idx])[0]

        # --- Output Visuale Ottimizzato ---
        st.divider()
        st.subheader("Risultato dell'Analisi")

        # Layout a metriche
        met_col1, met_col2 = st.columns(2)
        with met_col1:
            st.metric(label="Livello di Rischio", value=predicted_level)
        with met_col2:
            st.metric(label="Confidenza Modello",
                      value=f"{confidence*100:.1f}%")

        # Box di stato contestuale
        if predicted_level == "Major":
            st.error(
                "🚨 **Rischio Elevato (Major):** Co-somministrazione sconsigliata. Elevata probabilità di reazioni avverse severe.")
        elif predicted_level == "Moderate":
            st.warning(
                "⚠️ **Rischio Moderato (Moderate):** Possibili interazioni. Richiede valutazione clinica per la co-somministrazione.")
        else:
            st.success(
                "✅ **Rischio Lieve (Minor):** Nessuna interazione severa prevista basandosi sui dati attuali.")

        # --- Modulo Spiegazione con Vero Ragionatore Prolog (KBS) ---
        st.subheader("Spiegazione Logico-Deduttiva (KBS)")

        from src.prolog_interpreter import Prolog, Var, walk
        pl = Prolog()

        # Definiamo le regole logiche nel KBS
        pl.add_rule(
            ("interact", Var("X"), Var("Y"), Var("Role")),
            [
                ("has_role", Var("X"), Var("Role")),
                ("has_role", Var("Y"), Var("Role"))
            ]
        )
        pl.add_rule(
            ("high_risk_interaction", Var("X"), Var("Y"), Var("Role")),
            [
                ("interact", Var("X"), Var("Y"), Var("Role")),
                ("critical_role", Var("Role"))
            ]
        )

        # Definiamo alcuni ruoli critici noti nel dominio medico (fatti a priori)
        critical_roles_db = {
            "chebi_50244",  # immunosuppressive agent
            "chebi_35610",  # anticoagulant
            "chebi_38215",  # cardiovascular drug
            "chebi_35703",  # central nervous system depressant
            "chebi_49061",  # cardiotonic drug
            "chebi_35623",  # antihypertensive agent
            "chebi_35488"   # anaesthetic
        }
        for crole in critical_roles_db:
            pl.add_fact(("critical_role", crole))

        # Popoliamo i fatti a runtime per la coppia di farmaci selezionati
        atom_a = drug_a.lower().replace(" ", "_")
        atom_b = drug_b.lower().replace(" ", "_")

        roles_a = [r.lower() for r in roles_a_str.split() if r != 'None' and r.strip() != '']
        roles_b = [r.lower() for r in roles_b_str.split() if r != 'None' and r.strip() != '']

        for r in roles_a:
            pl.add_fact(("has_role", atom_a, r))
        for r in roles_b:
            pl.add_fact(("has_role", atom_b, r))

        # Eseguiamo le query reali tramite il motore Prolog
        interact_results = []
        for subst in pl.query(("interact", atom_a, atom_b, Var("Role"))):
            role_val = walk(Var("Role"), subst)
            if role_val not in interact_results:
                interact_results.append(role_val)

        high_risk_results = []
        for subst in pl.query(("high_risk_interaction", atom_a, atom_b, Var("Role"))):
            role_val = walk(Var("Role"), subst)
            if role_val not in high_risk_results:
                high_risk_results.append(role_val)

        if interact_results:
            st.info(
                f"**Ragionamento Logico Completato:** Il KBS ha dedotto una sovrapposizione biologica o chimica. Trovati **{len(interact_results)}** ruoli condivisi in ChEBI.")

            st.markdown("#### Traccia delle Regole Deduttive (Prolog)")
            st.code(
                "interact(X, Y, Role) :- has_role(X, Role), has_role(Y, Role).\n"
                "high_risk_interaction(X, Y, Role) :- interact(X, Y, Role), critical_role(Role).",
                language="prolog"
            )

            st.markdown("#### Fatti Asseriti a Runtime (Grounding)")
            facts_str = ""
            for r in sorted(list(set(roles_a + roles_b))):
                if r in roles_a:
                    facts_str += f"has_role({atom_a}, {r}).\n"
                if r in roles_b:
                    facts_str += f"has_role({atom_b}, {r}).\n"
            st.code(facts_str, language="prolog")

            st.markdown("#### Deduzioni Ottenute")
            deductions = []
            for r in sorted(interact_results):
                is_crit = r in critical_roles_db
                crit_desc = " (⚠️ CRITICO - Rischio Elevato)" if is_crit else ""
                deductions.append(f"- **interact({atom_a}, {atom_b}, {r})**{crit_desc}")

            for r in sorted(high_risk_results):
                deductions.append(f"- **high_risk_interaction({atom_a}, {atom_b}, {r})** (Inferred High-Risk)")

            st.markdown("\n".join(deductions))
        else:
            st.info("Nessun ruolo biologico condiviso rilevato direttamente. L'interazione è governata dalle feature statistiche indirette apprese dal modello di Machine Learning.")
            st.code("% Nessuna deduzione logica diretta tramite regole di sovrapposizione.\n% Risoluzione delegata al modello statistico (Random Forest).", language="prolog")

        with st.expander("🔍 Dettagli Estrazione Ontologica (Raw Data)"):
            st.code(
                f"{drug_a}:\n{roles_a_str.replace(' ', ', ')}\n\n{drug_b}:\n{roles_b_str.replace(' ', ', ')}", language="text")
