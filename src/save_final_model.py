import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
df = pd.read_csv(os.path.join(base_dir, "data",
                 "processed", "dataset_final_tabular.csv"))

df['Roles_A'] = df['Roles_A'].fillna('None').str.replace('|', ' ')
df['Roles_B'] = df['Roles_B'].fillna('None').str.replace('|', ' ')
X_text = df['Roles_A'] + " " + df['Roles_B']

vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(X_text)

le = LabelEncoder()
y = le.fit_transform(df['Level'])

print("Addestramento modello finale su tutti i dati...")
model = RandomForestClassifier(
    n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X, y)

# Salvataggio artefatti
os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
joblib.dump(model, os.path.join(base_dir, "models", "random_forest_model.pkl"))
joblib.dump(vectorizer, os.path.join(base_dir, "models", "vectorizer.pkl"))
joblib.dump(le, os.path.join(base_dir, "models", "label_encoder.pkl"))
print("Modello e componenti salvati in /models/")
