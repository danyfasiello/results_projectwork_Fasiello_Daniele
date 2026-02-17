import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import gradio as gr

# ==========================================
# CONFIGURAZIONE E RIPRODUCIBILITÀ
# ==========================================
RANDOM_SEED = 42
RESULTS_DIR = "results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# ==========================================
# GENERAZIONE DATASET
# ==========================================
def generate_synthetic_data(n=300):
    """Genera un dataset sintetico di recensioni hotel."""
    data = {
        'Housekeeping': ["pulizia", "camera sporca", "lenzuola", "bagno", "polvere", "asciugamani"],
        'Reception': ["check-in", "personale", "accoglienza", "chiavi", "prenotazione", "receptionist"],
        'F&B': ["colazione", "ristorante", "cibo", "cena", "buffet", "camerieri"]
    }
    reviews = []
    np.random.seed(RANDOM_SEED) # Seed per la generazione dati
    for _ in range(n):
        dept = np.random.choice(list(data.keys()))
        word = np.random.choice(data[dept])
        sent = np.random.choice(['Positivo', 'Negativo'])
        text = f"Il {word} era {'ottimo' if sent == 'Positivo' else 'pessimo'}."
        reviews.append({'review_text': text, 'department': dept, 'sentiment': sent})
    return pd.DataFrame(reviews)

# ==========================================
# ADDESTRAMENTO E SALVATAGGIO DATI (PER IL PROF)
# ==========================================
def train_and_evaluate(df):
    """Addestra i modelli e salva i file richiesti (CSV e Grafici)."""
    
    # Suddivisione Training e Test (Seed fisso per riproducibilità)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    tfidf = TfidfVectorizer(max_features=1000)
    Xtr = tfidf.fit_transform(train_df['review_text'])
    Xts = tfidf.transform(test_df['review_text'])

    # Target
    ydtr, ydts = train_df['department'], test_df['department']
    ystr, ysts = train_df['sentiment'], test_df['sentiment']

    # Modelli
    clf_dept = RandomForestClassifier(random_state=RANDOM_SEED)
    clf_dept.fit(Xtr, ydtr)
    
    clf_sent = LogisticRegression(random_state=RANDOM_SEED)
    clf_sent.fit(Xtr, ystr)

    # SALVATAGGIO DATASET RICHIESTI DAL PROFESSORE
    train_df.to_csv(os.path.join(RESULTS_DIR, "dataset_training.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(RESULTS_DIR, "dataset_test.csv"), index=False, encoding="utf-8")
    
    # GENERAZIONE GRAFICI
    y_pred = clf_dept.predict(Xts)
    
    # 1. Matrice di Confusione
    cm = confusion_matrix(ydts, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clf_dept.classes_, yticklabels=clf_dept.classes_)
    plt.title('Matrice di Confusione Reparto')
    plt.ylabel('Reale')
    plt.xlabel('Predetto')
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    # 2. Grafico F1-Score
    report = classification_report(ydts, y_pred, output_dict=True)
    f1_scores = {k: v['f1-score'] for k, v in report.items() if k in clf_dept.classes_}
    plt.figure(figsize=(8, 5))
    plt.bar(f1_scores.keys(), f1_scores.values(), color='skyblue')
    plt.title('F1-Score per ogni reparto')
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(RESULTS_DIR, "f1_bars.png"))
    plt.close()

    return tfidf, clf_dept, clf_sent

# Esecuzione logica
df_data = generate_synthetic_data()
tfidf_vectorizer, model_dept, model_sent = train_and_evaluate(df_data)

# Funzione di predizione per Gradio
def predict_review(text):
    if not text.strip(): return "Inserire testo", "N/A"
    vec = tfidf_vectorizer.transform([text])
    dept = model_dept.predict(vec)[0]
    sent = model_sent.predict(vec)[0]
    return dept, sent

# ==========================================
# INTERFACCIA GRADIO AVANZATA
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏨 Hotel Review Analyzer
        ### Dashboard Intelligente per il Management Alberghiero
        Inserisci il testo di una recensione qui sotto per smistarla automaticamente al reparto competente e analizzarne il tono.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Testo della Recensione", 
                placeholder="Es: Il personale alla reception è stato gentilissimo...",
                lines=4
            )
            btn = gr.Button("Analizza Feedback", variant="primary")
        
        with gr.Column():
            output_dept = gr.Textbox(label="📍 Reparto Destinatario")
            output_sent = gr.Textbox(label="📊 Sentiment Rilevato")

    gr.Examples(
        examples=[
            ["Il check-in è stato lentissimo, ho aspettato un'ora in piedi."],
            ["La colazione a buffet era varia e di ottima qualità."],
            ["La camera non era pulita bene, c'era polvere ovunque."],
        ],
        inputs=input_text
    )

    btn.click(fn=predict_review, inputs=input_text, outputs=[output_dept, output_sent])

if __name__ == "__main__":
    print(f"✅ Dati e grafici salvati nella cartella '{RESULTS_DIR}'")
    print("🚀 Dashboard in avvio...")
    demo.launch()