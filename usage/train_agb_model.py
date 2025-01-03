import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os


# 1. Daten laden
def load_data(file_path):
    print("Daten werden geladen...")
    df = pd.read_csv(file_path)
    print(f"Daten erfolgreich geladen! Anzahl der Einträge: {len(df)}")
    return df


# 2. Daten vorverarbeiten
def preprocess_data(df, tokenizer):
    print("Daten werden vorverarbeitet...")

    # Text in Token umwandeln
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    # Konvertiere die Labels in Integer-Form
    if 'label' not in df.columns:
        raise ValueError("Die 'label' Spalte fehlt im Datensatz!")
    df["label"] = df["label"].astype(int)

    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


# 3. Modell initialisieren
def initialize_model():
    print("Modell wird geladen...")
    model_name = "distilbert-base-german-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 Labels: valid/invalid
    return tokenizer, model


# 4. Training vorbereiten
def train_model(dataset, model, tokenizer):
    # Aufteilen in Training und Validierung
    dataset = dataset.train_test_split(test_size=0.2)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10,
        save_total_limit=2,
        eval_strategy="epoch",  # Eval während des Trainings
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],  # Eval-Dataset übergeben
        tokenizer=tokenizer,
    )

    trainer.train()

    print("Training abgeschlossen!")

    # Modell nach dem Training speichern
    print("Modell wird gespeichert...")

    # Sicherstellen, dass der Ordner existiert
    if not os.path.exists("models"):
        os.makedirs("models")

    model.save_pretrained("models/agb_ki_model")
    tokenizer.save_pretrained("models/agb_ki_model")  # Speichert auch den Tokenizer
    print("Modell erfolgreich gespeichert!")


# Hauptfunktion
if __name__ == "__main__":
    # Datei-Pfad
    file_path = "corpus/agb_data.csv"

    # Daten laden
    df = load_data(file_path)

    # Modell und Tokenizer laden
    tokenizer, model = initialize_model()

    # Daten vorverarbeiten
    tokenized_dataset = preprocess_data(df, tokenizer)

    # Training starten
    train_model(tokenized_dataset, model, tokenizer)
