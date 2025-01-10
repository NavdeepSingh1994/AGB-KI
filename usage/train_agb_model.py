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

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

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
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return tokenizer, model


# 4. Training vorbereiten
def train_model(dataset, model, tokenizer):
    dataset = dataset.train_test_split(test_size=0.2)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10,
        save_total_limit=2,
        eval_strategy="epoch",
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    trainer.train()

    print("Training abgeschlossen!")

    # Modell nach dem Training speichern
    print("Modell wird gespeichert...")
    if not os.path.exists("models"):
        os.makedirs("models")

    model.save_pretrained("models/agb_ki_model")
    tokenizer.save_pretrained("models/agb_ki_model")
    print("Modell erfolgreich gespeichert!")


# 5. Vorhersage mit Prompt
def predict_prompt(model, tokenizer):
    prompt = input("Gib deinen Prompt ein: ")
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    if prediction == 1:
        print("Diese Bedingung ist gültig.")
    else:
        print("Diese Bedingung ist nicht gültig.")


# Hauptfunktion
if __name__ == "__main__":
    file_path = "corpus/agb_data.csv"

    df = load_data(file_path)
    tokenizer, model = initialize_model()
    tokenized_dataset = preprocess_data(df, tokenizer)
    train_model(tokenized_dataset, model, tokenizer)
    predict_prompt(model, tokenizer)
