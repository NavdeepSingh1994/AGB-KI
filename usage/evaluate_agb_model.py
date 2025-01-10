import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# 1. Daten laden
def load_data(file_path):
    print("Daten werden geladen...")
    df = pd.read_csv(file_path)
    print(f"Daten erfolgreich geladen! Anzahl der Einträge: {len(df)}")
    return df

# 2. Daten vorverarbeiten
def preprocess_data(df, tokenizer):
    print("Daten werden vorverarbeitet...")

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    df["label"] = df["label"].astype(int)
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# 3. Modell initialisieren
def initialize_model():
    print("Modell wird geladen...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.config.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model

# 4. Metriken definieren
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 5. Trainingskurven für Training und Validierung plotten
def plot_training_and_validation_loss(trainer):
    logs = trainer.state.log_history
    if not logs:
        print("Keine Loss-Daten vorhanden.")
        return

    train_loss = [log['loss'] for log in logs if 'loss' in log]
    eval_loss = [log['eval_loss'] for log in logs if 'eval_loss' in log]

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Trainingsverlust")
    plt.plot(eval_loss, label="Validierungsverlust")
    plt.xlabel("Schritte")
    plt.ylabel("Loss")
    plt.title("Trainings- und Validierungsverlust")
    plt.legend()
    plt.show()

# 6. Modell evaluieren
def evaluate_model(dataset, model, tokenizer):
    dataset = dataset.train_test_split(test_size=0.2)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10,
        save_total_limit=2,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    print("Training abgeschlossen!")
    return trainer

# Hauptfunktion
if __name__ == "__main__":
    file_path = "corpus/agb_data.csv"

    df = load_data(file_path)
    tokenizer, model = initialize_model()
    tokenized_dataset = preprocess_data(df, tokenizer)

    trainer = evaluate_model(tokenized_dataset, model, tokenizer)

    # Trainings- und Validierungskurven anzeigen
    plot_training_and_validation_loss(trainer)
