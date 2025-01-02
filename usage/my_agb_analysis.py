import pandas as pd

# Datenpfad
data_path = "corpus/agb_data.csv"

# Daten einlesen
df = pd.read_csv(data_path)
texts = df["text"].tolist()
labels = df["label"].tolist()

print("Erste Texte:", texts[:5])
print("Erste Labels:", labels[:5])
