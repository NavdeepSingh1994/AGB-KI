import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lime.lime_text import LimeTextExplainer
import torch
import random

# Lade das Modell und den Tokenizer aus dem übergeordneten Verzeichnis
model = AutoModelForSequenceClassification.from_pretrained("./models/agb_ki_model")
tokenizer = AutoTokenizer.from_pretrained("./models/agb_ki_model")

# Lade den Testdatensatz aus der CSV-Datei
df_test = pd.read_csv('corpus/agb_data.csv')  # Lade die CSV-Datei
X_test = df_test['text'].tolist()  # Ersetze 'text' mit dem tatsächlichen Namen der Textspalte, falls notwendig

# Überprüfe die Länge von X_test
print(f"Länge von X_test: {len(X_test)}")

# Initialisiere den LIME-Explainer
explainer = LimeTextExplainer(class_names=['negativ', 'positiv'])  # Passe an, falls du andere Klassen hast

# Zufälligen Index wählen, der im Bereich von 0 bis len(X_test)-1 liegt
idx = random.randint(0, len(X_test) - 1)

# Wähle das Beispiel aus, das du erklären möchtest
text_example = X_test[idx]

# Vorhersage-Funktion für LIME anpassen
def predict_probabilities(texts):
    # Tokenisiere den Text
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.numpy()

# Erkläre das Beispiel mit LIME
explanation = explainer.explain_instance(text_example, predict_probabilities, num_features=10)

# Schön formatierte Ausgabe
print("\nErklärung der Vorhersage:")
print(f"Textbeispiel: {text_example}")
print("\nWichtige Merkmale (Wörter) und ihre Einflüsse auf die Vorhersage:")

# Positive und negative Features separat ausgeben
positive_features = []
negative_features = []

# Teile die Features in positive und negative auf
for feature, score in explanation.as_list():
    if score > 0:
        positive_features.append((feature, score))
    else:
        negative_features.append((feature, score))

# Ausgabe der positiven Features
print("\nPositive Merkmale (Fördern die Vorhersage):")
for feature, score in positive_features:
    print(f" - {feature}: {score:.4f}")

# Ausgabe der negativen Features
print("\nNegative Merkmale (Hemmend für die Vorhersage):")
for feature, score in negative_features:
    print(f" - {feature}: {score:.4f}")
