import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lime.lime_text import LimeTextExplainer
import torch
import random

# Lade das DistilBERT Modell und den Tokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-german-cased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-german-cased")

# Lade den Testdatensatz aus der CSV-Datei
df_test = pd.read_csv('corpus/agb_data.csv')
X_test = df_test['text'].tolist()
y_test = df_test['label'].tolist()

# Überprüfe die Länge von X_test
print(f"Länge von Daten: {len(X_test)}")

# Initialisiere den LIME-Explainer
explainer = LimeTextExplainer(class_names=['negativ', 'positiv'])

# Zufälligen Index wählen
idx = random.randint(0, len(X_test) - 1)
text_example = X_test[idx]
real_label = y_test[idx]

# Vorhersage-Funktion für LIME anpassen
def predict_probabilities(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.numpy()

# Erkläre das Beispiel mit LIME
explanation = explainer.explain_instance(text_example, predict_probabilities, num_features=10)

# Vorhersage berechnen und mit dem tatsächlichen Label vergleichen
predicted_class = torch.argmax(torch.tensor(predict_probabilities([text_example])), dim=1).item()
print("\nErklärung der Vorhersage:")
print(f"Textbeispiel: {text_example}")
print(f"Vorhergesagte Klasse: {predicted_class}")
print(f"Reales Label: {real_label}")

# Klarere Ausgabe der Vorhersagebewertung
if predicted_class == real_label:
    print("✅ Die Vorhersage ist korrekt!")
else:
    print("❌ Die Vorhersage ist nicht korrekt.")

# Positive und negative Features separat ausgeben
positive_features = []
negative_features = []

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
