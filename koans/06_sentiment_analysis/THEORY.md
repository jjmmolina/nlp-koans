# Teoría: Sentiment Analysis

## 📚 Tabla de Contenidos
1. [Introducción](#introducción)
2. [Enfoques](#enfoques)
3. [Lexicon-Based Methods](#lexicon)
4. [Machine Learning](#ml)
5. [Deep Learning](#dl)
6. [Herramientas](#herramientas)
7. [Aplicaciones](#aplicaciones)

---

## 🎯 Introducción {#introducción}

### ¿Qué es Sentiment Analysis?

Determinar la actitud emocional (positiva, negativa, neutral) expresada en texto.

```python
"I love this product!" → POSITIVE 😊
"This is terrible"     → NEGATIVE 😞
"It's okay"           → NEUTRAL 😐
```

### Niveles de Análisis

**Document-level:**
```python
Review completo → Un sentimiento general
"Great hotel, loved it!" → POSITIVE
```

**Sentence-level:**
```python
Cada oración → Su propio sentimiento
"Great hotel. But expensive rooms." 
→ Sentence 1: POSITIVE
→ Sentence 2: NEGATIVE
```

**Aspect-level:**
```python
Opiniones sobre aspectos específicos
"Great location but noisy rooms"
→ location: POSITIVE
→ rooms: NEGATIVE
```

---

## 🔄 Enfoques {#enfoques}

### 1. Rule-Based (Lexicon)

Diccionarios de palabras con polaridad.

### 2. Machine Learning

Entrenar clasificadores con datos etiquetados.

### 3. Deep Learning

Redes neuronales (RNN, LSTM, Transformers).

### 4. Hybrid

Combina reglas + ML.

---

## 📖 Lexicon-Based Methods {#lexicon}

### VADER (Valence Aware Dictionary)

Especializado en redes sociales.

```python
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

texts = [
    "I love this!",
    "This is terrible",
    "It's okay",
    "AMAZING!!! :)",
]

for text in texts:
    scores = sia.polarity_scores(text)
    print(f"{text:20} → {scores}")

# I love this!         → {'neg': 0.0, 'neu': 0.192, 'pos': 0.808, 'compound': 0.6369}
# This is terrible     → {'neg': 0.508, 'neu': 0.492, 'pos': 0.0, 'compound': -0.4767}
# It's okay            → {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
# AMAZING!!! :)        → {'neg': 0.0, 'neu': 0.213, 'pos': 0.787, 'compound': 0.8877}
```

**Compound Score:**
```
compound >= 0.05  → POSITIVE
compound <= -0.05 → NEGATIVE
else              → NEUTRAL
```

**Características de VADER:**
- ✅ Entiende intensificadores: "very good" > "good"
- ✅ Maneja mayúsculas: "GOOD" > "good"
- ✅ Reconoce emojis: 😊, 😞
- ✅ Entiende negación: "not good" es negativo
- ✅ Puntuación: "!!!" intensifica

### TextBlob

```python
from textblob import TextBlob

text = "I love this product! It's amazing!"
blob = TextBlob(text)

print(f"Polarity: {blob.sentiment.polarity}")      # -1 a 1
print(f"Subjectivity: {blob.sentiment.subjectivity}") # 0 a 1

# Polarity: 0.625 (positivo)
# Subjectivity: 0.6 (bastante subjetivo)
```

**Polarity:**
```
-1.0 → Muy negativo
 0.0 → Neutral
+1.0 → Muy positivo
```

**Subjectivity:**
```
0.0 → Objetivo (hechos)
1.0 → Subjetivo (opiniones)
```

### Ventajas y Desventajas

**Ventajas:**
- ⚡ Rápido
- ✅ Sin entrenamiento necesario
- 📖 Interpretable

**Desventajas:**
- ❌ No capta contexto complejo
- ❌ Dominio-dependiente
- ❌ No aprende de datos

---

## 🤖 Machine Learning {#ml}

### Con Features Tradicionales

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Dataset
texts = [
    "I love this product",
    "Terrible experience",
    "Amazing quality",
    "Waste of money",
]
labels = [1, 0, 1, 0]  # 1=positive, 0=negative

# Split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2
)

# Vectorización
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Modelo
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predicción
new_review = ["This is fantastic"]
new_vec = vectorizer.transform(new_review)
print(model.predict(new_vec))  # [1] (positive)
```

### Datasets Populares

**IMDb Movie Reviews:**
```python
from sklearn.datasets import load_files

# 50,000 reviews (25k train, 25k test)
# Balanceado: 50% positive, 50% negative
```

**Twitter Sentiment140:**
```
1.6 millones de tweets
Positive / Negative
```

**Amazon Reviews:**
```
Millones de reviews
1-5 estrellas
```

---

## 🧠 Deep Learning {#dl}

### Con Pre-trained Transformers

```python
from transformers import pipeline

# Modelo pre-entrenado
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

texts = [
    "I love this product!",
    "This is terrible",
    "It's okay"
]

results = sentiment_pipeline(texts)

for text, result in zip(texts, results):
    print(f"{text:25} → {result['label']} ({result['score']:.3f})")

# I love this product!      → POSITIVE (0.999)
# This is terrible          → NEGATIVE (0.999)
# It's okay                 → POSITIVE (0.679)
```

### Modelos Populares

| Modelo | Tamaño | Precisión | Uso |
|--------|--------|-----------|-----|
| **distilbert-base** | 66M | ⭐⭐⭐⭐ | General |
| **roberta-base** | 125M | ⭐⭐⭐⭐⭐ | Mejor precisión |
| **bert-base** | 110M | ⭐⭐⭐⭐ | Baseline |

### Fine-tuning

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# Cargar modelo base
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenizar datos
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Entrenar
trainer.train()
```

---

## 🛠️ Herramientas {#herramientas}

### VADER

```python
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
scores = sia.polarity_scores("I love this!")
```

**Mejor para:** Redes sociales, textos cortos

### TextBlob

```python
from textblob import TextBlob

blob = TextBlob("I love this product")
polarity = blob.sentiment.polarity
```

**Mejor para:** Análisis rápido, prototipado

### Transformers

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this product")
```

**Mejor para:** Máxima precisión, producción

### spaCy con spacytextblob

```python
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")

doc = nlp("I love this product but it's expensive")

# Document-level
print(doc._.polarity)  # 0.25

# Sentence-level
for sent in doc.sents:
    print(sent.text, sent._.polarity)
```

---

## 💼 Aplicaciones {#aplicaciones}

### 1. Product Reviews Analysis

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

reviews = [
    "Great product, highly recommend!",
    "Terrible quality, waste of money",
    "It's okay, nothing special",
]

for review in reviews:
    result = classifier(review)[0]
    print(f"{review[:30]:30} → {result['label']}")

# Great product, highly recomme → POSITIVE
# Terrible quality, waste of mo → NEGATIVE
# It's okay, nothing special    → POSITIVE
```

### 2. Social Media Monitoring

```python
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

tweets = [
    "@company Your service is amazing! #happy",
    "@company Worst experience ever #disappointed",
]

for tweet in tweets:
    score = sia.polarity_scores(tweet)['compound']
    
    if score >= 0.05:
        sentiment = "POSITIVE 😊"
    elif score <= -0.05:
        sentiment = "NEGATIVE 😞"
    else:
        sentiment = "NEUTRAL 😐"
    
    print(f"{tweet[:40]:40} → {sentiment}")
```

### 3. Customer Feedback Analysis

```python
# Análisis agregado
feedbacks = [
    "Great service",
    "Poor quality",
    "Love it",
    "Disappointed",
    "Excellent",
]

positive_count = 0
negative_count = 0

for feedback in feedbacks:
    result = classifier(feedback)[0]
    if result['label'] == 'POSITIVE':
        positive_count += 1
    else:
        negative_count += 1

print(f"Positive: {positive_count} ({positive_count/len(feedbacks)*100:.1f}%)")
print(f"Negative: {negative_count} ({negative_count/len(feedbacks)*100:.1f}%)")

# Positive: 3 (60.0%)
# Negative: 2 (40.0%)
```

### 4. Aspect-Based Sentiment

```python
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

review = "The food was delicious but the service was terrible"

doc = nlp(review)

# Extraer aspectos (sustantivos)
aspects = [token.text for token in doc if token.pos_ == "NOUN"]

# Análisis por oración (simplificado)
for sent in doc.sents:
    sentiment = sia.polarity_scores(sent.text)['compound']
    aspect_in_sent = [asp for asp in aspects if asp in sent.text]
    
    if aspect_in_sent:
        print(f"{aspect_in_sent[0]}: {sentiment:.2f}")

# food: 0.65 (positive)
# service: -0.64 (negative)
```

---

## 📊 Best Practices

### 1. Elegir el Enfoque Correcto

```python
# Análisis rápido, sin datos etiquetados
→ VADER o TextBlob

# Alta precisión, tienes datos
→ Machine Learning (Logistic Regression)

# Máxima precisión, recursos disponibles
→ Transformers (BERT, RoBERTa)
```

### 2. Manejar Casos Especiales

```python
# Negación
"not good" → debe ser NEGATIVE (no POSITIVE)

# Sarcasmo (difícil)
"Oh great, another bug" → NEGATIVE (no POSITIVE)

# Contexto
"This movie is not bad" → POSITIVE
```

### 3. Validación

```python
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Verificar ambas clases están balanceadas
```

---

## 🎓 Resumen

**Conceptos Clave:**
- Sentiment Analysis clasifica emociones en texto
- Enfoques: Lexicon, ML, DL
- Niveles: Document, Sentence, Aspect
- VADER para redes sociales, Transformers para precisión

**Herramientas:**
- VADER → Rápido, redes sociales
- TextBlob → Simple, prototipado
- Transformers → Máxima precisión

**Próximos Pasos:**
- **Koan 7**: Word Embeddings (mejor representación)
- **Koan 8**: Transformers (arquitectura detallada)

¡El sentiment analysis es crucial para entender opiniones! 🚀
