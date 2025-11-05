# 💡 Pistas para Koan 06: Sentiment Analysis

## 🎯 Objetivo del Koan

Aprender a **analizar sentimientos** en texto usando:
- TextBlob (reglas)
- Transformers (modelos pre-entrenados)
- Análisis multilingüe

---

## 📝 Función 1: `analyze_sentiment_textblob()`

### Nivel 1: Concepto
TextBlob analiza sentimiento basándose en un léxico de palabras con polaridad.

### Nivel 2: Implementación
```python
from textblob import TextBlob
blob = TextBlob(text)
# blob.sentiment tiene .polarity y .subjectivity
```

### Nivel 3: Casi la solución
```python
sentiment = blob.sentiment
return {
    "polarity": sentiment.polarity,      # -1 (negativo) a 1 (positivo)
    "subjectivity": sentiment.subjectivity  # 0 (objetivo) a 1 (subjetivo)
}
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def analyze_sentiment_textblob(text: str) -> dict:
    from textblob import TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment
    
    return {
        "polarity": sentiment.polarity,
        "subjectivity": sentiment.subjectivity
    }
```
</details>

---

## 📝 Función 2: `classify_sentiment_polarity()`

### Nivel 1: Concepto
Convierte polaridad numérica en categoría (positive/negative/neutral).

### Nivel 2: Reglas
```python
if polarity > threshold:
    return "positive"
elif polarity < -threshold:
    return "negative"
else:
    return "neutral"
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def classify_sentiment_polarity(polarity: float, threshold: float = 0.1) -> str:
    if polarity > threshold:
        return "positive"
    elif polarity < -threshold:
        return "negative"
    else:
        return "neutral"
```
</details>

---

## 📝 Función 3: `analyze_sentiment_transformers()`

### Nivel 1: Concepto
Usa modelos pre-entrenados de Hugging Face para análisis de sentimientos.

### Nivel 2: Implementación
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model=model_name)
result = classifier(text)[0]  # Retorna lista, toma primer elemento
```

### Nivel 3: Casi la solución
```python
result = classifier(text)[0]
return {
    "label": result["label"],
    "score": result["score"]
}
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def analyze_sentiment_transformers(text: str, 
                                    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english") -> dict:
    from transformers import pipeline
    
    classifier = pipeline("sentiment-analysis", model=model_name)
    result = classifier(text)[0]
    
    return {
        "label": result["label"],
        "score": result["score"]
    }
```
</details>

---

## 📝 Función 4: `analyze_sentiment_spanish()`

### Nivel 1: Concepto
Usa un modelo específico para español, como el de pysentimiento.

### Nivel 2: Modelos para español
```python
# Opción 1: pysentimiento/robertuito-sentiment-analysis
# Opción 2: cardiffnlp/twitter-xlm-roberta-base-sentiment
model_name = "pysentimiento/robertuito-sentiment-analysis"
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def analyze_sentiment_spanish(text: str) -> dict:
    from transformers import pipeline
    
    # Modelo entrenado para español
    classifier = pipeline(
        "sentiment-analysis",
        model="pysentimiento/robertuito-sentiment-analysis"
    )
    
    result = classifier(text)[0]
    
    return {
        "label": result["label"],
        "score": result["score"]
    }
```
</details>

---

## 📝 Función 5: `batch_sentiment_analysis()`

### Nivel 1: Concepto
Analiza múltiples textos de forma eficiente usando batch processing.

### Nivel 2: Implementación
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model=model_name)
# Pipeline acepta listas de textos directamente
results = classifier(texts)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def batch_sentiment_analysis(texts: List[str], 
                              model_name: str = "distilbert-base-uncased-finetuned-sst-2-english") -> List[dict]:
    from transformers import pipeline
    
    classifier = pipeline("sentiment-analysis", model=model_name)
    results = classifier(texts)
    
    return [{"label": r["label"], "score": r["score"]} for r in results]
```
</details>

---

## 📝 Función 6: `aggregate_sentiments()`

### Nivel 1: Concepto
Calcula estadísticas agregadas de una lista de análisis de sentimientos.

### Nivel 2: Pasos
1. Cuenta cuántos de cada label (POSITIVE/NEGATIVE/NEUTRAL)
2. Calcula score promedio
3. Encuentra el sentimiento dominante

### Nivel 3: Casi la solución
```python
from collections import Counter
labels = [s["label"] for s in sentiments]
label_counts = Counter(labels)
avg_score = sum(s["score"] for s in sentiments) / len(sentiments)
dominant = label_counts.most_common(1)[0][0]
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def aggregate_sentiments(sentiments: List[dict]) -> dict:
    from collections import Counter
    
    if not sentiments:
        return {"error": "No sentiments to aggregate"}
    
    labels = [s["label"] for s in sentiments]
    label_counts = Counter(labels)
    
    avg_score = sum(s["score"] for s in sentiments) / len(sentiments)
    dominant_sentiment = label_counts.most_common(1)[0][0]
    
    return {
        "total": len(sentiments),
        "label_counts": dict(label_counts),
        "average_score": avg_score,
        "dominant_sentiment": dominant_sentiment
    }
```
</details>

---

## 📝 Función 7: `compare_sentiment_methods()`

### Nivel 1: Concepto
Compara resultados de diferentes métodos de análisis de sentimientos.

### Nivel 2: Implementación
```python
textblob_result = analyze_sentiment_textblob(text)
transformers_result = analyze_sentiment_transformers(text, model_name)

return {
    "textblob": textblob_result,
    "transformers": transformers_result
}
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def compare_sentiment_methods(text: str, 
                               transformer_model: str = "distilbert-base-uncased-finetuned-sst-2-english") -> dict:
    textblob_result = analyze_sentiment_textblob(text)
    transformers_result = analyze_sentiment_transformers(text, transformer_model)
    
    # Clasificar polaridad de TextBlob
    textblob_label = classify_sentiment_polarity(textblob_result["polarity"])
    
    return {
        "textblob": {
            **textblob_result,
            "label": textblob_label
        },
        "transformers": transformers_result
    }
```
</details>

---

## 🎯 Conceptos Clave

### Métodos de Sentiment Analysis

| Método | Ventajas | Desventajas |
|--------|----------|-------------|
| **Léxico (TextBlob)** | Rápido, interpretable | Limitado, no contextual |
| **Machine Learning** | Personalizable | Necesita datos etiquetados |
| **Transformers** | Alta precisión, contextual | Lento, requiere GPU |

### TextBlob Scores

**Polarity**: -1.0 a 1.0
- `-1.0`: Muy negativo
- `0.0`: Neutral
- `1.0`: Muy positivo

**Subjectivity**: 0.0 a 1.0
- `0.0`: Muy objetivo (hechos)
- `1.0`: Muy subjetivo (opiniones)

```python
# Ejemplo
text = "This is an amazing product!"
# polarity: 0.8 (positivo)
# subjectivity: 0.9 (opinión)

text = "Water boils at 100 degrees Celsius"
# polarity: 0.0 (neutral)
# subjectivity: 0.0 (hecho)
```

### Modelos de Transformers Populares

**Para Inglés**:
- `distilbert-base-uncased-finetuned-sst-2-english` (rápido)
- `cardiffnlp/twitter-roberta-base-sentiment-latest` (tweets)
- `nlptown/bert-base-multilingual-uncased-sentiment` (multilingüe)

**Para Español**:
- `pysentimiento/robertuito-sentiment-analysis`
- `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- `lxyuan/distilbert-base-multilingual-cased-sentiments-student`

## 💡 Tips Prácticos

### 1. Preprocessing mejora resultados
```python
text = text.lower().strip()
text = text.replace("!!!", "!")  # Normalizar énfasis
```

### 2. Batch processing es más eficiente
```python
# Malo: 100 llamadas individuales
for text in texts:
    analyze_sentiment_transformers(text)

# Bueno: 1 llamada batch
batch_sentiment_analysis(texts)
```

### 3. Combina métodos para robustez
```python
results = compare_sentiment_methods(text)
if results["textblob"]["label"] == results["transformers"]["label"]:
    # Alta confianza: ambos métodos coinciden
    confidence = "high"
```

### 4. Contexto importa
```python
# Sarcasmo es difícil de detectar
"Oh great, another bug" # Probablemente negativo, pero tiene "great"
```

## 🚀 Casos de Uso

### Análisis de reseñas de productos
```python
reviews = ["Amazing product!", "Terrible quality", "It's okay"]
sentiments = batch_sentiment_analysis(reviews)
stats = aggregate_sentiments(sentiments)
print(f"Dominant sentiment: {stats['dominant_sentiment']}")
```

### Monitoreo de redes sociales
```python
tweets = get_tweets_about("my_brand")
sentiments = batch_sentiment_analysis(tweets, 
    model_name="cardiffnlp/twitter-roberta-base-sentiment-latest")
positive_ratio = sentiments.count("POSITIVE") / len(sentiments)
```

### Análisis de feedback de clientes
```python
feedback = "El servicio fue bueno pero la comida horrible"
result = analyze_sentiment_spanish(feedback)
# Detecta sentimiento mixto
```

### Dashboard de sentimientos
```python
daily_comments = load_comments_from_db()
sentiments = batch_sentiment_analysis(daily_comments)
stats = aggregate_sentiments(sentiments)
# Visualizar en gráfico
```

## 🔧 Troubleshooting

### Problema: Modelo muy lento
**Solución**:
```python
# Usa modelos distilled (más pequeños)
model = "distilbert-base-uncased-finetuned-sst-2-english"

# O procesa en batches
batch_sentiment_analysis(texts[:100])  # Procesa en chunks
```

### Problema: Resultados en español son malos
**Solución**:
```python
# Usa modelo específico para español
analyze_sentiment_spanish(text)
```

### Problema: Out of memory
**Solución**:
```python
# Reduce batch size
for i in range(0, len(texts), 32):
    batch = texts[i:i+32]
    batch_sentiment_analysis(batch)
```

### Problema: Sarcasmo no detectado
**Solución**: Los modelos básicos no detectan sarcasmo bien. Considera:
- Modelos especializados en sarcasmo
- Análisis de emojis/puntuación
- Contexto adicional

## 🚀 Siguiente Paso

Una vez completo, ve al **Koan 07: Word Embeddings**!
