# 📚 NLP Koans - Cheat Sheet

Referencia rápida de conceptos y código aprendido en cada koan.

---

## 🔹 Koan 01: Tokenización

### Conceptos Clave
- **Tokenización**: Dividir texto en unidades (tokens)
- **Tipos**: Palabras, oraciones, caracteres
- **Herramientas**: NLTK (clásico), spaCy (industrial)

### Código Esencial

```python
# NLTK - Tokenización de palabras
from nltk.tokenize import word_tokenize
tokens = word_tokenize("Hola, ¿cómo estás?")
# ['Hola', ',', '¿cómo', 'estás', '?']

# NLTK - Tokenización de oraciones
from nltk.tokenize import sent_tokenize
sentences = sent_tokenize("Hola. ¿Cómo estás? Bien.")
# ['Hola.', '¿Cómo estás?', 'Bien.']

# spaCy - Tokenización avanzada
import spacy
nlp = spacy.load("es_core_news_sm")
doc = nlp("El Dr. García ganó $1,000.")
tokens = [token.text for token in doc]

# Tokenización personalizada
tokens = text.split("-")  # Delimitador personalizado

# Contar frecuencias
from collections import Counter
tokens = word_tokenize(text.lower())
frecuencias = Counter(tokens)

# Eliminar puntuación
import string
tokens_limpios = [t for t in tokens if t not in string.punctuation]
```

### 💡 Tips
- spaCy maneja mejor abreviaturas y números
- NLTK es más rápido para tareas simples
- Siempre normaliza (lowercase) antes de contar

---

## 🔹 Koan 02: Stemming y Lemmatization

### Conceptos Clave
- **Stemming**: Corta sufijos (rápido, aproximado)
- **Lemmatization**: Forma canónica (preciso, usa diccionario)
- **Cuándo usar**: Stemming para IR, Lemmatization para análisis

### Código Esencial

```python
# Porter Stemmer (solo inglés)
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stem = stemmer.stem("running")  # "run"

# Snowball Stemmer (multiidioma)
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("spanish")
stem = stemmer.stem("corriendo")  # "corr"

# WordNet Lemmatizer (inglés)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemma = lemmatizer.lemmatize("running", pos="v")  # "run"

# spaCy Lemmatization (automático)
import spacy
nlp = spacy.load("es_core_news_sm")
doc = nlp("Los gatos corrían")
lemmas = [token.lemma_ for token in doc]
# ['el', 'gato', 'correr']
```

### Comparación

| Aspecto | Stemming | Lemmatization |
|---------|----------|---------------|
| Velocidad | Rápido ⚡ | Lento 🐢 |
| Precisión | Aproximado | Preciso ✓ |
| Resultado | Puede no ser palabra | Siempre palabra válida |
| Ejemplo | "corriendo" → "corr" | "corriendo" → "correr" |

### 💡 Tips
- Para español: usa Snowball o spaCy
- POS tag mejora lemmatization (n, v, a, r)
- spaCy hace lemmatization gratis al procesar

---

## 🔹 Koan 03: POS Tagging

### Conceptos Clave
- **POS**: Part of Speech (categoría gramatical)
- **Universal Dependencies**: 17 etiquetas estándar
- **Uso**: Filtrar palabras, análisis sintáctico

### Código Esencial

```python
# NLTK POS Tagging (inglés, Penn Treebank)
from nltk import pos_tag, word_tokenize
tokens = word_tokenize("The cat sits")
tags = pos_tag(tokens)
# [('The', 'DT'), ('cat', 'NN'), ('sits', 'VBZ')]

# spaCy POS Tagging (Universal Dependencies)
import spacy
nlp = spacy.load("es_core_news_sm")
doc = nlp("El gato grande")
for token in doc:
    print(f"{token.text}: {token.pos_} ({token.tag_})")

# Extraer por categoría
sustantivos = [token.text for token in doc if token.pos_ == "NOUN"]
verbos = [token.text for token in doc if token.pos_ == "VERB"]
adjetivos = [token.text for token in doc if token.pos_ == "ADJ"]

# Contar categorías
from collections import Counter
pos_counts = Counter([token.pos_ for token in doc])
```

### Etiquetas Universales Principales

| Etiqueta | Tipo | Ejemplo |
|----------|------|---------|
| NOUN | Sustantivo | casa, perro |
| VERB | Verbo | correr, comer |
| ADJ | Adjetivo | grande, azul |
| ADV | Adverbio | rápidamente |
| PRON | Pronombre | él, ella |
| DET | Determinante | el, la, un |
| ADP | Preposición | de, en, por |
| PUNCT | Puntuación | . , ; |

### 💡 Tips
- spaCy es mejor para producción
- Usa `.pos_` para universal, `.tag_` para específico
- Filtra por múltiples POS para análisis complejo

---

## 🔹 Koan 04: Named Entity Recognition (NER)

### Conceptos Clave
- **NER**: Identificar entidades nombradas
- **Tipos**: Personas, organizaciones, lugares, fechas
- **Uso**: Extracción de información, anonimización

### Código Esencial

```python
# spaCy NER (mejor opción)
import spacy
nlp = spacy.load("es_core_news_sm")
doc = nlp("María García trabaja en Google en Madrid")

# Extraer todas las entidades
entities = [(ent.text, ent.label_) for ent in doc.ents]
# [('María García', 'PER'), ('Google', 'ORG'), ('Madrid', 'LOC')]

# Filtrar por tipo
personas = [ent.text for ent in doc.ents if ent.label_ == "PER"]
lugares = [ent.text for ent in doc.ents if ent.label_ == "LOC"]
organizaciones = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

# Con contexto (posición)
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_}) en posición {ent.start_char}-{ent.end_char}")

# Contar tipos
from collections import Counter
entity_counts = Counter([ent.label_ for ent in doc.ents])
```

### Tipos de Entidades

**Español (spaCy)**:
- `PER`: Persona
- `LOC`: Lugar
- `ORG`: Organización
- `MISC`: Misceláneo

**Inglés (spaCy)**:
- `PERSON`: Persona
- `GPE`: Entidad geopolítica
- `ORG`: Organización
- `DATE`: Fecha
- `MONEY`: Dinero

### 💡 Tips
- Mayúsculas mejoran detección
- Modelos grandes (md, lg) son más precisos
- Usa contexto para desambiguar

---

## 🔹 Koan 05: Text Classification

### Conceptos Clave
- **Pipeline**: Vectorización → Modelo ML → Predicción
- **TF-IDF**: Frecuencia de términos con peso por rareza
- **Clasificadores**: Naive Bayes, Logistic Regression, SVM

### Código Esencial

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Vectorización TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# 2. Entrenar clasificador
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# 3. Predecir
predictions = classifier.predict(X_test)

# 4. Evaluar
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

# Pipeline (recomendado)
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=1000)),
    ('classifier', MultinomialNB())
])
pipeline.fit(train_texts, y_train)
predictions = pipeline.predict(test_texts)

# Feature importance
feature_names = vectorizer.get_feature_names_out()
coef = classifier.coef_[0]
top_features = sorted(zip(feature_names, coef), key=lambda x: x[1], reverse=True)[:10]
```

### Métricas de Evaluación

| Métrica | Significado |
|---------|-------------|
| **Accuracy** | % predicciones correctas |
| **Precision** | De los positivos predichos, % correctos |
| **Recall** | De los positivos reales, % encontrados |
| **F1-Score** | Media armónica de precision y recall |

### 💡 Tips
- Usa Pipeline para evitar errores
- Más features ≠ mejor (prueba 100-10000)
- Logistic Regression es interpretable

---

## 🔹 Koan 06: Sentiment Analysis

### Conceptos Clave
- **Métodos**: Léxico (TextBlob), ML, Transformers
- **Polaridad**: -1 (negativo) a +1 (positivo)
- **Subjetividad**: 0 (objetivo) a 1 (subjetivo)

### Código Esencial

```python
# TextBlob (reglas, rápido)
from textblob import TextBlob
blob = TextBlob("This is amazing!")
polarity = blob.sentiment.polarity  # 0.0 a 1.0
subjectivity = blob.sentiment.subjectivity

# Transformers (mejor precisión)
from transformers import pipeline

# Inglés
classifier = pipeline("sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Español
classifier = pipeline("sentiment-analysis",
    model="pysentimiento/robertuito-sentiment-analysis")
result = classifier("Me encanta este producto!")

# Batch processing (más eficiente)
results = classifier(["Text 1", "Text 2", "Text 3"])

# Clasificar polaridad
def classify_sentiment(polarity, threshold=0.1):
    if polarity > threshold:
        return "positive"
    elif polarity < -threshold:
        return "negative"
    return "neutral"
```

### Modelos Recomendados

**Inglés**:
- `distilbert-base-uncased-finetuned-sst-2-english` (rápido)
- `cardiffnlp/twitter-roberta-base-sentiment-latest` (tweets)

**Español**:
- `pysentimiento/robertuito-sentiment-analysis`
- `cardiffnlp/twitter-xlm-roberta-base-sentiment`

### 💡 Tips
- TextBlob para prototipos rápidos
- Transformers para producción
- Batch processing es 10x más rápido
- Sarcasmo es difícil de detectar

---

## 🔹 Koan 07: Word Embeddings

### Conceptos Clave
- **Embeddings**: Representación densa de palabras como vectores
- **Similitud**: Palabras similares → vectores cercanos
- **Aritmética**: vector("rey") - vector("hombre") + vector("mujer") ≈ vector("reina")

### Código Esencial

```python
# spaCy Word Vectors (necesita md o lg)
import spacy
import numpy as np

nlp = spacy.load("en_core_web_md")  # o es_core_news_md
doc = nlp("cat")
vector = doc.vector  # numpy array (300 dims)

# Similitud coseno
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2)

# Similitud entre palabras
doc1 = nlp("cat")
doc2 = nlp("dog")
similarity = doc1.similarity(doc2)  # 0.8

# Word2Vec (gensim)
from gensim.models import KeyedVectors

# Cargar modelo pre-entrenado
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors.bin', binary=True)

# Similitud
similarity = model.similarity('king', 'queen')

# Palabras más similares
similar = model.most_similar('king', topn=5)

# Analogías
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
# ['queen']

# Sentence embedding (promedio)
doc = nlp("The cat sits on the mat")
sentence_vector = doc.vector  # spaCy promedia automáticamente
```

### Propiedades Importantes

1. **Similitud Semántica**: Palabras relacionadas tienen vectores cercanos
2. **Aritmética Vectorial**: Captura relaciones (género, geografía, tiempo)
3. **Contextual vs Estático**:
   - Word2Vec: mismo vector siempre
   - BERT: vector cambia según contexto

### 💡 Tips
- Modelos `sm` NO tienen vectores
- Usa `md` o `lg` para embeddings
- Normaliza palabras (lowercase)
- Word2Vec es estático, BERT es contextual

---

## 🔹 Koan 08: Transformers

### Conceptos Clave
- **Transformers**: Arquitectura con self-attention
- **Modelos**: BERT (encoder), GPT (decoder), T5 (seq2seq)
- **Pipelines**: API simple de Hugging Face

### Código Esencial

```python
from transformers import pipeline

# 1. Generación de texto (GPT-2)
generator = pipeline("text-generation", model="gpt2")
text = generator("Once upon a time", max_length=50)[0]["generated_text"]

# 2. Fill-mask (BERT)
unmasker = pipeline("fill-mask", model="bert-base-uncased")
results = unmasker("Paris is the [MASK] of France.")
# [{'token_str': 'capital', 'score': 0.999}]

# 3. Question Answering
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
result = qa(
    question="What is the capital of France?",
    context="Paris is the capital of France. It has 2.2M inhabitants."
)
# {'answer': 'Paris', 'score': 0.98}

# 4. Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(long_article, max_length=130, min_length=30)[0]["summary_text"]

# 5. Embeddings contextuales (BERT)
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1)

# Control de generación
generator = pipeline("text-generation", model="gpt2")
text = generator(
    "The future of AI",
    max_length=100,
    temperature=0.7,    # Creatividad
    top_p=0.9,          # Nucleus sampling
    top_k=50,           # Top-K sampling
    do_sample=True
)
```

### Modelos Principales

| Modelo | Tipo | Mejor para |
|--------|------|------------|
| BERT | Encoder | Clasificación, NER, Q&A |
| GPT-2/3 | Decoder | Generación de texto |
| T5 | Encoder-Decoder | Traducción, resumen |
| BART | Encoder-Decoder | Resumen, paráfrasis |
| DistilBERT | Encoder | BERT más rápido (60%) |

### 💡 Tips
- Usa modelos distilled para velocidad
- GPU acelera 10-100x
- `temperature=0.7` para texto balanceado
- BERT max 512 tokens

---

## 🔹 Koan 09: Language Models

### Conceptos Clave
- **Perplexity**: Qué tan "sorprendido" está el modelo (menor = mejor)
- **Prompt Engineering**: Diseñar prompts efectivos
- **Control**: Temperature, top-k, top-p, repetition_penalty

### Código Esencial

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# 1. Perplexity
def calculate_perplexity(text, model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    
    return torch.exp(loss).item()

# 2. Generación controlada
generator = pipeline("text-generation", model="gpt2")

text = generator(
    "The future of AI",
    max_length=100,
    min_length=20,
    temperature=0.8,           # 0.1-2.0 (creatividad)
    top_k=50,                  # Top K tokens
    top_p=0.95,                # Nucleus sampling
    repetition_penalty=1.2,    # Penaliza repetición
    no_repeat_ngram_size=3,    # No repite 3-gramas
    num_return_sequences=3     # Genera 3 versiones
)

# 3. Prompt Engineering

# Zero-shot
prompt = "Translate to Spanish: Hello"
result = generator(prompt, max_length=50)

# Few-shot
prompt = """
Translate to Spanish:
English: Hello → Spanish: Hola
English: Goodbye → Spanish: Adiós
English: Thank you → Spanish:
"""
result = generator(prompt, max_length=len(prompt.split()) + 10)

# 4. Token probabilities
def get_next_token_probs(text, model_name="gpt2", top_k=5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
    
    top_probs, top_indices = torch.topk(probs, top_k)
    results = [(tokenizer.decode([idx]), prob.item()) 
               for prob, idx in zip(top_probs, top_indices)]
    return results

# 5. Evaluación de calidad
def evaluate_generation(text):
    tokens = text.split()
    diversity = len(set(tokens)) / len(tokens)  # Ratio tokens únicos
    perplexity = calculate_perplexity(text)
    
    return {
        "diversity": diversity,
        "perplexity": perplexity,
        "length": len(tokens)
    }
```

### Parámetros de Generación

| Parámetro | Rango | Efecto |
|-----------|-------|--------|
| `temperature` | 0.1-2.0 | Creatividad (↑ más aleatorio) |
| `top_k` | 1-100 | Considera top K tokens |
| `top_p` | 0.1-1.0 | Nucleus sampling |
| `repetition_penalty` | 1.0-2.0 | Penaliza repetición |
| `no_repeat_ngram_size` | 2-5 | Evita n-gramas repetidos |

### Estrategias de Prompting

1. **Zero-shot**: Instrucción directa
2. **Few-shot**: Dar ejemplos
3. **Chain-of-thought**: Razonamiento paso a paso
4. **Instruction-following**: Instrucciones explícitas

### 💡 Tips
- Temperature bajo (0.3) para código/facts
- Temperature alto (1.5) para creatividad
- Few-shot mejora resultados dramáticamente
- Perplexity < 50 = buen texto

---

## 🚀 Comandos Útiles

```bash
# Ejecutar tests
pytest koans/01_tokenization/test_tokenization.py -v
pytest koans/01_tokenization/test_tokenization.py::TestTokenizationBasics -v

# Verificar progreso
python check_progress.ps1  # Windows
./check_progress.sh        # Linux/Mac

# Instalar modelos spaCy
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md

# Descargar recursos NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

---

## 📖 Recursos Adicionales

- **Hugging Face**: https://huggingface.co/
- **spaCy Docs**: https://spacy.io/usage
- **NLTK Book**: https://www.nltk.org/book/
- **Papers With Code**: https://paperswithcode.com/area/natural-language-processing

---

**¡Consulta este cheat sheet siempre que necesites recordar algo rápidamente!** 🎯
