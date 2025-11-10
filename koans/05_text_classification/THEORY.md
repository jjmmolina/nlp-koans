# Teoría: Text Classification

## 📚 Tabla de Contenidos
1. [Introducción](#introducción)
2. [Feature Engineering](#features)
3. [Modelos Clásicos](#modelos)
4. [Pipeline Completo](#pipeline)
5. [Evaluación](#evaluación)
6. [Aplicaciones](#aplicaciones)

---

## 🎯 Introducción {#introducción}

### ¿Qué es Text Classification?

Asignar una o más categorías a un documento de texto.

```python
Texto: "This movie was amazing! I loved it."
Categoría: POSITIVE

Texto: "Python is a programming language"
Categoría: TECHNOLOGY

Texto: "Breaking news: Elections tomorrow"
Categoría: POLITICS, NEWS
```

### Tipos de Clasificación

**Binary Classification:**
```
spam / not spam
positive / negative
```

**Multi-class Classification:**
```
Categories: {sports, politics, technology, entertainment}
Each document → ONE category
```

**Multi-label Classification:**
```
Tags: {python, tutorial, beginner, video}
Each document → MULTIPLE tags
```

---

## 🔧 Feature Engineering {#features}

### 1. Bag of Words (BoW)

Cuenta frecuencia de palabras, ignora orden.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love Python",
    "Python is great",
    "I love programming"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
# ['great' 'is' 'love' 'programming' 'python']

print(X.toarray())
# [[0 0 1 0 1]  # "I love Python"
#  [1 1 0 0 1]  # "Python is great"
#  [0 0 1 1 0]] # "I love programming"
```

**Ventajas:** Simple, rápido
**Desventajas:** Ignora orden, contexto

### 2. TF-IDF

Term Frequency - Inverse Document Frequency.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Palabras comunes (como "the", "is") tienen menos peso
# Palabras raras/específicas tienen más peso
```

**Fórmulas:**
```
TF(t, d) = (número de veces que t aparece en d) / (total palabras en d)

IDF(t) = log(total docs / docs que contienen t)

TF-IDF(t, d) = TF(t, d) × IDF(t)
```

### 3. N-grams

Secuencias de N palabras.

```python
from sklearn.feature_extraction.text import CountVectorizer

# Unigrams (1 palabra)
vectorizer = CountVectorizer(ngram_range=(1, 1))
# Features: ["I", "love", "Python"]

# Bigrams (2 palabras)
vectorizer = CountVectorizer(ngram_range=(2, 2))
# Features: ["I love", "love Python"]

# Unigrams + Bigrams
vectorizer = CountVectorizer(ngram_range=(1, 2))
# Features: ["I", "love", "Python", "I love", "love Python"]
```

**Ejemplo:**
```python
text = "I love Python"

# (1,1): ["I", "love", "Python"]
# (2,2): ["I love", "love Python"]
# (1,2): ["I", "love", "Python", "I love", "love Python"]
# (1,3): todas las anteriores + ["I love Python"]
```

### 4. Word Embeddings

Vectores densos pre-entrenados (veremos en Koan 7).

```python
from gensim.models import Word2Vec

# Cada palabra → vector de 100-300 dimensiones
"python" → [0.1, -0.5, 0.8, ..., 0.3]  # 300 dims
```

---

## 🤖 Modelos Clásicos {#modelos}

### 1. Naive Bayes

Basado en probabilidades.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Datos
texts = ["I love this", "I hate this", "This is great"]
labels = [1, 0, 1]  # 1=positive, 0=negative

# Features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Modelo
model = MultinomialNB()
model.fit(X, labels)

# Predicción
new_text = ["I love Python"]
X_new = vectorizer.transform(new_text)
print(model.predict(X_new))  # [1]
```

**Ventajas:**
- ⚡ Muy rápido
- ✅ Funciona bien con poco datos
- 📊 Probabilístico

**Desventajas:**
- Asume independencia entre features

### 2. Logistic Regression

Modelo lineal con función sigmoide.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Probabilidades
probs = model.predict_proba(X_test)
# [[0.2, 0.8], [0.9, 0.1], ...]  # [prob_clase_0, prob_clase_1]
```

**Ventajas:**
- ✅ Interpretable
- ⚡ Rápido
- 📊 Probabilidades calibradas

### 3. Support Vector Machines (SVM)

Encuentra hiperplano óptimo que separa clases.

```python
from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(X_train, y_train)
```

**Ventajas:**
- ⭐ Alta precisión
- ✅ Funciona bien en alta dimensionalidad

**Desventajas:**
- 🐢 Lento con muchos datos
- No da probabilidades directamente

### 4. Random Forest

Ensemble de árboles de decisión.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
```

**Ventajas:**
- ⭐ Robusto
- ✅ Maneja features no lineales
- 📊 Feature importance

**Desventajas:**
- 💾 Requiere más memoria
- 🐢 Más lento que modelos lineales

### 5. Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

**Ventajas:**
- ⭐⭐ Mejor precisión
- ✅ Maneja features complejas

---

## 🔄 Pipeline Completo {#pipeline}

### Pipeline Básico

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Pipeline: vectorización + modelo
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Entrenar
pipeline.fit(X_train, y_train)

# Predecir (automáticamente vectoriza)
predictions = pipeline.predict(X_test)
```

### Pipeline con Preprocesamiento

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import string

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remover puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Aplicar preprocesamiento
X_train_clean = [preprocess(text) for text in X_train]

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train_clean, y_train)
```

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Parámetros a probar
parameters = {
    'tfidf__max_features': [1000, 5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__C': [0.1, 1, 10]
}

# Grid search
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

---

## 📊 Evaluación {#evaluación}

### Métricas

**Accuracy:**
```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
# ¿Qué % de predicciones son correctas?
```

**Precision, Recall, F1:**
```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))

#              precision    recall  f1-score   support
#           0       0.85      0.92      0.88       100
#           1       0.91      0.83      0.87       100
#    accuracy                           0.88       200
```

**Confusion Matrix:**
```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
#        Pred 0  Pred 1
# True 0   92      8
# True 1   17     83
```

**ROC-AUC:**
```python
from sklearn.metrics import roc_auc_score

# Requiere probabilidades
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_true, probs)
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
print(f"F1: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## 💼 Aplicaciones {#aplicaciones}

### 1. Spam Detection

```python
# Dataset
emails = [
    ("Win free money now!", "spam"),
    ("Meeting at 3pm", "ham"),
    ("Congratulations! You won!", "spam"),
]

X = [email[0] for email in emails]
y = [email[1] for email in emails]

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

pipeline.fit(X, y)

# Predicción
new_email = ["Free prize awaits you"]
print(pipeline.predict(new_email))  # ['spam']
```

### 2. Sentiment Analysis

```python
reviews = [
    ("This movie was amazing!", "positive"),
    ("Terrible waste of time", "negative"),
    ("I loved it", "positive"),
]

# Ver Koan 6 para análisis completo
```

### 3. Topic Classification

```python
articles = [
    ("Python tutorial for beginners", "technology"),
    ("Breaking: Elections tomorrow", "politics"),
    ("Lakers win championship", "sports"),
]

# Multi-class classification
```

### 4. Intent Classification (Chatbots)

```python
queries = [
    ("What's the weather?", "weather"),
    ("Tell me a joke", "entertainment"),
    ("Set alarm for 7am", "alarm"),
]

# Clasificar intención del usuario
```

---

## 🎓 Resumen

**Conceptos Clave:**
- Text Classification asigna categorías a documentos
- Features: BoW, TF-IDF, N-grams, Embeddings
- Modelos: Naive Bayes, Logistic Regression, SVM, Random Forest
- Pipeline: preprocesamiento → vectorización → modelo

**Mejores Prácticas:**
- Usar TF-IDF como baseline
- Probar múltiples modelos
- Usar cross-validation
- Grid search para hiperparámetros

**Próximos Pasos:**
- **Koan 6**: Sentiment Analysis (caso específico)
- **Koan 7**: Word Embeddings (mejores features)
- **Koan 8**: Transformers (DL para clasificación)

¡La clasificación es ubicua en NLP! 🚀
