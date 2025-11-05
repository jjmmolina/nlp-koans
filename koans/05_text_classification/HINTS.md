# 💡 Pistas para Koan 05: Text Classification

## 🎯 Objetivo del Koan

Aprender a **clasificar textos** usando Machine Learning:
- Convertir texto a números (vectorización)
- Entrenar clasificadores
- Evaluar modelos
- Predecir categorías de nuevos textos

---

## 📝 Función 1: `vectorize_text_tfidf()`

### Nivel 1: Concepto
TF-IDF (Term Frequency-Inverse Document Frequency) convierte texto en vectores numéricos.

### Nivel 2: Implementación
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=max_features)
# fit_transform para entrenar y transformar
# transform para solo transformar
```

### Nivel 3: Casi la solución
```python
if train:
    return vectorizer.fit_transform(texts), vectorizer
else:
    return vectorizer.transform(texts)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def vectorize_text_tfidf(texts: List[str], max_features: int = 1000, 
                         vectorizer=None, train: bool = True):
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    if train:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(texts)
        return X, vectorizer
    else:
        if vectorizer is None:
            raise ValueError("Vectorizer is required when train=False")
        return vectorizer.transform(texts)
```
</details>

---

## 📝 Función 2: `train_naive_bayes()`

### Nivel 1: Concepto
Naive Bayes es un clasificador probabilístico simple pero efectivo para texto.

### Nivel 2: Implementación
```python
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
# Usa fit(X, y) para entrenar
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def train_naive_bayes(X_train, y_train):
    from sklearn.naive_bayes import MultinomialNB
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    return classifier
```
</details>

---

## 📝 Función 3: `train_logistic_regression()`

### Nivel 1: Concepto
Logistic Regression es un clasificador lineal que funciona bien para texto.

### Nivel 2: Implementación
```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=max_iter)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def train_logistic_regression(X_train, y_train, max_iter: int = 1000):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(max_iter=max_iter)
    classifier.fit(X_train, y_train)
    return classifier
```
</details>

---

## 📝 Función 4: `predict_text_class()`

### Nivel 1: Concepto
Usa el modelo entrenado para predecir la clase de nuevos textos.

### Nivel 2: Pasos
1. Vectoriza los textos con el vectorizer
2. Usa classifier.predict() para obtener predicciones

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def predict_text_class(texts: List[str], vectorizer, classifier) -> List[str]:
    X = vectorizer.transform(texts)
    return classifier.predict(X).tolist()
```
</details>

---

## 📝 Función 5: `evaluate_classifier()`

### Nivel 1: Concepto
Calcula métricas de rendimiento: accuracy, precision, recall, f1-score.

### Nivel 2: Implementación
```python
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, output_dict=True)
```

### Nivel 3: Casi la solución
```python
accuracy = accuracy_score(y_true, y_pred)
report_dict = classification_report(y_true, y_pred, output_dict=True)
return {
    "accuracy": accuracy,
    "precision": report_dict["weighted avg"]["precision"],
    "recall": report_dict["weighted avg"]["recall"],
    "f1_score": report_dict["weighted avg"]["f1-score"]
}
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def evaluate_classifier(y_true: List[str], y_pred: List[str]) -> dict:
    from sklearn.metrics import accuracy_score, classification_report
    
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        "accuracy": accuracy,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"]
    }
```
</details>

---

## 📝 Función 6: `create_text_classifier_pipeline()`

### Nivel 1: Concepto
Pipeline combina vectorización y clasificación en un solo objeto.

### Nivel 2: Implementación
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=1000)),
    ('classifier', MultinomialNB())
])
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def create_text_classifier_pipeline(classifier_type: str = "naive_bayes", 
                                     max_features: int = 1000):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    
    if classifier_type == "naive_bayes":
        classifier = MultinomialNB()
    elif classifier_type == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=max_features)),
        ('classifier', classifier)
    ])
    
    return pipeline
```
</details>

---

## 📝 Función 7: `get_feature_importance()`

### Nivel 1: Concepto
Identifica las palabras más importantes para cada clase.

### Nivel 2: Pasos
1. Obtén los nombres de features del vectorizer
2. Obtén los coeficientes del clasificador
3. Encuentra los top N features con mayor/menor coeficiente

### Nivel 3: Casi la solución
```python
import numpy as np
feature_names = vectorizer.get_feature_names_out()
coef = classifier.coef_[class_idx]
top_indices = np.argsort(coef)[-top_n:][::-1]
return [feature_names[i] for i in top_indices]
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def get_feature_importance(vectorizer, classifier, class_idx: int = 0, 
                           top_n: int = 10) -> List[str]:
    import numpy as np
    
    feature_names = vectorizer.get_feature_names_out()
    
    if hasattr(classifier, 'coef_'):
        coef = classifier.coef_[class_idx]
        top_indices = np.argsort(coef)[-top_n:][::-1]
        return [feature_names[i] for i in top_indices]
    else:
        raise ValueError("Classifier does not have coefficients (coef_)")
```
</details>

---

## 🎯 Conceptos Clave

### Pipeline de Clasificación de Texto

```
Texto crudo → Vectorización → Modelo ML → Predicción
```

1. **Vectorización**: Convertir texto a números
   - TF-IDF: Frecuencia de términos con peso por rareza
   - Count Vectorizer: Solo frecuencias
   - Word Embeddings: Vectores semánticos

2. **Clasificadores**:
   - Naive Bayes: Rápido, simple, bueno para texto
   - Logistic Regression: Interpretable, robusto
   - SVM: Bueno para alta dimensionalidad
   - Random Forest: Ensemble, no lineal

3. **Métricas de Evaluación**:
   - **Accuracy**: % de predicciones correctas
   - **Precision**: % de positivos correctos
   - **Recall**: % de positivos encontrados
   - **F1-Score**: Media armónica de precision y recall

### TF-IDF Explicado

**TF (Term Frequency)**: Cuántas veces aparece una palabra en un documento

**IDF (Inverse Document Frequency)**: Qué tan rara es una palabra en el corpus

```
TF-IDF = TF × IDF
```

**Palabras importantes**: Alta frecuencia en pocos documentos
**Palabras sin importancia**: Alta frecuencia en todos los documentos (stop words)

## 💡 Tips Prácticos

### 1. Preprocesamiento es clave
```python
# Mejora la clasificación
texts = [text.lower().strip() for text in texts]
```

### 2. Ajusta max_features
```python
# Pocos features = rápido pero menos preciso
vectorizer = TfidfVectorizer(max_features=100)

# Muchos features = lento pero más preciso
vectorizer = TfidfVectorizer(max_features=10000)
```

### 3. Usa Pipeline para evitar errores
```python
# Malo: fácil equivocarse
X_train, vectorizer = vectorize_text_tfidf(train_texts)
X_test = vectorizer.transform(test_texts)
classifier.fit(X_train, y_train)

# Bueno: pipeline maneja todo
pipeline = create_text_classifier_pipeline()
pipeline.fit(train_texts, y_train)
pipeline.predict(test_texts)
```

### 4. Split train/test correctamente
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
```

## 🚀 Casos de Uso

### Clasificación de emails (spam/no spam)
```python
pipeline = create_text_classifier_pipeline("naive_bayes")
pipeline.fit(emails, labels)  # labels = ["spam", "ham", ...]
prediction = pipeline.predict(["Click here to win $1000!!!"])
```

### Análisis de sentimientos básico
```python
texts = ["Me encanta", "Es horrible"]
labels = ["positivo", "negativo"]
pipeline.fit(texts, labels)
```

### Categorización de noticias
```python
categories = ["deportes", "política", "tecnología"]
pipeline = create_text_classifier_pipeline("logistic_regression")
```

## 🔧 Troubleshooting

### Problema: Accuracy muy baja
**Solución**: 
- Más datos de entrenamiento
- Mejor preprocesamiento
- Probar otros clasificadores

### Problema: Overfitting
**Solución**:
- Reducir max_features
- Regularización en LogisticRegression
- Más datos de entrenamiento

### Problema: Predicciones sesgadas
**Solución**:
- Balancear clases en entrenamiento
- Usar class_weight='balanced'

## 🚀 Siguiente Paso

Una vez completo, ve al **Koan 06: Sentiment Analysis**!
