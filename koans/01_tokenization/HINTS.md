# 💡 Pistas para Koan 01: Tokenización

## 📝 Función 1: `tokenize_words_nltk()`

### Nivel 1: Pista General
Necesitas importar y usar la función `word_tokenize` de NLTK.

### Nivel 2: Pista Específica
```python
from nltk.tokenize import word_tokenize
# Usa word_tokenize(text) y retórnalo
```

### Nivel 3: Casi la Solución
```python
def tokenize_words_nltk(text: str) -> List[str]:
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    return tokens
```

### ✅ Solución Completa
<details>
<summary>Click para ver la solución (¡intenta resolverlo primero!)</summary>

```python
def tokenize_words_nltk(text: str) -> List[str]:
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)
```
</details>

---

## 📝 Función 2: `tokenize_sentences_nltk()`

### Nivel 1: Pista General
Similar a `word_tokenize`, pero existe `sent_tokenize` para oraciones.

### Nivel 2: Pista Específica
```python
from nltk.tokenize import sent_tokenize
# Úsalo igual que word_tokenize pero para oraciones
```

### ✅ Solución Completa
<details>
<summary>Click para ver la solución</summary>

```python
def tokenize_sentences_nltk(text: str) -> List[str]:
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)
```
</details>

---

## 📝 Función 3: `tokenize_words_spacy()`

### Nivel 1: Pista General
spaCy requiere:
1. Cargar un modelo (español o inglés)
2. Procesar el texto
3. Extraer tokens

### Nivel 2: Pista Específica
```python
import spacy
# Carga modelo: "es_core_news_sm" para español, "en_core_web_sm" para inglés
nlp = spacy.load(modelo)
doc = nlp(text)
# Extrae tokens: [token.text for token in doc]
```

### Nivel 3: Casi la Solución
```python
def tokenize_words_spacy(text: str, lang: str = "es") -> List[str]:
    import spacy
    model = "es_core_news_sm" if lang == "es" else "en_core_web_sm"
    nlp = spacy.load(model)
    doc = nlp(text)
    return [token.text for token in doc]
```

### ✅ Solución Completa
<details>
<summary>Click para ver la solución</summary>

```python
def tokenize_words_spacy(text: str, lang: str = "es") -> List[str]:
    import spacy
    model_name = "es_core_news_sm" if lang == "es" else "en_core_web_sm"
    nlp = spacy.load(model_name)
    doc = nlp(text)
    return [token.text for token in doc]
```
</details>

---

## 📝 Función 4: `custom_tokenize()`

### Nivel 1: Pista General
Usa el método `.split()` de strings de Python.

### Nivel 2: Pista Específica
```python
# text.split(delimitador) separa por el delimitador
return text.split(delimiter)
```

### ✅ Solución Completa
<details>
<summary>Click para ver la solución</summary>

```python
def custom_tokenize(text: str, delimiter: str = " ") -> List[str]:
    return text.split(delimiter)
```
</details>

---

## 📝 Función 5: `count_tokens()`

### Nivel 1: Pista General
Necesitas:
1. Tokenizar el texto
2. Convertir a minúsculas
3. Contar frecuencias

### Nivel 2: Pista Específica
```python
from collections import Counter
tokens = tokenize_words_nltk(text)
# Convierte a minúsculas: [t.lower() for t in tokens]
# Usa Counter para contar
```

### ✅ Solución Completa
<details>
<summary>Click para ver la solución</summary>

```python
def count_tokens(text: str) -> dict:
    from collections import Counter
    tokens = tokenize_words_nltk(text)
    tokens_lower = [token.lower() for token in tokens]
    return dict(Counter(tokens_lower))
```
</details>

---

## 📝 Función 6: `remove_punctuation_tokens()`

### Nivel 1: Pista General
Usa `string.punctuation` que contiene todos los signos de puntuación.

### Nivel 2: Pista Específica
```python
import string
# string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
# Filtra tokens que NO estén en string.punctuation
```

### Nivel 3: Casi la Solución
```python
import string
return [token for token in tokens if token not in string.punctuation]
```

### ✅ Solución Completa
<details>
<summary>Click para ver la solución</summary>

```python
def remove_punctuation_tokens(tokens: List[str]) -> List[str]:
    import string
    return [token for token in tokens if token not in string.punctuation]
```
</details>

---

## 🎯 Consejos Generales

1. **Ejecuta los tests frecuentemente**: `pytest test_tokenization.py -v`
2. **Lee los mensajes de error**: Te dicen exactamente qué falta
3. **Usa print() para debug**: Imprime resultados intermedios
4. **Consulta la documentación**:
   - NLTK: https://www.nltk.org/
   - spaCy: https://spacy.io/

## 🚀 Siguiente Paso

Una vez que todos los tests pasen, ve al **Koan 02: Stemming y Lemmatization**!
