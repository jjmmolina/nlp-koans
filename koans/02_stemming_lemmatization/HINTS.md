# 💡 Pistas para Koan 02: Stemming y Lemmatization

## 🎯 Objetivo del Koan

Aprender a **normalizar palabras** reduciéndolas a su forma base:
- **Stemming**: Corta sufijos (rápido pero tosco)
- **Lemmatization**: Usa reglas lingüísticas (preciso pero lento)

---

## 📝 Función 1: `stem_word_porter()`

### Nivel 1: Concepto
El algoritmo Porter es el stemmer más usado para inglés. Corta sufijos de palabras.

### Nivel 2: Implementación
```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
# Usa stemmer.stem(palabra)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def stem_word_porter(word: str) -> str:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    return stemmer.stem(word)
```
</details>

---

## 📝 Función 2: `stem_word_snowball()`

### Nivel 1: Concepto
Snowball es mejor que Porter y soporta **múltiples idiomas** (español, inglés, etc.)

### Nivel 2: Implementación
```python
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer(language)  # "spanish" o "english"
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def stem_word_snowball(word: str, language: str = "spanish") -> str:
    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer(language)
    return stemmer.stem(word)
```
</details>

---

## 📝 Función 3: `stem_sentence()`

### Nivel 1: Concepto
Aplica stemming a cada palabra de una oración.

### Nivel 2: Pasos
1. Tokeniza la oración en palabras
2. Aplica `stem_word_snowball()` a cada palabra
3. Une las palabras con espacios

### Nivel 3: Casi la solución
```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize(sentence)
stemmed = [stem_word_snowball(token, language) for token in tokens]
return " ".join(stemmed)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def stem_sentence(sentence: str, language: str = "spanish") -> str:
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(sentence)
    stemmed_tokens = [stem_word_snowball(token, language) for token in tokens]
    return " ".join(stemmed_tokens)
```
</details>

---

## 📝 Función 4: `lemmatize_word_nltk()`

### Nivel 1: Concepto
Lemmatization encuentra la **forma canónica** de una palabra usando un diccionario.

### Nivel 2: POS tags
El parámetro `pos` es importante:
- `'n'` = noun (sustantivo)
- `'v'` = verb (verbo)
- `'a'` = adjective (adjetivo)
- `'r'` = adverb (adverbio)

### Nivel 3: Implementación
```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
return lemmatizer.lemmatize(word, pos=pos)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def lemmatize_word_nltk(word: str, pos: str = "n") -> str:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word, pos=pos)
```
</details>

---

## 📝 Función 5: `lemmatize_with_spacy()`

### Nivel 1: Concepto
spaCy hace lemmatization **automáticamente** cuando procesas texto.

### Nivel 2: Implementación
```python
import spacy
model = "es_core_news_sm" if lang == "es" else "en_core_web_sm"
nlp = spacy.load(model)
doc = nlp(text)
# Cada token tiene un atributo .lemma_
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def lemmatize_with_spacy(text: str, lang: str = "es") -> List[str]:
    import spacy
    model_name = "es_core_news_sm" if lang == "es" else "en_core_web_sm"
    nlp = spacy.load(model_name)
    doc = nlp(text)
    return [token.lemma_ for token in doc]
```
</details>

---

## 📝 Función 6: `compare_stem_vs_lemma()`

### Nivel 1: Concepto
Compara los resultados de stemming vs lemmatization para la misma palabra.

### Nivel 2: Pasos
1. Obtén el stem con `stem_word_snowball()`
2. Obtén el lema con `lemmatize_with_spacy()`
3. Retorna diccionario con ambos

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def compare_stem_vs_lemma(word: str, language: str = "spanish") -> dict:
    stem = stem_word_snowball(word, language)
    lemmas = lemmatize_with_spacy(word, lang="es" if language == "spanish" else "en")
    lemma = lemmas[0] if lemmas else word
    
    return {
        "original": word,
        "stem": stem,
        "lemma": lemma
    }
```
</details>

---

## 📝 Función 7: `normalize_text()`

### Nivel 1: Concepto
Normaliza un texto completo usando stemming o lemmatization según el método elegido.

### Nivel 2: Pasos
```python
if method == "stem":
    # Usa stem_sentence()
elif method == "lemma":
    # Usa lemmatize_with_spacy() y une con espacios
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def normalize_text(text: str, method: str = "lemma", language: str = "spanish") -> str:
    if method == "stem":
        return stem_sentence(text, language)
    elif method == "lemma":
        lemmas = lemmatize_with_spacy(text, lang="es" if language == "spanish" else "en")
        return " ".join(lemmas)
    else:
        return text
```
</details>

---

## 🎯 Conceptos Clave

### Diferencia Stem vs Lemma

| Aspecto | Stemming | Lemmatization |
|---------|----------|---------------|
| **Velocidad** | Rápido | Más lento |
| **Precisión** | Aproximado | Preciso |
| **Resultado** | Puede no ser palabra real | Siempre palabra válida |
| **Ejemplo** | "corriendo" → "corr" | "corriendo" → "correr" |

### ¿Cuándo usar cada uno?

**Usa Stemming cuando**:
- Necesitas velocidad
- Precisión no es crítica
- Búsqueda de texto / IR

**Usa Lemmatization cuando**:
- Necesitas precisión
- Análisis lingüístico
- Interpretación humana

## 💡 Tips

1. **Porter solo funciona bien para inglés**
2. **Snowball soporta 15+ idiomas**
3. **spaCy hace lemmatization gratis** cuando procesas texto
4. **Para español, lemmatization > stemming**

## 🚀 Siguiente Paso

Una vez completo, ve al **Koan 03: POS Tagging**!
