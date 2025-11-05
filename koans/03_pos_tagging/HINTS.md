# 💡 Pistas para Koan 03: POS Tagging

## 🎯 Objetivo del Koan

Aprender a **etiquetar categorías gramaticales** (Part-of-Speech):
- Identificar verbos, sustantivos, adjetivos, etc.
- Usar etiquetas universales (Universal Dependencies)
- Extraer palabras por categoría

---

## 📝 Función 1: `pos_tag_nltk()`

### Nivel 1: Concepto
NLTK tiene un tagger entrenado para inglés que asigna etiquetas POS a cada palabra.

### Nivel 2: Implementación
```python
from nltk import pos_tag, word_tokenize
tokens = word_tokenize(text)
# pos_tag(tokens) retorna lista de tuplas (palabra, etiqueta)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def pos_tag_nltk(text: str) -> List[Tuple[str, str]]:
    from nltk import pos_tag, word_tokenize
    tokens = word_tokenize(text)
    return pos_tag(tokens)
```
</details>

---

## 📝 Función 2: `pos_tag_spacy()`

### Nivel 1: Concepto
spaCy hace POS tagging automáticamente. Cada token tiene:
- `token.pos_`: Etiqueta universal (NOUN, VERB, ADJ, etc.)
- `token.tag_`: Etiqueta específica del idioma

### Nivel 2: Implementación
```python
import spacy
nlp = spacy.load("es_core_news_sm")
doc = nlp(text)
# Retorna lista de (texto, pos_, tag_)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def pos_tag_spacy(text: str, lang: str = "es") -> List[Tuple[str, str, str]]:
    import spacy
    model = "es_core_news_sm" if lang == "es" else "en_core_web_sm"
    nlp = spacy.load(model)
    doc = nlp(text)
    return [(token.text, token.pos_, token.tag_) for token in doc]
```
</details>

---

## 📝 Función 3: `get_nouns()`

### Nivel 1: Concepto
Extrae solo los sustantivos de un texto usando spaCy.

### Nivel 2: Pasos
1. Procesa texto con spaCy
2. Filtra tokens donde `token.pos_ == "NOUN"`
3. Retorna lista de textos

### Nivel 3: Casi la solución
```python
doc = nlp(text)
return [token.text for token in doc if token.pos_ == "NOUN"]
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def get_nouns(text: str, lang: str = "es") -> List[str]:
    import spacy
    model = "es_core_news_sm" if lang == "es" else "en_core_web_sm"
    nlp = spacy.load(model)
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ == "NOUN"]
```
</details>

---

## 📝 Función 4: `get_verbs()`

### Nivel 1: Concepto
Similar a `get_nouns()`, pero filtra por `token.pos_ == "VERB"`.

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def get_verbs(text: str, lang: str = "es") -> List[str]:
    import spacy
    model = "es_core_news_sm" if lang == "es" else "en_core_web_sm"
    nlp = spacy.load(model)
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ == "VERB"]
```
</details>

---

## 📝 Función 5: `get_adjectives()`

### Nivel 1: Concepto
Filtra por `token.pos_ == "ADJ"` para obtener adjetivos.

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def get_adjectives(text: str, lang: str = "es") -> List[str]:
    import spacy
    model = "es_core_news_sm" if lang == "es" else "en_core_web_sm"
    nlp = spacy.load(model)
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ == "ADJ"]
```
</details>

---

## 📝 Función 6: `filter_by_pos()`

### Nivel 1: Concepto
Versión genérica que filtra por **cualquier** etiqueta POS.

### Nivel 2: Pasos
```python
doc = nlp(text)
# Convierte pos_tags a set para búsqueda rápida
pos_set = set(pos_tags)
return [token.text for token in doc if token.pos_ in pos_set]
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def filter_by_pos(text: str, pos_tags: List[str], lang: str = "es") -> List[str]:
    import spacy
    model = "es_core_news_sm" if lang == "es" else "en_core_web_sm"
    nlp = spacy.load(model)
    doc = nlp(text)
    pos_set = set(pos_tags)
    return [token.text for token in doc if token.pos_ in pos_set]
```
</details>

---

## 📝 Función 7: `analyze_sentence_structure()`

### Nivel 1: Concepto
Cuenta cuántas palabras de cada categoría hay en el texto.

### Nivel 2: Implementación
```python
from collections import Counter
doc = nlp(text)
pos_counts = Counter([token.pos_ for token in doc])
return dict(pos_counts)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def analyze_sentence_structure(text: str, lang: str = "es") -> dict:
    import spacy
    from collections import Counter
    model = "es_core_news_sm" if lang == "es" else "en_core_web_sm"
    nlp = spacy.load(model)
    doc = nlp(text)
    pos_counts = Counter([token.pos_ for token in doc])
    return dict(pos_counts)
```
</details>

---

## 🎯 Etiquetas POS Universales

### Principales etiquetas (Universal Dependencies)

| Etiqueta | Nombre | Ejemplo ES | Ejemplo EN |
|----------|--------|------------|------------|
| **NOUN** | Sustantivo | casa, perro | house, dog |
| **VERB** | Verbo | comer, correr | eat, run |
| **ADJ** | Adjetivo | grande, azul | big, blue |
| **ADV** | Adverbio | rápidamente | quickly |
| **PRON** | Pronombre | él, ella | he, she |
| **DET** | Determinante | el, la, un | the, a |
| **ADP** | Preposición | de, en, por | of, in, by |
| **CONJ** | Conjunción | y, o, pero | and, or, but |
| **PUNCT** | Puntuación | . , ; | . , ; |
| **NUM** | Número | uno, 42 | one, 42 |

### NLTK vs spaCy Tags

**NLTK (Penn Treebank)**:
- Solo inglés
- Etiquetas específicas: NN, NNS, VB, VBD, JJ, etc.
- Más de 36 etiquetas

**spaCy (Universal Dependencies)**:
- Multiidioma
- Etiquetas universales: 17 categorías principales
- Consistente entre idiomas

## 💡 Tips

1. **spaCy es mejor para producción** (más rápido y preciso)
2. **Usa `token.pos_` para etiquetas universales** (NOUN, VERB)
3. **Usa `token.tag_` para etiquetas específicas** (NN, VB, etc.)
4. **Filtra por múltiples POS** con `filter_by_pos()`

## 🚀 Casos de Uso

### Extracción de información
```python
# Obtener todos los nombres propios
nouns = get_nouns("María vive en Madrid")
# ['María', 'Madrid']
```

### Análisis de estilo
```python
# Verificar densidad de adjetivos
structure = analyze_sentence_structure(text)
adj_ratio = structure.get('ADJ', 0) / sum(structure.values())
```

### Simplificación de texto
```python
# Mantener solo contenido importante
important = filter_by_pos(text, ['NOUN', 'VERB', 'ADJ'])
```

## 🚀 Siguiente Paso

Una vez completo, ve al **Koan 04: Named Entity Recognition**!
