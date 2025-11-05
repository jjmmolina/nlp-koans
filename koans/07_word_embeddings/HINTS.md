# 💡 Pistas para Koan 07: Word Embeddings

## 🎯 Objetivo del Koan

Aprender sobre **representaciones vectoriales de palabras**:
- Vectores densos capturan significado semántico
- Palabras similares tienen vectores similares
- Operaciones vectoriales revelan relaciones

---

## 📝 Función 1: `get_word_vector_spacy()`

### Nivel 1: Concepto
spaCy proporciona word vectors pre-entrenados con modelos medianos/grandes.

### Nivel 2: Implementación
```python
import spacy
import numpy as np
nlp = spacy.load("en_core_web_md")  # Necesita modelo con vectores
doc = nlp(word)
return doc.vector  # Vector numpy array
```

### Nivel 3: Importante
⚠️ Los modelos `sm` (small) NO tienen vectores. Necesitas `md` o `lg`.

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def get_word_vector_spacy(word: str, lang: str = "en") -> np.ndarray:
    import spacy
    import numpy as np
    
    # Modelos medianos tienen vectores (300 dimensiones)
    model = "en_core_web_md" if lang == "en" else "es_core_news_md"
    nlp = spacy.load(model)
    
    doc = nlp(word)
    return doc.vector
```
</details>

---

## 📝 Función 2: `cosine_similarity()`

### Nivel 1: Concepto
Mide similitud entre vectores: 1 = idénticos, 0 = no relacionados, -1 = opuestos.

### Nivel 2: Fórmula
```
similarity = (A · B) / (||A|| × ||B||)
```

### Nivel 3: Implementación
```python
import numpy as np
dot_product = np.dot(vec1, vec2)
norm1 = np.linalg.norm(vec1)
norm2 = np.linalg.norm(vec2)
return dot_product / (norm1 * norm2)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    import numpy as np
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    return float(dot_product / (norm_vec1 * norm_vec2))
```
</details>

---

## 📝 Función 3: `word_similarity()`

### Nivel 1: Concepto
Calcula similitud semántica entre dos palabras usando sus vectores.

### Nivel 2: Pasos
1. Obtén vectores de ambas palabras
2. Calcula cosine_similarity entre ellos

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def word_similarity(word1: str, word2: str, lang: str = "en") -> float:
    vec1 = get_word_vector_spacy(word1, lang)
    vec2 = get_word_vector_spacy(word2, lang)
    return cosine_similarity(vec1, vec2)
```
</details>

---

## 📝 Función 4: `most_similar_words()`

### Nivel 1: Concepto
Encuentra las palabras más similares a una palabra dada en un vocabulario.

### Nivel 2: Pasos
1. Obtén vector de la palabra objetivo
2. Calcula similitud con cada palabra del vocabulario
3. Ordena por similitud y retorna top_n

### Nivel 3: Casi la solución
```python
target_vec = get_word_vector_spacy(word, lang)
similarities = []
for vocab_word in vocabulary:
    if vocab_word != word:
        vec = get_word_vector_spacy(vocab_word, lang)
        sim = cosine_similarity(target_vec, vec)
        similarities.append((vocab_word, sim))

similarities.sort(key=lambda x: x[1], reverse=True)
return similarities[:top_n]
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def most_similar_words(word: str, vocabulary: List[str], 
                       lang: str = "en", top_n: int = 5) -> List[Tuple[str, float]]:
    target_vector = get_word_vector_spacy(word, lang)
    
    similarities = []
    for vocab_word in vocabulary:
        if vocab_word.lower() != word.lower():
            vocab_vector = get_word_vector_spacy(vocab_word, lang)
            similarity = cosine_similarity(target_vector, vocab_vector)
            similarities.append((vocab_word, similarity))
    
    # Ordenar por similitud descendente
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_n]
```
</details>

---

## 📝 Función 5: `load_word2vec_model()`

### Nivel 1: Concepto
Carga modelos Word2Vec pre-entrenados de Google o propios.

### Nivel 2: Implementación
```python
from gensim.models import KeyedVectors
# Para formato binario de Google
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def load_word2vec_model(model_path: str):
    from gensim.models import KeyedVectors
    # Carga modelo pre-entrenado (ej: GoogleNews-vectors-negative300.bin)
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model
```
</details>

---

## 📝 Función 6: `word_analogy()`

### Nivel 1: Concepto
Resuelve analogías: "rey es a reina como hombre es a ?"

**Respuesta**: mujer (usando aritmética vectorial)

### Nivel 2: Fórmula
```
resultado = vector(rey) - vector(hombre) + vector(mujer)
# Encuentra palabra más cercana a resultado
```

### Nivel 3: Implementación con gensim
```python
# word2vec_model tiene método most_similar que acepta positive/negative
result = word2vec_model.most_similar(
    positive=[word_a, word_c],  # "rey", "mujer"
    negative=[word_b],           # "hombre"
    topn=1
)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def word_analogy(word2vec_model, word_a: str, word_b: str, word_c: str) -> str:
    """
    Resuelve: word_a es a word_b como word_c es a ?
    Ejemplo: rey (a) es a hombre (b) como reina (c) es a mujer (?)
    """
    try:
        # Aritmética vectorial: resultado = a - b + c
        result = word2vec_model.most_similar(
            positive=[word_a, word_c],
            negative=[word_b],
            topn=1
        )
        return result[0][0]  # Retorna la palabra más similar
    except KeyError as e:
        return f"Word not in vocabulary: {e}"
```
</details>

---

## 📝 Función 7: `get_sentence_embedding()`

### Nivel 1: Concepto
Representa una oración completa como un solo vector promediando word vectors.

### Nivel 2: Pasos
1. Tokeniza la oración
2. Obtén vector de cada palabra
3. Calcula promedio de todos los vectores

### Nivel 3: Casi la solución
```python
import spacy
import numpy as np
nlp = spacy.load("en_core_web_md")
doc = nlp(sentence)
# doc.vector ya hace esto automáticamente!
return doc.vector
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def get_sentence_embedding(sentence: str, lang: str = "en") -> np.ndarray:
    import spacy
    import numpy as np
    
    model = "en_core_web_md" if lang == "en" else "es_core_news_md"
    nlp = spacy.load(model)
    
    doc = nlp(sentence)
    # spaCy promedia automáticamente los vectores de tokens
    return doc.vector
```
</details>

---

## 🎯 Conceptos Clave

### ¿Qué son Word Embeddings?

**Representación densa** de palabras como vectores de números reales:

```
"king"   → [0.32, -0.45, 0.18, ..., 0.67]  (300 dimensiones)
"queen"  → [0.35, -0.42, 0.21, ..., 0.69]  (similar a king)
"car"    → [-0.12, 0.73, -0.56, ..., 0.05] (muy diferente)
```

### Propiedades Importantes

1. **Similitud Semántica**:
   - Palabras similares → vectores cercanos
   - "perro" y "gato" más cercanos que "perro" y "avión"

2. **Aritmética Vectorial**:
   ```
   vector("rey") - vector("hombre") + vector("mujer") ≈ vector("reina")
   vector("Madrid") - vector("España") + vector("Francia") ≈ vector("París")
   ```

3. **Capturan Relaciones**:
   - Género: hombre/mujer, rey/reina
   - Geografía: país/capital
   - Tiempo: presente/pasado

### Técnicas Principales

| Método | Año | Características |
|--------|-----|-----------------|
| **Word2Vec** | 2013 | Rápido, eficiente (CBOW, Skip-gram) |
| **GloVe** | 2014 | Basado en estadísticas de co-ocurrencia |
| **FastText** | 2016 | Maneja palabras fuera de vocabulario |
| **spaCy vectors** | - | Pre-entrenados, fácil de usar |

## 💡 Tips Prácticos

### 1. Necesitas modelos con vectores
```python
# ❌ NO funciona
nlp = spacy.load("en_core_web_sm")  # Sin vectores
doc = nlp("hello")
print(doc.vector)  # Vector de ceros

# ✅ Funciona
nlp = spacy.load("en_core_web_md")  # Con vectores
doc = nlp("hello")
print(doc.vector)  # Vector real [300 dims]
```

### 2. Descarga modelos grandes
```bash
# Inglés con vectores (300D)
python -m spacy download en_core_web_md

# Español con vectores (300D)
python -m spacy download es_core_news_md
```

### 3. Normaliza palabras
```python
# Minúsculas para mejor matching
word_similarity("King", "king")  # Misma palabra
```

### 4. Maneja palabras fuera de vocabulario
```python
try:
    vec = model["palabra_rara_xyz"]
except KeyError:
    print("Palabra no en vocabulario")
    vec = np.zeros(300)  # Vector de ceros como fallback
```

## 🚀 Casos de Uso

### Búsqueda semántica
```python
query = "feliz"
docs = ["alegre", "contento", "triste", "carro"]
similarities = [(doc, word_similarity(query, doc, "es")) for doc in docs]
similarities.sort(key=lambda x: x[1], reverse=True)
# [("alegre", 0.85), ("contento", 0.78), ...]
```

### Detección de duplicados
```python
sentence1 = "The cat sits on the mat"
sentence2 = "A cat is sitting on a rug"
emb1 = get_sentence_embedding(sentence1)
emb2 = get_sentence_embedding(sentence2)
similarity = cosine_similarity(emb1, emb2)
# High similarity → likely duplicates
```

### Recomendación de contenido
```python
user_liked = ["machine learning", "AI", "neural networks"]
all_topics = ["deep learning", "cooking", "sports", "NLP"]

for topic in all_topics:
    avg_sim = np.mean([word_similarity(topic, liked) for liked in user_liked])
    print(f"{topic}: {avg_sim}")
```

### Clustering de palabras
```python
from sklearn.cluster import KMeans
words = ["dog", "cat", "car", "truck", "bird"]
vectors = [get_word_vector_spacy(w) for w in words]
kmeans = KMeans(n_clusters=2).fit(vectors)
# Clusters: [animales], [vehículos]
```

## 🔧 Troubleshooting

### Problema: `doc.vector` es todo ceros
**Solución**: Usa modelo con vectores (`md` o `lg`, no `sm`)

### Problema: KeyError al buscar palabra
**Solución**: Palabra no está en vocabulario
```python
if word in model.vocab:
    vec = model[word]
else:
    print("Palabra no encontrada")
```

### Problema: Similitud siempre baja
**Solución**: 
- Verifica que palabras estén en vocabulario
- Normaliza texto (minúsculas, lemmatización)

### Problema: Memoria insuficiente
**Solución**: Modelos grandes (GoogleNews 3.6GB)
```python
# Carga solo las palabras que necesitas
model = KeyedVectors.load_word2vec_format(path, binary=True, limit=100000)
```

## 🚀 Siguiente Paso

Una vez completo, ve al **Koan 08: Transformers**!
