# Teoría: Word Embeddings

## 📚 Tabla de Contenidos
1. [Introducción](#introducción)
2. [Word2Vec](#word2vec)
3. [GloVe](#glove)
4. [FastText](#fasttext)
5. [Visualización y Análisis](#visualización)
6. [Aplicaciones](#aplicaciones)

---

## 🎯 Introducción {#introducción}

### ¿Qué son los Word Embeddings?

Representaciones vectoriales densas de palabras que capturan su significado semántico.

```python
# One-hot encoding (tradicional) - SPARSE
"cat" → [0, 0, 0, 1, 0, 0, ..., 0]  # 10,000 dimensiones, un solo 1

# Word embedding - DENSE
"cat" → [0.2, -0.5, 0.8, 0.1, -0.3]  # 300 dimensiones, todos números reales
```

### Propiedades Mágicas

**1. Similitud Semántica:**
```python
distance("king", "queen") < distance("king", "car")
```

**2. Analogías:**
```python
king - man + woman ≈ queen
Paris - France + Spain ≈ Madrid
```

**3. Clustering:**
```python
# Palabras similares están cerca en el espacio vectorial
{"car", "truck", "vehicle"} → cluster de vehículos
{"happy", "joyful", "glad"} → cluster de emociones positivas
```

### Ventajas sobre One-Hot

| Aspecto | One-Hot | Embeddings |
|---------|---------|------------|
| **Dimensionalidad** | Vocabulario (10k-100k) | 50-300 |
| **Esparsidad** | Muy sparse (99.99% ceros) | Denso |
| **Similitud** | ❌ No captura | ✅ Captura |
| **Memoria** | ❌ Mucha | ✅ Poca |

---

## 🔤 Word2Vec {#word2vec}

### Concepto

Predice palabras basándose en contexto (o viceversa).

**Dos Arquitecturas:**

**1. CBOW (Continuous Bag of Words):**
```
Contexto → Palabra

["The", "cat", "on", "the"] → "sat"
```

**2. Skip-gram:**
```
Palabra → Contexto

"sat" → ["The", "cat", "on", "the"]
```

### Entrenamiento

```python
from gensim.models import Word2Vec

sentences = [
    ["I", "love", "machine", "learning"],
    ["I", "love", "natural", "language", "processing"],
    ["machine", "learning", "is", "great"],
]

# Entrenar modelo
model = Word2Vec(
    sentences,
    vector_size=100,  # Dimensionalidad
    window=5,         # Ventana de contexto
    min_count=1,      # Mínima frecuencia
    workers=4,        # Procesos paralelos
    sg=0              # 0=CBOW, 1=Skip-gram
)

# Obtener embedding
vector = model.wv['love']
print(vector.shape)  # (100,)
```

### Operaciones con Vectores

```python
# Similitud
similarity = model.wv.similarity('machine', 'learning')
print(f"Similarity: {similarity:.3f}")

# Palabras más similares
similar = model.wv.most_similar('learning', topn=5)
print(similar)
# [('machine', 0.95), ('processing', 0.87), ...]

# Analogías
result = model.wv.most_similar(
    positive=['king', 'woman'],
    negative=['man'],
    topn=1
)
print(result)  # [('queen', 0.85)]

# No relacionadas
odd_one = model.wv.doesnt_match(['cat', 'dog', 'car'])
print(odd_one)  # 'car'
```

### Pre-trained Models

```python
import gensim.downloader as api

# Cargar modelo pre-entrenado
model = api.load('word2vec-google-news-300')

# 3 millones de palabras, 300 dimensiones
vector = model['python']
print(vector.shape)  # (300,)

# Analogías
result = model.most_similar(
    positive=['Paris', 'Germany'],
    negative=['France'],
    topn=1
)
print(result)  # [('Berlin', 0.76)]
```

---

## 🌐 GloVe {#glove}

### Global Vectors for Word Representation

Basado en co-ocurrencias globales en el corpus.

**Idea:**
```
Si "ice" y "steam" aparecen frecuentemente con "water"
pero raramente juntas entre sí,
sus vectores deben reflejar eso.
```

### Uso con Gensim

```python
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Convertir formato GloVe a Word2Vec
glove2word2vec(
    glove_input_file='glove.6B.100d.txt',
    word2vec_output_file='glove.6B.100d.word2vec.txt'
)

# Cargar
model = KeyedVectors.load_word2vec_format(
    'glove.6B.100d.word2vec.txt',
    binary=False
)

# Usar igual que Word2Vec
vector = model['computer']
similar = model.most_similar('computer')
```

### Pre-trained GloVe Models

```
glove.6B   → 6 billion tokens, Wikipedia + Gigaword
             50d, 100d, 200d, 300d

glove.42B  → 42 billion tokens, Common Crawl
             300d

glove.840B → 840 billion tokens, Common Crawl
             300d
```

### Word2Vec vs GloVe

| Aspecto | Word2Vec | GloVe |
|---------|----------|-------|
| **Método** | Predicción local | Estadísticas globales |
| **Velocidad** | ⚡⚡ | ⚡⚡⚡ |
| **Calidad** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Analogías** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 📝 FastText {#fasttext}

### Subword Information

A diferencia de Word2Vec/GloVe, FastText descompone palabras en n-gramas de caracteres.

```python
# Word2Vec: trata "running" como átomo
"running" → vector único

# FastText: usa n-gramas
"running" → ["<ru", "run", "unn", "nni", "nin", "ing", "ng>"]
           → combina vectores de n-gramas
```

**Ventaja:** Maneja palabras desconocidas (OOV).

```python
from gensim.models import FastText

sentences = [
    ["I", "love", "programming"],
    ["I", "love", "coding"],
]

# Entrenar
model = FastText(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    min_n=3,  # N-grama mínimo
    max_n=6   # N-grama máximo
)

# Palabra en vocabulario
vector_known = model.wv['programming']

# Palabra NO en vocabulario (OOV)
# FastText puede generar vector basándose en n-gramas
vector_oov = model.wv['programmmming']  # typo
print(vector_oov.shape)  # (100,) ✅ Funciona!
```

### Pre-trained FastText

```python
import gensim.downloader as api

# Modelo pre-entrenado
model = api.load('fasttext-wiki-news-subwords-300')

# Maneja palabras no vistas
vector = model['COVID-19']  # Palabra reciente
print(vector.shape)  # (300,)
```

---

## 📊 Visualización y Análisis {#visualización}

### Reducción de Dimensionalidad

**PCA:**
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Obtener vectores
words = ['king', 'queen', 'man', 'woman', 'boy', 'girl']
vectors = [model.wv[word] for word in words]

# Reducir a 2D
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Plotear
plt.figure(figsize=(10, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])

for i, word in enumerate(words):
    plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]))

plt.show()
```

**t-SNE:**
```python
from sklearn.manifold import TSNE

# Reducir a 2D (mejor para visualización)
tsne = TSNE(n_components=2, random_state=42)
vectors_2d = tsne.fit_transform(vectors)

# Plotear igual que PCA
```

### Análisis de Clusters

```python
from sklearn.cluster import KMeans

# Clustering
words = list(model.wv.index_to_key[:100])  # Top 100 palabras
vectors = [model.wv[word] for word in words]

kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(vectors)

# Palabras por cluster
for cluster_id in range(10):
    cluster_words = [words[i] for i, c in enumerate(clusters) if c == cluster_id]
    print(f"Cluster {cluster_id}: {cluster_words[:5]}")
```

---

## 💼 Aplicaciones {#aplicaciones}

### 1. Text Classification

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Representar documentos como promedio de word vectors
def document_vector(doc, model):
    vectors = [model.wv[word] for word in doc if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Vectorizar corpus
X = [document_vector(doc, model) for doc in documents]
y = labels

# Clasificar
clf = LogisticRegression()
clf.fit(X, y)
```

### 2. Document Similarity

```python
from scipy.spatial.distance import cosine

def doc_similarity(doc1, doc2, model):
    vec1 = document_vector(doc1, model)
    vec2 = document_vector(doc2, model)
    return 1 - cosine(vec1, vec2)

doc1 = ["I", "love", "machine", "learning"]
doc2 = ["I", "like", "deep", "learning"]

similarity = doc_similarity(doc1, doc2, model)
print(f"Similarity: {similarity:.3f}")
```

### 3. Word Analogies

```python
# Encontrar relaciones
def solve_analogy(a, b, c, model):
    """a is to b as c is to ?"""
    result = model.wv.most_similar(
        positive=[b, c],
        negative=[a],
        topn=1
    )
    return result[0][0]

# Ejemplos
print(solve_analogy('man', 'king', 'woman', model))  # queen
print(solve_analogy('Paris', 'France', 'Madrid', model))  # Spain
```

### 4. Semantic Search

```python
# Buscar documentos similares a query
query = "machine learning tutorial"
query_vec = document_vector(query.split(), model)

# Calcular similitud con todos los docs
similarities = []
for doc in documents:
    doc_vec = document_vector(doc, model)
    sim = 1 - cosine(query_vec, doc_vec)
    similarities.append(sim)

# Top-K documentos más similares
top_k = np.argsort(similarities)[-5:][::-1]
print("Most similar documents:", top_k)
```

### 5. Feature Engineering

```python
# Usar embeddings como features para ML
def extract_embedding_features(text, model):
    words = text.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    
    if not vectors:
        return np.zeros(model.vector_size * 3)
    
    # Features estadísticos
    mean_vec = np.mean(vectors, axis=0)
    max_vec = np.max(vectors, axis=0)
    min_vec = np.min(vectors, axis=0)
    
    # Concatenar
    features = np.concatenate([mean_vec, max_vec, min_vec])
    return features
```

---

## 🎓 Resumen

**Conceptos Clave:**
- Word Embeddings son representaciones densas de palabras
- Capturan similitud semántica y relaciones
- Word2Vec: CBOW y Skip-gram
- GloVe: Co-ocurrencias globales
- FastText: Maneja OOV con subwords

**Modelos:**
- Word2Vec → Estándar, bueno para analogías
- GloVe → Rápido, buenas analogías
- FastText → Mejor para OOV, morfología

**Próximos Pasos:**
- **Koan 8**: Transformers (embeddings contextuales)
- **Koan 12**: Semantic Search (usando embeddings)

¡Los embeddings son la base del NLP moderno! 🚀
