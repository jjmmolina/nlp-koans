# Teoría: Semantic Search & Vector Databases

## 📚 Tabla de Contenidos
1. [Introducción a Búsqueda Semántica](#introducción)
2. [Embeddings: Representaciones Vectoriales](#embeddings)
3. [Métricas de Similitud](#métricas)
4. [Vector Databases](#vector-databases)
5. [Estrategias de Búsqueda](#estrategias)
6. [Reranking](#reranking)
7. [Optimización y Escalado](#optimización)
8. [Casos de Uso](#casos-uso)

---

## 🔍 Introducción a Búsqueda Semántica {#introducción}

### Búsqueda Tradicional vs Semántica

**Búsqueda Tradicional (Keyword-based):**
```
Query: "python programming"
Método: Busca documentos que contengan exactamente "python" y "programming"

Encuentra:
✅ "Learn Python programming in 10 days"
❌ "Master the art of coding with Python" (no tiene "programming")
❌ "Python tutorial for beginners" (no tiene "programming")
```

**Búsqueda Semántica (Meaning-based):**
```
Query: "python programming"
Método: Entiende el SIGNIFICADO y busca contenido similar semánticamente

Encuentra:
✅ "Learn Python programming in 10 days"
✅ "Master the art of coding with Python" (coding ≈ programming)
✅ "Python tutorial for beginners" (tutorial implica programming)
✅ "Build applications with Python" (similar semánticamente)
```

### ¿Por qué Búsqueda Semántica?

**Ventajas:**
- 🎯 Entiende intención, no solo palabras exactas
- 🌍 Funciona con diferentes idiomas
- 📝 Maneja sinónimos y paráfrasis
- 🧠 Captura conceptos y relaciones
- ✨ Mejores resultados para queries naturales

**Desventajas:**
- 💰 Más costoso computacionalmente
- 🐌 Más lento que búsqueda de keywords
- 🔧 Requiere embeddings pre-calculados
- 📊 Necesita modelos de embeddings

### Evolución de la Búsqueda

```
1990s: TF-IDF, BM25 (estadística)
  ↓
2000s: PageRank, Link Analysis
  ↓
2013: Word2Vec (primeros embeddings útiles)
  ↓
2018: BERT (embeddings contextuales)
  ↓
2019: Sentence Transformers (embeddings de oraciones)
  ↓
2020: Dense Retrieval supera a BM25
  ↓
2023: Embeddings multimodales (texto + imagen)
  ↓
2024: Embeddings de alta dimensión (OpenAI, Cohere)
       Vector databases en producción
```

---

## 🎯 Embeddings: Representaciones Vectoriales {#embeddings}

### ¿Qué es un Embedding?

Un **embedding** es una representación numérica (vector) de texto que captura su significado semántico.

```python
"perro"     → [0.2, -0.5, 0.8, ..., 0.1]  # 384 dimensiones
"gato"      → [0.3, -0.4, 0.7, ..., 0.2]  # Similar a "perro"
"ordenador" → [-0.8, 0.2, -0.3, ..., 0.9] # Muy diferente
```

### Propiedades Clave

**1. Similitud Semántica**
```
Textos similares → Vectores cercanos en el espacio
"rey" - "hombre" + "mujer" ≈ "reina"
```

**2. Dimensionalidad**
- Típicamente: 384, 768, 1536, 3072 dimensiones
- Más dimensiones = más precisión (pero más costo)

**3. Normalización**
- Vectores suelen normalizarse (longitud = 1)
- Facilita cálculo de similitud coseno

### Tipos de Embeddings

#### Word Embeddings (Nivel Palabra)

**Word2Vec (2013)**
```python
# Predice palabra siguiente o palabra en contexto
"The cat sat on the __"  → probabilidades de palabras
```

**Características:**
- Una representación por palabra
- No captura contexto
- "banco" (dinero) = "banco" (asiento)

**GloVe (2014)**
- Basado en co-ocurrencias globales
- Similar a Word2Vec en práctica

#### Contextualizados (Nivel Oración)

**BERT (2018)**
```python
# Embeddings dependen del contexto
"Fui al banco a sacar dinero" → embedding_banco_1
"Me senté en el banco del parque" → embedding_banco_2
# embedding_banco_1 ≠ embedding_banco_2
```

**Sentence-BERT (2019)**
- Optimizado para embeddings de oraciones completas
- Rápido y eficiente
- Estado del arte para búsqueda semántica

#### Modernos (2023-2024)

**OpenAI text-embedding-3**
```python
# Dimensiones configurables: 1536 o 3072
# Multilingüe
# Optimizado para búsqueda
```

**Cohere Embed v3**
```python
# Soporta documentos largos (hasta 512 tokens)
# Embeddings comprimibles
```

### Modelos Populares

| Modelo | Proveedor | Dim | Costo | Calidad | Uso |
|--------|-----------|-----|-------|---------|-----|
| **text-embedding-3-small** | OpenAI | 1536 | 💰 | ⭐⭐⭐⭐ | Producción |
| **text-embedding-3-large** | OpenAI | 3072 | 💰💰 | ⭐⭐⭐⭐⭐ | Máxima calidad |
| **all-MiniLM-L6-v2** | HuggingFace | 384 | Free | ⭐⭐⭐ | Desarrollo |
| **all-mpnet-base-v2** | HuggingFace | 768 | Free | ⭐⭐⭐⭐ | Balance |
| **multilingual-e5-large** | HuggingFace | 1024 | Free | ⭐⭐⭐⭐ | Multilingüe |

### Generando Embeddings

**OpenAI API:**
```python
from openai import OpenAI

client = OpenAI()
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Texto a convertir en embedding"
)

embedding = response.data[0].embedding  # Lista de 1536 floats
```

**Sentence Transformers (local):**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("Texto a convertir")  # Array numpy de 384 floats
```

**Batch Processing:**
```python
# ✅ Eficiente: procesar múltiples textos a la vez
texts = ["texto 1", "texto 2", ..., "texto N"]
embeddings = model.encode(texts, batch_size=32)
```

### Dimensionalidad

**Ventajas de más dimensiones:**
- ✅ Mayor precisión
- ✅ Captura más matices semánticos
- ✅ Mejor para tareas complejas

**Desventajas:**
- ❌ Más memoria
- ❌ Búsqueda más lenta
- ❌ Mayor costo (APIs)

**Reducción de Dimensionalidad:**
```python
from sklearn.decomposition import PCA

# Reducir de 1536 a 512 dimensiones
pca = PCA(n_components=512)
embeddings_reduced = pca.fit_transform(embeddings)
```

---

## 📏 Métricas de Similitud {#métricas}

### 1. Cosine Similarity (Similitud Coseno)

**Concepto:**
Mide el ángulo entre dos vectores.

```
cosine_sim = (A · B) / (||A|| × ||B||)

Donde:
A · B = producto punto
||A|| = magnitud (norma) de A
```

**Rango:**
- `-1`: Vectores opuestos
- `0`: Vectores perpendiculares (no relacionados)
- `+1`: Vectores idénticos

**En práctica con embeddings normalizados:**
- Rango típico: `0.0` a `1.0`
- `> 0.8`: Muy similar
- `0.5 - 0.8`: Similar
- `< 0.5`: Poco similar

**Implementación:**
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Embeddings normalizados
vec_a = np.array([0.5, 0.8, 0.2])
vec_b = np.array([0.6, 0.7, 0.3])

similarity = cosine_similarity([vec_a], [vec_b])[0][0]
# similarity ≈ 0.95
```

**Ventajas:**
- ✅ No afectado por magnitud (solo dirección)
- ✅ Funciona bien con embeddings normalizados
- ✅ Interpretable intuitivamente

### 2. Euclidean Distance (Distancia Euclidiana)

**Concepto:**
Distancia en línea recta entre dos puntos.

```
euclidean_dist = √(Σ(a_i - b_i)²)
```

**Rango:**
- `0`: Vectores idénticos
- `∞`: Sin límite superior

**Implementación:**
```python
from scipy.spatial.distance import euclidean

distance = euclidean(vec_a, vec_b)
```

**Nota:**
- Para embeddings normalizados, cosine similarity y euclidean distance están relacionados:
  `euclidean_dist = √(2 - 2 * cosine_sim)`

### 3. Dot Product (Producto Punto)

**Concepto:**
Suma de productos elemento a elemento.

```
dot_product = Σ(a_i × b_i)
```

**Para vectores normalizados:**
- Equivalente a cosine similarity
- Más rápido de calcular

**Implementación:**
```python
dot_prod = np.dot(vec_a, vec_b)
```

### Comparativa

| Métrica | Velocidad | Mejor Para | Sensible a Magnitud |
|---------|-----------|------------|---------------------|
| **Cosine** | ⚡⚡ | Embeddings generales | No |
| **Euclidean** | ⚡⚡⚡ | Clustering | Sí |
| **Dot Product** | ⚡⚡⚡⚡ | Vectores normalizados | Sí |

**Recomendación:**
- Para embeddings normalizados: **Dot Product** (más rápido)
- Para embeddings sin normalizar: **Cosine Similarity**
- Para clustering: **Euclidean Distance**

---

## 🗄️ Vector Databases {#vector-databases}

### ¿Qué es una Vector Database?

Una base de datos optimizada para almacenar y buscar vectores (embeddings) eficientemente.

**Problema a Resolver:**
```python
# ❌ Búsqueda ingenua: O(n) - muy lento
def naive_search(query_embedding, all_embeddings):
    similarities = []
    for emb in all_embeddings:  # 1 millón de embeddings
        sim = cosine_similarity(query_embedding, emb)
        similarities.append(sim)
    return top_k(similarities, k=10)

# Tiempo: ~segundos para millones de vectores
```

```python
# ✅ Vector DB: O(log n) o mejor
def vector_db_search(query_embedding):
    results = vector_db.search(query_embedding, k=10)
    return results

# Tiempo: ~milisegundos para millones de vectores
```

### Características Clave

**1. Indexación Eficiente**
- Algoritmos ANN (Approximate Nearest Neighbors)
- Trade-off: velocidad vs precisión

**2. Escalabilidad**
- Manejo de millones/billones de vectores
- Distribución horizontal

**3. Filtrado de Metadata**
```python
# Buscar embeddings + filtrar por metadata
results = db.search(
    query_embedding,
    filter={"category": "technology", "date": ">2024-01-01"}
)
```

**4. Actualizaciones en Tiempo Real**
- Añadir/eliminar vectores dinámicamente
- Re-indexación incremental

### Vector Databases Populares

#### 1. **ChromaDB**

**Características:**
- 🎯 Diseñada para simplicidad
- 💾 In-memory o persistente
- 🐍 Python-first
- 🆓 Open-source y gratuita

**Cuándo Usar:**
- Prototipos y desarrollo
- Aplicaciones pequeñas/medianas (< 1M vectores)
- Embeddings generados localmente

**Ejemplo:**
```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

# Añadir documentos (embeddings automáticos)
collection.add(
    documents=["Doc 1", "Doc 2"],
    ids=["id1", "id2"],
    metadatas=[{"source": "web"}, {"source": "pdf"}]
)

# Buscar
results = collection.query(
    query_texts=["consulta"],
    n_results=5
)
```

#### 2. **FAISS (Facebook AI Similarity Search)**

**Características:**
- ⚡ Extremadamente rápido
- 🎓 Algoritmos de investigación de Facebook
- 💻 CPU y GPU support
- 🆓 Open-source

**Cuándo Usar:**
- Necesitas máxima velocidad
- Tienes millones de vectores
- Quieres control total sobre índices

**Tipos de Índices:**
```python
import faiss

# Flat (exacto pero lento para escala)
index = faiss.IndexFlatL2(dimension)

# IVF (particionado - balance velocidad/precisión)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# HNSW (graph-based - rápido y preciso)
index = faiss.IndexHNSWFlat(dimension, M)
```

#### 3. **Pinecone**

**Características:**
- ☁️ Totalmente cloud/SaaS
- 📈 Escalado automático
- 🔌 API simple
- 💰 De pago

**Cuándo Usar:**
- Producción sin infrastructure management
- Necesitas escalado automático
- Budget disponible

**Ejemplo:**
```python
import pinecone

pinecone.init(api_key="...")
index = pinecone.Index("my-index")

# Upsert vectores
index.upsert(vectors=[
    ("id1", [0.1, 0.2, ...], {"meta": "data"}),
    ("id2", [0.3, 0.4, ...], {"meta": "data"})
])

# Query
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=10,
    filter={"category": "tech"}
)
```

#### 4. **Weaviate**

**Características:**
- 🎯 GraphQL API
- 🧠 Módulos de AI integrados
- 🔄 Vectorización automática
- 🆓 Open-source + cloud

#### 5. **Qdrant**

**Características:**
- 🦀 Escrito en Rust (rápido)
- 🎯 Filtrado avanzado
- 📊 Payloads ricos
- 🆓 Open-source

#### 6. **Milvus**

**Características:**
- 🏢 Enterprise-grade
- 📈 Petabyte-scale
- 🔌 Múltiples índices
- 🆓 Open-source

### Comparativa

| Database | Facilidad | Velocidad | Escala | Costo | Mejor Para |
|----------|-----------|-----------|--------|-------|------------|
| **ChromaDB** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Free | Prototipos |
| **FAISS** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free | Alta performance |
| **Pinecone** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$$ | Producción managed |
| **Weaviate** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free/$ | GraphQL apps |
| **Qdrant** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Free/$ | Balance óptimo |
| **Milvus** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Free | Enterprise |

---

## 🎯 Estrategias de Búsqueda {#estrategias}

### 1. Dense Retrieval (Búsqueda Densa)

Solo embeddings semánticos.

```python
# 1. Generar embedding de consulta
query_emb = model.encode("Python programming")

# 2. Buscar en vector DB
results = vector_db.search(query_emb, k=10)
```

**Pros:**
- ✅ Entiende semántica
- ✅ Funciona con sinónimos
- ✅ Queries naturales

**Cons:**
- ❌ Puede fallar con nombres propios
- ❌ Fechas y números exactos problemáticos

### 2. Sparse Retrieval (Búsqueda Dispersa)

Métodos tradicionales: BM25, TF-IDF.

```python
from rank_bm25 import BM25Okapi

# 1. Tokenizar documentos
tokenized_docs = [doc.split() for doc in documents]

# 2. Crear índice BM25
bm25 = BM25Okapi(tokenized_docs)

# 3. Buscar
scores = bm25.get_scores(query.split())
```

**Pros:**
- ✅ Rápido
- ✅ Bueno para matches exactos
- ✅ Nombres propios, IDs

**Cons:**
- ❌ No entiende semántica
- ❌ Sinónimos son problema

### 3. Hybrid Search (Búsqueda Híbrida)

Combina dense + sparse.

```python
# 1. Búsqueda densa
dense_results = vector_db.search(query_emb, k=20)
dense_scores = {doc_id: score for doc_id, score in dense_results}

# 2. Búsqueda dispersa (BM25)
sparse_results = bm25_search(query, k=20)
sparse_scores = {doc_id: score for doc_id, score in sparse_results}

# 3. Combinar scores
final_scores = {}
for doc_id in set(dense_scores) | set(sparse_scores):
    dense = dense_scores.get(doc_id, 0)
    sparse = sparse_scores.get(doc_id, 0)
    
    # Weighted combination
    final_scores[doc_id] = alpha * dense + (1 - alpha) * sparse

# 4. Ordenar y retornar top-k
top_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:10]
```

**Ventajas:**
- ✅ Lo mejor de ambos mundos
- ✅ Robusto para diferentes tipos de queries

**Parámetro `alpha`:**
- `alpha = 1.0`: Solo dense (semántica)
- `alpha = 0.5`: Balance 50-50
- `alpha = 0.0`: Solo sparse (keywords)

### 4. MMR (Maximal Marginal Relevance)

Balancea relevancia y diversidad.

```python
def mmr(query_emb, doc_embeddings, lambda_param=0.5, k=10):
    selected = []
    candidates = list(range(len(doc_embeddings)))
    
    while len(selected) < k and candidates:
        mmr_scores = []
        
        for i in candidates:
            # Relevancia a query
            relevance = cosine_sim(query_emb, doc_embeddings[i])
            
            # Similitud a docs ya seleccionados
            if selected:
                max_sim = max([cosine_sim(doc_embeddings[i], doc_embeddings[j]) 
                              for j in selected])
            else:
                max_sim = 0
            
            # MMR = lambda * relevancia - (1-lambda) * redundancia
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append((i, mmr_score))
        
        # Seleccionar mejor
        best_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected.append(best_idx)
        candidates.remove(best_idx)
    
    return selected
```

**Cuándo usar:**
- Quieres resultados diversos (no todos iguales)
- Exploratory search
- Recomendaciones

---

## 🎖️ Reranking {#reranking}

### Concepto

**Two-Stage Retrieval:**

```
Stage 1: Retrieval Rápido (Bi-Encoder)
         ↓ 1000 documentos candidatos
Stage 2: Reranking Preciso (Cross-Encoder)
         ↓ Top 10 documentos finales
```

### Bi-Encoder vs Cross-Encoder

**Bi-Encoder:**
```python
# Procesa query y docs INDEPENDIENTEMENTE
query_emb = encoder(query)
doc_embs = [encoder(doc) for doc in docs]

# Compara embeddings
scores = [cosine_sim(query_emb, doc_emb) for doc_emb in doc_embs]
```

**Cross-Encoder:**
```python
# Procesa query Y doc JUNTOS
scores = [cross_encoder(query, doc) for doc in docs]
```

**Comparación:**

| Aspecto | Bi-Encoder | Cross-Encoder |
|---------|------------|---------------|
| **Velocidad** | ⚡⚡⚡⚡⚡ | ⚡⚡ |
| **Precisión** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Escalabilidad** | ✅ Millones | ❌ Solo top-K |
| **Uso** | Stage 1 | Stage 2 |

### Implementación

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

# Stage 1: Bi-encoder retrieval
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
query_emb = bi_encoder.encode(query)
doc_embs = bi_encoder.encode(documents)

# Buscar top 100
similarities = cosine_similarity([query_emb], doc_embs)[0]
top_100_indices = np.argsort(similarities)[-100:][::-1]
top_100_docs = [documents[i] for i in top_100_indices]

# Stage 2: Cross-encoder reranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
pairs = [[query, doc] for doc in top_100_docs]
scores = cross_encoder.predict(pairs)

# Final top 10
top_10_indices = np.argsort(scores)[-10:][::-1]
final_results = [top_100_docs[i] for i in top_10_indices]
```

### Modelos de Cross-Encoder

| Modelo | Tamaño | Velocidad | Calidad |
|--------|--------|-----------|---------|
| **ms-marco-TinyBERT-L-2-v2** | 17MB | ⚡⚡⚡⚡ | ⭐⭐⭐ |
| **ms-marco-MiniLM-L-6-v2** | 80MB | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| **ms-marco-MiniLM-L-12-v2** | 130MB | ⚡⚡ | ⭐⭐⭐⭐⭐ |

### Mejora de Performance

Reranking típicamente mejora:
- **NDCG@10**: +10-20%
- **MRR**: +15-25%
- **Precision@10**: +10-15%

---

## ⚡ Optimización y Escalado {#optimización}

### 1. Indexación

**HNSW (Hierarchical Navigable Small World)**

Algoritmo de grafos multi-capa.

```python
import faiss

# Crear índice HNSW
M = 32  # Número de conexiones por nodo
index = faiss.IndexHNSWFlat(dimension, M)

# Configurar
index.hnsw.efConstruction = 40  # Calidad de construcción
index.hnsw.efSearch = 16        # Calidad de búsqueda

# Añadir vectores
index.add(embeddings)
```

**Parámetros:**
- `M`: Más alto = más preciso pero más memoria
- `efConstruction`: Calidad al construir
- `efSearch`: Trade-off velocidad/precisión

**IVF (Inverted File Index)**

Particiona el espacio en clusters.

```python
# Cuantizador
quantizer = faiss.IndexFlatL2(dimension)

# IVF con nlist particiones
nlist = 100  # Número de clusters
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Entrenar
index.train(embeddings)
index.add(embeddings)

# Búsqueda
index.nprobe = 10  # Buscar en 10 clusters más cercanos
```

### 2. Compresión

**Product Quantization (PQ)**

Reduce tamaño de vectores.

```python
# Original: 768 floats × 4 bytes = 3KB por vector
# Con PQ: 96 bytes por vector (32x compresión)

m = 96  # Número de subvectores
nbits = 8  # Bits por subvector

index = faiss.IndexPQ(dimension, m, nbits)
index.train(embeddings)
index.add(embeddings)
```

**Trade-off:**
- ✅ 10-100x menos memoria
- ❌ Pérdida de precisión (~5-10%)

### 3. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text):
    return model.encode(text)

# Queries repetidas son instantáneas
emb1 = get_embedding("python")  # Calcula
emb2 = get_embedding("python")  # Cache hit!
```

### 4. Batch Processing

```python
# ❌ Lento: uno a la vez
for text in texts:
    embedding = model.encode(text)

# ✅ Rápido: en batch
embeddings = model.encode(texts, batch_size=32)
```

### 5. GPU Acceleration

```python
# FAISS con GPU
import faiss.contrib.torch_utils

# Mover índice a GPU
res = faiss.StandardGpuResources()
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)

# Búsquedas 10-100x más rápidas
```

### Benchmark: Escalabilidad

| Vectores | FAISS Flat | FAISS HNSW | Pinecone |
|----------|------------|------------|----------|
| **1K** | 10ms | 5ms | 20ms |
| **10K** | 100ms | 10ms | 25ms |
| **100K** | 1s | 15ms | 30ms |
| **1M** | 10s | 20ms | 35ms |
| **10M** | 100s | 30ms | 40ms |

---

## 💼 Casos de Uso {#casos-uso}

### 1. **Búsqueda de Documentos**

```python
# Empresa con base de conocimiento
docs = ["Manual de usuario...", "FAQ...", "Política..."]
doc_embeddings = model.encode(docs)

# Usuario busca
query = "¿Cómo cambiar mi contraseña?"
query_emb = model.encode(query)

# Encontrar documentos relevantes
similarities = cosine_similarity([query_emb], doc_embeddings)[0]
top_doc = docs[np.argmax(similarities)]
```

### 2. **Recomendaciones**

```python
# Usuario vio producto
product_embedding = get_embedding(product_description)

# Encontrar similares
similar_products = vector_db.search(product_embedding, k=10)
```

### 3. **Detección de Duplicados**

```python
# Comparar nuevo contenido con existente
new_article_emb = model.encode(new_article)
existing_embs = model.encode(existing_articles)

similarities = cosine_similarity([new_article_emb], existing_embs)[0]
if max(similarities) > 0.95:
    print("Posible duplicado detectado")
```

### 4. **Clustering Semántico**

```python
from sklearn.cluster import KMeans

# Agrupar documentos similares
embeddings = model.encode(documents)
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(embeddings)
```

### 5. **Q&A Systems**

```python
# Base de Q&A
questions = ["¿Qué es Python?", "¿Cómo instalo pip?", ...]
answers = ["Python es...", "Pip se instala con...", ...]
q_embeddings = model.encode(questions)

# Usuario pregunta
user_q = "¿Cómo usar pip?"
user_q_emb = model.encode(user_q)

# Encontrar pregunta más similar
similarities = cosine_similarity([user_q_emb], q_embeddings)[0]
best_match_idx = np.argmax(similarities)
answer = answers[best_match_idx]
```

---

## 📚 Recursos Adicionales

### Papers Importantes

- [Sentence-BERT](https://arxiv.org/abs/1908.10084)
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)

### Documentación

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [ChromaDB Docs](https://docs.trychroma.com/)

### Benchmarks

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Evalúa modelos de embeddings

---

## 🎓 Próximos Pasos

- **Koan 13: RAG** - Usa búsqueda semántica para RAG
- **LangChain Retrievers** - Integra con agentes
- **Fine-tuning Embeddings** - Personaliza para tu dominio

¡Domina la búsqueda semántica! 🚀
