# Hints para Koan 12: Semantic Search & Vector Databases

## Pista 1: create_openai_embedding()

<details>
<summary>Ver Pista Nivel 1</summary>

OpenAI ofrece modelos de embeddings:
- `text-embedding-3-small`: Rápido y económico
- `text-embedding-3-large`: Más preciso
- `text-embedding-ada-002`: Modelo anterior
- Requiere API key: `OPENAI_API_KEY`

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from openai import OpenAI

client = OpenAI()
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)
return response.data[0].embedding
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from openai import OpenAI
from typing import List

def create_openai_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    client = OpenAI()
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding
```

</details>

---

## Pista 2: create_sentence_transformer_embedding()

<details>
<summary>Ver Pista Nivel 1</summary>

Sentence Transformers: embeddings locales sin API:
- Instala: `sentence-transformers`
- Modelos populares:
  - `all-MiniLM-L6-v2`: Pequeño y rápido
  - `all-mpnet-base-v2`: Mejor calidad
  - Multilingüe: `paraphrase-multilingual-MiniLM-L12-v2`

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(text)
return embedding
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def create_sentence_transformer_embedding(text: str, model: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model)
    embedding = model.encode(text)
    return embedding
```

</details>

---

## Pista 3: cosine_similarity_search()

<details>
<summary>Ver Pista Nivel 1</summary>

Similitud coseno mide el ángulo entre vectores:
- Valores de -1 a 1 (embeddings suelen usar 0 a 1)
- Más cercano a 1 = más similar
- Formula: `dot(A, B) / (norm(A) * norm(B))`
- Usa numpy o scikit-learn

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

query = np.array(query_embedding).reshape(1, -1)
docs = np.array(document_embeddings)

similarities = cosine_similarity(query, docs)[0]
top_indices = np.argsort(similarities)[-top_k:][::-1]

return [(idx, similarities[idx]) for idx in top_indices]
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple

def cosine_similarity_search(query_embedding: List[float], 
                            document_embeddings: List[List[float]], 
                            top_k: int = 5) -> List[Tuple[int, float]]:
    query = np.array(query_embedding).reshape(1, -1)
    docs = np.array(document_embeddings)
    
    similarities = cosine_similarity(query, docs)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = [(int(idx), float(similarities[idx])) for idx in top_indices]
    return results
```

</details>

---

## Pista 4: create_chromadb_collection()

<details>
<summary>Ver Pista Nivel 1</summary>

ChromaDB: Vector database open-source y simple:
- Instala: `chromadb`
- No requiere servidor (modo in-memory)
- Soporta embeddings automáticos
- Usa `chromadb.Client()` para empezar

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection(name=collection_name)
return collection
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
import chromadb

def create_chromadb_collection(collection_name: str):
    client = chromadb.Client()
    collection = client.get_or_create_collection(name=collection_name)
    return collection
```

</details>

---

## Pista 5: add_documents_to_chromadb()

<details>
<summary>Ver Pista Nivel 1</summary>

Añadir documentos a ChromaDB:
- Usa `collection.add()`
- Parámetros: `documents`, `ids`, `metadatas` (opcional)
- ChromaDB genera embeddings automáticamente
- IDs deben ser únicos (usa str(i) o UUIDs)

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
ids = [str(i) for i in range(len(documents))]
collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from typing import List

def add_documents_to_chromadb(collection, documents: List[str], metadatas: List[dict] = None):
    ids = [str(i) for i in range(len(documents))]
    
    if metadatas:
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
    else:
        collection.add(
            documents=documents,
            ids=ids
        )
```

</details>

---

## Pista 6: search_chromadb()

<details>
<summary>Ver Pista Nivel 1</summary>

Búsqueda en ChromaDB:
- Usa `collection.query()`
- Parámetros: `query_texts` (lista), `n_results`
- Devuelve diccionario con: `documents`, `distances`, `metadatas`
- Distancias: menores = más similares

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
results = collection.query(
    query_texts=[query],
    n_results=top_k
)
return results
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from typing import List

def search_chromadb(collection, query: str, top_k: int = 5) -> List[dict]:
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results
```

</details>

---

## Pista 7: create_faiss_index()

<details>
<summary>Ver Pista Nivel 1</summary>

FAISS (Facebook AI Similarity Search):
- Instala: `faiss-cpu` o `faiss-gpu`
- Extremadamente rápido para millones de vectores
- `IndexFlatL2`: Búsqueda exacta (L2 distance)
- `IndexFlatIP`: Inner product (para cosine con vectores normalizados)

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
import faiss
import numpy as np

embeddings = np.array(embeddings).astype('float32')
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
return index
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
import faiss
import numpy as np

def create_faiss_index(embeddings: np.ndarray):
    embeddings = embeddings.astype('float32')
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
```

</details>

---

## Pista 8: semantic_search_with_reranking()

<details>
<summary>Ver Pista Nivel 1</summary>

Reranking mejora resultados de búsqueda:
1. Búsqueda inicial (retrieval rápido)
2. Rerank con modelo más preciso (cross-encoder)
3. Cross-encoder: calcula score para par (query, doc)

Modelos cross-encoder: `cross-encoder/ms-marco-MiniLM-L-12-v2`

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

# 1. Embedding inicial
model = SentenceTransformer('all-MiniLM-L6-v2')
query_emb = model.encode(query)
doc_embs = model.encode(documents)

# 2. Búsqueda inicial
similarities = cosine_similarity([query_emb], doc_embs)[0]
top_indices = np.argsort(similarities)[-top_k*2:][::-1]

# 3. Rerank
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
pairs = [(query, documents[i]) for i in top_indices]
scores = cross_encoder.predict(pairs)

# 4. Re-sort
final_indices = np.argsort(scores)[-top_k:][::-1]
return [(documents[top_indices[i]], scores[i]) for i in final_indices]
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple

def semantic_search_with_reranking(query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    # 1. Initial retrieval
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = model.encode(query)
    doc_embs = model.encode(documents)
    
    similarities = cosine_similarity([query_emb], doc_embs)[0]
    top_indices = np.argsort(similarities)[-(top_k*2):][::-1]
    
    # 2. Reranking
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    pairs = [(query, documents[i]) for i in top_indices]
    scores = cross_encoder.predict(pairs)
    
    # 3. Final sorting
    final_indices = np.argsort(scores)[-top_k:][::-1]
    results = [(documents[top_indices[i]], float(scores[i])) for i in final_indices]
    
    return results
```

</details>

---

## Conceptos Clave

### Embeddings
Representación numérica (vectores) del significado semántico de texto:
- Textos similares → vectores cercanos
- Dimensiones típicas: 384, 768, 1536
- Tipos: palabra, oración, documento

### Vector Databases
Bases de datos optimizadas para búsqueda vectorial:

| Database | Pros | Cons |
|----------|------|------|
| ChromaDB | Simple, local | Escala limitada |
| FAISS | Muy rápido | Solo in-memory |
| Pinecone | Cloud, escalable | De pago |
| Weaviate | Features avanzados | Setup complejo |

### Métricas de Similitud
- **Cosine Similarity**: Ángulo entre vectores (-1 a 1)
- **Euclidean Distance**: Distancia L2
- **Dot Product**: Producto punto (requiere normalización)

### Estrategias de Búsqueda

1. **Semantic Search**: Solo embeddings
2. **Keyword Search**: BM25, TF-IDF
3. **Hybrid Search**: Combina ambos
4. **Reranking**: Dos fases (rápido + preciso)

### Pipeline de Búsqueda Semántica

```
Query → Embedding → Vector DB → Top-K Results → (Optional Reranking) → Final Results
```

### Cross-Encoder vs Bi-Encoder

**Bi-Encoder** (Sentence Transformers):
- Codifica query y docs independientemente
- Rápido (puede pre-calcular embeddings)
- Menos preciso

**Cross-Encoder**:
- Codifica par (query, doc) juntos
- Lento (debe procesar cada par)
- Más preciso

→ Solución: Bi-encoder para retrieval, Cross-encoder para reranking

### Mejores Prácticas
- Normaliza embeddings para cosine similarity
- Usa batch processing para múltiples documentos
- Pre-calcula embeddings de documentos
- Considera índices aproximados (ANN) para escala
- Evalúa con métricas: Precision@K, Recall@K, MRR

### Modelos Recomendados

**Inglés**:
- `all-MiniLM-L6-v2`: Rápido, balanceado
- `all-mpnet-base-v2`: Mejor calidad

**Multilingüe**:
- `paraphrase-multilingual-MiniLM-L12-v2`
- `sentence-transformers/LaBSE`

**Cross-Encoders**:
- `cross-encoder/ms-marco-MiniLM-L-12-v2`
- `cross-encoder/ms-marco-TinyBERT-L-2-v2` (más rápido)

## Recursos
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [FAISS Tutorial](https://github.com/facebookresearch/faiss/wiki)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
