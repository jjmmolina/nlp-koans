# Teoría: RAG (Retrieval-Augmented Generation)

## 📚 Tabla de Contenidos
1. [Introducción a RAG](#introducción)
2. [Arquitectura de RAG](#arquitectura)
3. [Document Processing & Chunking](#chunking)
4. [Retrieval Strategies](#retrieval)
5. [Generation & Prompting](#generation)
6. [Advanced RAG Patterns](#advanced)
7. [Evaluation](#evaluation)
8. [Production Best Practices](#production)

---

## 🎯 Introducción a RAG {#introducción}

### ¿Qué es RAG?

**RAG (Retrieval-Augmented Generation)** combina búsqueda de información con generación de lenguaje para crear respuestas más precisas y actualizadas.

```
Sin RAG:
Usuario: "¿Cuáles son las últimas features de Python 3.12?"
LLM: "No tengo información sobre Python 3.12..." ❌

Con RAG:
Usuario: "¿Cuáles son las últimas features de Python 3.12?"
Sistema:
  1. 🔍 Busca en documentación actualizada
  2. 📄 Encuentra: "Python 3.12 añade type parameter syntax..."
  3. 🤖 LLM genera respuesta basada en docs encontrados
LLM: "Python 3.12 introduce las siguientes features:
      - Type parameter syntax (PEP 695)
      - Improved error messages..." ✅
```

### Problema que Resuelve

**Limitaciones de LLMs Puros:**

1. **Knowledge Cutoff**
```python
# LLM entrenado hasta enero 2023
pregunta = "¿Qué empresas compraron startups de IA en marzo 2024?"
respuesta = llm(pregunta)
# ❌ "No tengo información más allá de enero 2023"
```

2. **Alucinaciones**
```python
pregunta = "¿Cuál es el teléfono de nuestra oficina en Madrid?"
respuesta = llm(pregunta)
# ❌ "+34 912 345 678" (inventado)
```

3. **Información Privada**
```python
pregunta = "¿Cuál es la política de vacaciones de nuestra empresa?"
respuesta = llm(pregunta)
# ❌ No tiene acceso a documentos internos
```

**Solución con RAG:**

```python
pregunta = "¿Cuál es la política de vacaciones?"

# 1. Recuperar documentos relevantes
docs = retriever.search(pregunta)
# docs = ["Manual de empleados: Las vacaciones son 22 días..."]

# 2. Generar respuesta basada en docs
respuesta = llm(f"Basándote en: {docs}\nResponde: {pregunta}")
# ✅ "Según el manual de empleados, tienes 22 días de vacaciones..."
```

### Ventajas de RAG

| Ventaja | Descripción | Ejemplo |
|---------|-------------|---------|
| **🔄 Actualizable** | Información siempre fresca | Añadir docs nuevos sin reentrenar |
| **📚 Conocimiento Específico** | Dominio especializado | Manuales técnicos, políticas internas |
| **🎯 Reducción de Alucinaciones** | Respuestas basadas en hechos | Citas y referencias verificables |
| **💰 Cost-Effective** | No requiere fine-tuning | Actualizar base de datos vs reentrenar |
| **🔍 Trazabilidad** | Citar fuentes | "Según documento X, página Y..." |

### RAG vs Alternativas

**RAG vs Fine-Tuning:**

| Aspecto | RAG | Fine-Tuning |
|---------|-----|-------------|
| **Costo** | $ | $$$ |
| **Actualización** | Instantánea | Requiere reentrenamiento |
| **Conocimiento Específico** | ✅ Excelente | ✅ Excelente |
| **Flexibilidad** | ✅ Alta | ❌ Baja |
| **Trazabilidad** | ✅ Citas claras | ❌ Black box |

**RAG vs Prompt Engineering:**

| Aspecto | RAG | Prompt Engineering |
|---------|-----|-------------------|
| **Límite de Contexto** | ✅ Ilimitado (retrieval) | ❌ Limitado (tokens) |
| **Precisión** | ✅ Alta | ⚠️ Media |
| **Costo por Query** | $ | $$ |

---

## 🏗️ Arquitectura de RAG {#arquitectura}

### Pipeline Básico

```
┌──────────────┐
│   Usuario    │
│   Pregunta   │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ 1. INGESTION (Offline)                  │
│                                          │
│  Documentos → Chunks → Embeddings       │
│                            ↓             │
│                    ┌──────────────┐     │
│                    │ Vector Store │     │
│                    └──────────────┘     │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ 2. RETRIEVAL (Query time)               │
│                                          │
│  Query → Embedding → Search Vector DB   │
│                            ↓             │
│                  Top K Documents         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 3. GENERATION                           │
│                                          │
│  Prompt Template:                       │
│  "Contexto: {docs}"                     │
│  "Pregunta: {query}"                    │
│             ↓                            │
│           LLM                            │
│             ↓                            │
│        Respuesta                         │
└─────────────────────────────────────────┘
```

### Componentes Clave

**1. Document Loaders**
```python
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredHTMLLoader
)

# Cargar diferentes formatos
loader = PyPDFLoader("manual.pdf")
documents = loader.load()
```

**2. Text Splitters**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)
```

**3. Embeddings**
```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
```

**4. Vector Store**
```python
from langchain.vectorstores import Chroma

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)
```

**5. Retriever**
```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

**6. LLM**
```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)
```

**7. Chain**
```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
```

---

## 📄 Document Processing & Chunking {#chunking}

### ¿Por qué Chunking?

**Problema:** Documentos largos no caben en contexto del LLM.

```python
# Contexto típico de LLM: 4k-128k tokens
# Documento: 1 millón de tokens ❌
```

**Solución:** Dividir en chunks manejables.

```python
# Documento → [Chunk1, Chunk2, ..., ChunkN]
# Cada chunk: 500-1500 tokens ✅
```

### Estrategias de Chunking

#### 1. **Fixed Size (Tamaño Fijo)**

```python
def fixed_size_chunks(text, chunk_size=1000):
    return [text[i:i+chunk_size] 
            for i in range(0, len(text), chunk_size)]
```

**Pros:**
- ✅ Simple
- ✅ Predecible

**Cons:**
- ❌ Rompe en medio de oraciones
- ❌ Pierde contexto

#### 2. **Fixed Size con Overlap**

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200  # 20% overlap
)
```

**Ventaja del Overlap:**
```
Chunk 1: "...al final de este párrafo hablamos de Python..."
Chunk 2: "...Python es un lenguaje..." ← Contexto mantenido
```

**Recomendación:**
- Overlap: 10-20% del chunk size
- Chunk size: 500-1500 tokens

#### 3. **Semantic Chunking (por Significado)**

Divide por cambios de tema o estructura.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Prioriza splits por:
# 1. Párrafos (\n\n)
# 2. Líneas (\n)
# 3. Oraciones (.)
# 4. Palabras ( )

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

#### 4. **Document-Aware Chunking**

Respeta estructura del documento.

```python
# Markdown
from langchain.text_splitter import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
)

# HTML
from langchain.text_splitter import HTMLHeaderTextSplitter

splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "Header 1"),
        ("h2", "Header 2"),
    ]
)
```

#### 5. **Sentence-Based Chunking**

```python
import nltk

def sentence_chunks(text, max_sentences=5):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i+max_sentences])
        chunks.append(chunk)
    
    return chunks
```

### Metadata Enrichment

Añadir metadatos a chunks para mejor retrieval.

```python
from langchain.schema import Document

chunks = []
for i, text in enumerate(split_texts):
    chunk = Document(
        page_content=text,
        metadata={
            "source": "manual.pdf",
            "page": page_num,
            "chunk_id": i,
            "section": section_name,
            "author": author,
            "date": "2024-01-15"
        }
    )
    chunks.append(chunk)
```

**Beneficios:**
- Filtrado preciso
- Citas con contexto
- Auditabilidad

### Chunking Best Practices

| Factor | Recomendación | Razón |
|--------|---------------|-------|
| **Chunk Size** | 500-1500 tokens | Balance contexto/precisión |
| **Overlap** | 10-20% | Mantiene continuidad |
| **Método** | RecursiveCharacter | Respeta estructura |
| **Metadata** | Siempre incluir | Trazabilidad y filtrado |

---

## 🔍 Retrieval Strategies {#retrieval}

### 1. **Basic Similarity Search**

Búsqueda por similitud coseno.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

docs = retriever.get_relevant_documents("¿Qué es Python?")
```

**Pros:**
- ✅ Simple y rápido
- ✅ Funciona bien en la mayoría de casos

**Cons:**
- ❌ Solo considera similitud semántica
- ❌ Puede recuperar documentos redundantes

### 2. **MMR (Maximal Marginal Relevance)**

Balancea relevancia y diversidad.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20,  # Candidatos iniciales
        "lambda_mult": 0.5  # Balance relevancia/diversidad
    }
)
```

**Parámetros:**
- `fetch_k`: Documentos candidatos
- `lambda_mult`:
  - `1.0`: Solo relevancia (como similarity)
  - `0.5`: Balance
  - `0.0`: Máxima diversidad

**Ventaja:**
```
Similarity: ["Intro Python", "Python basics", "Python tutorial"]
            ↑ Todos muy similares entre sí

MMR: ["Intro Python", "Python advanced", "Python vs Java"]
     ↑ Relevantes pero diversos
```

### 3. **Similarity Score Threshold**

Solo documentos con score mínimo.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.8,  # Mínimo 0.8 similitud
        "k": 4
    }
)
```

**Ventaja:**
- ✅ Filtra resultados irrelevantes
- ✅ Respuestas más precisas

**Precaución:**
- ⚠️ Puede retornar 0 documentos si threshold muy alto

### 4. **Metadata Filtering**

Filtrar por metadatos antes de buscar.

```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {
            "source": "manual_2024.pdf",
            "section": "installation"
        }
    }
)
```

**Casos de uso:**
```python
# Por fecha
filter={"date": {"$gte": "2024-01-01"}}

# Por categoría
filter={"category": {"$in": ["python", "programming"]}}

# Combinado
filter={
    "author": "John Doe",
    "verified": True,
    "date": {"$gte": "2023-01-01"}
}
```

### 5. **Compression (Contextual Compression)**

Comprime documentos recuperados para extraer partes relevantes.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Compressor (extrae partes relevantes con LLM)
compressor = LLMChainExtractor.from_llm(llm)

# Compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)
```

**Ventaja:**
```
Antes: [Doc completo 2000 tokens]
Después: [Solo párrafo relevante 200 tokens]
```

**Beneficio:**
- ✅ Más contexto útil en menos tokens
- ✅ Respuestas más precisas

### 6. **Multi-Query Retrieval**

Genera múltiples variantes de la query.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# Usuario: "¿Cómo instalar Python?"
# Sistema genera:
#   - "¿Cómo instalar Python?"
#   - "Pasos para instalación de Python"
#   - "Guía de instalación Python"
#   - "Configurar Python en mi sistema"
```

**Ventaja:**
- ✅ Captura diferentes formulaciones
- ✅ Más robusto a queries ambiguas

### 7. **Ensemble Retriever**

Combina múltiples retrievers.

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

# Dense retriever (embeddings)
dense_retriever = vectorstore.as_retriever()

# Sparse retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(documents)

# Combinar
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # 50% cada uno
)
```

**Ventaja:**
- ✅ Mejor de ambos mundos (semantic + keyword)

---

## 🤖 Generation & Prompting {#generation}

### Prompt Templates para RAG

#### Template Básico

```python
from langchain.prompts import PromptTemplate

template = """Usa el siguiente contexto para responder la pregunta.
Si no sabes la respuesta, di que no lo sabes.

Contexto: {context}

Pregunta: {question}

Respuesta:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)
```

#### Template con Instrucciones Específicas

```python
template = """Eres un asistente experto en Python. Usa el contexto proporcionado 
para responder la pregunta del usuario.

REGLAS:
1. Solo usa información del contexto
2. Si la respuesta no está en el contexto, di "No tengo esa información"
3. Incluye ejemplos de código cuando sea relevante
4. Cita la fuente del contexto

Contexto:
{context}

Pregunta: {question}

Respuesta (siguiendo las reglas):"""
```

#### Template con Few-Shot Examples

```python
template = """Responde basándote en el contexto proporcionado.

EJEMPLOS:
Pregunta: ¿Qué es Python?
Respuesta: Según el documento "Intro.pdf", Python es un lenguaje...

Pregunta: ¿Cuánto cuesta la licencia?
Respuesta: No encuentro información sobre precios en los documentos proporcionados.

---

Contexto actual:
{context}

Pregunta: {question}

Respuesta:"""
```

### Chain Types

#### 1. **Stuff Chain**

Concatena todos los docs en un solo prompt.

```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)
```

**Ventajas:**
- ✅ Simple
- ✅ Una sola llamada al LLM
- ✅ Respuesta coherente

**Desventajas:**
- ❌ Limitado por contexto del LLM
- ❌ No escala con muchos docs

**Cuándo usar:**
- Pocos documentos (< 4)
- Documentos pequeños

#### 2. **Map-Reduce Chain**

Procesa cada doc independientemente, luego combina.

```python
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=retriever
)
```

**Proceso:**
```
Doc1 → LLM → Summary1 ┐
Doc2 → LLM → Summary2 ├→ Combinar → Respuesta Final
Doc3 → LLM → Summary3 ┘
```

**Ventajas:**
- ✅ Escala a muchos docs
- ✅ Paralelizable

**Desventajas:**
- ❌ Múltiples llamadas al LLM (costoso)
- ❌ Puede perder contexto entre docs

#### 3. **Refine Chain**

Refina la respuesta iterativamente.

```python
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=retriever
)
```

**Proceso:**
```
Doc1 → LLM → Answer1
Answer1 + Doc2 → LLM → Answer2
Answer2 + Doc3 → LLM → Answer3 (final)
```

**Ventajas:**
- ✅ Respuesta mejorada iterativamente
- ✅ Mantiene contexto

**Desventajas:**
- ❌ Secuencial (no paralelizable)
- ❌ Costoso (N llamadas)

#### 4. **Map-Rerank Chain**

LLM asigna score de confianza a cada respuesta.

```python
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_rerank",
    retriever=retriever
)
```

**Proceso:**
```
Doc1 → LLM → (Answer1, Score: 0.8)
Doc2 → LLM → (Answer2, Score: 0.95) ← Seleccionada
Doc3 → LLM → (Answer3, Score: 0.6)
```

**Ventaja:**
- ✅ Selecciona mejor respuesta automáticamente

### Citations (Citas)

Incluir fuentes en la respuesta.

```python
template = """Usa el contexto para responder. SIEMPRE cita la fuente.

Contexto:
{context}

Pregunta: {question}

Respuesta con formato:
[Tu respuesta]

Fuentes:
- Documento X, página Y
"""

# O usando return_source_documents
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

result = qa({"query": "¿Qué es Python?"})
answer = result["result"]
sources = result["source_documents"]
```

---

## 🚀 Advanced RAG Patterns {#advanced}

### 1. **Multi-Query RAG**

Ya visto en retrieval. Genera múltiples queries.

### 2. **RAG Fusion**

Combina resultados de múltiples queries con Reciprocal Rank Fusion.

```python
def reciprocal_rank_fusion(results_lists, k=60):
    """
    results_lists: [[doc1, doc2, ...], [doc3, doc1, ...], ...]
    """
    doc_scores = {}
    
    for results in results_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.metadata.get("id", str(doc))
            
            # RRF score
            score = 1 / (k + rank + 1)
            
            if doc_id in doc_scores:
                doc_scores[doc_id] += score
            else:
                doc_scores[doc_id] = score
    
    # Ordenar por score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs
```

**Ventaja:**
- ✅ Más robusto que single query
- ✅ Combina perspectivas diferentes

### 3. **Conversational RAG**

Mantiene historial de conversación.

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Primera pregunta
response1 = qa({"question": "¿Qué es Python?"})

# Pregunta de seguimiento (usa contexto)
response2 = qa({"question": "¿Cuáles son sus ventajas?"})
# Entiende que "sus" se refiere a Python
```

**Técnicas:**

**Query Rewriting:**
```python
# Usuario: "¿Y sus desventajas?" (ambiguo)
# Sistema reescribe: "¿Cuáles son las desventajas de Python?"
```

**Condensed Question:**
```python
# Historial:
# User: "¿Qué es Python?"
# AI: "Python es un lenguaje..."
# User: "Dame ejemplos"

# Sistema condensa:
# "Dame ejemplos de Python" o "Dame ejemplos de código Python"
```

### 4. **Self-Query Retriever**

LLM extrae filtros de la query en lenguaje natural.

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="source",
        description="El origen del documento",
        type="string"
    ),
    AttributeInfo(
        name="date",
        description="Fecha de publicación",
        type="date"
    ),
]

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Documentación de Python",
    metadata_field_info=metadata_field_info
)

# Query: "documentos sobre funciones publicados después de 2023"
# Sistema extrae:
#   - Query semántica: "funciones"
#   - Filtro: date > 2023-01-01
```

### 5. **Parent Document Retriever**

Busca en chunks pequeños, retorna documentos completos.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Store para documentos completos
docstore = InMemoryStore()

# Retriever que busca chunks pero retorna parents
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,  # Chunks pequeños
    parent_splitter=parent_splitter,  # Docs completos (opcional)
)
```

**Ventaja:**
```
Búsqueda: chunk de 200 tokens (preciso) ✅
Contexto al LLM: documento completo de 2000 tokens (contexto rico) ✅
```

### 6. **Hypothetical Document Embeddings (HyDE)**

LLM genera documento hipotético, se usa para búsqueda.

```python
# 1. Usuario pregunta
query = "¿Cómo optimizar queries SQL?"

# 2. LLM genera respuesta hipotética (sin contexto)
hypothetical_doc = llm.predict(
    f"Escribe un artículo que responda: {query}"
)

# 3. Embedding del documento hipotético
hyde_embedding = embeddings.embed_query(hypothetical_doc)

# 4. Buscar con ese embedding
docs = vectorstore.similarity_search_by_vector(hyde_embedding)
```

**Ventaja:**
- ✅ Mejor que buscar con query corta
- ✅ Captura intención más completa

---

## 📊 Evaluation {#evaluation}

### Métricas Clave

#### 1. **Retrieval Metrics**

**Precision@K:**
```python
def precision_at_k(relevant_docs, retrieved_docs, k):
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = len(set(relevant_docs) & set(retrieved_k))
    return relevant_retrieved / k
```

**Recall@K:**
```python
def recall_at_k(relevant_docs, retrieved_docs, k):
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = len(set(relevant_docs) & set(retrieved_k))
    return relevant_retrieved / len(relevant_docs)
```

**MRR (Mean Reciprocal Rank):**
```python
def mrr(relevant_docs, retrieved_docs):
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1 / (i + 1)
    return 0
```

**NDCG (Normalized Discounted Cumulative Gain):**
```python
from sklearn.metrics import ndcg_score

# relevance_scores: [3, 2, 3, 0, 1] (graded relevance)
# retrieved_scores: scores de retrieval
ndcg = ndcg_score([relevance_scores], [retrieved_scores])
```

#### 2. **Generation Metrics**

**Faithfulness:**
¿La respuesta está basada en el contexto?

```python
from ragas.metrics import faithfulness

score = faithfulness.score(
    question=question,
    answer=answer,
    contexts=[doc.page_content for doc in retrieved_docs]
)
```

**Answer Relevancy:**
¿La respuesta responde la pregunta?

```python
from ragas.metrics import answer_relevancy

score = answer_relevancy.score(
    question=question,
    answer=answer
)
```

**Context Relevancy:**
¿El contexto recuperado es relevante?

```python
from ragas.metrics import context_relevancy

score = context_relevancy.score(
    question=question,
    contexts=[doc.page_content for doc in retrieved_docs]
)
```

#### 3. **End-to-End Metrics**

**Correctness:**
Comparación con ground truth.

```python
from ragas.metrics import answer_correctness

score = answer_correctness.score(
    question=question,
    answer=answer,
    ground_truth=expected_answer
)
```

### Frameworks de Evaluación

#### RAGAS (RAG Assessment)

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# Dataset de evaluación
eval_dataset = {
    "question": [...],
    "answer": [...],
    "contexts": [...],
    "ground_truths": [...]
}

# Evaluar
result = evaluate(
    eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ],
)

print(result)
# {
#   "faithfulness": 0.92,
#   "answer_relevancy": 0.88,
#   "context_recall": 0.85,
#   "context_precision": 0.90
# }
```

#### TruLens

```python
from trulens_eval import TruChain, Feedback, Tru

# Crear feedback functions
f_qa_relevance = Feedback(relevance_scorer).on_input_output()
f_groundedness = Feedback(groundedness_scorer).on(contexts).on_output()

# Wrap chain
tru_chain = TruChain(
    qa_chain,
    app_id="my_rag_app",
    feedbacks=[f_qa_relevance, f_groundedness]
)

# Usar normalmente
result = tru_chain.query("¿Qué es Python?")

# Ver dashboard
tru = Tru()
tru.run_dashboard()
```

### A/B Testing

```python
def compare_rag_systems(systemA, systemB, test_queries):
    results = {"A": [], "B": []}
    
    for query in test_queries:
        # Sistema A
        responseA = systemA.query(query)
        scoreA = evaluate_response(query, responseA)
        results["A"].append(scoreA)
        
        # Sistema B
        responseB = systemB.query(query)
        scoreB = evaluate_response(query, responseB)
        results["B"].append(scoreB)
    
    # Comparar
    import numpy as np
    print(f"A: {np.mean(results['A']):.2f}")
    print(f"B: {np.mean(results['B']):.2f}")
```

---

## 🏭 Production Best Practices {#production}

### 1. **Chunking Strategy**

```python
# ✅ Recomendado para producción
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

**Experimentar con:**
- Chunk sizes: 500, 1000, 1500 tokens
- Overlaps: 10%, 20%, 30%
- Medir impacto en métricas

### 2. **Indexing Pipeline**

```python
# Pipeline robusto con error handling
def index_documents(documents):
    try:
        # 1. Validar documentos
        validated = validate_documents(documents)
        
        # 2. Chunking
        chunks = splitter.split_documents(validated)
        
        # 3. Enriquecer metadata
        enriched = enrich_metadata(chunks)
        
        # 4. Generar embeddings (batch)
        embeddings = embedding_model.embed_documents(
            [c.page_content for c in enriched],
            batch_size=32
        )
        
        # 5. Añadir a vector store
        vectorstore.add_documents(enriched)
        
        # 6. Log success
        logger.info(f"Indexed {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise
```

### 3. **Caching**

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_embedding(text):
    return embedding_model.embed_query(text)

# Cache de respuestas completas
class RAGCache:
    def __init__(self):
        self.cache = {}
    
    def get(self, query):
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return self.cache.get(query_hash)
    
    def set(self, query, response):
        query_hash = hashlib.md5(query.encode()).hexdigest()
        self.cache[query_hash] = response
```

### 4. **Monitoring**

```python
import logging
from datetime import datetime

class RAGMonitor:
    def log_query(self, query, response, latency, docs_retrieved):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response_length": len(response),
            "latency_ms": latency,
            "docs_retrieved": len(docs_retrieved),
            "avg_doc_score": np.mean([d.score for d in docs_retrieved])
        }
        
        logging.info(log_entry)
        
        # Alertas
        if latency > 5000:  # > 5 segundos
            self.alert("High latency detected")
        
        if len(docs_retrieved) == 0:
            self.alert("No documents retrieved")
```

### 5. **Rate Limiting**

```python
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=50, period=60)  # 50 llamadas por minuto
def query_rag_system(query):
    return rag_chain.run(query)
```

### 6. **Error Handling**

```python
def robust_rag_query(query, max_retries=3):
    for attempt in range(max_retries):
        try:
            # Validar query
            if not query or len(query) > 1000:
                raise ValueError("Invalid query")
            
            # Retrieval
            docs = retriever.get_relevant_documents(query)
            
            if not docs:
                return "No encontré información relevante."
            
            # Generation
            response = llm_chain.run(
                context=docs,
                question=query
            )
            
            return response
            
        except RateLimitError:
            if attempt < max_retries - 1:
                sleep(2 ** attempt)  # Exponential backoff
            else:
                return "Sistema temporalmente ocupado. Intenta más tarde."
        
        except Exception as e:
            logger.error(f"Error: {e}")
            return "Ocurrió un error. Por favor intenta de nuevo."
```

### 7. **Incremental Updates**

```python
# Añadir nuevos documentos sin reindexar todo
def add_new_documents(new_docs):
    # 1. Procesar
    chunks = splitter.split_documents(new_docs)
    
    # 2. Check duplicados
    existing_ids = vectorstore.get_all_ids()
    chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]
    
    # 3. Añadir
    if chunks:
        vectorstore.add_documents(chunks)
        logger.info(f"Added {len(chunks)} new chunks")
```

### 8. **Vector Store Optimization**

```python
# Para FAISS
import faiss

# Usar GPU si disponible
if faiss.get_num_gpus() > 0:
    index = faiss.index_cpu_to_gpu(
        faiss.StandardGpuResources(),
        0,
        index
    )

# Product Quantization para reducir memoria
index = faiss.IndexIVFPQ(
    quantizer,
    dimension,
    nlist=100,
    M=8,
    nbits=8
)
```

### 9. **Testing**

```python
import pytest

def test_rag_pipeline():
    # Test retrieval
    docs = retriever.get_relevant_documents("test query")
    assert len(docs) > 0
    assert all(hasattr(d, "page_content") for d in docs)
    
    # Test generation
    response = qa_chain.run("test query")
    assert len(response) > 0
    assert "no sé" not in response.lower() or len(docs) == 0

def test_edge_cases():
    # Query vacía
    response = rag_system.query("")
    assert "invalid" in response.lower()
    
    # Query muy larga
    long_query = "test " * 1000
    response = rag_system.query(long_query)
    assert response is not None
```

### 10. **Deployment Checklist**

- [ ] Embeddings cacheados
- [ ] Vector store optimizado
- [ ] Rate limiting configurado
- [ ] Monitoring activo
- [ ] Error handling robusto
- [ ] Tests automatizados
- [ ] Documentación completa
- [ ] Backup de vector store
- [ ] Plan de escalado
- [ ] Security (API keys, authentication)

---

## 🎯 Casos de Uso Reales

### 1. **Customer Support**

```python
# Base de conocimiento de FAQs
qa_system = RAGSystem(
    documents=load_faq_documents(),
    chunk_size=500,
    retriever_k=3
)

# Usuario pregunta
answer = qa_system.query("¿Cómo reseteo mi contraseña?")
```

### 2. **Code Documentation Assistant**

```python
# Indexar código + docs
code_qa = RAGSystem(
    documents=load_code_and_docs(),
    chunk_size=1500,  # Mayor para código
    metadata_filters={"type": "code"}
)

# Developer pregunta
answer = code_qa.query("¿Cómo usar la API de autenticación?")
```

### 3. **Legal Document Analysis**

```python
# Documentos legales
legal_qa = RAGSystem(
    documents=load_legal_docs(),
    chunk_size=1000,
    return_source_documents=True  # Citas esenciales
)

# Abogado busca precedentes
result = legal_qa.query("Casos similares a fraude corporativo 2020-2024")
answer = result["answer"]
sources = result["sources"]  # Para verificación
```

### 4. **Research Assistant**

```python
# Papers científicos
research_qa = ConversationalRAG(
    documents=load_research_papers(),
    memory=ConversationBufferMemory()
)

# Investigador hace preguntas iterativas
research_qa.query("¿Qué dice la literatura sobre transformers?")
research_qa.query("¿Y sobre sus limitaciones?")
research_qa.query("Dame papers específicos")
```

---

## 📚 Recursos Adicionales

### Papers Fundamentales

- [RAG (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) - Paper original
- [Self-RAG (Asai et al., 2023)](https://arxiv.org/abs/2310.11511)
- [Corrective RAG (Yan et al., 2024)](https://arxiv.org/abs/2401.15884)

### Frameworks

- [LangChain](https://python.langchain.com/) - Framework completo para RAG
- [LlamaIndex](https://www.llamaindex.ai/) - Alternativa especializada en RAG
- [Haystack](https://haystack.deepset.ai/) - Pipeline de NLP

### Herramientas

- [RAGAS](https://github.com/explodinggradients/ragas) - Evaluación de RAG
- [TruLens](https://www.trulens.org/) - Monitoring y debugging
- [Weights & Biases](https://wandb.ai/) - Experiment tracking

---

## 🎓 Próximos Pasos

Después de dominar RAG:

1. **Advanced RAG Architectures**
   - Self-RAG con reflexión
   - Corrective RAG
   - Adaptive RAG

2. **Multi-Modal RAG**
   - Texto + imágenes
   - Texto + tablas
   - Texto + código

3. **Agent-Based RAG**
   - Agentes que deciden cuándo usar RAG
   - RAG como tool en sistemas multi-agente

4. **Fine-Tuning para RAG**
   - Embeddings específicos de dominio
   - LLMs optimizados para RAG

¡Domina RAG y construye aplicaciones inteligentes! 🚀
