# Hints para Koan 13: RAG (Retrieval-Augmented Generation)

## Pista 1: chunk_documents()

<details>
<summary>Ver Pista Nivel 1</summary>

Text splitting para RAG:
- Documentos largos → chunks pequeños
- `RecursiveCharacterTextSplitter` de LangChain
- Parámetros importantes: `chunk_size`, `chunk_overlap`
- Overlap ayuda a mantener contexto entre chunks

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=overlap
)
chunks = splitter.split_text(document)
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def chunk_documents(documents: List[str], chunk_size: int = 500, overlap: int = 50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc)
        all_chunks.extend(chunks)
    
    return all_chunks
```

</details>

---

## Pista 2: create_vector_store()

<details>
<summary>Ver Pista Nivel 1</summary>

Vector store con LangChain:
- `Chroma` para ChromaDB
- Necesita embeddings (OpenAI o HuggingFace)
- `from_texts()` crea store desde lista de strings
- Requiere `OPENAI_API_KEY` para OpenAI embeddings

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_texts(documents, embeddings)
return vector_store
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List

def create_vector_store(documents: List[str]):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_texts(
        texts=documents,
        embedding=embeddings
    )
    return vector_store
```

</details>

---

## Pista 3: create_retriever()

<details>
<summary>Ver Pista Nivel 1</summary>

Retriever para búsqueda:
- `vector_store.as_retriever()`
- Tipos de búsqueda: `similarity`, `mmr`, `similarity_score_threshold`
- `k`: número de documentos a recuperar
- MMR (Maximal Marginal Relevance) reduce redundancia

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": k}
)
return retriever
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
def create_retriever(vector_store, search_type: str = "similarity", k: int = 4):
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )
    return retriever
```

</details>

---

## Pista 4: basic_rag_chain()

<details>
<summary>Ver Pista Nivel 1</summary>

RAG Chain básico:
- `RetrievalQA` de LangChain
- Combina: retriever + LLM
- `chain_type="stuff"`: Concatena todos los docs
- Otros tipos: `map_reduce`, `refine`, `map_rerank`

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)
return chain
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from langchain.chains import RetrievalQA

def basic_rag_chain(retriever, llm):
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return chain
```

</details>

---

## Pista 5: rag_with_citations()

<details>
<summary>Ver Pista Nivel 1</summary>

RAG con referencias:
- Usa `return_source_documents=True`
- Resultado incluye: `result`, `source_documents`
- Extrae metadata de source_documents
- Formatea respuesta con fuentes

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

result = chain({"query": query})
answer = result["result"]
sources = [doc.page_content for doc in result["source_documents"]]

return {"answer": answer, "sources": sources}
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from langchain.chains import RetrievalQA
from typing import Dict

def rag_with_citations(query: str, retriever, llm) -> Dict[str, any]:
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    result = chain({"query": query})
    
    return {
        "answer": result["result"],
        "sources": [doc.page_content for doc in result["source_documents"]]
    }
```

</details>

---

## Pista 6: multi_query_rag()

<details>
<summary>Ver Pista Nivel 1</summary>

Multi-query RAG mejora recall:
- `MultiQueryRetriever` genera múltiples versiones de la query
- Busca con cada versión
- Combina resultados (deduplica)
- Mejor cobertura de información relevante

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langchain.retrievers import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

docs = multi_retriever.get_relevant_documents(query)
# Usa docs para generar respuesta con LLM
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import RetrievalQA

def multi_query_rag(query: str, retriever, llm, num_queries: int = 3) -> str:
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=multi_retriever
    )
    
    result = chain({"query": query})
    return result["result"]
```

</details>

---

## Pista 7: rag_fusion()

<details>
<summary>Ver Pista Nivel 1</summary>

RAG Fusion combina múltiples estrategias:
- Usa diferentes retrievers (similarity, MMR, etc.)
- Reciprocal Rank Fusion (RRF) para combinar scores
- Formula RRF: `1 / (k + rank)` para cada doc
- Suma scores de todos los retrievers

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langchain.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=retrievers,
    weights=[0.5, 0.5]  # Pesos para cada retriever
)

docs = ensemble_retriever.get_relevant_documents(query)
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from typing import List

def rag_fusion(query: str, retrievers: List, llm) -> str:
    ensemble_retriever = EnsembleRetriever(
        retrievers=retrievers,
        weights=[1.0/len(retrievers)] * len(retrievers)
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=ensemble_retriever
    )
    
    result = chain({"query": query})
    return result["result"]
```

</details>

---

## Pista 8: conversational_rag()

<details>
<summary>Ver Pista Nivel 1</summary>

RAG conversacional con memoria:
- `ConversationalRetrievalChain`
- Mantiene historial de conversación
- Reformula preguntas con contexto
- `chat_history`: lista de tuplas (pregunta, respuesta)

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langchain.chains import ConversationalRetrievalChain

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

result = chain({
    "question": question,
    "chat_history": chat_history
})
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from langchain.chains import ConversationalRetrievalChain
from typing import List, Tuple, Dict

def conversational_rag(chat_history: List[Tuple[str, str]], 
                      question: str, 
                      retriever, 
                      llm) -> Dict[str, any]:
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    result = chain({
        "question": question,
        "chat_history": chat_history
    })
    
    return {
        "answer": result["answer"],
        "sources": [doc.page_content for doc in result.get("source_documents", [])]
    }
```

</details>

---

## Pista 9: evaluate_rag_response()

<details>
<summary>Ver Pista Nivel 1</summary>

Evaluación de RAG:
- **Faithfulness**: ¿La respuesta se basa en el contexto?
- **Answer Relevancy**: ¿La respuesta responde la pregunta?
- **Context Precision**: ¿El contexto es relevante?
- **Context Recall**: ¿Se recuperó toda la info necesaria?

Frameworks: RAGAS, TruLens

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
# Métricas simples sin frameworks

def evaluate_rag_response(question, answer, context, ground_truth=None):
    metrics = {}
    
    # Faithfulness: ¿respuesta basada en contexto?
    # Usa LLM para verificar
    
    # Answer relevancy: similitud semántica pregunta-respuesta
    # Usa embeddings
    
    # Context precision: % contexto relevante
    
    return metrics
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_rag_response(question: str, answer: str, context: List[str], 
                         ground_truth: str = None) -> Dict[str, float]:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Answer Relevancy: similitud pregunta-respuesta
    q_emb = model.encode([question])
    a_emb = model.encode([answer])
    relevancy = float(cosine_similarity(q_emb, a_emb)[0][0])
    
    metrics = {
        "answer_relevancy": relevancy
    }
    
    # Context Precision: si hay ground truth
    if ground_truth:
        gt_emb = model.encode([ground_truth])
        precision = float(cosine_similarity(a_emb, gt_emb)[0][0])
        metrics["precision"] = precision
    
    return metrics
```

</details>

---

## Conceptos Clave

### ¿Qué es RAG?
Retrieval-Augmented Generation combina:
1. **Retrieval**: Buscar información relevante
2. **Augmentation**: Añadir contexto al prompt
3. **Generation**: LLM genera respuesta con contexto

### Pipeline RAG Básico
```
Query → Retriever → Relevant Docs → LLM Prompt → Answer
```

### Ventajas de RAG
- ✅ Información actualizada sin reentrenar
- ✅ Reduce alucinaciones
- ✅ Cita fuentes (verificable)
- ✅ Menos costoso que fine-tuning
- ✅ Funciona con documentos privados

### Componentes de RAG

1. **Document Ingestion**:
   - Load → Split → Embed → Store

2. **Retrieval**:
   - Query → Embed → Search → Top-K

3. **Generation**:
   - Context + Query → Prompt → LLM → Answer

### Estrategias de Chunking

| Método | Pros | Cons |
|--------|------|------|
| Fixed Size | Simple | Puede cortar contexto |
| Sentence | Mantiene coherencia | Tamaño variable |
| Recursive | Inteligente | Más complejo |
| Semantic | Grupos semánticos | Requiere modelo |

### Chain Types en LangChain

- **stuff**: Mete todos los docs en un prompt (simple, limitado por contexto)
- **map_reduce**: Procesa docs por separado, luego combina (escala mejor)
- **refine**: Itera sobre docs refinando respuesta (más tokens)
- **map_rerank**: Genera múltiples respuestas y elige la mejor

### Técnicas Avanzadas de RAG

1. **Multi-Query**: Genera múltiples versiones de la query
2. **RAG Fusion**: Combina múltiples estrategias de retrieval
3. **Self-RAG**: LLM decide cuándo recuperar info
4. **Corrective RAG**: Valida y corrige retrieval
5. **Adaptive RAG**: Adapta estrategia según complejidad

### Retrieval Strategies

- **Similarity Search**: Más similar por embedding
- **MMR** (Maximal Marginal Relevance): Balance similitud-diversidad
- **Similarity Score Threshold**: Solo docs sobre umbral
- **Hybrid Search**: Keyword (BM25) + Semantic

### Evaluación de RAG

**Métricas de Retrieval**:
- Precision@K
- Recall@K
- MRR (Mean Reciprocal Rank)

**Métricas de Generation**:
- Faithfulness (fidelidad al contexto)
- Answer Relevancy (relevancia)
- BLEU, ROUGE (con ground truth)

**Frameworks de Evaluación**:
- RAGAS
- TruLens
- LlamaIndex evaluation

### Mejores Prácticas

1. **Chunking**:
   - Chunk size: 500-1000 caracteres
   - Overlap: 10-20% del tamaño
   - Preserva estructura (párrafos, oraciones)

2. **Retrieval**:
   - k=4-10 documentos típicamente
   - Considera hybrid search
   - Reranking mejora calidad

3. **Prompting**:
   - Instruye usar solo el contexto
   - Pide citas/referencias
   - Maneja "no sé" si info no disponible

4. **Optimización**:
   - Cache embeddings
   - Índices optimizados (HNSW, IVF)
   - Batch processing

### Problemas Comunes

**Lost in the Middle**: LLM ignora info en medio del contexto
- Solución: Reordenar docs por relevancia

**Alucinaciones**: LLM inventa info
- Solución: Prompt engineering, temperature baja

**Retrieval Pobre**: No encuentra docs relevantes
- Solución: Mejor chunking, hybrid search, query expansion

**Contexto Limitado**: No caben todos los docs
- Solución: Compression, summarization, map_reduce

## Recursos
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [RAGAS Evaluation](https://docs.ragas.io/)
- [RAG Patterns](https://www.anthropic.com/research/retrieval-augmented-generation)
- [Advanced RAG Techniques](https://arxiv.org/abs/2312.10997)
