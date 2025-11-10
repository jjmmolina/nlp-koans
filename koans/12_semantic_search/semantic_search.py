"""
Koan 12: Semantic Search & Vector Databases

Búsqueda semántica usando embeddings y vector databases:
- OpenAI Embeddings
- Sentence Transformers
- ChromaDB, FAISS
- Similarity search

Librerías: openai, sentence-transformers, chromadb, faiss-cpu
"""

from typing import List, Tuple
import numpy as np


def create_openai_embedding(
    text: str, model: str = "text-embedding-3-small"
) -> List[float]:
    """Crea embedding con OpenAI API"""
    # TODO: Implementar con openai.embeddings.create()
    pass


def create_sentence_transformer_embedding(
    text: str, model: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """Crea embedding con Sentence Transformers (local, gratis)"""
    # TODO: from sentence_transformers import SentenceTransformer
    pass


def cosine_similarity_search(
    query_embedding: List[float], document_embeddings: List[List[float]], top_k: int = 5
) -> List[Tuple[int, float]]:
    """Encuentra documentos más similares por similitud coseno"""
    # TODO: Implementar búsqueda por similitud
    pass


def create_chromadb_collection(collection_name: str):
    """Crea colección en ChromaDB para búsqueda vectorial"""
    # TODO: import chromadb; client = chromadb.Client()
    pass


def add_documents_to_chromadb(
    collection, documents: List[str], metadatas: List[dict] = None
):
    """Añade documentos a ChromaDB con embeddings automáticos"""
    # TODO: collection.add(documents=documents, metadatas=metadatas, ids=[...])
    pass


def search_chromadb(collection, query: str, top_k: int = 5) -> List[dict]:
    """Busca en ChromaDB"""
    # TODO: return collection.query(query_texts=[query], n_results=top_k)
    pass


def create_faiss_index(embeddings: np.ndarray):
    """Crea índice FAISS para búsqueda rápida"""
    # TODO: import faiss; return faiss.IndexFlatL2(dimension)
    pass


def semantic_search_with_reranking(
    query: str, documents: List[str], top_k: int = 5
) -> List[Tuple[str, float]]:
    """Búsqueda semántica con reranking"""
    # TODO: 1. Embedding, 2. Búsqueda inicial, 3. Rerank con cross-encoder
    pass
