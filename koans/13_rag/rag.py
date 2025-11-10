"""
Koan 13: Retrieval-Augmented Generation (RAG)

RAG combina búsqueda de información con LLMs:
- Document chunking
- Vector stores
- Retrieval chains
- Citations & sources
- Advanced RAG patterns

Librerías: langchain, openai, chromadb
"""

from typing import List, Dict, Tuple


def chunk_documents(
    documents: List[str], chunk_size: int = 500, overlap: int = 50
) -> List[str]:
    """Divide documentos en chunks para RAG"""
    # TODO: Usa RecursiveCharacterTextSplitter de LangChain
    pass


def create_vector_store(documents: List[str]):
    """Crea vector store con ChromaDB"""
    # TODO: Importa Chroma de langchain.vectorstores y crea el store
    pass


def create_retriever(vector_store, search_type: str = "similarity", k: int = 4):
    """Crea retriever para búsqueda"""
    # TODO: Convierte el vector store en retriever con parámetros de búsqueda
    pass


def basic_rag_chain(retriever, llm):
    """Crea RAG chain básico con LangChain"""
    # TODO: Usa RetrievalQA de langchain.chains
    pass


def rag_with_citations(query: str, retriever, llm) -> Dict[str, any]:
    """RAG con referencias a fuentes"""
    # TODO: Recupera documentos y genera respuesta con fuentes
    pass


def multi_query_rag(query: str, retriever, llm, num_queries: int = 3) -> str:
    """RAG con expansión de consultas múltiples"""
    # TODO: Usa MultiQueryRetriever de langchain.retrievers
    pass


def rag_fusion(query: str, retrievers: List, llm) -> str:
    """RAG con múltiples estrategias de retrieval"""
    # TODO: Combina resultados de múltiples retrievers
    pass


def conversational_rag(
    chat_history: List[Tuple[str, str]], question: str, retriever, llm
) -> Dict[str, any]:
    """RAG conversacional con historial"""
    # TODO: Usa ConversationalRetrievalChain de langchain.chains
    pass


def evaluate_rag_response(
    question: str, answer: str, context: List[str], ground_truth: str = None
) -> Dict[str, float]:
    """Evalúa calidad de respuesta RAG"""
    # TODO: Calcula métricas: faithfulness, answer_relevancy, context_precision
    pass
