"""
Tests for Koan 13: RAG (Retrieval-Augmented Generation)
"""

import pytest
import os


class TestDocumentChunking:
    """Tests for document chunking"""

    def test_chunk_documents(self):
        from koans.rag.rag import chunk_documents

        docs = ["This is a long document. " * 100]
        chunks = chunk_documents(docs, chunk_size=100, overlap=20)

        assert isinstance(chunks, list)
        assert len(chunks) > 1


class TestVectorStore:
    """Tests for vector store creation"""

    def test_create_vector_store(self):
        from koans.rag.rag import create_vector_store

        docs = ["Doc 1", "Doc 2", "Doc 3"]
        store = create_vector_store(docs)
        assert store is not None

    def test_create_retriever(self):
        from koans.rag.rag import create_vector_store, create_retriever

        docs = ["Python programming", "Machine learning"]
        store = create_vector_store(docs)
        retriever = create_retriever(store, k=2)
        assert retriever is not None


class TestBasicRAG:
    """Tests for basic RAG functionality"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
    )
    def test_basic_rag_chain(self):
        from koans.rag.rag import create_vector_store, create_retriever, basic_rag_chain
        from langchain_openai import ChatOpenAI

        docs = ["Python is a programming language", "ML uses Python"]
        store = create_vector_store(docs)
        retriever = create_retriever(store)
        llm = ChatOpenAI(temperature=0)

        chain = basic_rag_chain(retriever, llm)
        assert chain is not None


class TestRAGWithCitations:
    """Tests for RAG with citations"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
    )
    def test_rag_with_citations(self):
        from koans.rag.rag import (
            create_vector_store,
            create_retriever,
            rag_with_citations,
        )
        from langchain_openai import ChatOpenAI

        docs = ["Python is used for AI", "Machine learning is popular"]
        store = create_vector_store(docs)
        retriever = create_retriever(store)
        llm = ChatOpenAI(temperature=0)

        result = rag_with_citations("What is Python used for?", retriever, llm)
        assert "answer" in result
        assert "sources" in result


class TestAdvancedRAG:
    """Tests for advanced RAG patterns"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
    )
    def test_multi_query_rag(self):
        from koans.rag.rag import create_vector_store, create_retriever, multi_query_rag
        from langchain_openai import ChatOpenAI

        docs = ["Python programming", "Data science"]
        store = create_vector_store(docs)
        retriever = create_retriever(store)
        llm = ChatOpenAI(temperature=0)

        answer = multi_query_rag("Tell me about Python", retriever, llm, num_queries=2)
        assert isinstance(answer, str)


class TestConversationalRAG:
    """Tests for conversational RAG"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
    )
    def test_conversational_rag(self):
        from koans.rag.rag import (
            create_vector_store,
            create_retriever,
            conversational_rag,
        )
        from langchain_openai import ChatOpenAI

        docs = ["Python is a language", "AI uses Python"]
        store = create_vector_store(docs)
        retriever = create_retriever(store)
        llm = ChatOpenAI(temperature=0)

        history = [("What is Python?", "Python is a programming language")]
        result = conversational_rag(history, "What is it used for?", retriever, llm)

        assert "answer" in result or isinstance(result, str)


class TestRAGEvaluation:
    """Tests for RAG evaluation"""

    def test_evaluate_rag_response(self):
        from koans.rag.rag import evaluate_rag_response

        question = "What is Python?"
        answer = "Python is a programming language"
        context = ["Python is a high-level programming language"]

        metrics = evaluate_rag_response(question, answer, context)
        assert isinstance(metrics, dict)
