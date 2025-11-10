"""
Tests for Koan 12: Semantic Search & Vector Databases
"""

import pytest
import os


class TestOpenAIEmbeddings:
    """Tests for OpenAI embeddings"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
    )
    def test_create_openai_embedding(self):
        from koans.semantic_search.semantic_search import create_openai_embedding

        embedding = create_openai_embedding("Hello world")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)


class TestSentenceTransformers:
    """Tests for Sentence Transformers embeddings"""

    def test_create_sentence_transformer_embedding(self):
        from koans.semantic_search.semantic_search import (
            create_sentence_transformer_embedding,
        )

        embedding = create_sentence_transformer_embedding("Hello world")
        assert embedding is not None
        assert len(embedding) > 0


class TestCosineSimilarity:
    """Tests for cosine similarity search"""

    def test_cosine_similarity_search(self):
        from koans.semantic_search.semantic_search import cosine_similarity_search

        query_emb = [1.0, 0.0, 0.0]
        doc_embs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]]

        results = cosine_similarity_search(query_emb, doc_embs, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


class TestChromaDB:
    """Tests for ChromaDB operations"""

    def test_create_chromadb_collection(self):
        from koans.semantic_search.semantic_search import create_chromadb_collection

        collection = create_chromadb_collection("test_collection")
        assert collection is not None

    def test_add_documents_to_chromadb(self):
        from koans.semantic_search.semantic_search import (
            create_chromadb_collection,
            add_documents_to_chromadb,
        )

        collection = create_chromadb_collection("test_docs")
        docs = ["doc1", "doc2"]
        add_documents_to_chromadb(collection, docs)
        assert collection.count() == 2

    def test_search_chromadb(self):
        from koans.semantic_search.semantic_search import (
            create_chromadb_collection,
            add_documents_to_chromadb,
            search_chromadb,
        )

        collection = create_chromadb_collection("test_search")
        docs = ["Python programming", "Machine learning", "Data science"]
        add_documents_to_chromadb(collection, docs)

        results = search_chromadb(collection, "Python code", top_k=2)
        assert len(results) > 0


class TestFAISS:
    """Tests for FAISS operations"""

    def test_create_faiss_index(self):
        from koans.semantic_search.semantic_search import create_faiss_index
        import numpy as np

        embeddings = np.random.rand(10, 384).astype("float32")
        index = create_faiss_index(embeddings)
        assert index is not None
        assert index.ntotal == 10


class TestSemanticSearch:
    """Tests for semantic search with reranking"""

    def test_semantic_search_with_reranking(self):
        from koans.semantic_search.semantic_search import semantic_search_with_reranking

        docs = [
            "Python is a programming language",
            "Machine learning is AI",
            "Data science uses statistics",
        ]

        results = semantic_search_with_reranking("coding in Python", docs, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
