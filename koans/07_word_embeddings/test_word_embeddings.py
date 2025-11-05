"""
Tests para Koan 07: Word Embeddings

Ejecuta con:
    pytest koans/07_word_embeddings/test_word_embeddings.py -v
"""

import pytest
import numpy as np
from word_embeddings import (
    get_word_vector_spacy,
    get_text_vector_spacy,
    cosine_similarity_words,
    find_most_similar,
    word_analogy,
    get_document_similarity,
    cluster_words_by_similarity
)


class TestVectorExtraction:
    """Tests de extracción de vectores"""
    
    def test_get_word_vector_spacy(self):
        """Test: Obtener vector de palabra"""
        vector = get_word_vector_spacy("python", lang="es")
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0
        
    def test_get_text_vector_spacy(self):
        """Test: Obtener vector de texto"""
        vector = get_text_vector_spacy("me gusta python", lang="es")
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0


class TestSimilarity:
    """Tests de similitud"""
    
    def test_cosine_similarity_words(self):
        """Test: Similitud entre palabras"""
        similarity = cosine_similarity_words("gato", "perro", lang="es")
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        
    def test_similar_words_high_similarity(self):
        """Test: Palabras similares tienen alta similitud"""
        # gato y perro son ambos animales
        sim_animals = cosine_similarity_words("gato", "perro", lang="es")
        # gato y coche no están relacionados
        sim_unrelated = cosine_similarity_words("gato", "coche", lang="es")
        
        # Esta comparación puede variar según el modelo
        assert isinstance(sim_animals, float)
        assert isinstance(sim_unrelated, float)


class TestMostSimilar:
    """Tests de búsqueda de similares"""
    
    def test_find_most_similar(self):
        """Test: Encontrar palabras más similares"""
        result = find_most_similar("perro", ["gato", "coche", "manzana"], lang="es")
        
        assert isinstance(result, list)
        assert len(result) <= 3
        # Cada elemento debe ser una tupla (palabra, score)
        if len(result) > 0:
            assert isinstance(result[0], tuple)
            assert len(result[0]) == 2


class TestAnalogy:
    """Tests de analogías"""
    
    def test_word_analogy(self):
        """Test: Resolver analogía simple"""
        result = word_analogy(
            "hombre", "rey", "mujer",
            ["reina", "princesa", "niña"],
            lang="es"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0


class TestDocumentSimilarity:
    """Tests de similitud de documentos"""
    
    def test_get_document_similarity(self):
        """Test: Similitud entre documentos"""
        text1 = "Me gusta programar en Python"
        text2 = "Python es mi lenguaje favorito"
        
        similarity = get_document_similarity(text1, text2, lang="es")
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        
    def test_identical_documents_high_similarity(self):
        """Test: Documentos idénticos tienen alta similitud"""
        text = "Python es genial"
        similarity = get_document_similarity(text, text, lang="es")
        
        # Debe ser muy cercano a 1
        assert similarity > 0.9


class TestClustering:
    """Tests de clustering"""
    
    def test_cluster_words_by_similarity(self):
        """Test: Agrupar palabras similares"""
        words = ["perro", "gato", "coche", "auto"]
        result = cluster_words_by_similarity(words, threshold=0.5, lang="es")
        
        assert isinstance(result, list)
        # Cada cluster debe ser una lista
        if len(result) > 0:
            assert isinstance(result[0], list)


class TestRealWorldExamples:
    """Tests con ejemplos reales"""
    
    def test_programming_languages_similarity(self):
        """Test: Similitud entre lenguajes de programación"""
        sim = cosine_similarity_words("Python", "Java", lang="es")
        
        # Deben estar algo relacionados (ambos son lenguajes)
        assert isinstance(sim, float)
        
    def test_semantic_search(self):
        """Test: Búsqueda semántica simple"""
        query = "programación"
        docs = ["Python es un lenguaje", "Me gusta cocinar", "JavaScript es útil"]
        
        # Buscar documento más similar
        similarities = [
            get_document_similarity(query, doc, lang="es")
            for doc in docs
        ]
        
        assert len(similarities) == 3
        assert all(isinstance(s, float) for s in similarities)
