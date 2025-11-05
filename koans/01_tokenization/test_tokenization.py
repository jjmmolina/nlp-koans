"""
Tests para Koan 01: Tokenización

Ejecuta estos tests con:
    pytest koans/01_tokenization/test_tokenization.py -v

Los tests fallarán hasta que implementes las funciones en tokenization.py
"""

import pytest
from tokenization import (
    tokenize_words_nltk,
    tokenize_sentences_nltk,
    tokenize_words_spacy,
    custom_tokenize,
    count_tokens,
    remove_punctuation_tokens
)


class TestTokenizationBasics:
    """Tests básicos de tokenización"""
    
    def test_tokenize_words_nltk_spanish(self):
        """Test: Tokenización de palabras en español con NLTK"""
        text = "Hola, ¿cómo estás?"
        result = tokenize_words_nltk(text)
        
        assert isinstance(result, list), "Debe retornar una lista"
        assert len(result) > 0, "La lista no debe estar vacía"
        assert "Hola" in result, "Debe contener la palabra 'Hola'"
        assert "cómo" in result, "Debe contener la palabra 'cómo'"
        
    def test_tokenize_words_nltk_english(self):
        """Test: Tokenización de palabras en inglés con NLTK"""
        text = "Hello, how are you?"
        result = tokenize_words_nltk(text)
        
        assert "Hello" in result
        assert "how" in result
        assert "?" in result
        
    def test_tokenize_sentences_nltk(self):
        """Test: Tokenización de oraciones con NLTK"""
        text = "Hola mundo. ¿Cómo estás? Yo estoy bien."
        result = tokenize_sentences_nltk(text)
        
        assert isinstance(result, list)
        assert len(result) == 3, "Debe haber exactamente 3 oraciones"
        assert "Hola mundo." in result[0]


class TestTokenizationSpacy:
    """Tests de tokenización con spaCy"""
    
    def test_tokenize_words_spacy_spanish(self):
        """Test: Tokenización con spaCy en español"""
        text = "El Dr. García ganó 1,000 euros."
        result = tokenize_words_spacy(text, lang="es")
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert "Dr." in result or "Dr" in result, "Debe manejar abreviaturas"
        assert "García" in result
        
    def test_tokenize_words_spacy_english(self):
        """Test: Tokenización con spaCy en inglés"""
        text = "I'm learning NLP!"
        result = tokenize_words_spacy(text, lang="en")
        
        assert isinstance(result, list)
        assert len(result) > 0


class TestCustomTokenization:
    """Tests de tokenización personalizada"""
    
    def test_custom_tokenize_spaces(self):
        """Test: Tokenización por espacios"""
        text = "Hola mundo Python"
        result = custom_tokenize(text, delimiter=" ")
        
        assert result == ["Hola", "mundo", "Python"]
        
    def test_custom_tokenize_custom_delimiter(self):
        """Test: Tokenización con delimitador personalizado"""
        text = "rojo-verde-azul"
        result = custom_tokenize(text, delimiter="-")
        
        assert result == ["rojo", "verde", "azul"]
        assert len(result) == 3


class TestTokenCounting:
    """Tests de conteo de tokens"""
    
    def test_count_tokens_simple(self):
        """Test: Contar frecuencia de tokens"""
        text = "el gato y el perro"
        result = count_tokens(text)
        
        assert isinstance(result, dict)
        assert result.get("el") == 2, "La palabra 'el' aparece 2 veces"
        assert result.get("gato") == 1
        assert result.get("perro") == 1
        
    def test_count_tokens_case_insensitive(self):
        """Test: El conteo debe ser insensible a mayúsculas"""
        text = "Python python PYTHON"
        result = count_tokens(text)
        
        # Debe contar como la misma palabra
        assert result.get("python") == 3


class TestPunctuationRemoval:
    """Tests de eliminación de puntuación"""
    
    def test_remove_punctuation_tokens(self):
        """Test: Eliminar signos de puntuación"""
        tokens = ["Hola", ",", "mundo", "!", "¿", "cómo", "?"]
        result = remove_punctuation_tokens(tokens)
        
        assert "Hola" in result
        assert "mundo" in result
        assert "cómo" in result
        assert "," not in result
        assert "!" not in result
        assert "?" not in result
        
    def test_remove_punctuation_empty_list(self):
        """Test: Lista vacía debe retornar lista vacía"""
        result = remove_punctuation_tokens([])
        assert result == []


class TestRealWorldExamples:
    """Tests con ejemplos del mundo real"""
    
    def test_tweet_tokenization(self):
        """Test: Tokenizar un tweet"""
        tweet = "¡Me encanta #Python y #NLP! 🚀"
        tokens = tokenize_words_nltk(tweet)
        
        assert len(tokens) > 0
        assert any("#Python" in t or "Python" in t for t in tokens)
        
    def test_multiline_text(self):
        """Test: Texto con múltiples líneas"""
        text = """Primera línea.
        Segunda línea.
        Tercera línea."""
        
        sentences = tokenize_sentences_nltk(text)
        assert len(sentences) >= 3
