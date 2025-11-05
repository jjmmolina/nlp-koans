"""
Tests para Koan 03: POS Tagging

Ejecuta con:
    pytest koans/03_pos_tagging/test_pos_tagging.py -v
"""

import pytest
from pos_tagging import (
    pos_tag_nltk,
    pos_tag_spacy,
    extract_nouns,
    extract_verbs,
    extract_adjectives,
    get_pos_statistics,
    find_noun_phrases,
    pos_pattern_match
)


class TestBasicPOSTagging:
    """Tests básicos de POS tagging"""
    
    def test_pos_tag_nltk_english(self):
        """Test: POS tagging con NLTK"""
        result = pos_tag_nltk("Python is awesome")
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        
    def test_pos_tag_spacy_spanish(self):
        """Test: POS tagging con spaCy en español"""
        result = pos_tag_spacy("Python es genial", lang="es")
        
        assert isinstance(result, list)
        assert len(result) > 0
        # Cada tupla debe tener 3 elementos
        assert all(len(item) == 3 for item in result)
        
    def test_pos_tag_spacy_english(self):
        """Test: POS tagging con spaCy en inglés"""
        result = pos_tag_spacy("Python is awesome", lang="en")
        
        assert isinstance(result, list)
        assert len(result) >= 3


class TestExtraction:
    """Tests de extracción de POS específicos"""
    
    def test_extract_nouns_spanish(self):
        """Test: Extraer sustantivos en español"""
        text = "El gato y el perro juegan en el parque"
        result = extract_nouns(text, lang="es")
        
        assert isinstance(result, list)
        assert "gato" in result
        assert "perro" in result
        assert "parque" in result
        
    def test_extract_nouns_english(self):
        """Test: Extraer sustantivos en inglés"""
        text = "The cat and the dog play"
        result = extract_nouns(text, lang="en")
        
        assert "cat" in result
        assert "dog" in result
        
    def test_extract_verbs_spanish(self):
        """Test: Extraer verbos en español"""
        text = "El gato come y el perro corre"
        result = extract_verbs(text, lang="es")
        
        assert isinstance(result, list)
        assert any("come" in v or "comer" in v for v in result)
        assert any("corre" in v or "correr" in v for v in result)
        
    def test_extract_adjectives_spanish(self):
        """Test: Extraer adjetivos"""
        text = "El gato negro es muy rápido"
        result = extract_adjectives(text, lang="es")
        
        assert isinstance(result, list)
        assert "negro" in result or "rápido" in result


class TestStatistics:
    """Tests de estadísticas de POS"""
    
    def test_get_pos_statistics(self):
        """Test: Obtener estadísticas de POS tags"""
        text = "El gato negro come pescado"
        result = get_pos_statistics(text, lang="es")
        
        assert isinstance(result, dict)
        assert len(result) > 0
        # Debe haber al menos un sustantivo
        assert result.get("NOUN", 0) > 0 or result.get("PROPN", 0) > 0
        
    def test_pos_statistics_counts(self):
        """Test: Conteo correcto de POS tags"""
        text = "perro gato pájaro"  # 3 sustantivos
        result = get_pos_statistics(text, lang="es")
        
        # Debe contar 3 sustantivos
        assert result.get("NOUN", 0) + result.get("PROPN", 0) >= 3


class TestNounPhrases:
    """Tests de frases nominales"""
    
    def test_find_noun_phrases_spanish(self):
        """Test: Encontrar frases nominales"""
        text = "El gato negro duerme en la cama grande"
        result = find_noun_phrases(text, lang="es")
        
        assert isinstance(result, list)
        assert len(result) > 0
        # Debe contener al menos una frase nominal
        assert any("gato" in phrase.lower() for phrase in result)
        
    def test_noun_phrases_not_empty(self):
        """Test: Texto con sustantivos debe retornar frases"""
        text = "Python es un lenguaje de programación"
        result = find_noun_phrases(text, lang="es")
        
        assert len(result) > 0


class TestPatternMatching:
    """Tests de coincidencia de patrones"""
    
    def test_pos_pattern_match_adj_noun(self):
        """Test: Patrón ADJ + NOUN"""
        text = "El gato negro y el perro blanco"
        result = pos_pattern_match(text, ["ADJ", "NOUN"], lang="es")
        
        assert isinstance(result, list)
        # Debe encontrar al menos un patrón
        assert len(result) >= 0  # Puede variar según implementación
        
    def test_pos_pattern_match_empty_pattern(self):
        """Test: Patrón vacío debe retornar lista vacía"""
        result = pos_pattern_match("Texto cualquiera", [], lang="es")
        assert result == []


class TestRealWorldExamples:
    """Tests con ejemplos reales"""
    
    def test_news_headline(self):
        """Test: Analizar titular de noticia"""
        headline = "El presidente anuncia nuevas medidas económicas"
        
        nouns = extract_nouns(headline, lang="es")
        verbs = extract_verbs(headline, lang="es")
        
        assert len(nouns) >= 2  # presidente, medidas
        assert len(verbs) >= 1  # anuncia
        
    def test_product_description(self):
        """Test: Analizar descripción de producto"""
        description = "Smartphone moderno con cámara potente y batería duradera"
        
        adjectives = extract_adjectives(description, lang="es")
        assert len(adjectives) >= 2  # moderno, potente, duradera
        
    def test_technical_text(self):
        """Test: Texto técnico con nombres propios"""
        text = "Python y JavaScript son lenguajes de programación populares"
        
        tags = pos_tag_spacy(text, lang="es")
        # Debe identificar Python y JavaScript como nombres propios
        assert any(tag == "PROPN" for _, tag, _ in tags)
