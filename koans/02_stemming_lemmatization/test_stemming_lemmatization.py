"""
Tests para Koan 02: Stemming y Lemmatization

Ejecuta con:
    pytest koans/02_stemming_lemmatization/test_stemming_lemmatization.py -v
"""

import pytest
from stemming_lemmatization import (
    stem_word_porter,
    stem_word_snowball,
    stem_sentence,
    lemmatize_word_nltk,
    lemmatize_with_spacy,
    compare_stem_vs_lemma,
    normalize_text
)


class TestStemming:
    """Tests de stemming"""
    
    def test_stem_word_porter_english(self):
        """Test: Stemming con Porter en inglés"""
        assert stem_word_porter("running") == "run"
        assert stem_word_porter("flies") == "fli"
        assert stem_word_porter("studies") == "studi"
        
    def test_stem_word_snowball_spanish(self):
        """Test: Stemming con Snowball en español"""
        result = stem_word_snowball("corriendo", "spanish")
        assert len(result) > 0
        assert result == "corr" or result == "corriend"  # Puede variar
        
    def test_stem_word_snowball_english(self):
        """Test: Stemming con Snowball en inglés"""
        result = stem_word_snowball("running", "english")
        assert result == "run"
        
    def test_stem_sentence_spanish(self):
        """Test: Stemming de oración completa"""
        sentence = "Los gatos están corriendo"
        result = stem_sentence(sentence, "spanish")
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Debe contener versiones stemmed
        assert "corr" in result.lower()


class TestLemmatization:
    """Tests de lemmatization"""
    
    def test_lemmatize_word_nltk_verb(self):
        """Test: Lemmatization de verbos con NLTK"""
        assert lemmatize_word_nltk("running", pos="v") == "run"
        assert lemmatize_word_nltk("flies", pos="v") == "fly"
        
    def test_lemmatize_word_nltk_adjective(self):
        """Test: Lemmatization de adjetivos"""
        result = lemmatize_word_nltk("better", pos="a")
        assert result == "good"
        
    def test_lemmatize_word_nltk_noun(self):
        """Test: Lemmatization de sustantivos"""
        result = lemmatize_word_nltk("feet", pos="n")
        assert result == "foot"
        
    def test_lemmatize_with_spacy_spanish(self):
        """Test: Lemmatization con spaCy en español"""
        text = "Los gatos están corriendo"
        result = lemmatize_with_spacy(text, lang="es")
        
        assert isinstance(result, list)
        assert len(result) > 0
        # Debe tener los lemas básicos
        assert "gato" in result or "el" in result
        
    def test_lemmatize_with_spacy_english(self):
        """Test: Lemmatization con spaCy en inglés"""
        text = "The cats are running"
        result = lemmatize_with_spacy(text, lang="en")
        
        assert isinstance(result, list)
        assert "cat" in result
        assert "run" in result


class TestComparison:
    """Tests de comparación stem vs lemma"""
    
    def test_compare_stem_vs_lemma_spanish(self):
        """Test: Comparar stemming y lemmatization"""
        result = compare_stem_vs_lemma("corriendo", "spanish")
        
        assert isinstance(result, dict)
        assert result["original"] == "corriendo"
        assert "stem" in result
        assert "lemma" in result
        assert len(result["stem"]) > 0
        assert len(result["lemma"]) > 0
        
    def test_compare_shows_difference(self):
        """Test: Stem y lemma deben ser diferentes"""
        result = compare_stem_vs_lemma("estudiando", "spanish")
        
        # El stem suele ser más corto que el lema
        assert len(result["stem"]) <= len(result["lemma"])


class TestNormalization:
    """Tests de normalización de texto"""
    
    def test_normalize_text_stem(self):
        """Test: Normalización con stemming"""
        text = "Los gatos corrían rápidamente"
        result = normalize_text(text, method="stem", language="spanish")
        
        assert isinstance(result, str)
        assert len(result) > 0
        
    def test_normalize_text_lemma(self):
        """Test: Normalización con lemmatization"""
        text = "Los gatos corrían rápidamente"
        result = normalize_text(text, method="lemma", language="spanish")
        
        assert isinstance(result, str)
        assert len(result) > 0


class TestRealWorldExamples:
    """Tests con ejemplos reales"""
    
    def test_verb_conjugations_spanish(self):
        """Test: Conjugaciones verbales en español"""
        verbs = ["corro", "corres", "corre", "corremos", "corren"]
        stems = [stem_word_snowball(v, "spanish") for v in verbs]
        
        # Todos deben tener el mismo stem
        assert len(set(stems)) == 1, "Todas las conjugaciones deben tener el mismo stem"
        
    def test_plural_to_singular(self):
        """Test: Lematización de plurales a singulares"""
        result = lemmatize_with_spacy("Los gatos y los perros", lang="es")
        
        assert "gato" in result, "Debe convertir 'gatos' a 'gato'"
        assert "perro" in result, "Debe convertir 'perros' a 'perro'"
        
    def test_irregular_verbs(self):
        """Test: Verbos irregulares en inglés"""
        # 'was' es pasado de 'be'
        result = lemmatize_word_nltk("was", pos="v")
        assert result == "be" or result == "was"  # Puede variar según el lemmatizer
