"""
Tests para Koan 06: Sentiment Analysis

Ejecuta con:
    pytest koans/06_sentiment_analysis/test_sentiment_analysis.py -v
    
NOTA: Estos tests requieren descargar modelos (pueden tardar la primera vez)
"""

import pytest
from sentiment_analysis import (
    analyze_sentiment_simple,
    analyze_sentiment_batch,
    get_sentiment_label,
    get_sentiment_score,
    classify_sentiment_simple,
    analyze_text_emotions,
    sentiment_statistics
)


@pytest.mark.slow
class TestBasicSentiment:
    """Tests básicos de análisis de sentimientos"""
    
    def test_analyze_sentiment_simple_positive(self):
        """Test: Analizar texto positivo"""
        result = analyze_sentiment_simple("Me encanta Python, es genial!")
        
        assert isinstance(result, dict)
        assert 'label' in result or 'score' in result
        
    def test_analyze_sentiment_simple_negative(self):
        """Test: Analizar texto negativo"""
        result = analyze_sentiment_simple("Odio los bugs, son horribles")
        
        assert isinstance(result, dict)


@pytest.mark.slow
class TestBatchProcessing:
    """Tests de procesamiento por lotes"""
    
    def test_analyze_sentiment_batch(self):
        """Test: Analizar múltiples textos"""
        texts = ["Me gusta", "No me gusta", "Es normal"]
        result = analyze_sentiment_batch(texts)
        
        assert isinstance(result, list)
        assert len(result) == 3


@pytest.mark.slow
class TestSentimentExtraction:
    """Tests de extracción de sentimiento"""
    
    def test_get_sentiment_label(self):
        """Test: Obtener solo label"""
        label = get_sentiment_label("Excelente producto")
        
        assert isinstance(label, str)
        assert len(label) > 0
        
    def test_get_sentiment_score(self):
        """Test: Obtener solo score"""
        score = get_sentiment_score("Me encanta!")
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestSimpleClassification:
    """Tests de clasificación simple"""
    
    def test_classify_sentiment_simple_positive(self):
        """Test: Clasificar como positivo"""
        result = classify_sentiment_simple("Me gusta mucho Python")
        
        assert result in ['positivo', 'negativo', 'neutral']
        
    def test_classify_sentiment_simple_negative(self):
        """Test: Clasificar como negativo"""
        result = classify_sentiment_simple("Odio los errores")
        
        assert result in ['positivo', 'negativo', 'neutral']


@pytest.mark.slow
class TestEmotionAnalysis:
    """Tests de análisis de emociones"""
    
    def test_analyze_text_emotions(self):
        """Test: Analizar emociones"""
        result = analyze_text_emotions("I am very happy!")
        
        assert isinstance(result, dict)


class TestStatistics:
    """Tests de estadísticas de sentimiento"""
    
    def test_sentiment_statistics(self):
        """Test: Calcular estadísticas"""
        texts = ["Bueno", "Malo", "Normal"]
        result = sentiment_statistics(texts)
        
        assert isinstance(result, dict)
        assert 'total' in result
        assert result['total'] == 3
