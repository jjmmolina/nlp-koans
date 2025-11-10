"""
Koan 06: Sentiment Analysis - Análisis de Sentimientos

El análisis de sentimientos determina si un texto es positivo, negativo o neutral.

Usaremos modelos de Transformers pre-entrenados de Hugging Face.

Ejemplos:
- "Me encanta Python!" → POSITIVO
- "Odio los bugs" → NEGATIVO
- "Python es un lenguaje" → NEUTRAL
"""

from transformers import pipeline
from typing import Dict, List


def analyze_sentiment_simple(text: str, lang: str = "es") -> Dict:
    """
    Analiza el sentimiento de un texto usando un modelo pre-entrenado.

    Ejemplo:
        >>> analyze_sentiment_simple("Me encanta Python!")
        {'label': 'POSITIVE', 'score': 0.9987}

    Args:
        text: Texto a analizar
        lang: Idioma ('es' o 'en')

    Returns:
        Diccionario con label y score
    """
    # TODO: Implementa análisis de sentimientos con transformers
    # Pista: Usa pipeline de transformers para "sentiment-analysis"
    # Consulta HINTS.md para más detalles
    pass


def analyze_sentiment_batch(texts: List[str], lang: str = "es") -> List[Dict]:
    """
    Analiza el sentimiento de múltiples textos.

    Ejemplo:
        >>> texts = ["Me gusta", "No me gusta"]
        >>> analyze_sentiment_batch(texts)
        [{'label': 'POSITIVE', ...}, {'label': 'NEGATIVE', ...}]

    Args:
        texts: Lista de textos
        lang: Idioma

    Returns:
        Lista de resultados
    """
    # TODO: Procesa múltiples textos
    # Pista: Los pipelines pueden procesar listas de textos
    pass


def get_sentiment_label(text: str, lang: str = "es") -> str:
    """
    Retorna solo la etiqueta del sentimiento (sin score).

    Ejemplo:
        >>> get_sentiment_label("Excelente producto")
        'POSITIVE'

    Args:
        text: Texto a analizar
        lang: Idioma

    Returns:
        Etiqueta de sentimiento
    """
    # TODO: Usa la función anterior y extrae solo el label
    pass


def get_sentiment_score(text: str, lang: str = "es") -> float:
    """
    Retorna solo el score de confianza del sentimiento.

    Ejemplo:
        >>> get_sentiment_score("Me encanta!")
        0.9987

    Args:
        text: Texto a analizar
        lang: Idioma

    Returns:
        Score de confianza (0-1)
    """
    # TODO: Extrae solo el score
    pass


def classify_sentiment_simple(text: str) -> str:
    """
    Clasifica sentimiento en categorías simples: positivo, negativo, neutral.

    Ejemplo:
        >>> classify_sentiment_simple("Me gusta Python")
        'positivo'

    Args:
        text: Texto a analizar

    Returns:
        'positivo', 'negativo', o 'neutral'
    """
    # TODO: Implementa clasificación simple
    # Mapea los resultados a categorías en español
    pass


def analyze_text_emotions(text: str) -> Dict[str, float]:
    """
    Analiza emociones específicas en el texto.

    Puede detectar: alegría, tristeza, enojo, miedo, sorpresa.

    Ejemplo:
        >>> analyze_text_emotions("Estoy muy feliz!")
        {'joy': 0.95, 'sadness': 0.01, ...}

    Args:
        text: Texto a analizar

    Returns:
        Diccionario con emociones y scores
    """
    # TODO: Implementa análisis de emociones
    # Pista: Necesitas un modelo específico de emociones
    pass


def sentiment_statistics(texts: List[str]) -> Dict:
    """
    Calcula estadísticas de sentimiento sobre múltiples textos.

    Ejemplo:
        >>> texts = ["Me gusta", "Odio esto", "Es normal"]
        >>> sentiment_statistics(texts)
        {
            'total': 3,
            'positive': 1,
            'negative': 1,
            'neutral': 1,
            'avg_score': 0.65
        }

    Args:
        texts: Lista de textos

    Returns:
        Diccionario con estadísticas
    """
    # TODO: Calcula estadísticas agregadas
    # Analiza todos los textos y cuenta por categoría
    pass
