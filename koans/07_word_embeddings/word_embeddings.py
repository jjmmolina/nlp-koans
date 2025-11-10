"""
Koan 07: Word Embeddings - Representaciones Vectoriales

Los word embeddings representan palabras como vectores numéricos,
capturando relaciones semánticas.

Ejemplo:
- "rey" - "hombre" + "mujer" ≈ "reina"
- Similar("gato", "perro") > Similar("gato", "coche")

Librerías:
- spaCy: Embeddings pre-entrenados
- gensim: Word2Vec, FastText
"""

import spacy
import numpy as np
from typing import List, Tuple
from scipy.spatial.distance import cosine


def get_word_vector_spacy(word: str, lang: str = "es") -> np.ndarray:
    """
    Obtiene el vector de una palabra usando spaCy.

    Ejemplo:
        >>> vector = get_word_vector_spacy("python")
        >>> vector.shape
        (96,)  # Depende del modelo

    Args:
        word: Palabra a vectorizar
        lang: Idioma

    Returns:
        Vector numpy
    """
    # TODO: Implementa obtención de vector
    # Pista: spaCy puede procesar texto y cada documento tiene un atributo de vector
    # Consulta HINTS.md para más detalles
    pass


def get_text_vector_spacy(text: str, lang: str = "es") -> np.ndarray:
    """
    Obtiene el vector promedio de un texto.

    Ejemplo:
        >>> vector = get_text_vector_spacy("me gusta python")

    Args:
        text: Texto a vectorizar
        lang: Idioma

    Returns:
        Vector numpy
    """
    # TODO: Similar a get_word_vector_spacy pero con texto completo
    pass


def cosine_similarity_words(word1: str, word2: str, lang: str = "es") -> float:
    """
    Calcula la similitud coseno entre dos palabras.

    Valores cercanos a 1 = muy similares
    Valores cercanos a 0 = no relacionadas

    Ejemplo:
        >>> cosine_similarity_words("gato", "perro")
        0.85  # Alta similitud (ambos son animales)
        >>> cosine_similarity_words("gato", "coche")
        0.23  # Baja similitud

    Args:
        word1: Primera palabra
        word2: Segunda palabra
        lang: Idioma

    Returns:
        Similitud (0-1)
    """
    # TODO: Implementa similitud coseno
    # Pista: Necesitas obtener vectores y calcular similitud coseno (ya está importada)
    pass


def find_most_similar(
    word: str, candidates: List[str], lang: str = "es", top_n: int = 3
) -> List[Tuple[str, float]]:
    """
    Encuentra las palabras más similares a una dada.

    Ejemplo:
        >>> find_most_similar("perro", ["gato", "coche", "pájaro", "manzana"])
        [('gato', 0.85), ('pájaro', 0.72), ('manzana', 0.15)]

    Args:
        word: Palabra de referencia
        candidates: Lista de palabras candidatas
        lang: Idioma
        top_n: Número de resultados

    Returns:
        Lista de tuplas (palabra, similitud) ordenadas por similitud
    """
    # TODO: Implementa búsqueda de similares
    # Pista: Calcula similitud con cada candidato y ordena
    pass


def word_analogy(
    word_a: str, word_b: str, word_c: str, candidates: List[str], lang: str = "es"
) -> str:
    """
    Resuelve analogías: A es a B como C es a ?

    Ejemplo clásico:
        >>> word_analogy("rey", "hombre", "mujer", ["reina", "princesa", "niña"])
        "reina"  # rey - hombre + mujer ≈ reina

    Args:
        word_a: Palabra A
        word_b: Palabra B
        word_c: Palabra C
        candidates: Palabras candidatas para la respuesta
        lang: Idioma

    Returns:
        Palabra más apropiada
    """
    # TODO: Implementa analogía
    # Pista: Los vectores se pueden sumar y restar algebraicamente
    pass


def get_document_similarity(text1: str, text2: str, lang: str = "es") -> float:
    """
    Calcula la similitud entre dos documentos.

    Ejemplo:
        >>> text1 = "Me gusta programar en Python"
        >>> text2 = "Python es mi lenguaje favorito"
        >>> get_document_similarity(text1, text2)
        0.78  # Alta similitud

    Args:
        text1: Primer texto
        text2: Segundo texto
        lang: Idioma

    Returns:
        Similitud (0-1)
    """
    # TODO: Calcula similitud entre documentos
    # Combina funciones anteriores
    pass


def cluster_words_by_similarity(
    words: List[str], threshold: float = 0.7, lang: str = "es"
) -> List[List[str]]:
    """
    Agrupa palabras similares en clusters.

    Ejemplo:
        >>> words = ["perro", "gato", "coche", "auto", "pájaro"]
        >>> cluster_words_by_similarity(words, threshold=0.7)
        [['perro', 'gato', 'pájaro'], ['coche', 'auto']]

    Args:
        words: Lista de palabras
        threshold: Umbral de similitud mínima
        lang: Idioma

    Returns:
        Lista de clusters (listas de palabras similares)
    """
    # TODO: Implementa clustering simple
    # Agrupa palabras con similitud por encima del threshold
    pass
