"""
Koan 01: Tokenización - Dividiendo el Texto en Unidades

La tokenización es el proceso de dividir texto en unidades más pequeñas (tokens),
que pueden ser palabras, oraciones, o incluso caracteres.

Este es el primer paso fundamental en casi cualquier pipeline de NLP.

Librerías usadas:
- NLTK: Natural Language Toolkit (clásico)
- spaCy: Procesamiento industrial de NLP
"""

import nltk
from typing import List
from nltk.tokenize import word_tokenize, sent_tokenize

# TODO: Descarga los recursos necesarios de NLTK
# Descomenta estas líneas cuando las necesites:
# nltk.download('punkt')
# nltk.download('punkt_tab')


def tokenize_words_nltk(text: str) -> List[str]:
    """
    Tokeniza un texto en palabras usando NLTK.

    Ejemplo:
        >>> tokenize_words_nltk("Hola, ¿cómo estás?")
        ['Hola', ',', '¿', 'cómo', 'estás', '?']

    Args:
        text: Texto a tokenizar

    Returns:
        Lista de tokens (palabras y signos de puntuación)
    """
    # TODO: Implementa la tokenización de palabras
    # Pista: Necesitas usar la función word_tokenize que ya está importada arriba
    # ¿Qué debes retornar?
    pass


def tokenize_sentences_nltk(text: str) -> List[str]:
    """
    Tokeniza un texto en oraciones usando NLTK.

    Ejemplo:
        >>> text = "Hola mundo. ¿Cómo estás? Yo estoy bien."
        >>> tokenize_sentences_nltk(text)
        ['Hola mundo.', '¿Cómo estás?', 'Yo estoy bien.']

    Args:
        text: Texto a tokenizar

    Returns:
        Lista de oraciones
    """
    # TODO: Implementa la tokenización de oraciones
    # Pista: Similar a word_tokenize, pero para oraciones. Ya está importada arriba.
    pass


def tokenize_words_spacy(text: str, lang: str = "es") -> List[str]:
    """
    Tokeniza un texto en palabras usando spaCy.

    spaCy es más sofisticado que NLTK y maneja mejor casos especiales.

    Ejemplo:
        >>> tokenize_words_spacy("Dr. Smith ganó $1,000 dólares.")
        ['Dr.', 'Smith', 'ganó', '$', '1,000', 'dólares', '.']

    Args:
        text: Texto a tokenizar
        lang: Idioma ('es' para español, 'en' para inglés)

    Returns:
        Lista de tokens
    """
    # TODO: Implementa la tokenización con spaCy
    # Pistas:
    # 1. Importa spacy
    # 2. Carga el modelo correcto según el idioma
    # 3. Procesa el texto con el modelo
    # 4. Extrae los tokens (cada token tiene un atributo .text)
    # Consulta HINTS.md si necesitas más ayuda
    pass


def custom_tokenize(text: str, delimiter: str = " ") -> List[str]:
    """
    Tokenización simple usando un delimitador.

    A veces una simple división es suficiente.

    Ejemplo:
        >>> custom_tokenize("Hola-mundo-Python", delimiter="-")
        ['Hola', 'mundo', 'Python']

    Args:
        text: Texto a tokenizar
        delimiter: Delimitador para separar tokens

    Returns:
        Lista de tokens
    """
    # TODO: Implementa una tokenización simple
    # Pista: Los strings en Python tienen un método que divide por un delimitador
    pass


def count_tokens(text: str) -> dict:
    """
    Cuenta la frecuencia de cada token en un texto.

    Ejemplo:
        >>> count_tokens("el gato y el perro")
        {'el': 2, 'gato': 1, 'y': 1, 'perro': 1}

    Args:
        text: Texto a analizar

    Returns:
        Diccionario con frecuencias de tokens
    """
    # TODO: Implementa el conteo de tokens
    # Pistas:
    # 1. Primero tokeniza el texto (puedes usar una función que ya creaste)
    # 2. Normaliza a minúsculas
    # 3. Cuenta las frecuencias (mira collections.Counter o usa un dict)
    pass


def remove_punctuation_tokens(tokens: List[str]) -> List[str]:
    """
    Elimina signos de puntuación de una lista de tokens.

    Ejemplo:
        >>> remove_punctuation_tokens(['Hola', ',', 'mundo', '!'])
        ['Hola', 'mundo']

    Args:
        tokens: Lista de tokens

    Returns:
        Lista de tokens sin puntuación
    """
    # TODO: Filtra los tokens que NO sean puntuación
    # Pistas:
    # 1. El módulo string tiene una constante con todos los signos de puntuación
    # 2. Usa una list comprehension para filtrar
    pass
