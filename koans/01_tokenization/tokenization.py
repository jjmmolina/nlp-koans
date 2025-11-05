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
    # TODO: Implementa la tokenización de palabras con nltk.word_tokenize()
    # Pista: from nltk.tokenize import word_tokenize
    return []


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
    # TODO: Implementa la tokenización de oraciones con nltk.sent_tokenize()
    return []


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
    # 1. import spacy
    # 2. Carga el modelo: nlp = spacy.load("es_core_news_sm") o "en_core_web_sm"
    # 3. Procesa el texto: doc = nlp(text)
    # 4. Extrae los tokens: [token.text for token in doc]
    return []


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
    # TODO: Implementa una tokenización simple con str.split()
    return []


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
    # 1. Usa tokenize_words_nltk() para obtener los tokens
    # 2. Usa collections.Counter o un diccionario para contar
    # 3. Convierte a minúsculas para normalizar
    return {}


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
    # Pista: import string; usa string.punctuation
    return []
