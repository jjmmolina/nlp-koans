"""
Koan 03: POS Tagging - Etiquetado Gramatical

POS (Part-of-Speech) Tagging identifica la categoría gramatical de cada palabra:
sustantivo, verbo, adjetivo, etc.

Ejemplos:
- "Python" → NOUN (sustantivo)
- "es" → VERB (verbo)
- "genial" → ADJ (adjetivo)

Librerías:
- spaCy: POS tagging moderno y preciso
- NLTK: POS tagging clásico
"""

import nltk
from typing import List, Tuple, Dict

# TODO: Descarga recursos
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')


def pos_tag_nltk(text: str) -> List[Tuple[str, str]]:
    """
    Etiqueta gramaticalmente un texto usando NLTK.

    Ejemplo:
        >>> pos_tag_nltk("Python is awesome")
        [('Python', 'NNP'), ('is', 'VBZ'), ('awesome', 'JJ')]

    Args:
        text: Texto a etiquetar

    Returns:
        Lista de tuplas (palabra, etiqueta)
    """
    # TODO: Implementa POS tagging con NLTK
    # Pistas:
    # 1. Primero necesitas tokenizar el texto
    # 2. Luego aplica nltk.pos_tag() a los tokens
    pass


def pos_tag_spacy(text: str, lang: str = "es") -> List[Tuple[str, str, str]]:
    """
    Etiqueta gramaticalmente usando spaCy.

    spaCy proporciona etiquetas más detalladas y universales.

    Ejemplo:
        >>> pos_tag_spacy("Python es genial")
        [('Python', 'PROPN', 'nombre propio'),
         ('es', 'AUX', 'verbo auxiliar'),
         ('genial', 'ADJ', 'adjetivo')]

    Args:
        text: Texto a etiquetar
        lang: Idioma ('es' o 'en')

    Returns:
        Lista de tuplas (palabra, etiqueta_universal, etiqueta_detallada)
    """
    # TODO: Implementa con spaCy
    # Pistas:
    # 1. Carga el modelo de spaCy
    # 2. Procesa el texto
    # 3. Cada token tiene atributos: .text, .pos_, .tag_
    pass


def extract_nouns(text: str, lang: str = "es") -> List[str]:
    """
    Extrae todos los sustantivos de un texto.

    Ejemplo:
        >>> extract_nouns("El gato y el perro juegan")
        ['gato', 'perro']

    Args:
        text: Texto a analizar
        lang: Idioma

    Returns:
        Lista de sustantivos
    """
    # TODO: Extrae solo los tokens con POS == 'NOUN' o 'PROPN'
    # Pista: Usa la función pos_tag_spacy y filtra por categoría
    pass


def extract_verbs(text: str, lang: str = "es") -> List[str]:
    """
    Extrae todos los verbos de un texto.

    Ejemplo:
        >>> extract_verbs("El gato come y el perro corre")
        ['come', 'corre']

    Args:
        text: Texto a analizar
        lang: Idioma

    Returns:
        Lista de verbos
    """
    # TODO: Extrae tokens con POS == 'VERB' o 'AUX'
    # Similar a extract_nouns pero con diferentes categorías
    pass


def extract_adjectives(text: str, lang: str = "es") -> List[str]:
    """
    Extrae todos los adjetivos de un texto.

    Ejemplo:
        >>> extract_adjectives("El gato negro es muy rápido")
        ['negro', 'rápido']

    Args:
        text: Texto a analizar
        lang: Idioma

    Returns:
        Lista de adjetivos
    """
    # TODO: Extrae tokens con POS == 'ADJ'
    pass


def get_pos_statistics(text: str, lang: str = "es") -> Dict[str, int]:
    """
    Calcula estadísticas de POS tags en un texto.

    Ejemplo:
        >>> get_pos_statistics("El gato negro come")
        {'DET': 1, 'NOUN': 1, 'ADJ': 1, 'VERB': 1}

    Args:
        text: Texto a analizar
        lang: Idioma

    Returns:
        Diccionario con conteo de cada POS tag
    """
    # TODO: Cuenta la frecuencia de cada POS tag
    # Pista: collections.Counter es muy útil aquí
    pass


def find_noun_phrases(text: str, lang: str = "es") -> List[str]:
    """
    Encuentra frases nominales (noun phrases) en un texto.

    Una frase nominal es un grupo de palabras con un sustantivo como núcleo.
    Ejemplo: "el gato negro", "mi casa grande"

    Ejemplo:
        >>> find_noun_phrases("El gato negro duerme")
        ['El gato negro']

    Args:
        text: Texto a analizar
        lang: Idioma

    Returns:
        Lista de frases nominales
    """
    # TODO: Implementa extracción de noun chunks con spaCy
    # Pista: spaCy tiene un atributo especial en el doc para esto
    pass


def pos_pattern_match(
    text: str, pattern: List[str], lang: str = "es"
) -> List[List[str]]:
    """
    Encuentra secuencias de palabras que coinciden con un patrón de POS tags.

    Ejemplo:
        >>> # Buscar patrón ADJ + NOUN (adjetivo + sustantivo)
        >>> pos_pattern_match("El gato negro y el perro blanco", ["ADJ", "NOUN"])
        [['negro', 'gato'], ['blanco', 'perro']]

    Args:
        text: Texto a analizar
        pattern: Lista de POS tags a buscar (en orden)
        lang: Idioma

    Returns:
        Lista de secuencias que coinciden con el patrón
    """
    # TODO: Implementa búsqueda de patrones
    # Pistas:
    # 1. Obtén los POS tags del texto
    # 2. Recorre el texto buscando ventanas del tamaño del patrón
    # 3. Compara los POS tags de cada ventana con el patrón
    pass
