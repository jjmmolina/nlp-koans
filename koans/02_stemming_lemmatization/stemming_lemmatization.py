"""
Koan 02: Stemming y Lemmatization - Normalización de Palabras

Stemming y Lemmatization son técnicas para reducir palabras a su forma base.

- Stemming: Corta el final de las palabras (rápido pero tosco)
  Ejemplo: "corriendo" → "corr"

- Lemmatization: Usa reglas lingüísticas (preciso pero más lento)
  Ejemplo: "corriendo" → "correr"

Librerías:
- NLTK: Varios stemmers (Porter, Snowball, etc.)
- spaCy: Lemmatization integrada
"""

import nltk
from typing import List

# TODO: Descarga recursos necesarios
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def stem_word_porter(word: str) -> str:
    """
    Aplica stemming usando el algoritmo Porter.

    El Porter Stemmer es el más común para inglés.

    Ejemplo:
        >>> stem_word_porter("running")
        'run'
        >>> stem_word_porter("flies")
        'fli'

    Args:
        word: Palabra a procesar

    Returns:
        Stem de la palabra
    """
    # TODO: Implementa con PorterStemmer de NLTK
    # Pista: Necesitas importar la clase y crear una instancia
    # Consulta HINTS.md para más detalles
    pass


def stem_word_snowball(word: str, language: str = "spanish") -> str:
    """
    Aplica stemming usando el algoritmo Snowball.

    Snowball soporta múltiples idiomas, incluyendo español.

    Ejemplo:
        >>> stem_word_snowball("corriendo", "spanish")
        'corr'
        >>> stem_word_snowball("running", "english")
        'run'

    Args:
        word: Palabra a procesar
        language: Idioma ('spanish', 'english', etc.)

    Returns:
        Stem de la palabra
    """
    # TODO: Implementa con SnowballStemmer de NLTK
    # Pista: Este stemmer acepta un parámetro de idioma al crearse
    pass


def stem_sentence(sentence: str, language: str = "spanish") -> str:
    """
    Aplica stemming a todas las palabras de una oración.

    Ejemplo:
        >>> stem_sentence("Los gatos están corriendo")
        'los gat est corr'

    Args:
        sentence: Oración a procesar
        language: Idioma

    Returns:
        Oración con palabras stemmed
    """
    # TODO: Implementa stemming de oración
    # Pistas:
    # 1. Divide la oración en palabras (tokenización simple)
    # 2. Aplica stemming a cada palabra (usa una función que ya creaste)
    # 3. Une las palabras procesadas
    pass


def lemmatize_word_nltk(word: str, pos: str = "n") -> str:
    """
    Lemmatiza una palabra usando NLTK WordNet.

    Ejemplo:
        >>> lemmatize_word_nltk("running", pos="v")
        'run'
        >>> lemmatize_word_nltk("better", pos="a")
        'good'

    Args:
        word: Palabra a lemmatizar
        pos: Part-of-speech tag ('n'=noun, 'v'=verb, 'a'=adjective, 'r'=adverb)

    Returns:
        Lema de la palabra
    """
    # TODO: Implementa con WordNetLemmatizer
    # Pista: Crea una instancia del lemmatizer y usa su método lemmatize()
    pass


def lemmatize_with_spacy(text: str, lang: str = "es") -> List[str]:
    """
    Lemmatiza un texto usando spaCy.

    spaCy hace lemmatization automáticamente al procesar texto.

    Ejemplo:
        >>> lemmatize_with_spacy("Los gatos están corriendo")
        ['el', 'gato', 'estar', 'correr']

    Args:
        text: Texto a lemmatizar
        lang: Idioma ('es' o 'en')

    Returns:
        Lista de lemas
    """
    # TODO: Implementa lemmatization con spaCy
    # Pistas:
    # 1. Carga el modelo spaCy apropiado
    # 2. Procesa el texto
    # 3. Cada token tiene un atributo con su lema
    pass


def compare_stem_vs_lemma(word: str, language: str = "spanish") -> dict:
    """
    Compara el resultado de stemming vs lemmatization.

    Ejemplo:
        >>> compare_stem_vs_lemma("corriendo")
        {
            'original': 'corriendo',
            'stem': 'corr',
            'lemma': 'correr'
        }

    Args:
        word: Palabra a analizar
        language: Idioma

    Returns:
        Diccionario con resultados
    """
    # TODO: Implementa la comparación
    # Usa las funciones que ya creaste para stemming y lemmatization
    return {
        "original": word,
        "stem": None,  # TODO: Aplica stemming aquí
        "lemma": None,  # TODO: Aplica lemmatization aquí
    }


def normalize_text(text: str, method: str = "lemma", language: str = "spanish") -> str:
    """
    Normaliza un texto usando stemming o lemmatization.

    Ejemplo:
        >>> normalize_text("Los gatos corrían rápidamente", method="stem")
        'los gat corr rapid'
        >>> normalize_text("Los gatos corrían rápidamente", method="lemma")
        'el gato correr rápidamente'

    Args:
        text: Texto a normalizar
        method: 'stem' o 'lemma'
        language: Idioma

    Returns:
        Texto normalizado
    """
    # TODO: Implementa normalización de texto
    # Pista: Usa if/else para elegir entre stemming y lemmatization
    # Cada método requiere un procesamiento diferente del texto
    pass
