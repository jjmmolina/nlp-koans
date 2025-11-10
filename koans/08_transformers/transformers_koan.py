"""
Koan 08: Transformers - Modelos de Lenguaje Modernos

Los Transformers son la arquitectura más poderosa actual para NLP.

Modelos famosos:
- BERT: Bidirectional Encoder (comprensión)
- GPT: Generative Pre-trained (generación)
- T5: Text-to-Text Transfer Transformer

Usaremos Hugging Face Transformers.
"""

from transformers import pipeline, AutoTokenizer, AutoModel
from typing import List, Dict
import torch


def load_pretrained_pipeline(
    task: str, model: str = None, lang: str = "es"
) -> pipeline:
    """
    Carga un pipeline pre-entrenado de Hugging Face.

    Ejemplo:
        >>> pipe = load_pretrained_pipeline("sentiment-analysis")

    Args:
        task: Tarea (sentiment-analysis, ner, qa, etc.)
        model: Nombre del modelo (opcional)
        lang: Idioma

    Returns:
        Pipeline de Hugging Face
    """
    # TODO: Implementa carga de pipeline
    # Pista: La librería transformers tiene una función pipeline muy útil
    pass


def extract_features_bert(
    text: str, model_name: str = "bert-base-multilingual-cased"
) -> torch.Tensor:
    """
    Extrae características usando BERT.

    BERT genera representaciones contextuales de palabras.

    Ejemplo:
        >>> features = extract_features_bert("Python es genial")
        >>> features.shape
        torch.Size([1, 5, 768])  # [batch, tokens, hidden_size]

    Args:
        text: Texto a procesar
        model_name: Nombre del modelo BERT

    Returns:
        Tensor con características
    """
    # TODO: Implementa extracción de features con BERT
    # Pista: Necesitas tokenizar el texto y pasarlo por el modelo
    # Consulta HINTS.md para el proceso completo
    pass


def question_answering(context: str, question: str, lang: str = "es") -> Dict:
    """
    Responde preguntas sobre un contexto usando un modelo QA.

    Ejemplo:
        >>> context = "Python fue creado por Guido van Rossum en 1991"
        >>> question = "¿Quién creó Python?"
        >>> question_answering(context, question)
        {'answer': 'Guido van Rossum', 'score': 0.95}

    Args:
        context: Texto de contexto
        question: Pregunta
        lang: Idioma

    Returns:
        Diccionario con respuesta y score
    """
    # TODO: Implementa QA con transformers
    # Pista: Existe un pipeline específico para question-answering
    pass


def fill_mask(text: str, lang: str = "es") -> List[Dict]:
    """
    Rellena palabras enmascaradas en un texto.

    Usa [MASK] o <mask> para marcar la palabra a predecir.

    Ejemplo:
        >>> fill_mask("Python es un [MASK] de programación")
        [{'token_str': 'lenguaje', 'score': 0.87}, ...]

    Args:
        text: Texto con [MASK]
        lang: Idioma

    Returns:
        Lista de predicciones con scores
    """
    # TODO: Implementa fill-mask
    # Pista: Pipeline de "fill-mask"
    pass


def zero_shot_classification(
    text: str, candidate_labels: List[str], lang: str = "es"
) -> Dict:
    """
    Clasifica texto sin entrenamiento específico.

    Zero-shot learning permite clasificar en categorías nunca vistas.

    Ejemplo:
        >>> text = "Este código tiene un bug"
        >>> labels = ["problema", "éxito", "neutral"]
        >>> zero_shot_classification(text, labels)
        {'labels': ['problema', 'neutral', 'éxito'], 'scores': [0.89, 0.08, 0.03]}

    Args:
        text: Texto a clasificar
        candidate_labels: Etiquetas posibles
        lang: Idioma

    Returns:
        Clasificación con scores
    """
    # TODO: Implementa zero-shot classification
    pass


def summarize_text(
    text: str, max_length: int = 130, min_length: int = 30, lang: str = "es"
) -> str:
    """
    Resume un texto automáticamente.

    Ejemplo:
        >>> long_text = "Python es un lenguaje... (texto largo)"
        >>> summarize_text(long_text)
        "Python es un lenguaje interpretado y de alto nivel..."

    Args:
        text: Texto a resumir
        max_length: Longitud máxima del resumen
        min_length: Longitud mínima del resumen
        lang: Idioma

    Returns:
        Texto resumido
    """
    # TODO: Implementa summarization
    pass


def translate_text(text: str, source_lang: str = "es", target_lang: str = "en") -> str:
    """
    Traduce texto entre idiomas.

    Ejemplo:
        >>> translate_text("Hola mundo", source_lang="es", target_lang="en")
        "Hello world"

    Args:
        text: Texto a traducir
        source_lang: Idioma origen
        target_lang: Idioma destino

    Returns:
        Texto traducido
    """
    # TODO: Implementa traducción
    # Pista: Busca modelos de traducción específicos para el par de idiomas
    pass


def compare_models_performance(
    text: str, models: List[str], task: str = "sentiment-analysis"
) -> Dict[str, Dict]:
    """
    Compara el rendimiento de diferentes modelos en la misma tarea.

    Ejemplo:
        >>> text = "Me encanta Python"
        >>> models = ["model1", "model2"]
        >>> compare_models_performance(text, models, "sentiment-analysis")
        {
            'model1': {'label': 'POSITIVE', 'score': 0.99, 'time': 0.5},
            'model2': {'label': 'POSITIVE', 'score': 0.95, 'time': 0.3}
        }

    Args:
        text: Texto de prueba
        models: Lista de modelos a comparar
        task: Tarea a realizar

    Returns:
        Diccionario con resultados por modelo
    """
    # TODO: Implementa comparación de modelos
    # Pista: Mide el tiempo de cada modelo y compara resultados
    pass
