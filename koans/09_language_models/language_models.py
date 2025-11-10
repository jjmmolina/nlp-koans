"""
Koan 09: Language Models - Modelos de Lenguaje Generativos

Los modelos de lenguaje generan texto coherente basándose en patrones aprendidos.

Modelos populares:
- GPT (Generative Pre-trained Transformer)
- GPT-2, GPT-3, GPT-4
- BLOOM, LLaMA

Aplicaciones:
- Generación de texto
- Completado automático
- Chatbots
- Asistentes de código
"""

from transformers import (
    pipeline,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from typing import List, Dict
import torch


def generate_text_simple(
    prompt: str, max_length: int = 50, model_name: str = "gpt2"
) -> str:
    """
    Genera texto a partir de un prompt.

    Ejemplo:
        >>> generate_text_simple("Python es un lenguaje")
        "Python es un lenguaje de programación interpretado y de alto nivel..."

    Args:
        prompt: Texto inicial
        max_length: Longitud máxima del texto generado
        model_name: Nombre del modelo

    Returns:
        Texto generado
    """
    # TODO: Implementa generación de texto con transformers pipeline
    # Pista: Usa la tarea 'text-generation'
    pass


def generate_multiple_completions(
    prompt: str, num_completions: int = 3, max_length: int = 50
) -> List[str]:
    """
    Genera múltiples completados diferentes del mismo prompt.

    Ejemplo:
        >>> generate_multiple_completions("El mejor lenguaje es", num_completions=3)
        ["El mejor lenguaje es Python porque...",
         "El mejor lenguaje es JavaScript ya que...",
         "El mejor lenguaje es Go debido a..."]

    Args:
        prompt: Texto inicial
        num_completions: Número de completados
        max_length: Longitud máxima

    Returns:
        Lista de textos generados
    """
    # TODO: Genera múltiples completados
    # Pista: Busca el parámetro num_return_sequences
    pass


def generate_with_temperature(
    prompt: str, temperature: float = 1.0, max_length: int = 50
) -> str:
    """
    Genera texto con temperatura ajustable.

    Temperature controla la aleatoriedad:
    - Baja (0.1-0.5): Más predecible y conservador
    - Alta (1.0-2.0): Más creativo y aleatorio

    Ejemplo:
        >>> # Conservador
        >>> generate_with_temperature("Python es", temperature=0.3)
        "Python es un lenguaje de programación"
        >>> # Creativo
        >>> generate_with_temperature("Python es", temperature=1.5)
        "Python es mágico y transforma ideas en realidad"

    Args:
        prompt: Texto inicial
        temperature: Control de aleatoriedad
        max_length: Longitud máxima

    Returns:
        Texto generado
    """
    # TODO: Implementa generación con temperatura
    pass


def generate_with_top_k(prompt: str, top_k: int = 50, max_length: int = 50) -> str:
    """
    Genera texto con top-k sampling.

    Top-k sampling limita las opciones a las k palabras más probables.

    Ejemplo:
        >>> generate_with_top_k("Python", top_k=10)
        "Python is a programming language..."

    Args:
        prompt: Texto inicial
        top_k: Número de tokens candidatos
        max_length: Longitud máxima

    Returns:
        Texto generado
    """
    # TODO: Implementa top-k sampling
    pass


def generate_with_top_p(prompt: str, top_p: float = 0.9, max_length: int = 50) -> str:
    """
    Genera texto con nucleus sampling (top-p).

    Top-p selecciona del conjunto mínimo de palabras cuya probabilidad suma >= top_p.

    Ejemplo:
        >>> generate_with_top_p("Python", top_p=0.95)
        "Python provides powerful tools..."

    Args:
        prompt: Texto inicial
        top_p: Probabilidad acumulada
        max_length: Longitud máxima

    Returns:
        Texto generado
    """
    # TODO: Implementa nucleus sampling
    pass


def chat_completion(messages: List[Dict[str, str]], model_name: str = "gpt2") -> str:
    """
    Genera respuesta en formato de chat.

    Ejemplo:
        >>> messages = [
        ...     {"role": "user", "content": "¿Qué es Python?"}
        ... ]
        >>> chat_completion(messages)
        "Python es un lenguaje de programación..."

    Args:
        messages: Lista de mensajes con role y content
        model_name: Modelo a usar

    Returns:
        Respuesta del asistente
    """
    # TODO: Implementa chat completion
    # Pista: Formatea los mensajes como texto antes de generar
    pass


def calculate_perplexity(text: str, model_name: str = "gpt2") -> float:
    """
    Calcula la perplejidad de un texto.

    Perplejidad mide qué tan "sorprendido" está el modelo con el texto.
    Menor perplejidad = texto más natural/esperado

    Ejemplo:
        >>> calculate_perplexity("Python is a programming language")
        15.2  # Baja perplejidad (texto natural)
        >>> calculate_perplexity("xyzabc qwerty asdfgh")
        450.8  # Alta perplejidad (texto raro)

    Args:
        text: Texto a evaluar
        model_name: Modelo a usar

    Returns:
        Valor de perplejidad
    """
    # TODO: Implementa cálculo de perplejidad
    # Pista: perplejidad = exp(loss), donde loss viene del modelo
    pass


def prompt_engineering_template(
    task: str, context: str = "", examples: List[str] = None
) -> str:
    """
    Crea un prompt bien estructurado para mejores resultados.

    Ejemplo:
        >>> prompt = prompt_engineering_template(
        ...     task="Traduce al inglés",
        ...     examples=["Hola -> Hello", "Adiós -> Goodbye"]
        ... )
        >>> # Usa este prompt para generación

    Args:
        task: Descripción de la tarea
        context: Contexto adicional
        examples: Ejemplos few-shot

    Returns:
        Prompt formateado
    """
    # TODO: Construye un prompt estructurado
    # Pista: Incluye task, context y examples en un formato claro
    pass


def compare_generation_strategies(
    prompt: str, strategies: List[str] = None
) -> Dict[str, str]:
    """
    Compara diferentes estrategias de generación.

    Estrategias: greedy, beam_search, sampling, top_k, top_p

    Ejemplo:
        >>> compare_generation_strategies("Python es", ["greedy", "sampling"])
        {
            'greedy': "Python es un lenguaje...",
            'sampling': "Python es fantástico..."
        }

    Args:
        prompt: Texto inicial
        strategies: Lista de estrategias a comparar

    Returns:
        Diccionario con resultados por estrategia
    """
    # TODO: Implementa comparación de estrategias
    if strategies is None:
        strategies = ["greedy", "sampling"]
    pass
