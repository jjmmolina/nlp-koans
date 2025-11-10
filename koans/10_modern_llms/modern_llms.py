"""
Koan 10: Modern LLMs & APIs - Usando Modelos de Lenguaje Avanzados

Este koan explora los LLMs más modernos y sus APIs:
- OpenAI (GPT-4, GPT-4o, o1)
- Anthropic (Claude)
- Google (Gemini)
- Function calling
- Streaming
- Mejores prácticas

Librerías:
- openai
- anthropic
- google-generativeai
"""

from typing import List, Dict, Any, Generator, Optional
import os


def call_openai_chat(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> str:
    """
    Llama a la API de OpenAI Chat Completion.

    Ejemplo:
        >>> messages = [
        ...     {"role": "system", "content": "Eres un asistente útil."},
        ...     {"role": "user", "content": "¿Qué es Python?"}
        ... ]
        >>> response = call_openai_chat(messages)
        >>> print(response)

    Args:
        messages: Lista de mensajes con rol y contenido
        model: Modelo a usar (gpt-4o-mini, gpt-4o, gpt-4, o1-mini, o1-preview)
        temperature: Creatividad (0.0-2.0)
        max_tokens: Máximo de tokens en la respuesta

    Returns:
        Contenido de la respuesta del modelo
    """
    # TODO: Implementa la llamada a OpenAI
    # Pistas:
    # 1. from openai import OpenAI
    # 2. client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # 3. response = client.chat.completions.create(...)
    # 4. return response.choices[0].message.content
    pass


def call_openai_streaming(
    messages: List[Dict[str, str]], model: str = "gpt-4o-mini"
) -> Generator[str, None, None]:
    """
    Llama a OpenAI con streaming (respuesta en tiempo real).

    Ejemplo:
        >>> messages = [{"role": "user", "content": "Cuenta hasta 5"}]
        >>> for chunk in call_openai_streaming(messages):
        ...     print(chunk, end="", flush=True)

    Args:
        messages: Lista de mensajes
        model: Modelo a usar

    Yields:
        Fragmentos de texto conforme se generan
    """
    # TODO: Implementa streaming
    # Pistas:
    # 1. client.chat.completions.create(..., stream=True)
    # 2. for chunk in stream: yield chunk.choices[0].delta.content
    pass


def call_anthropic_claude(
    messages: List[Dict[str, str]],
    model: str = "claude-3-5-sonnet-20241022",
    max_tokens: int = 1000,
) -> str:
    """
    Llama a la API de Anthropic (Claude).

    Ejemplo:
        >>> messages = [{"role": "user", "content": "Explica qué es NLP"}]
        >>> response = call_anthropic_claude(messages)

    Args:
        messages: Lista de mensajes
        model: claude-3-5-sonnet-20241022, claude-3-opus-20240229, etc.
        max_tokens: Máximo de tokens

    Returns:
        Contenido de la respuesta
    """
    # TODO: Implementa llamada a Claude
    # Pistas:
    # 1. import anthropic
    # 2. client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    # 3. response = client.messages.create(...)
    # 4. return response.content[0].text
    pass


def call_google_gemini(prompt: str, model: str = "gemini-1.5-flash") -> str:
    """
    Llama a la API de Google Gemini.

    Ejemplo:
        >>> response = call_google_gemini("¿Qué es machine learning?")

    Args:
        prompt: Texto del prompt
        model: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-exp

    Returns:
        Respuesta del modelo
    """
    # TODO: Implementa llamada a Gemini
    # Pistas:
    # 1. import google.generativeai as genai
    # 2. genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    # 3. model = genai.GenerativeModel(model)
    # 4. response = model.generate_content(prompt)
    # 5. return response.text
    pass


def openai_function_calling(
    messages: List[Dict[str, str]],
    functions: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Usa function calling de OpenAI para que el modelo llame funciones.

    Ejemplo:
        >>> functions = [{
        ...     "name": "get_weather",
        ...     "description": "Obtiene el clima de una ciudad",
        ...     "parameters": {
        ...         "type": "object",
        ...         "properties": {
        ...             "city": {"type": "string"}
        ...         }
        ...     }
        ... }]
        >>> messages = [{"role": "user", "content": "¿Qué tiempo hace en Madrid?"}]
        >>> result = openai_function_calling(messages, functions)

    Args:
        messages: Lista de mensajes
        functions: Lista de definiciones de funciones
        model: Modelo a usar

    Returns:
        Dict con información de la función llamada
    """
    # TODO: Implementa function calling
    # Pistas:
    # 1. client.chat.completions.create(..., tools=[{"type": "function", "function": f} for f in functions])
    # 2. response.choices[0].message.tool_calls[0]
    # 3. return {"name": tool_call.function.name, "arguments": tool_call.function.arguments}
    pass


def calculate_token_cost(
    prompt_tokens: int, completion_tokens: int, model: str
) -> float:
    """
    Calcula el costo aproximado de una llamada a la API.

    Precios aproximados (Nov 2024):
    - gpt-4o: $2.50 / 1M input, $10 / 1M output
    - gpt-4o-mini: $0.15 / 1M input, $0.60 / 1M output
    - claude-3-5-sonnet: $3 / 1M input, $15 / 1M output
    - gemini-1.5-flash: $0.075 / 1M input, $0.30 / 1M output

    Ejemplo:
        >>> cost = calculate_token_cost(1000, 500, "gpt-4o-mini")
        >>> print(f"${cost:.4f}")

    Args:
        prompt_tokens: Tokens del prompt
        completion_tokens: Tokens de la respuesta
        model: Nombre del modelo

    Returns:
        Costo en dólares
    """
    # TODO: Implementa cálculo de costo
    # Pistas:
    # 1. Define diccionario con precios por modelo
    # 2. cost = (prompt_tokens / 1_000_000 * input_price) + (completion_tokens / 1_000_000 * output_price)
    pass


def compare_llm_outputs(
    prompt: str,
    models: List[str] = [
        "gpt-4o-mini",
        "claude-3-5-sonnet-20241022",
        "gemini-1.5-flash",
    ],
) -> Dict[str, str]:
    """
    Compara las respuestas de diferentes LLMs para el mismo prompt.

    Ejemplo:
        >>> results = compare_llm_outputs("Explica qué es un transformer en 2 líneas")
        >>> for model, response in results.items():
        ...     print(f"{model}: {response[:100]}...")

    Args:
        prompt: Texto del prompt
        models: Lista de modelos a comparar

    Returns:
        Dict con modelo: respuesta
    """
    # TODO: Implementa comparación
    # Pistas:
    # 1. Llama a cada API según el modelo
    # 2. Maneja errores con try/except
    # 3. return {model: response for model, response in ...}
    pass


def safe_llm_call(
    prompt: str, model: str = "gpt-4o-mini", max_retries: int = 3
) -> Optional[str]:
    """
    Llama a un LLM con manejo de errores y reintentos.

    Ejemplo:
        >>> response = safe_llm_call("Hola", max_retries=3)

    Args:
        prompt: Texto del prompt
        model: Modelo a usar
        max_retries: Número máximo de reintentos

    Returns:
        Respuesta o None si falla
    """
    # TODO: Implementa llamada segura
    # Pistas:
    # 1. Use for attempt in range(max_retries)
    # 2. try: llamar API, return response
    # 3. except Exception as e: log error, continuar
    # 4. time.sleep(2 ** attempt)  # exponential backoff
    pass
