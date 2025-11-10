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

    La API de OpenAI usa un formato de conversación con mensajes que tienen
    roles ('system', 'user', 'assistant'). El rol 'system' configura el
    comportamiento del asistente, 'user' son los mensajes del usuario, y
    'assistant' son las respuestas previas del modelo.

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
        temperature: Creatividad (0.0-2.0). Valores altos = más creativo/aleatorio
        max_tokens: Máximo de tokens en la respuesta

    Returns:
        Contenido de la respuesta del modelo

    Nota:
        Necesitas la variable de entorno OPENAI_API_KEY configurada.
        Consulta THEORY.md para entender los diferentes modelos disponibles.
    """
    # TODO: Implementa la llamada a OpenAI Chat API
    # Pista: Necesitas crear un cliente de OpenAI e invocar chat.completions
    # Consulta HINTS.md para detalles sobre la estructura de la respuesta
    pass


def call_openai_streaming(
    messages: List[Dict[str, str]], model: str = "gpt-4o-mini"
) -> Generator[str, None, None]:
    """
    Llama a OpenAI con streaming (respuesta en tiempo real).

    El streaming permite recibir la respuesta del modelo en fragmentos conforme
    se va generando, en lugar de esperar a que termine completamente. Esto mejora
    la experiencia del usuario al ver el texto aparecer progresivamente, similar
    a ChatGPT.

    Ejemplo:
        >>> messages = [{"role": "user", "content": "Cuenta hasta 5"}]
        >>> for chunk in call_openai_streaming(messages):
        ...     print(chunk, end="", flush=True)

    Args:
        messages: Lista de mensajes
        model: Modelo a usar

    Yields:
        Fragmentos de texto conforme se generan

    Nota:
        Esta función es un generador. Debes iterar sobre ella para obtener
        los fragmentos de texto. Cada fragmento puede ser una palabra, parte
        de una palabra, o varios caracteres.
    """
    # TODO: Implementa streaming de OpenAI
    # Pista: Activa streaming y procesa chunks en un loop
    # Consulta HINTS.md para entender la estructura de los chunks
    pass


def call_anthropic_claude(
    messages: List[Dict[str, str]],
    model: str = "claude-3-5-sonnet-20241022",
    max_tokens: int = 1000,
) -> str:
    """
    Llama a la API de Anthropic (Claude).

    Claude de Anthropic es conocido por su capacidad de seguir instrucciones
    complejas y mantener contextos largos. La API es similar a OpenAI pero
    con algunas diferencias en los parámetros y estructura de respuesta.

    Ejemplo:
        >>> messages = [{"role": "user", "content": "Explica qué es NLP"}]
        >>> response = call_anthropic_claude(messages)

    Args:
        messages: Lista de mensajes (mismo formato que OpenAI)
        model: claude-3-5-sonnet-20241022, claude-3-opus-20240229, etc.
        max_tokens: Máximo de tokens (requerido por la API de Anthropic)

    Returns:
        Contenido de la respuesta

    Nota:
        A diferencia de OpenAI, Anthropic requiere que especifiques max_tokens
        obligatoriamente. La estructura de la respuesta también es diferente.
        Consulta THEORY.md para comparar las APIs.
    """
    # TODO: Implementa llamada a Claude
    # Pista: La librería anthropic tiene un patrón diferente a OpenAI
    # Consulta HINTS.md para detalles de la API de Anthropic
    pass


def call_google_gemini(prompt: str, model: str = "gemini-1.5-flash") -> str:
    """
    Llama a la API de Google Gemini.

    Gemini es la familia de modelos de Google, conocidos por su velocidad
    y capacidad multimodal (texto, imágenes, video). La API tiene una
    interfaz más simple que OpenAI o Anthropic para casos de uso básicos.

    Ejemplo:
        >>> response = call_google_gemini("¿Qué es machine learning?")

    Args:
        prompt: Texto del prompt (interfaz más simple que formato de mensajes)
        model: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-exp

    Returns:
        Respuesta del modelo

    Nota:
        A diferencia de OpenAI y Anthropic, Gemini puede recibir un prompt
        simple en lugar de una lista de mensajes (aunque también soporta
        conversaciones). Ver THEORY.md para más detalles sobre las diferencias.
    """
    # TODO: Implementa llamada a Gemini
    # Pista: google.generativeai tiene un patrón diferente
    # Consulta HINTS.md para la configuración de Gemini
    pass


def openai_function_calling(
    messages: List[Dict[str, str]],
    functions: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Usa function calling de OpenAI para que el modelo llame funciones.

    Function calling permite que el LLM decida cuándo y cómo llamar funciones
    externas basándose en el contexto de la conversación. Es fundamental para
    crear agentes que puedan interactuar con APIs, bases de datos, o cualquier
    herramienta externa.

    El modelo analiza el mensaje del usuario y decide si necesita llamar alguna
    función. Si es así, devuelve los parámetros necesarios en formato estructurado.

    Ejemplo:
        >>> functions = [{
        ...     "name": "get_weather",
        ...     "description": "Obtiene el clima de una ciudad",
        ...     "parameters": {
        ...         "type": "object",
        ...         "properties": {
        ...             "city": {"type": "string", "description": "Nombre de la ciudad"}
        ...         },
        ...         "required": ["city"]
        ...     }
        ... }]
        >>> messages = [{"role": "user", "content": "¿Qué tiempo hace en Madrid?"}]
        >>> result = openai_function_calling(messages, functions)
        >>> # result contendrá: {"name": "get_weather", "arguments": '{"city": "Madrid"}'}

    Args:
        messages: Lista de mensajes
        functions: Lista de definiciones de funciones en formato JSON Schema
        model: Modelo a usar (debe soportar function calling)

    Returns:
        Dict con 'name' (nombre de la función) y 'arguments' (argumentos en JSON)

    Nota:
        La API devuelve los argumentos como string JSON que necesitas parsear.
        El modelo NO ejecuta la función, solo te dice qué función llamar y con
        qué argumentos. Ver THEORY.md para entender el flujo completo.
    """
    # TODO: Implementa function calling
    # Pista: Usa el formato 'tools' y procesa 'tool_calls' en la respuesta
    # Consulta HINTS.md para la estructura completa
    pass


def calculate_token_cost(
    prompt_tokens: int, completion_tokens: int, model: str
) -> float:
    """
    Calcula el costo aproximado de una llamada a la API.

    Los LLMs comerciales cobran por tokens (piezas de texto). Generalmente
    el costo de generar tokens (output) es más caro que leer tokens (input).
    Entender los costos es crucial para optimizar aplicaciones en producción.

    Precios aproximados (Nov 2024) por 1M de tokens:
    - gpt-4o: $2.50 input, $10.00 output
    - gpt-4o-mini: $0.15 input, $0.60 output
    - claude-3-5-sonnet: $3.00 input, $15.00 output
    - gemini-1.5-flash: $0.075 input, $0.30 output

    Ejemplo:
        >>> cost = calculate_token_cost(1000, 500, "gpt-4o-mini")
        >>> print(f"${cost:.4f}")  # Aprox $0.0004

    Args:
        prompt_tokens: Número de tokens en el input/prompt
        completion_tokens: Número de tokens en el output/respuesta
        model: Nombre del modelo usado

    Returns:
        Costo estimado en dólares (USD)

    Nota:
        Estos precios son aproximados y pueden cambiar. Verifica los precios
        actuales en las páginas oficiales de cada proveedor. Ver THEORY.md
        para entender cómo se calculan los tokens.
    """
    # TODO: Implementa cálculo de costo
    # Pista: Crea un diccionario con precios por modelo
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

    Diferentes modelos pueden dar respuestas variadas en estilo, longitud,
    y enfoque. Comparar múltiples modelos ayuda a:
    - Validar respuestas críticas
    - Elegir el mejor modelo para tu caso de uso
    - Detectar sesgos o limitaciones de un modelo específico

    Ejemplo:
        >>> results = compare_llm_outputs("Explica qué es un transformer en 2 líneas")
        >>> for model, response in results.items():
        ...     print(f"\n{model}:\n{response[:100]}...")

    Args:
        prompt: Texto del prompt a enviar a todos los modelos
        models: Lista de nombres de modelos a comparar

    Returns:
        Dict con modelo: respuesta para cada modelo exitoso

    Nota:
        Esta función debe manejar errores gracefully. Si un modelo falla
        (por falta de API key, rate limit, etc.), debe continuar con los demás
        y registrar el error. Ver HINTS.md para estrategias de error handling.
    """
    # TODO: Implementa comparación multi-modelo
    # Pista: Identifica qué API llamar según el nombre del modelo
    # Usa try/except para manejar errores independientemente
    pass


def safe_llm_call(
    prompt: str, model: str = "gpt-4o-mini", max_retries: int = 3
) -> Optional[str]:
    """
    Llama a un LLM con manejo robusto de errores y reintentos.

    En producción, las llamadas a APIs pueden fallar por múltiples razones:
    - Rate limits (demasiadas peticiones)
    - Timeouts de red
    - Errores del servidor (500, 503)
    - Errores de autenticación

    Una estrategia de reintentos con backoff exponencial mejora la resiliencia.

    Ejemplo:
        >>> response = safe_llm_call("Hola, ¿cómo estás?", max_retries=3)
        >>> if response:
        ...     print(response)
        ... else:
        ...     print("Falló después de todos los reintentos")

    Args:
        prompt: Texto del prompt
        model: Modelo a usar
        max_retries: Número máximo de intentos

    Returns:
        Respuesta del modelo o None si todos los intentos fallan

    Nota:
        Implementa backoff exponencial: espera 1s, 2s, 4s, etc. entre reintentos.
        Esto evita saturar la API y respeta los rate limits. Ver THEORY.md
        para mejores prácticas de resiliencia.
    """
    # TODO: Implementa llamada con reintentos
    # Pista: Usa backoff exponencial con time.sleep(2 ** attempt)
    pass
