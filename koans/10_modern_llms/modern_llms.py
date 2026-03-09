"""
Koan 10: Modern LLMs & APIs - Usando Modelos de Lenguaje Avanzados

Este koan explora los LLMs más modernos y sus APIs:
- OpenAI (GPT-4.1, GPT-4o, o3, o4-mini)
- Anthropic (Claude 3.7 Sonnet, extended thinking)
- Google (Gemini 2.5 Pro/Flash)
- Meta / Open Source (Llama 4, DeepSeek)
- Function calling y Structured Outputs
- Streaming
- Reasoning Models
- OpenAI Responses API
- Mejores prácticas

Librerías:
- openai
- anthropic
- google-generativeai
"""

from typing import List, Dict, Any, Generator, Optional, Type
import os


def call_openai_chat(
    messages: List[Dict[str, str]],
    model: str = "gpt-4.1-mini",
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
        model: Modelo a usar (gpt-4.1-mini, gpt-4.1, gpt-4o, o3-mini, o4-mini)
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
    messages: List[Dict[str, str]], model: str = "gpt-4.1-mini"
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
    model: str = "claude-3-7-sonnet-20250219",
    max_tokens: int = 1000,
) -> str:
    """
    Llama a la API de Anthropic (Claude).

    Claude de Anthropic es conocido por su capacidad de seguir instrucciones
    complejas y mantener contextos largos. La API es similar a OpenAI pero
    con algunas diferencias en los parámetros y estructura de respuesta.

    Example:
        >>> messages = [{"role": "user", "content": "Explica qué es NLP"}]
        >>> response = call_anthropic_claude(messages)

    Args:
        messages: Lista de mensajes (mismo formato que OpenAI)
        model: claude-3-7-sonnet-20250219, claude-opus-4-20250514, claude-3-5-haiku-20241022
        max_tokens: Máximo de tokens (requerido por la API de Anthropic)

    Returns:
        Contenido de la respuesta

    Nota:
        A diferencia de OpenAI, Anthropic requiere que especifiques max_tokens
        obligatoriamente. La estructura de la respuesta también es diferente.
        Consulta THEORY.md para comparar las APIs.
        Claude 3.7 Sonnet soporta "extended thinking" para razonamiento profundo.
    """
    # TODO: Implementa llamada a Claude
    # Pista: La librería anthropic tiene un patrón diferente a OpenAI
    # Consulta HINTS.md para detalles de la API de Anthropic
    pass


def call_google_gemini(prompt: str, model: str = "gemini-2.0-flash") -> str:
    """
    Llama a la API de Google Gemini.

    Gemini es la familia de modelos de Google, conocidos por su velocidad
    y capacidad multimodal (texto, imágenes, video). La API tiene una
    interfaz más simple que OpenAI o Anthropic para casos de uso básicos.

    Ejemplo:
        >>> response = call_google_gemini("¿Qué es machine learning?")

    Args:
        prompt: Texto del prompt (interfaz más simple que formato de mensajes)
        model: gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash, gemini-2.0-flash-thinking

    Returns:
        Respuesta del modelo

    Nota:
        A diferencia de OpenAI y Anthropic, Gemini puede recibir un prompt
        simple en lugar de una lista de mensajes (aunque también soporta
        conversaciones). Ver THEORY.md para más detalles sobre las diferencias.
        Gemini 2.5 Pro tiene ventanas de contexto masivas (hasta 2M tokens).
    """
    # TODO: Implementa llamada a Gemini
    # Pista: google.generativeai tiene un patrón diferente
    # Consulta HINTS.md para la configuración de Gemini
    pass


def openai_function_calling(
    messages: List[Dict[str, str]],
    functions: List[Dict[str, Any]],
    model: str = "gpt-4.1-mini",
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

    Precios aproximados (2026) por 1M de tokens:
    - gpt-4.1: $2.00 input, $8.00 output
    - gpt-4.1-mini: $0.40 input, $1.60 output
    - gpt-4o: $2.50 input, $10.00 output
    - gpt-4o-mini: $0.15 input, $0.60 output
    - o3: $10.00 input, $40.00 output
    - o4-mini: $1.10 input, $4.40 output
    - claude-3-7-sonnet: $3.00 input, $15.00 output
    - claude-3-5-haiku: $0.80 input, $4.00 output
    - gemini-2.5-pro: $1.25 input, $10.00 output
    - gemini-2.0-flash: $0.10 input, $0.40 output

    Ejemplo:
        >>> cost = calculate_token_cost(1000, 500, "gpt-4.1-mini")
        >>> print(f"${cost:.4f}")  # Aprox $0.0012

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
        "gpt-4.1-mini",
        "claude-3-7-sonnet-20250219",
        "gemini-2.0-flash",
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


def call_reasoning_model(
    prompt: str,
    model: str = "o4-mini",
    effort: str = "medium",
) -> Dict[str, Any]:
    """
    Llama a un modelo de razonamiento (OpenAI o3/o4, Claude extended thinking).

    Los modelos de razonamiento "piensan antes de responder" usando una cadena
    de razonamiento interna (Chain-of-Thought). Son especialmente útiles para:
    - Matemáticas y lógica compleja
    - Problemas de programación difíciles
    - Razonamiento multi-paso
    - Análisis de argumentos

    **Modelos de razonamiento OpenAI:**
    - o3: El más potente, costoso pero muy preciso
    - o4-mini: Balance razonamiento/velocidad/costo
    - o3-mini: Deprecated, usar o4-mini

    **Esfuerzo de razonamiento (effort):**
    - 'low': Razonamiento mínimo (más rápido/barato)
    - 'medium': Balance calidad/costo
    - 'high': Razonamiento profundo (más lento/caro)

    Ejemplo:
        >>> result = call_reasoning_model(
        ...     "¿Cuántos ceros tiene 100! (factorial)?",
        ...     model="o4-mini",
        ...     effort="high"
        ... )
        >>> print(result["output"])  # "24 ceros"
        >>> print(result["reasoning_tokens"])  # tokens de razonamiento usados

    Args:
        prompt: Problema o pregunta a resolver
        model: Modelo de razonamiento ('o3', 'o4-mini', 'o3-mini')
        effort: Nivel de esfuerzo ('low', 'medium', 'high')

    Returns:
        Dict con 'output' (respuesta final) y 'reasoning_tokens' (tokens usados en pensar)

    Nota:
        Los modelos de razonamiento no soportan 'temperature' (siempre usan 1).
        El 'system' message se incluye diferente. Consulta HINTS.md para detalles.
    """
    # TODO: Implementa llamada a modelo de razonamiento
    # Pista: Usa reasoning_effort en place de temperature
    # Los modelos 'o' no aceptan parámetro temperature
    pass


def call_with_structured_output(
    messages: List[Dict[str, str]],
    schema: Dict[str, Any],
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    Obtiene respuesta estructurada (JSON) garantizada del LLM.

    Structured Outputs garantizan que el modelo devuelva exactamente el
    formato JSON Schema especificado — sin necesidad de parsear texto libre.
    Esto es crucial para aplicaciones que procesan datos estructurados.

    **Diferencia con JSON mode:**
    - JSON mode: Garantiza JSON válido, pero no el schema
    - Structured Outputs: Garantiza que cumple el schema exactamente

    Ejemplo:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        ...         "score": {"type": "number"},
        ...         "explanation": {"type": "string"}
        ...     },
        ...     "required": ["sentiment", "score", "explanation"]
        ... }
        >>> messages = [{"role": "user", "content": "Analiza: 'Me encanta Python'"}]
        >>> result = call_with_structured_output(messages, schema)
        >>> print(result["sentiment"])  # "positive"
        >>> print(result["score"])      # 0.95

    Args:
        messages: Lista de mensajes de la conversación
        schema: JSON Schema que define la estructura esperada
        model: Modelo a usar (gpt-4.1 o gpt-4o recomendados)

    Returns:
        Diccionario que cumple exactamente el schema especificado

    Nota:
        Solo disponible para modelos OpenAI que soporten Structured Outputs.
        La librería `instructor` facilita aún más esto con tipos Pydantic.
        Ver HINTS.md para implementación con Pydantic y instructor.
    """
    # TODO: Implementa structured outputs
    # Pista: Usa response_format con json_schema
    # Recuerda hacer json.loads() del content para obtener el dict
    pass


def call_openai_responses_api(
    input_text: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    model: str = "gpt-4.1",
    store: bool = True,
) -> Dict[str, Any]:
    """
    Usa la nueva OpenAI Responses API (lanzada en 2025).

    La Responses API es el reemplazo de la API de Chat Completions para
    aplicaciones agénticas. Características principales:
    - Estado persistente del servidor (no necesitas reenviar historial)
    - Herramientas built-in: web_search_preview, file_search, code_interpreter
    - Diseñada para workflows multi-paso
    - Computer use (control de navegador/computadora)

    **Herramientas built-in disponibles:**
    - web_search_preview: Búsqueda web en tiempo real
    - file_search: Búsqueda en vectorstores
    - code_interpreter: Ejecuta código Python

    Ejemplo:
        >>> result = call_openai_responses_api(
        ...     "¿Cuál es la noticia más reciente sobre IA?",
        ...     tools=[{"type": "web_search_preview"}]
        ... )
        >>> print(result["output_text"])
        >>> print(result["response_id"])  # Para continuar la conversación

    Args:
        input_text: Texto de entrada del usuario
        tools: Lista de herramientas (web_search_preview, file_search, etc.)
        model: Modelo a usar (gpt-4.1, gpt-4o)
        store: Si almacenar el estado en el servidor (para conversación multi-turno)

    Returns:
        Dict con 'output_text' (respuesta) y 'response_id' (para continuar)

    Nota:
        La Responses API reemplaza gradualmente a Chat Completions para uso agéntico.
        El 'response_id' permite continuar conversaciones sin reenviar historial.
        Ver THEORY.md para comparar ambas APIs.
    """
    # TODO: Implementa la Responses API de OpenAI
    # Pista: Usa client.responses.create() en lugar de client.chat.completions.create()
    pass


def call_claude_extended_thinking(
    prompt: str,
    budget_tokens: int = 5000,
    model: str = "claude-3-7-sonnet-20250219",
) -> Dict[str, Any]:
    """
    Usa el modo "Extended Thinking" de Claude 3.7 Sonnet.

    Extended Thinking permite a Claude razonar de forma más profunda antes
    de responder, similar a los modelos de razonamiento de OpenAI pero con
    la ventaja de que puedes ver el proceso de pensamiento completo.

    **Extended Thinking vs respuesta normal:**
    - Normal: Claude responde directamente
    - Extended Thinking: Claude "piensa en voz alta" antes de responder
    - El 'thinking' está disponible en la respuesta para auditoría

    **Cuándo usar Extended Thinking:**
    - Problemas matemáticos o lógicos complejos
    - Análisis de código difícil
    - Decisiones con múltiples factores
    - Cualquier tarea donde la precisión es crítica

    Ejemplo:
        >>> result = call_claude_extended_thinking(
        ...     "Demuestra por qué sqrt(2) es irracional",
        ...     budget_tokens=8000
        ... )
        >>> print(result["thinking"])   # El razonamiento interno
        >>> print(result["response"])   # La respuesta final

    Args:
        prompt: Problema o pregunta compleja
        budget_tokens: Tokens máximos para el proceso de pensamiento (1024-100000)
        model: Modelo Claude que soporte extended thinking

    Returns:
        Dict con 'thinking' (proceso de razonamiento) y 'response' (respuesta final)

    Nota:
        El thinking NO se envía al usuario por defecto — es para debugging.
        Cuantos más budget_tokens, más profundo el razonamiento (y mayor costo).
        Ver HINTS.md para la estructura exacta de la respuesta.
    """
    # TODO: Implementa extended thinking de Claude
    # Pista: Activa betas y usa thinking={"type": "enabled", "budget_tokens": ...}
    pass


def call_local_llm(
    messages: List[Dict[str, str]],
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
) -> str:
    """
    Llama a un LLM local usando Ollama (alternativa gratuita sin API key).

    Ollama permite ejecutar modelos de código abierto localmente:
    - Llama 4 (Meta) - último modelo de Meta, multimodal
    - Llama 3.3 70B - excelente calidad de razonamiento
    - DeepSeek-R1 - modelo de razonamiento open source
    - Qwen 2.5 - modelos eficientes de Alibaba
    - Mistral - modelos eficientes y rápidos
    - Phi-4 - modelo compacto de Microsoft

    **Ventajas de modelos locales:**
    ✅ Gratuito (solo coste de hardware)
    ✅ Privacidad total (datos no salen)
    ✅ Sin rate limits
    ✅ Funciona sin internet

    **Instalación:**
    ```bash
    # Windows/Mac: Descarga de https://ollama.ai
    ollama pull llama3.2
    ollama serve  # Inicia el servidor en localhost:11434
    ```

    Ejemplo:
        >>> messages = [{"role": "user", "content": "¿Qué es NLP?"}]
        >>> response = call_local_llm(messages, model="llama3.2")
        >>> print(response)

    Args:
        messages: Lista de mensajes (mismo formato API)
        model: Nombre del modelo en Ollama (llama3.2, llama3.3, deepseek-r1, etc.)
        base_url: URL del servidor Ollama local

    Returns:
        Respuesta del modelo local

    Nota:
        Requiere tener Ollama instalado y el modelo descargado.
        La API es compatible con OpenAI (usa openai client con base_url).
        Ver HINTS.md para la configuración completa.
    """
    # TODO: Implementa llamada a Ollama local
    # Pista: Usa cliente OpenAI con base_url apuntando a Ollama
    # Ollama expone una API compatible con OpenAI en /v1
    pass
