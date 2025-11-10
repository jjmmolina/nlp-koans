# 💡 Pistas para Koan 10: Modern LLMs & APIs

## 🎯 Objetivo del Koan

Aprender a usar los **LLMs más modernos** a través de sus APIs:
- OpenAI (GPT-4, GPT-4o)
- Anthropic (Claude)
- Google (Gemini)
- Function calling
- Streaming
- Gestión de costos

---

## 📝 Función 1: `call_openai_chat()`

### Nivel 1: Concepto
La API de OpenAI usa el formato de "chat completions" con mensajes de sistema, usuario y asistente.

### Nivel 2: Implementación
```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=temperature,
    max_tokens=max_tokens
)
return response.choices[0].message.content
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def call_openai_chat(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> str:
    from openai import OpenAI
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content
```
</details>

---

## 📝 Función 2: `call_openai_streaming()`

### Nivel 1: Concepto
Streaming permite recibir la respuesta en tiempo real, token por token, mejorando UX.

### Nivel 2: Implementación
```python
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

stream = client.chat.completions.create(
    model=model,
    messages=messages,
    stream=True  # ¡Clave!
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        yield chunk.choices[0].delta.content
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def call_openai_streaming(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini"
) -> Generator[str, None, None]:
    from openai import OpenAI
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```
</details>

---

## 📝 Función 3: `call_anthropic_claude()`

### Nivel 1: Concepto
Claude tiene una API similar pero con algunas diferencias (no hay rol "system" separado en versiones antiguas).

### Nivel 2: Implementación
```python
import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model=model,
    max_tokens=max_tokens,
    messages=messages
)

return response.content[0].text
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def call_anthropic_claude(
    messages: List[Dict[str, str]],
    model: str = "claude-3-5-sonnet-20241022",
    max_tokens: int = 1000
) -> str:
    import anthropic
    
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages
    )
    
    return response.content[0].text
```
</details>

---

## 📝 Función 4: `call_google_gemini()`

### Nivel 1: Concepto
Gemini de Google tiene una API más simple para casos básicos.

### Nivel 2: Implementación
```python
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(model)
response = model.generate_content(prompt)
return response.text
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def call_google_gemini(
    prompt: str,
    model: str = "gemini-1.5-flash"
) -> str:
    import google.generativeai as genai
    
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    model_instance = genai.GenerativeModel(model)
    response = model_instance.generate_content(prompt)
    
    return response.text
```
</details>

---

## 📝 Función 5: `openai_function_calling()`

### Nivel 1: Concepto
Function calling permite que el LLM decida llamar funciones con parámetros específicos.

### Nivel 2: Formato de tools
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "function_name",
            "description": "What it does",
            "parameters": {...}
        }
    }
]
```

### Nivel 3: Implementación
```python
from openai import OpenAI
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

tools = [{"type": "function", "function": f} for f in functions]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools
)

tool_call = response.choices[0].message.tool_calls[0]
return {
    "name": tool_call.function.name,
    "arguments": tool_call.function.arguments
}
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def openai_function_calling(
    messages: List[Dict[str, str]],
    functions: List[Dict[str, Any]],
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    from openai import OpenAI
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Convertir functions a formato tools
    tools = [{"type": "function", "function": f} for f in functions]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools
    )
    
    # Extraer la primera tool call
    message = response.choices[0].message
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        return {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments
        }
    
    return {}
```
</details>

---

## 📝 Función 6: `calculate_token_cost()`

### Nivel 1: Concepto
Cada modelo tiene diferentes precios por token de entrada (prompt) y salida (completion).

### Nivel 2: Precios (Nov 2024)
```python
PRICES = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30}
}
# Precios por 1M tokens
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def calculate_token_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str
) -> float:
    # Precios por 1M tokens (Nov 2024)
    PRICES = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30}
    }
    
    if model not in PRICES:
        return 0.0
    
    input_cost = (prompt_tokens / 1_000_000) * PRICES[model]["input"]
    output_cost = (completion_tokens / 1_000_000) * PRICES[model]["output"]
    
    return input_cost + output_cost
```
</details>

---

## 📝 Función 7: `compare_llm_outputs()`

### Nivel 1: Concepto
Comparar respuestas de diferentes modelos ayuda a elegir el mejor para tu caso de uso.

### Nivel 2: Manejo de APIs
```python
results = {}
messages = [{"role": "user", "content": prompt}]

if "gpt" in model:
    results[model] = call_openai_chat(messages, model=model)
elif "claude" in model:
    results[model] = call_anthropic_claude(messages, model=model)
elif "gemini" in model:
    results[model] = call_google_gemini(prompt, model=model)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def compare_llm_outputs(
    prompt: str,
    models: List[str] = ["gpt-4o-mini", "claude-3-5-sonnet-20241022", "gemini-1.5-flash"]
) -> Dict[str, str]:
    results = {}
    messages = [{"role": "user", "content": prompt}]
    
    for model in models:
        try:
            if "gpt" in model or "o1" in model:
                results[model] = call_openai_chat(messages, model=model, max_tokens=500)
            elif "claude" in model:
                results[model] = call_anthropic_claude(messages, model=model, max_tokens=500)
            elif "gemini" in model:
                results[model] = call_google_gemini(prompt, model=model)
            else:
                results[model] = f"Unknown model type: {model}"
        except Exception as e:
            results[model] = f"Error: {str(e)}"
    
    return results
```
</details>

---

## 📝 Función 8: `safe_llm_call()`

### Nivel 1: Concepto
Las llamadas a APIs pueden fallar (rate limits, timeouts, errores de red). Necesitamos manejo robusto.

### Nivel 2: Exponential Backoff
```python
import time

for attempt in range(max_retries):
    try:
        # Intentar llamada
        return response
    except Exception as e:
        if attempt == max_retries - 1:
            return None
        wait_time = 2 ** attempt  # 1s, 2s, 4s...
        time.sleep(wait_time)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def safe_llm_call(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 3
) -> Optional[str]:
    import time
    
    messages = [{"role": "user", "content": prompt}]
    
    for attempt in range(max_retries):
        try:
            if "gpt" in model or "o1" in model:
                return call_openai_chat(messages, model=model)
            elif "claude" in model:
                return call_anthropic_claude(messages, model=model)
            elif "gemini" in model:
                return call_google_gemini(prompt, model=model)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt == max_retries - 1:
                return None
            
            # Exponential backoff
            wait_time = 2 ** attempt
            time.sleep(wait_time)
    
    return None
```
</details>

---

## 🎯 Conceptos Clave

### Modelos Principales (Nov 2024)

| Proveedor | Modelo | Contexto | Mejor para |
|-----------|--------|----------|------------|
| **OpenAI** | gpt-4o | 128K | Balance precio/calidad |
| | gpt-4o-mini | 128K | Tareas simples, económico |
| | o1-preview | 128K | Razonamiento complejo |
| **Anthropic** | claude-3-5-sonnet | 200K | Código, análisis largo |
| | claude-3-opus | 200K | Tareas complejas |
| **Google** | gemini-1.5-pro | 2M | Contexto ultra-largo |
| | gemini-2.0-flash-exp | 1M | Rápido, multimodal |

### Parámetros Importantes

**Temperature** (0.0-2.0):
- `0.0-0.3`: Determinista, factual
- `0.7-1.0`: Balanceado (default)
- `1.5-2.0`: Creativo, aleatorio

**Max Tokens**:
- Controla longitud de respuesta
- No confundir con window de contexto

**Top-p** (Nucleus Sampling):
- Alternativa a temperature
- 0.9 = considera tokens hasta 90% prob acumulada

### Function Calling

Permite que el LLM:
1. **Detecte** cuándo necesita una herramienta
2. **Extraiga** parámetros del contexto
3. **Devuelva** JSON con la llamada a función

**Ejemplo**:
```python
User: "¿Qué tiempo hace en Madrid?"
LLM: {"name": "get_weather", "arguments": {"city": "Madrid"}}
```

## 💡 Tips Prácticos

### 1. Optimiza Costos

```python
# Usa modelos mini para tareas simples
model = "gpt-4o-mini"  # vs "gpt-4o"

# Limita tokens de salida
max_tokens = 100  # vs 4000

# Usa caching de prompts (Claude, Gemini)
```

### 2. Maneja Rate Limits

```python
import time
from functools import wraps

def rate_limit(calls_per_minute=60):
    interval = 60.0 / calls_per_minute
    
    def decorator(func):
        last_called = [0.0]
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        
        return wrapper
    return decorator

@rate_limit(calls_per_minute=30)
def my_llm_call():
    ...
```

### 3. System Prompts Efectivos

```python
# Malo: vago
"Eres un asistente útil"

# Bueno: específico
"""Eres un experto en Python. 
Proporciona código limpio, bien documentado.
Explica decisiones de diseño.
Usa type hints siempre."""
```

### 4. Structured Outputs

```python
# OpenAI response_format (beta)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    response_format={"type": "json_object"}
)
```

## 🚀 Casos de Uso

### Chatbot Conversacional
```python
conversation_history = []

while True:
    user_input = input("User: ")
    conversation_history.append({"role": "user", "content": user_input})
    
    response = call_openai_chat(conversation_history)
    print(f"Bot: {response}")
    
    conversation_history.append({"role": "assistant", "content": response})
```

### Extracción de Información con Function Calling
```python
functions = [{
    "name": "save_contact",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "phone": {"type": "string"}
        }
    }
}]

text = "Contacta con Juan Pérez al juan@example.com o al 555-1234"
result = openai_function_calling([{"role": "user", "content": text}], functions)
# Extrae automáticamente nombre, email, teléfono
```

### Comparación de Modelos
```python
prompt = "Explica qué es un transformer en 3 líneas"
results = compare_llm_outputs(prompt)

for model, response in results.items():
    print(f"\n{model}:")
    print(response)
    print(f"Cost: ${calculate_token_cost(100, 50, model):.4f}")
```

## 🔧 Troubleshooting

### Problema: API Key inválida
**Solución**:
```bash
# Linux/Mac
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."

# Python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

### Problema: Rate limit exceeded
**Solución**:
- Implementa exponential backoff
- Usa tier más alto (paga más)
- Distribuye requests en el tiempo

### Problema: Respuestas inconsistentes
**Solución**:
- Baja temperature (0.0-0.3)
- Usa seed parameter (OpenAI)
- Mejora system prompt

### Problema: Costo muy alto
**Solución**:
- Usa modelos mini
- Reduce max_tokens
- Cachea respuestas comunes
- Implementa presupuesto por usuario

## 📚 Recursos

- **OpenAI Docs**: https://platform.openai.com/docs
- **Anthropic Docs**: https://docs.anthropic.com
- **Google AI Docs**: https://ai.google.dev/docs
- **LLM Pricing**: https://artificialanalysis.ai/models

## 🚀 Siguiente Paso

Una vez completo, ve al **Koan 11: AI Agents**!
