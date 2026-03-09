# Teoría: Modern LLMs & APIs

## 📚 Tabla de Contenidos
1. [Introducción a los LLMs Modernos](#introducción)
2. [Arquitectura de Transformers](#arquitectura)
3. [Principales Proveedores y Modelos](#proveedores)
4. [Anatomía de una Llamada a API](#llamadas-api)
5. [Streaming de Respuestas](#streaming)
6. [Function Calling y Structured Outputs](#function-calling)
7. [Modelos de Razonamiento](#reasoning-models)
8. [OpenAI Responses API](#responses-api)
9. [Extended Thinking (Claude)](#extended-thinking)
10. [LLMs Locales y Open Source](#open-source)
11. [Tokens y Costos](#tokens-costos)
12. [Mejores Prácticas](#mejores-prácticas)

---

## 🌟 Introducción a los LLMs Modernos {#introducción}

### ¿Qué es un LLM?

Un **Large Language Model** (Modelo de Lenguaje Grande) es una red neuronal entrenada con billones de palabras para:
- Entender lenguaje natural
- Generar texto coherente
- Seguir instrucciones complejas
- Razonar sobre información
- Traducir, resumir, programar, y más

### Evolución de los LLMs

```
2018: GPT-1 (117M parámetros)
  ↓
2019: GPT-2 (1.5B parámetros)
  ↓
2020: GPT-3 (175B parámetros) ← Primera API comercial
  ↓
2022: ChatGPT (basado en GPT-3.5)
  ↓
2023: GPT-4, Claude 2 & 3, Gemini, Llama 2
  ↓
2024: GPT-4o, o1, Claude 3.5, Gemini 1.5/2.0
      Modelos multimodales, Function Calling avanzado
  ↓
2025: GPT-4.1, o3, o4-mini, Claude 3.7 (Extended Thinking)
      Gemini 2.5 Pro, Llama 4, DeepSeek V3/R1
      OpenAI Responses API, MCP Protocol
  ↓
2026: Modelos de razonamiento mainstream
      Agentes autónomos en producción
      Multimodalidad: texto, imagen, audio, video
```

### ¿Por qué usar APIs de LLMs?

**Ventajas:**
- ✅ No necesitas infraestructura de GPU
- ✅ Modelos de última generación listos para usar
- ✅ Escalado automático
- ✅ Actualizaciones constantes
- ✅ Pago por uso (no costos fijos)

**Desventajas:**
- ❌ Costo por token
- ❌ Latencia de red
- ❌ Dependencia de terceros
- ❌ Limitaciones de privacidad (tus datos van al proveedor)

---

## 🏗️ Arquitectura de Transformers {#arquitectura}

### El Transformer

Todos los LLMs modernos están basados en la arquitectura **Transformer** (Vaswani et al., 2017):

```
Input Text → Tokenization → Embeddings → Transformer Layers → Output Logits → Text
```

#### Componentes Clave:

**1. Self-Attention**
- Permite al modelo "prestar atención" a diferentes partes del texto
- Captura relaciones entre palabras distantes
- Ejemplo: En "El gato que María alimentó estaba contento", "estaba" se relaciona con "gato"

**2. Multi-Head Attention**
- Múltiples mecanismos de atención en paralelo
- Cada "cabeza" aprende diferentes tipos de relaciones
- Típicamente 8-96 cabezas por capa

**3. Feed-Forward Networks**
- Redes neuronales densas después de cada capa de atención
- Procesan la información agregada

**4. Layer Normalization & Residual Connections**
- Estabilizan el entrenamiento
- Permiten entrenar modelos muy profundos (hasta 100+ capas)

### Tipos de Modelos

**Decoder-Only (como GPT)**
- Solo predicen el siguiente token
- Unidireccionales (solo ven el pasado)
- Óptimos para generación de texto
- Ejemplos: GPT-4, Claude, Gemini

**Encoder-Only (como BERT)**
- Procesamiento bidireccional
- Óptimos para clasificación y comprensión
- Ejemplos: BERT, RoBERTa

**Encoder-Decoder (como T5)**
- Combinan ambos
- Óptimos para traducción y resumen
- Ejemplos: T5, BART

---

## 🏢 Principales Proveedores y Modelos {#proveedores}

### OpenAI

**Historia:**
- Fundada en 2015 por Sam Altman, Elon Musk, y otros
- Popularizó los LLMs con ChatGPT (Nov 2022)
- Líder del mercado en LLMs comerciales

**Modelos Actuales (2026):**

| Modelo | Velocidad | Costo | Mejor Para |
|--------|-----------|-------|------------|
| **GPT-4.1** | ⚡⚡⚡ | 💰💰 | General, instrucciones complejas |
| **GPT-4.1-mini** | ⚡⚡⚡⚡ | 💰 | Tareas simples, bajo costo |
| **GPT-4o** | ⚡⚡⚡ | 💰💰 | Multimodal (texto + imagen) |
| **GPT-4o-mini** | ⚡⚡⚡⚡ | 💰 | Multimodal económico |
| **o3** | ⚡ | 💰💰💰💰 | Razonamiento profundo, STEM |
| **o4-mini** | ⚡⚡⚡ | 💰💰 | Razonamiento + velocidad |

**Modelos de Razonamiento (serie 'o'):**
- Razonan antes de responder (Chain-of-Thought interno)
- Especialmente buenos en matemáticas, código, análisis lógico
- Más lentos y caros, pero más precisos en tareas complejas
- Usan `reasoning_effort` en lugar de `temperature`

**Características Únicas:**
- 🎯 Function calling + Structured Outputs
- 🖼️ Visión nativa (análisis de imágenes)
- 🌐 Responses API con herramientas built-in (web search, code interpreter)
- 🎙️ Audio: Whisper (transcripción) y TTS
- 🎨 DALL-E 3 para generación de imágenes
- 📊 Batch API (50% descuento para procesamiento async)

### Anthropic

**Historia:**
- Fundada en 2021 por ex-empleados de OpenAI
- Enfocados en "AI segura y confiable"
- Conocidos por Claude

**Modelos Claude (2026):**

| Modelo | Contexto | Velocidad | Costo | Mejor Para |
|--------|----------|-----------|-------|------------|
| **Claude 3.7 Sonnet** | 200K | ⚡⚡⚡ | 💰💰 | Balance + Extended Thinking |
| **Claude Opus 4** | 200K | ⚡⚡ | 💰💰💰 | Máxima calidad razonamiento |
| **Claude 3.5 Haiku** | 200K | ⚡⚡⚡⚡ | 💰 | Velocidad y bajo costo |

**Extended Thinking (Claude 3.7+):**
Claude 3.7 Sonnet introdujo **Extended Thinking** — el modelo "piensa en voz alta"
antes de responder. Similar a los modelos de razonamiento de OpenAI, con la ventaja
de que el proceso de pensamiento es visible y auditable.

```python
# Activar Extended Thinking
client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": "Problema complejo..."}]
)
```

**Características Únicas:**
- 📖 Ventana de contexto masiva (200K tokens = ~150K palabras)
- 🧠 Extended Thinking para razonamiento auditable
- 🔒 Énfasis en seguridad (Constitutional AI)
- 🛠️ Tool use paralelo (llama múltiples tools a la vez)
- 📊 Análisis de documentos largos

### Google (Gemini)

**Historia:**
- Google ha estado en AI/ML desde siempre (TensorFlow, BERT, T5)
- Gemini lanzado en 2023 como respuesta a GPT-4
- Integrado con todo el ecosistema Google

**Modelos Gemini (2026):**

| Modelo | Contexto | Velocidad | Costo | Mejor Para |
|--------|----------|-----------|-------|------------|
| **Gemini 2.5 Pro** | 2M | ⚡⚡ | 💰💰 | Análisis de documentos masivos |
| **Gemini 2.5 Flash** | 1M | ⚡⚡⚡⚡ | 💰 | Velocidad + contexto |
| **Gemini 2.0 Flash** | 1M | ⚡⚡⚡⚡⚡ | 💰 | Ultra rápido y barato |
| **Gemini 2.0 Flash Thinking** | 1M | ⚡⚡⚡ | 💰 | Razonamiento economico |

**Características Únicas:**
- 🚀 Ventanas de contexto MASIVAS (hasta 2M tokens)
- 🎥 Multimodal nativo completo (texto, imagen, video, audio, código)
- 🆓 Tier gratuito muy generoso
- 🔗 Integración nativa con Google Workspace y Search
- 🤖 Gemini 2.0 Flash Thinking: razonamiento open source gratuito

### Open Source y Locales (2026)

**Meta (Llama):**

| Modelo | Parámetros | Mejor Para |
|--------|------------|------------|
| **Llama 4 Scout** | MoE | Razonamiento + multimodal |
| **Llama 4 Maverick** | MoE | Balance calidad/eficiencia |
| **Llama 3.3 70B** | 70B | Calidad sin GPU masiva |
| **Llama 3.2 3B/1B** | 1-3B | Edge devices, local |

**Otros Modelos Open Source:**

| Modelo | Organización | Puntos Fuertes |
|--------|-------------|----------------|
| **DeepSeek V3** | DeepSeek | Calidad comparable a GPT-4, gratis |
| **DeepSeek R1** | DeepSeek | Razonamiento open source |
| **Qwen 2.5 72B** | Alibaba | Multilingüe, código |
| **Mistral Large 2** | Mistral | Eficiente, europeo |
| **Phi-4** | Microsoft | Modelos pequeños/eficientes |
| **Gemma 3** | Google | Open, optimizado para inferencia |

**Uso Local con Ollama:**
```bash
ollama pull llama3.3      # Meta Llama 3.3 70B
ollama pull deepseek-r1  # DeepSeek R1 (razonamiento)
ollama pull qwen2.5      # Qwen 2.5
ollama serve              # API compatible con OpenAI en localhost:11434
```

### Comparativa Rápida (2026)

| Característica | OpenAI | Anthropic | Google | Open Source |
|----------------|--------|-----------|--------|-------------|
| **Calidad General** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Razonamiento** | ⭐⭐⭐⭐⭐ (o3) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ (R1) |
| **Velocidad** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Costo** | 💰💰 | 💰💰💰 | 💰 | 🆓 |
| **Contexto** | 1M | 200K | 2M | 128K+ |
| **Privacidad** | ❌ | ❌ | ❌ | ✅ (local) |
| **Documentación** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Variable |
| **Ecosistema** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 📡 Anatomía de una Llamada a API {#llamadas-api}

### OpenAI Chat Completions

**Request Structure:**

```python
{
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "system",
            "content": "Eres un asistente útil."
        },
        {
            "role": "user",
            "content": "¿Qué es Python?"
        }
    ],
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}
```

**Response Structure:**

```python
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o-mini",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Python es un lenguaje..."
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 150,
        "total_tokens": 170
    }
}
```

### Roles en Mensajes

**system**: Configura el comportamiento del asistente
```python
{"role": "system", "content": "Eres un experto en Python que responde concisamente."}
```

**user**: Mensajes del usuario
```python
{"role": "user", "content": "¿Cómo funciona una lista?"}
```

**assistant**: Respuestas previas del modelo (para contexto)
```python
{"role": "assistant", "content": "Una lista es una estructura..."}
```

### Parámetros Importantes

**temperature** (0.0 - 2.0):
- `0.0`: Determinista, siempre la respuesta más probable
- `0.7`: Balance (recomendado por defecto)
- `1.5+`: Muy creativo/aleatorio

**max_tokens**:
- Límite de tokens en la respuesta
- No confundir con el límite del modelo (contexto)

**top_p** (0.0 - 1.0):
- Nucleus sampling
- `1.0`: Considera todos los tokens posibles
- `0.9`: Solo considera el 90% más probable
- Alternativa a temperature

**frequency_penalty** (-2.0 - 2.0):
- Penaliza repetir tokens ya usados
- Positivo = menos repetición
- Negativo = más repetición

**presence_penalty** (-2.0 - 2.0):
- Similar a frequency pero no acumula
- Útil para fomentar nuevos temas

---

## 🌊 Streaming de Respuestas {#streaming}

### ¿Por qué Streaming?

Sin streaming:
```
Usuario espera... ⏳
Usuario espera... ⏳
Usuario espera... ⏳
[5 segundos después]
¡Respuesta completa aparece!
```

Con streaming:
```
"Python"
"Python es"
"Python es un"
"Python es un lenguaje..."
[Aparece palabra por palabra]
```

### Implementación

**OpenAI:**
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    stream=True  # ← Activar streaming
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**Anthropic:**
```python
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    messages=messages,
    max_tokens=1000
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### Ventajas del Streaming

✅ Mejor UX (respuesta instantánea)
✅ Menor tiempo percibido de espera
✅ Permite cancelar generaciones largas
✅ Ideal para chatbots en tiempo real

### Desventajas

❌ Más complejo de implementar
❌ Dificulta el manejo de errores
❌ No puedes procesar la respuesta completa hasta el final

---

## 🛠️ Function Calling {#function-calling}

### Concepto

Function calling permite que el LLM:
1. Detecte cuándo necesita información externa
2. Decida qué función llamar
3. Extraiga los parámetros necesarios
4. Te los devuelva en formato estructurado

**El LLM NO ejecuta la función**, solo te dice qué llamar y cómo.

### Flujo Completo

```
1. Usuario: "¿Qué tiempo hace en Madrid?"
   ↓
2. LLM analiza y decide: "Necesito get_weather(city='Madrid')"
   ↓
3. Tu código recibe: {"name": "get_weather", "arguments": '{"city": "Madrid"}'}
   ↓
4. Tu código ejecuta: weather = get_weather("Madrid")
   ↓
5. Envías resultado de vuelta al LLM con el contexto
   ↓
6. LLM genera respuesta: "En Madrid hace 22°C y está soleado."
```

### Definir Funciones

Las funciones se definen usando **JSON Schema**:

```python
functions = [
    {
        "name": "get_weather",
        "description": "Obtiene el clima actual de una ciudad",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Nombre de la ciudad"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Unidad de temperatura"
                }
            },
            "required": ["city"]
        }
    }
]
```

### Enviar a la API

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[
        {"type": "function", "function": f}
        for f in functions
    ]
)
```

### Procesar Respuesta

```python
message = response.choices[0].message

if message.tool_calls:
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    # Ejecutar tu función
    result = execute_function(function_name, arguments)
    
    # Enviar resultado de vuelta
    messages.append({
        "role": "function",
        "name": function_name,
        "content": str(result)
    })
    
    # Nueva llamada con el resultado
    final_response = client.chat.completions.create(...)
```

### Casos de Uso

- 🌐 Llamar APIs externas (clima, noticias, búsqueda)
- 💾 Consultar bases de datos
- 📧 Enviar emails o notificaciones
- 🔧 Ejecutar código o comandos
- 🤖 Crear agentes autónomos

---

## � Modelos de Razonamiento {#reasoning-models}

### ¿Qué son los Modelos de Razonamiento?

A partir de 2024-2025, surgió una nueva categoría de LLMs que "piensan antes de responder"
usando una cadena de razonamiento interna:

```
Modelos Normales:
  Input → [LLM] → Output (directo)

Modelos de Razonamiento:
  Input → [LLM piensa] → Cadena de razonamiento interno → Output
```

### Modelos Disponibles

**OpenAI (serie o):**
```
o1 (2024) → o3 (2025) → o4-mini (2025)
```

- **o3**: El más potente de OpenAI, SOTA en benchmarks matemáticos y científicos
- **o4-mini**: Balance óptimo razonamiento/velocidad/costo. Soporta tools + vision.

**Anthropic:**
- **Claude Extended Thinking**: Claude 3.7+ puede activar razonamiento profundo
- El "thinking" es visible y auditable (diferencia clave vs OpenAI)

**Google:**
- **Gemini Flash Thinking**: Razonamiento económico y rápido
- **Gemini 2.5 Pro**: Razonamiento integrado

**Open Source:**
- **DeepSeek R1**: Primer modelo de razonamiento open source de calidad comparable a o1

### Cuándo Usar Modelos de Razonamiento

✅ **Usar cuando:**
- Matemáticas, física, o ciencias complejas
- Debugging de código difícil
- Análisis multi-paso con muchas variables
- Decisiones estratégicas complejas
- Cualquier tarea donde la precisión es crítica

❌ **NO usar cuando:**
- Conversación casual o Q&A simple
- Extracción de información directa
- Reformateo de texto
- Clasificación simple

### API de Modelos de Razonamiento

```python
from openai import OpenAI

client = OpenAI()

# Los modelos 'o' usan 'reasoning_effort' no 'temperature'
response = client.chat.completions.create(
    model="o4-mini",
    reasoning_effort="high",  # 'low', 'medium', 'high'
    messages=[
        {"role": "user", "content": "¿Cuántos ceros tiene 100! (factorial de 100)?"}
    ]
)

# Acceder a tokens de razonamiento
usage = response.usage
print(f"Tokens de razonamiento: {usage.completion_tokens_details.reasoning_tokens}")
print(f"Tokens de respuesta: {usage.completion_tokens_details.text_tokens}")
```

### Diferencias Importantes

| Característica | Modelos Normales (GPT-4o) | Modelos Razonamiento (o3/o4) |
|----------------|--------------------------|------------------------------|
| `temperature` | ✅ Soportado | ❌ No (siempre 1) |
| `reasoning_effort` | ❌ N/A | ✅ 'low'/'medium'/'high' |
| Velocidad | ⚡⚡⚡ | ⚡ |
| Costo | 💰💰 | 💰💰💰💰 |
| Precisión STEM | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Conversación | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 🔌 OpenAI Responses API {#responses-api}

### ¿Por qué una nueva API?

La **Responses API** (lanzada en 2025) está diseñada para workflows **agénticos**,
donde el modelo necesita tomar múltiples pasos y usar herramientas.

**Chat Completions vs Responses API:**

```
Chat Completions (tradicional):
- Sin estado: debes reenviar todo el historial cada vez
- Sin herramientas built-in
- Para conversaciones simples

Responses API (nueva):
- Con estado: el servidor recuerda la conversación
- Herramientas built-in: web search, file search, code interpreter
- Para workflows agénticos multi-paso
```

### Herramientas Built-in

```python
from openai import OpenAI

client = OpenAI()

# Búsqueda web integrada
response = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "web_search_preview"}],
    input="¿Cuáles son las noticias de IA de esta semana?"
)

# Code Interpreter
response = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
    input="Analiza estos datos y crea una gráfica: [1, 5, 3, 7, 2]"
)

# File Search (buscar en documentos subidos)
response = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "file_search", "vector_store_ids": ["vs_abc123"]}],
    input="¿Qué dice el documento sobre los términos de uso?"
)
```

### Conversación Multi-turno Stateful

```python
# Primer mensaje
response1 = client.responses.create(
    model="gpt-4.1",
    input="Hola, soy Ana",
    store=True  # Guardar en servidor
)
response_id = response1.id

# Segundo mensaje - referencia al anterior
response2 = client.responses.create(
    model="gpt-4.1",
    input="¿Cómo me llamo?",
    previous_response_id=response_id  # ¡Sin reenviar historial!
)
print(response2.output_text)  # "Te llamas Ana"
```

---

## 🤔 Extended Thinking (Claude) {#extended-thinking}

### Concepto

**Extended Thinking** es la implementación de Anthropic para razonamiento profundo.
A diferencia de los modelos de razonamiento de OpenAI (proceso opaco), el "thinking"
de Claude es visible y auditable.

### Activación

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # Tokens máximos para pensar
    },
    messages=[{
        "role": "user",
        "content": "Demuestra que sqrt(2) es irracional"
    }]
)

# Separar thinking de respuesta
for block in response.content:
    if block.type == "thinking":
        print(f"Razonamiento:\n{block.thinking}")
    elif block.type == "text":
        print(f"\nRespuesta:\n{block.text}")
```

### budget_tokens

El `budget_tokens` controla cuánto puede "pensar" Claude:
- `1024`: Mínimo, razonamiento básico
- `10000`: Razonamiento moderado (recomendado por defecto)
- `100000`: Razonamiento muy profundo (problemas extremadamente complejos)

> **Nota:** Más budget_tokens = mejor calidad, pero mayor costo y latencia.

---

## 💻 LLMs Locales y Open Source {#open-source}

### Por qué correr modelos localmente

✅ **Privacidad**: Los datos no salen de tu organización
✅ **Sin costos de API**: Solo costo de hardware
✅ **Sin rate limits**: Velocidad limitada solo por tu hardware
✅ **Control total**: Personalización y fine-tuning completo
✅ **Offline**: Funciona sin internet

### Ollama (la forma más fácil)

```bash
# Instalar Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Descargar modelos
ollama pull llama3.3          # Meta Llama 3.3 70B
ollama pull deepseek-r1       # DeepSeek R1 (razonamiento)
ollama pull qwen2.5:72b       # Qwen 2.5 (multilingüe)
ollama pull mistral-large     # Mistral Large 2
ollama pull phi4              # Microsoft Phi-4

# Iniciar servidor
ollama serve  # localhost:11434
```

### Usar con cliente OpenAI

Ollama expone una API compatible con OpenAI:

```python
from openai import OpenAI

# Apuntar al servidor local de Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Cualquier string, no se usa
)

response = client.chat.completions.create(
    model="llama3.3",  # Nombre del modelo en Ollama
    messages=[{"role": "user", "content": "¿Qué es NLP?"}]
)
print(response.choices[0].message.content)
```

### Cuándo usar cada opción

| Escenario | Recomendación |
|-----------|--------------|
| Prototipado rápido | GPT-4.1-mini (barato y bueno) |
| Razonamiento complejo | o4-mini o Claude 3.7 Extended Thinking |
| Mucho contexto | Gemini 2.5 Pro (2M tokens) |
| Privacidad crítica | Ollama + Llama 4 local |
| Sin presupuesto API | Ollama + DeepSeek R1 (razonamiento gratuito) |
| Producción | Depende del caso de uso + benchmark |

---

### ¿Qué es un Token?

Un token es una unidad de texto. Aproximadamente:
- 1 token ≈ 4 caracteres en inglés
- 1 token ≈ 0.75 palabras en inglés
- 1 palabra en español ≈ 1-2 tokens

**Ejemplos:**

| Texto | Tokens |
|-------|--------|
| "Hello" | 1 |
| "Hello, world!" | 4 |
| "Hola, ¿cómo estás?" | 7 |
| "artificial intelligence" | 2 |
| "inteligencia artificial" | 4 |

### Tokenización

Los modelos usan tokenizadores específicos:

**GPT (BPE - Byte Pair Encoding):**
```python
import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o-mini")
tokens = encoder.encode("Hola, ¿cómo estás?")
print(len(tokens))  # 7
```

**Claude (sentencepiece):**
Similar pero con vocabulario diferente

### Estructura de Costos

```
Costo Total = (Input Tokens × Precio Input) + (Output Tokens × Precio Output)
```

**Precios (2026) por 1M tokens:**

| Modelo | Input | Output |
|--------|-------|--------|
| gpt-4.1 | $2.00 | $8.00 |
| gpt-4.1-mini | $0.40 | $1.60 |
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| o3 | $10.00 | $40.00 |
| o4-mini | $1.10 | $4.40 |
| claude-3-7-sonnet | $3.00 | $15.00 |
| claude-3-5-haiku | $0.80 | $4.00 |
| gemini-2.5-pro | $1.25 | $10.00 |
| gemini-2.0-flash | $0.10 | $0.40 |
| Ollama (local) | 🆓 | 🆓 |

### Ejemplos de Costo

**Conversación Simple** (100 tokens input, 200 tokens output):
- GPT-4o: $0.00025 + $0.002 = **$0.00225**
- GPT-4o-mini: $0.000015 + $0.00012 = **$0.000135**
- Gemini Flash: $0.0000075 + $0.00006 = **$0.0000675**

**Análisis de Documento** (10K tokens input, 1K tokens output):
- GPT-4o: $0.025 + $0.01 = **$0.035**
- Claude Sonnet: $0.03 + $0.015 = **$0.045**

**Uso Diario** (100K tokens/día):
- GPT-4.1-mini: ~$12/mes
- GPT-4o-mini: ~$9/mes
- Gemini Flash: ~$3/mes
- Ollama local: $0/mes (solo hardware)

### Optimizar Costos

**1. Usa el modelo más barato que funcione**
```python
# ❌ Usar GPT-4o para todo
response = call_gpt4o("Hola")

# ✅ Usar GPT-4o-mini para tareas simples
response = call_gpt4o_mini("Hola")
```

**2. Limita max_tokens**
```python
# ✅ Si solo necesitas respuestas cortas
response = client.chat.completions.create(
    ...,
    max_tokens=100  # Límita el output
)
```

**3. Cachea respuestas**
```python
cache = {}

def cached_llm_call(prompt):
    if prompt in cache:
        return cache[prompt]
    response = llm_call(prompt)
    cache[prompt] = response
    return response
```

**4. Usa prompts concisos**
```python
# ❌ Prompt verboso
prompt = "Te voy a dar un texto y me gustaría que por favor lo resumas de la manera más concisa posible, intentando capturar las ideas principales..."

# ✅ Prompt conciso
prompt = "Resume este texto:"
```

**5. Batch processing**
```python
# ✅ Procesa múltiples items en una llamada
prompt = "Resume cada uno:\n1. Texto1\n2. Texto2\n3. Texto3"
```

---

## ⚡ Mejores Prácticas {#mejores-prácticas}

### 1. Manejo de Errores

**Errores Comunes:**

- **401 Unauthorized**: API key inválida
- **429 Too Many Requests**: Rate limit excedido
- **500/503 Server Error**: Problemas del servidor
- **Timeout**: Respuesta demasiado lenta

**Implementación Robusta:**

```python
import time
from openai import OpenAI, OpenAIError

def robust_llm_call(prompt, max_retries=3):
    client = OpenAI()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )
            return response.choices[0].message.content
            
        except OpenAIError as e:
            print(f"Intento {attempt + 1} falló: {e}")
            
            if attempt < max_retries - 1:
                # Backoff exponencial: 1s, 2s, 4s
                wait_time = 2 ** attempt
                print(f"Reintentando en {wait_time}s...")
                time.sleep(wait_time)
            else:
                print("Todos los intentos fallaron")
                return None
```

### 2. Rate Limiting

**Implementa tu propio rate limiter:**

```python
import time
from collections import deque

class RateLimiter:
    def __init__(self, max_calls, time_window):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
    
    def wait_if_needed(self):
        now = time.time()
        
        # Eliminar llamadas antiguas
        while self.calls and self.calls[0] < now - self.time_window:
            self.calls.popleft()
        
        # Si alcanzamos el límite, esperar
        if len(self.calls) >= self.max_calls:
            sleep_time = self.calls[0] + self.time_window - now
            time.sleep(sleep_time)
        
        self.calls.append(now)

# Uso: 10 llamadas por minuto
limiter = RateLimiter(max_calls=10, time_window=60)

for prompt in prompts:
    limiter.wait_if_needed()
    response = llm_call(prompt)
```

### 3. Logging y Monitoring

```python
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitored_llm_call(prompt, model="gpt-4o-mini"):
    start_time = datetime.now()
    
    try:
        logger.info(f"Llamando a {model}")
        logger.debug(f"Prompt: {prompt[:100]}...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        tokens = response.usage.total_tokens
        cost = calculate_cost(response.usage, model)
        
        logger.info(f"✓ Completado en {duration:.2f}s | "
                   f"Tokens: {tokens} | Costo: ${cost:.4f}")
        
        return response.choices[0].message.content
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"✗ Error después de {duration:.2f}s: {e}")
        raise
```

### 4. Seguridad

**Variables de Entorno para API Keys:**

```python
# ✅ CORRECTO
import os
api_key = os.getenv("OPENAI_API_KEY")

# ❌ NUNCA hagas esto
api_key = "sk-proj-abc123..."  # Hardcoded en el código
```

**Archivo .env:**
```bash
# .env (NO subas esto a Git)
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

**Cargar con python-dotenv:**
```python
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
```

### 5. Testing

**Mock LLM calls en tests:**

```python
import pytest
from unittest.mock import patch

def test_llm_function():
    with patch('openai.OpenAI') as mock_client:
        # Mock de respuesta
        mock_client.return_value.chat.completions.create.return_value = {
            "choices": [{
                "message": {
                    "content": "Respuesta mockeada"
                }
            }]
        }
        
        result = my_llm_function("test prompt")
        assert result == "Respuesta mockeada"
```

### 6. Prompt Engineering

**Técnicas Efectivas:**

**a) Instrucciones Claras y Específicas:**
```python
# ❌ Vago
prompt = "Dame información sobre Python"

# ✅ Específico
prompt = "Lista 5 características clave de Python que lo hacen popular, en máximo 2 líneas cada una."
```

**b) Few-Shot Learning:**
```python
prompt = """Clasifica el sentimiento de estos tweets:

Tweet: "¡Me encanta este producto!"
Sentimiento: Positivo

Tweet: "No funciona, muy decepcionado"
Sentimiento: Negativo

Tweet: "Acabo de comprar el nuevo iPhone"
Sentimiento: """
```

**c) Chain of Thought:**
```python
prompt = """Resuelve paso a paso:
Juan tiene 3 manzanas. María le da 5 más. Juan come 2.
¿Cuántas manzanas tiene Juan?

Razonamiento paso a paso:"""
```

**d) Roles y Contexto:**
```python
system_message = """Eres un profesor de programación experto 
que explica conceptos complejos de forma simple, usando analogías 
y ejemplos prácticos."""
```

---

## 📚 Recursos Adicionales

### Documentación Oficial

- [OpenAI API Docs](https://platform.openai.com/docs)
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses)
- [Anthropic Claude Docs](https://docs.anthropic.com/)
- [Anthropic Extended Thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [Google Gemini API](https://ai.google.dev/docs)
- [Ollama Docs](https://ollama.ai/docs)

### Papers Importantes

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer original
- [GPT-3 Paper](https://arxiv.org/abs/2005.14165)
- [Constitutional AI](https://arxiv.org/abs/2212.08073) - Enfoque de seguridad de Claude
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - Primer reasoning model open source

### Herramientas

- [tiktoken](https://github.com/openai/tiktoken) - Tokenizador de OpenAI
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) - Web tool
- [LangChain](https://python.langchain.com/) - Framework para aplicaciones con LLMs
- [instructor](https://github.com/jxnl/instructor) - Structured outputs con Pydantic
- [Ollama](https://ollama.ai) - LLMs locales fáciles

### Comunidades

- [OpenAI Community Forum](https://community.openai.com/)
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)
- [Hugging Face Discord](https://hf.co/join/discord)

---

## 🎓 Próximos Pasos

Después de dominar este koan, continúa con:

- **Koan 11: AI Agents** - Construye agentes autónomos y comprende el Agentic Mode
- **Koan 12: Semantic Search** - Embeddings y búsqueda vectorial
- **Koan 13: RAG** - Retrieval-Augmented Generation

¡Buena suerte! 🚀
