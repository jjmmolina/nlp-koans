# Teoría: AI Agents - Agentes Autónomos con LLMs

## 📚 Tabla de Contenidos
1. [Introducción a AI Agents](#introducción)
2. [Arquitectura de Agentes](#arquitectura)
3. [El Patrón ReAct](#react)
4. [Herramientas (Tools)](#herramientas)
5. [Memoria Conversacional](#memoria)
6. [Frameworks: LangChain](#langchain)
7. [Sistemas Multi-Agente](#multi-agente)
8. [Callbacks y Monitoring](#callbacks)
9. [Mejores Prácticas](#mejores-prácticas)

---

## 🤖 Introducción a AI Agents {#introducción}

### ¿Qué es un Agente de IA?

Un **agente de IA** es un sistema autónomo que puede:
- Percibir su entorno (leer inputs)
- Razonar sobre qué hacer
- Tomar decisiones
- Actuar usando herramientas
- Aprender de los resultados

En el contexto de LLMs, un agente combina la capacidad de razonamiento del modelo
con herramientas externas para realizar tareas complejas.

### Diferencia: LLM vs Agente

**LLM Simple:**
```
Usuario: "¿Qué tiempo hace en Madrid?"
LLM: "No tengo acceso a información en tiempo real..."
```

**Agente con Tools:**
```
Usuario: "¿Qué tiempo hace en Madrid?"
Agente:
  → Piensa: "Necesito datos actuales del clima"
  → Usa tool: get_weather("Madrid")
  → Observa: {"temp": 22, "condition": "sunny"}
  → Responde: "En Madrid hace 22°C y está soleado"
```

### Historia y Evolución

```
2017: Transformers (Vaswani et al.)
  ↓
2020: GPT-3 y primeros experimentos con "prompts como programas"
  ↓
2021: Chain-of-Thought prompting
  ↓
2022: ReAct paper (Yao et al.) - Razonamiento + Acción
      Toolformer - LLMs aprenden a usar APIs
  ↓
2023: Function Calling en OpenAI
      LangChain populariza agentes
      AutoGPT: Agentes completamente autónomos
  ↓
2024: Agentes en producción
      CrewAI, LangGraph, Autogen
      Multi-agent systems mainstream
```

### Tipos de Agentes

| Tipo | Descripción | Uso |
|------|-------------|-----|
| **Simple Reflex** | Reacciona a reglas fijas | Chatbots básicos |
| **Model-Based** | Mantiene estado interno | Asistentes con contexto |
| **Goal-Based** | Planifica para objetivos | Automatización de tareas |
| **Utility-Based** | Optimiza métrica | Agentes de trading |
| **Learning** | Mejora con experiencia | Sistemas adaptativos |

Los agentes LLM modernos son principalmente **Goal-Based** con capacidades de
**Model-Based** (memoria).

---

## 🏗️ Arquitectura de Agentes {#arquitectura}

### Componentes Fundamentales

```
┌─────────────────────────────────────────┐
│           AGENTE DE IA                  │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────┐        ┌─────────────┐  │
│  │   LLM     │◄──────►│   Prompt    │  │
│  │ (Cerebro) │        │  Template   │  │
│  └───────────┘        └─────────────┘  │
│       ▲                                 │
│       │                                 │
│       ▼                                 │
│  ┌───────────────────────────────────┐ │
│  │      Agent Executor               │ │
│  │  (Loop de Razonamiento)           │ │
│  └───────────────────────────────────┘ │
│       │            │           │        │
│       ▼            ▼           ▼        │
│  ┌────────┐   ┌────────┐  ┌────────┐  │
│  │ Tool 1 │   │ Tool 2 │  │ Tool N │  │
│  └────────┘   └────────┘  └────────┘  │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │         Memory                  │   │
│  │  (Historial Conversacional)     │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

#### 1. **LLM (Large Language Model)**
- El "cerebro" del agente
- Realiza el razonamiento
- Decide qué herramientas usar
- GPT-4, Claude, Gemini, etc.

#### 2. **Prompt Template**
- Instrucciones de cómo comportarse
- Define el "rol" del agente
- Explica cómo usar las herramientas
- Crucial para el comportamiento correcto

#### 3. **Agent Executor**
- Loop que ejecuta el ciclo de razonamiento
- Llama al LLM
- Ejecuta herramientas
- Maneja errores
- Controla iteraciones máximas

#### 4. **Tools (Herramientas)**
- Funciones que el agente puede usar
- Búsqueda web, calculadora, APIs, etc.
- Extienden las capacidades del LLM

#### 5. **Memory**
- Almacena historial de conversación
- Permite contexto multi-turno
- Varios tipos (buffer, summary, etc.)

### Flujo de Ejecución

```
1. Usuario envía query
   ↓
2. Agent Executor recibe la query
   ↓
3. LLM analiza query + memoria + herramientas disponibles
   ↓
4. LLM decide: ¿Necesito usar una herramienta?
   
   SÍ:                              NO:
   ↓                                ↓
   5a. Selecciona herramienta       5b. Genera respuesta final
   6a. Ejecuta con argumentos       6b. Retorna al usuario
   7a. Observa resultado
   8a. Vuelve al paso 3 (loop)
```

### Límites y Control

**Iteraciones Máximas:**
```python
AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,  # Previene loops infinitos
    max_execution_time=30  # Timeout en segundos
)
```

**Manejo de Errores:**
- Tool falla → Agente recibe mensaje de error
- Timeout → Excepción
- Token limit → Truncar contexto

---

## 🔄 El Patrón ReAct {#react}

### Concepto

**ReAct** = **Re**asoning + **Act**ing

Paper original: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)

### Estructura

El agente alterna entre tres estados:

1. **Thought (Pensamiento)**
   - El agente razona sobre la situación
   - "¿Qué información necesito?"
   - "¿Qué herramienta debo usar?"

2. **Action (Acción)**
   - Selecciona y ejecuta una herramienta
   - `search["Python tutorials"]`
   - `calculator["sqrt(144)"]`

3. **Observation (Observación)**
   - Analiza el resultado de la acción
   - Decide si continuar o terminar

### Ejemplo Completo

**Query:** "¿Cuál es la capital de Francia y cuántos habitantes tiene?"

```
Thought 1: Necesito buscar información sobre Francia.
Action 1: search["capital de Francia"]
Observation 1: París es la capital de Francia.

Thought 2: Ahora necesito la población de París.
Action 2: search["población de París 2024"]
Observation 2: París tiene aproximadamente 2.2 millones de habitantes.

Thought 3: Tengo toda la información necesaria.
Final Answer: La capital de Francia es París, que tiene aproximadamente
2.2 millones de habitantes.
```

### Prompt Template de ReAct

```python
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
```

### Ventajas de ReAct

✅ **Transparencia**: Cada paso es explicable
✅ **Debugging**: Fácil ver dónde falla
✅ **Corrección**: El agente puede corregirse
✅ **Performance**: Mejor que solo prompting

### Comparación con Alternativas

| Método | Transparency | Performance | Cost |
|--------|--------------|-------------|------|
| **Zero-Shot** | ❌ Bajo | ⭐⭐ | 💰 |
| **Few-Shot** | ❌ Bajo | ⭐⭐⭐ | 💰💰 |
| **Chain-of-Thought** | ✅ Alto | ⭐⭐⭐ | 💰💰 |
| **ReAct** | ✅ Muy Alto | ⭐⭐⭐⭐ | 💰💰💰 |
| **ReAct + Self-Reflection** | ✅ Máximo | ⭐⭐⭐⭐⭐ | 💰💰💰💰 |

---

## 🛠️ Herramientas (Tools) {#herramientas}

### Anatomía de una Herramienta

```python
class Tool:
    name: str              # Identificador único
    description: str       # Qué hace y cuándo usarla
    function: Callable     # La función Python
    return_direct: bool    # Si retornar directo al usuario
    args_schema: Schema    # Validación de argumentos (opcional)
```

### Tipos de Herramientas

#### 1. **Búsqueda y Información**

**Web Search:**
```python
search_tool = Tool(
    name="web_search",
    description="Busca información actual en internet. Usa esto cuando necesites datos recientes o información que no conoces.",
    function=duckduckgo_search
)
```

**Wikipedia:**
```python
wiki_tool = Tool(
    name="wikipedia",
    description="Busca información enciclopédica. Útil para hechos históricos, biografías, y conceptos generales.",
    function=wikipedia_search
)
```

#### 2. **Computación y Matemáticas**

**Calculator:**
```python
calc_tool = Tool(
    name="calculator",
    description="Realiza cálculos matemáticos. Entrada: expresión matemática válida. Ejemplo: '5 + 3 * 2'",
    function=calculate
)
```

**Python REPL:**
```python
python_tool = Tool(
    name="python_repl",
    description="Ejecuta código Python. Útil para cálculos complejos, manipulación de datos, o lógica programática.",
    function=python_repl
)
```

#### 3. **APIs y Servicios**

**Weather API:**
```python
weather_tool = Tool(
    name="get_weather",
    description="Obtiene el clima actual de una ciudad. Entrada: nombre de la ciudad.",
    function=get_weather_api
)
```

**Database Query:**
```python
db_tool = Tool(
    name="query_database",
    description="Consulta la base de datos de productos. Entrada: consulta SQL o búsqueda de producto.",
    function=query_product_db
)
```

#### 4. **Acciones y Automatización**

**Send Email:**
```python
email_tool = Tool(
    name="send_email",
    description="Envía un email. Formato: 'destinatario: <email>, asunto: <asunto>, cuerpo: <mensaje>'",
    function=send_email
)
```

**File Operations:**
```python
file_tool = Tool(
    name="read_file",
    description="Lee el contenido de un archivo. Entrada: ruta del archivo.",
    function=read_file_content
)
```

### Diseño de Herramientas Efectivas

#### ✅ Buenas Prácticas

**1. Descripciones Claras:**
```python
# ❌ MAL
description="Hace búsquedas"

# ✅ BIEN
description="Busca información en internet. Usa esta herramienta cuando necesites información actualizada que no esté en tu conocimiento. Entrada: consulta de búsqueda en lenguaje natural."
```

**2. Ejemplos en la Descripción:**
```python
description="""Convierte unidades de medida.
Formato: "<cantidad> <unidad_origen> to <unidad_destino>"
Ejemplos:
- "100 celsius to fahrenheit"
- "5 miles to kilometers"
"""
```

**3. Manejo de Errores:**
```python
def robust_tool(input: str) -> str:
    try:
        result = process(input)
        return f"Success: {result}"
    except ValueError as e:
        return f"Error: Entrada inválida - {e}. Por favor proporciona..."
    except Exception as e:
        return f"Error inesperado: {e}"
```

**4. Validación de Entrada:**
```python
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="Nombre de la ciudad")
    units: str = Field(default="metric", description="celsius o fahrenheit")

weather_tool = Tool(
    name="weather",
    description="...",
    args_schema=WeatherInput,
    function=get_weather
)
```

### Herramientas Pre-construidas en LangChain

```python
from langchain.agents import load_tools

# Cargar múltiples herramientas a la vez
tools = load_tools(
    ["ddg-search", "llm-math", "wikipedia"],
    llm=llm
)

# Herramientas disponibles:
# - ddg-search: DuckDuckGo
# - google-search: Google (requiere API key)
# - wikipedia: Wikipedia
# - llm-math: Calculadora con LLM
# - python_repl: Ejecutar Python
# - requests_get/post: HTTP requests
# - terminal: Ejecutar comandos shell
```

---

## 🧠 Memoria Conversacional {#memoria}

### ¿Por qué Memoria?

Sin memoria:
```
Usuario: "Mi nombre es Ana"
Agente: "¡Hola Ana! ¿En qué puedo ayudarte?"

[Nueva conversación]
Usuario: "¿Cuál es mi nombre?"
Agente: "Lo siento, no tengo esa información."  # ❌
```

Con memoria:
```
Usuario: "Mi nombre es Ana"
Agente: "¡Hola Ana! ¿En qué puedo ayudarte?"

Usuario: "¿Cuál es mi nombre?"
Agente: "Tu nombre es Ana."  # ✅
```

### Tipos de Memoria en LangChain

#### 1. **ConversationBufferMemory**

Almacena TODO el historial.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Automáticamente guarda:
# Usuario: "Hola"
# Agente: "¡Hola! ¿Cómo estás?"
# Usuario: "Bien, gracias"
# Agente: "Me alegro..."
```

**Pros:**
- ✅ Contexto completo
- ✅ Simple de implementar

**Cons:**
- ❌ Crece infinitamente
- ❌ Costoso en tokens

#### 2. **ConversationBufferWindowMemory**

Mantiene solo las últimas N interacciones.

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=5,  # Últimos 5 pares de mensajes
    memory_key="chat_history",
    return_messages=True
)
```

**Pros:**
- ✅ Tokens controlados
- ✅ Contexto reciente relevante

**Cons:**
- ❌ Pierde información antigua

#### 3. **ConversationSummaryMemory**

Resume la conversación periódicamente.

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=llm,  # Usa un LLM para resumir
    memory_key="chat_history"
)

# Ejemplo de resumen:
# "El usuario Ana preguntó sobre Python. Se explicaron conceptos básicos
# de programación y se recomendaron recursos de aprendizaje."
```

**Pros:**
- ✅ Escala bien
- ✅ Retiene información importante

**Cons:**
- ❌ Costo de resumir
- ❌ Puede perder detalles

#### 4. **ConversationSummaryBufferMemory**

Híbrido: resumen de lo viejo + buffer de lo reciente.

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=2000,  # Límite de tokens
    memory_key="chat_history"
)
```

**Pros:**
- ✅ Balance óptimo
- ✅ Contexto completo + eficiencia

**Cons:**
- ❌ Más complejo

#### 5. **ConversationKGMemory** (Knowledge Graph)

Extrae y almacena relaciones estructuradas.

```python
from langchain.memory import ConversationKGMemory

memory = ConversationKGMemory(
    llm=llm,
    memory_key="chat_history"
)

# Extrae:
# Ana -> WORKS_AT -> Google
# Ana -> LIVES_IN -> Madrid
# Ana -> INTERESTED_IN -> Python
```

**Pros:**
- ✅ Información estructurada
- ✅ Queries complejas

**Cons:**
- ❌ Complejo de configurar
- ❌ Requiere LLM bueno

### Comparativa de Memorias

| Tipo | Tokens | Precisión | Complejidad | Mejor Para |
|------|--------|-----------|-------------|------------|
| **Buffer** | 💰💰💰 | ⭐⭐⭐⭐⭐ | ⚙️ | Conversaciones cortas |
| **Window** | 💰💰 | ⭐⭐⭐⭐ | ⚙️ | Chatbots generales |
| **Summary** | 💰 | ⭐⭐⭐ | ⚙️⚙️ | Conversaciones largas |
| **Summary+Buffer** | 💰💰 | ⭐⭐⭐⭐ | ⚙️⚙️⚙️ | Producción |
| **KG** | 💰💰 | ⭐⭐⭐⭐⭐ | ⚙️⚙️⚙️⚙️ | Asistentes personales |

### Persistencia de Memoria

Las memorias anteriores son in-memory. Para persistir:

```python
# Redis
from langchain.memory import RedisChatMessageHistory

message_history = RedisChatMessageHistory(
    url="redis://localhost:6379",
    session_id="user_123"
)

memory = ConversationBufferMemory(
    chat_memory=message_history
)

# PostgreSQL
from langchain.memory import PostgresChatMessageHistory

message_history = PostgresChatMessageHistory(
    connection_string="postgresql://...",
    session_id="user_123"
)
```

---

## 🔗 Frameworks: LangChain {#langchain}

### ¿Qué es LangChain?

**LangChain** es el framework más popular para construir aplicaciones con LLMs.

**Características:**
- 🤖 Agentes con múltiples estrategias
- 🧠 Sistemas de memoria
- 🛠️ Biblioteca extensiva de herramientas
- ⛓️ Chains para workflows complejos
- 🗄️ Integraciones con vector databases
- 📊 Callbacks y monitoring

### Arquitectura de LangChain

```
LangChain Framework
│
├── LangChain Core
│   ├── LLMs & Chat Models
│   ├── Prompts & Templates
│   ├── Output Parsers
│   └── Callbacks
│
├── LangChain Community
│   ├── Integraciones de terceros
│   ├── Herramientas adicionales
│   └── Retrievers
│
└── LangChain Hub
    └── Prompts pre-hechos y compartidos
```

### Crear un Agente en LangChain

**Paso 1: Inicializar LLM**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)
```

**Paso 2: Definir Herramientas**
```python
from langchain.tools import Tool

tools = [
    Tool(
        name="search",
        description="Busca en internet",
        func=search_function
    ),
    Tool(
        name="calculator",
        description="Calcula matemáticas",
        func=calc_function
    )
]
```

**Paso 3: Crear Agente**
```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain import hub

# Obtener prompt pre-hecho
prompt = hub.pull("hwchase17/openai-functions-agent")

# Crear agente
agent = create_tool_calling_agent(llm, tools, prompt)

# Envolver en executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
```

**Paso 4: Ejecutar**
```python
result = agent_executor.invoke({
    "input": "¿Cuál es la raíz cuadrada de 144 y busca información sobre Pythagor

as?"
})

print(result["output"])
```

### Tipos de Agentes en LangChain

| Tipo | Descripción | Cuándo Usar |
|------|-------------|-------------|
| **Tool Calling** | Usa function calling nativo | OpenAI, Anthropic (recomendado) |
| **ReAct** | Patrón ReAct explícito | Modelos sin function calling |
| **Structured Chat** | Para tools con inputs complejos | Múltiples parámetros |
| **Conversational ReAct** | ReAct + memoria | Chatbots |
| **Self-Ask** | Descompone preguntas | Preguntas multi-paso |
| **Plan-and-Execute** | Planifica → Ejecuta | Tareas muy complejas |

---

## 👥 Sistemas Multi-Agente {#multi-agente}

### Concepto

En lugar de un solo agente general, múltiples agentes especializados colaboran.

### Arquitecturas Comunes

#### 1. **Pipeline Secuencial**

```
Agente 1 (Researcher) → Agente 2 (Writer) → Agente 3 (Editor)
```

**Ejemplo:**
```python
# Researcher busca información
research = researcher_agent.run("Python best practices")

# Writer crea artículo
article = writer_agent.run(f"Write about: {research}")

# Editor mejora
final = editor_agent.run(f"Edit: {article}")
```

#### 2. **Supervisor Pattern**

```
           Supervisor
          /     |     \
         /      |      \
    Agent1   Agent2   Agent3
```

El supervisor delega tareas según expertise.

#### 3. **Hierarchical**

```
CEO Agent
├── Manager Agent 1
│   ├── Worker Agent 1a
│   └── Worker Agent 1b
└── Manager Agent 2
    ├── Worker Agent 2a
    └── Worker Agent 2b
```

#### 4. **Autonomous Collaboration**

Agentes se comunican libremente sin jerarquía.

### Frameworks Multi-Agente

#### LangGraph

Framework de Langchain para workflows complejos con grafos.

```python
from langgraph.graph import Graph

# Definir nodos (agentes)
workflow = Graph()
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)

# Definir flujo
workflow.add_edge("researcher", "writer")
workflow.set_entry_point("researcher")
workflow.set_finish_point("writer")

# Ejecutar
result = workflow.invoke("Create article about AI")
```

#### CrewAI

Framework específico para multi-agente con roles.

```python
from crewai import Agent, Task, Crew

# Definir agentes con roles
researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert researcher with...",
    tools=[search_tool]
)

writer = Agent(
    role="Writer",
    goal="Write engaging content",
    backstory="Talented writer...",
    tools=[]
)

# Definir tareas
task1 = Task(
    description="Research about AI",
    agent=researcher
)

task2 = Task(
    description="Write article",
    agent=writer
)

# Crear crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2]
)

# Ejecutar
result = crew.kickoff()
```

#### AutoGen (Microsoft)

Framework para conversaciones entre agentes.

```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant")
user_proxy = UserProxyAgent("user")

user_proxy.initiate_chat(
    assistant,
    message="Help me write a Python script"
)
```

---

## 📊 Callbacks y Monitoring {#callbacks}

### ¿Qué son Callbacks?

Hooks que se ejecutan en diferentes momentos del ciclo del agente.

### Eventos de Callback

```python
class CustomCallback(BaseCallbackHandler):
    def on_llm_start(self, prompts, **kwargs):
        """Cuando el LLM empieza"""
        print(f"LLM starting with: {prompts[0][:100]}")
    
    def on_llm_end(self, response, **kwargs):
        """Cuando el LLM termina"""
        print(f"LLM finished")
    
    def on_tool_start(self, tool, input, **kwargs):
        """Cuando una herramienta empieza"""
        print(f"Using tool: {tool}")
    
    def on_tool_end(self, output, **kwargs):
        """Cuando una herramienta termina"""
        print(f"Tool returned: {output}")
    
    def on_agent_action(self, action, **kwargs):
        """Cuando el agente toma una acción"""
        print(f"Agent action: {action}")
    
    def on_agent_finish(self, finish, **kwargs):
        """Cuando el agente termina"""
        print(f"Agent finished: {finish}")
```

### Uso

```python
agent = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[CustomCallback()]
)
```

### Callbacks Pre-hechos

**OpenAI Callback:**
```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = agent.invoke({"input": "..."})
    
    print(f"Tokens: {cb.total_tokens}")
    print(f"Cost: ${cb.total_cost}")
    print(f"Successful requests: {cb.successful_requests}")
```

**Streaming Callback:**
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

agent = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[StreamingStdOutCallbackHandler()]
)
```

---

## ⚡ Mejores Prácticas {#mejores-prácticas}

### 1. Diseño de Prompts

```python
# ✅ BIEN: Instrucciones claras
system_prompt = """You are a helpful assistant that:
- Always uses tools when you need external information
- Explains your reasoning before taking actions
- Provides sources for factual information
- Admits when you don't know something"""

# ❌ MAL: Vago
system_prompt = "You are a helpful assistant"
```

### 2. Límites de Seguridad

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,        # Evita loops infinitos
    max_execution_time=60,    # Timeout
    early_stopping_method="generate"  # Genera respuesta si se detiene
)
```

### 3. Manejo de Errores en Tools

```python
def safe_tool(input: str) -> str:
    try:
        result = risky_operation(input)
        return f"Success: {result}"
    except SpecificError as e:
        return f"Error: {str(e)}. Please try with different input."
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "An unexpected error occurred. Please try again."
```

### 4. Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def logged_agent_call(query):
    logger.info(f"Starting agent with query: {query}")
    
    try:
        result = agent.invoke({"input": query})
        logger.info(f"Success: {result['output'][:100]}")
        return result
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        raise
```

### 5. Testing de Agentes

```python
def test_agent():
    # Test simple
    result = agent.invoke({"input": "What is 2+2?"})
    assert "4" in result["output"]
    
    # Test con tool
    result = agent.invoke({"input": "Search for Python"})
    assert result["intermediate_steps"]  # Verificar que usó tool
    
    # Test de error handling
    result = agent.invoke({"input": "Invalid query ###"})
    assert result is not None  # No debe crashear
```

---

## 📚 Recursos Adicionales

### Documentación

- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [CrewAI](https://docs.crewai.com/)

### Papers Importantes

- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)

### Tutoriales

- [LangChain Agents Tutorial](https://python.langchain.com/docs/tutorials/agents/)
- [Building Production-Ready Agents](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/)

---

## 🎓 Próximos Pasos

Después de dominar agentes:

- **Koan 12: Semantic Search** - Para que tus agentes busquen en documentos
- **Koan 13: RAG** - Combina agentes con retrieval
- **LangGraph** - Para workflows multi-agente avanzados

¡Construye agentes increíbles! 🚀
