# Hints para Koan 11: AI Agents

## Pista 1: create_react_agent()

<details>
<summary>Ver Pista Nivel 1</summary>

Usa LangChain para crear un agente ReAct (Reasoning + Acting):
- Instala: `langchain`, `langchain-openai`
- El agente necesita: LLM + Tools + Prompt
- ReAct combina razonamiento y acción en iteraciones

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4", temperature=0)
# Define herramientas (tools)
# Crea el prompt del agente
# return create_react_agent(llm, tools, prompt)
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from langchain.agents import create_react_agent
from langchain_openai import ChatOpenAI
from langchain import hub

def create_react_agent(tools):
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    return agent
```

</details>

---

## Pista 2: run_agent()

<details>
<summary>Ver Pista Nivel 1</summary>

Para ejecutar un agente necesitas:
- AgentExecutor (envuelve el agente)
- Input como diccionario con "input" key
- El agente itera: Thought → Action → Observation

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langchain.agents import AgentExecutor

executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": query})
return result["output"]
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from langchain.agents import AgentExecutor

def run_agent(agent, tools, query: str) -> str:
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5
    )
    result = executor.invoke({"input": query})
    return result["output"]
```

</details>

---

## Pista 3: agent_with_memory()

<details>
<summary>Ver Pista Nivel 1</summary>

Memoria conversacional para agentes:
- `ConversationBufferMemory`: Guarda todo el historial
- Usa `memory_key="chat_history"`
- El AgentExecutor acepta parámetro `memory`

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

executor = AgentExecutor(agent=agent, tools=tools, memory=memory)
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory

def agent_with_memory(agent, tools):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )
    return executor
```

</details>

---

## Pista 4: create_calculator_tool()

<details>
<summary>Ver Pista Nivel 1</summary>

LangChain tiene tools predefinidas:
- `load_tools(["llm-math"], llm=llm)`
- O crea tu propia con `@tool` decorator
- Tools necesitan: nombre, descripción, función

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langchain.agents import load_tools

tools = load_tools(["llm-math"], llm=llm)
return tools[0]
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from langchain.agents import load_tools
from langchain_openai import ChatOpenAI

def create_calculator_tool():
    llm = ChatOpenAI(temperature=0)
    tools = load_tools(["llm-math"], llm=llm)
    return tools[0]
```

</details>

---

## Pista 5: create_search_tool()

<details>
<summary>Ver Pista Nivel 1</summary>

Herramienta de búsqueda web:
- Instala: `duckduckgo-search`
- `load_tools(["ddg-search"])`
- No requiere API key (DuckDuckGo es gratis)

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langchain.agents import load_tools

tools = load_tools(["ddg-search"])
return tools[0]
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from langchain.agents import load_tools

def create_search_tool():
    tools = load_tools(["ddg-search"])
    return tools[0]
```

</details>

---

## Pista 6: create_custom_tool()

<details>
<summary>Ver Pista Nivel 1</summary>

Crea tus propias herramientas con `@tool`:
```python
from langchain.tools import tool

@tool
def my_tool(query: str) -> str:
    """Descripción de qué hace la herramienta"""
    # Tu código aquí
    return result
```

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langchain.tools import tool

@tool
def get_word_length(word: str) -> int:
    """Calcula la longitud de una palabra."""
    return len(word)

return get_word_length
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from langchain.tools import tool

def create_custom_tool():
    @tool
    def get_word_length(word: str) -> int:
        """Returns the length of a word."""
        return len(word)
    
    return get_word_length
```

</details>

---

## Pista 7: agent_with_callbacks()

<details>
<summary>Ver Pista Nivel 1</summary>

Callbacks para monitorear agentes:
- `StdOutCallbackHandler`: Imprime en consola
- O crea tu propio handler heredando de `BaseCallbackHandler`
- Pásalo al AgentExecutor con `callbacks=[...]`

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langchain.callbacks import StdOutCallbackHandler

callbacks = [StdOutCallbackHandler()]

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=callbacks
)
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from langchain.agents import AgentExecutor
from langchain.callbacks import StdOutCallbackHandler

def agent_with_callbacks(agent, tools):
    callbacks = [StdOutCallbackHandler()]
    
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        callbacks=callbacks,
        verbose=True
    )
    return executor
```

</details>

---

## Pista 8: multi_agent_collaboration()

<details>
<summary>Ver Pista Nivel 1</summary>

Multi-agente con LangGraph:
- Define múltiples agentes con roles específicos
- Usa LangGraph para orquestar
- Cada agente puede tener sus propias herramientas

Alternativa simple: usar un agente "supervisor" que delega a otros

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
# Enfoque simple: crear agentes especializados

researcher = create_react_agent(research_tools)
writer = create_react_agent(writing_tools)

def collaborate(query):
    research_result = run_agent(researcher, research_tools, query)
    final_result = run_agent(writer, writing_tools, 
                            f"Write about: {research_result}")
    return final_result
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain import hub

def multi_agent_collaboration():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    prompt = hub.pull("hwchase17/react")
    
    # Agente 1: Researcher
    research_tools = [create_search_tool()]
    researcher = create_react_agent(llm, research_tools, prompt)
    researcher_executor = AgentExecutor(agent=researcher, tools=research_tools)
    
    # Agente 2: Writer
    writer_tools = []  # Solo LLM, sin herramientas
    writer = create_react_agent(llm, writer_tools, prompt)
    writer_executor = AgentExecutor(agent=writer, tools=writer_tools)
    
    return {
        "researcher": researcher_executor,
        "writer": writer_executor
    }
```

</details>

---

## Conceptos Clave

### ¿Qué es un Agente AI?
- Sistema autónomo que usa LLM para razonar
- Decide qué acciones tomar iterativamente
- Usa herramientas (tools) para interactuar con el mundo

### ReAct Pattern
```
Thought: Necesito buscar información sobre X
Action: search("X")
Observation: [resultado de búsqueda]
Thought: Ahora tengo la información, puedo responder
Final Answer: [respuesta]
```

### Componentes de un Agente
1. **LLM**: Motor de razonamiento
2. **Tools**: Funciones que el agente puede usar
3. **Prompt**: Instrucciones de cómo razonar
4. **Memory**: Historial conversacional (opcional)
5. **Callbacks**: Monitoreo y logging (opcional)

### Herramientas Comunes
- **llm-math**: Calculadora matemática
- **ddg-search**: Búsqueda en DuckDuckGo
- **wikipedia**: Búsqueda en Wikipedia
- **python_repl**: Ejecutar código Python
- **Custom tools**: Tus propias funciones

### Mejores Prácticas
- Usa GPT-4 para agentes (más razonamiento)
- Da descripciones claras a tus herramientas
- Limita `max_iterations` para evitar loops
- Usa `verbose=True` para debugging
- Añade memoria para conversaciones

### Frameworks para Agentes
- **LangChain**: Framework completo
- **LangGraph**: Multi-agente avanzado
- **AutoGPT**: Agentes autónomos
- **CrewAI**: Equipos de agentes

## Recursos
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangGraph Tutorial](https://langchain-ai.github.io/langgraph/)
