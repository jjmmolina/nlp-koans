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

---

## Pista 9: create_langgraph_agent()

<details>
<summary>Ver Pista Nivel 1</summary>

LangGraph modela el agente como un grafo:
- `StateGraph` define el grafo con un estado tipado
- Los nodos son funciones que reciben y devuelven estado
- Las aristas condicionales deciden el siguiente paso
- `ToolNode` ejecuta las herramientas automáticamente

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

workflow = StateGraph(AgentState)
workflow.add_node("agent", llm_node)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
app = workflow.compile()
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

def create_langgraph_agent(tools, model="gpt-4.1-mini", max_iterations=10):
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
    
    llm = ChatOpenAI(model=model, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    def llm_node(state):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    def should_continue(state):
        last = state["messages"][-1]
        if last.tool_calls:
            return "tools"
        return END
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", llm_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue,
                                   {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()
```

</details>

---

## Pista 10: create_agentic_pipeline()

<details>
<summary>Ver Pista Nivel 1</summary>

El patrón Plan-Execute divide la tarea en dos fases:
- **Fase 1 (Planning)**: Un LLM de razonamiento crea un plan estructurado
- **Fase 2 (Execution)**: Cada paso del plan se ejecuta con herramientas

Para el plan, usa structured output (JSON) para obtener pasos claros.

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
# 1. Planning con structured output
planning_llm = ChatOpenAI(model=planning_model)
plan_response = planning_llm.invoke([
    SystemMessage(content="Crea un plan estructurado en JSON con una lista 'steps'"),
    HumanMessage(content=f"Tarea: {task}")
])

# 2. Execution de cada paso
agent = create_react_agent(llm_with_tools, tools, prompt)
executor = AgentExecutor(agent=agent, tools=available_tools)

for step in plan["steps"]:
    result = executor.invoke({"input": step})
    steps_results.append(result["output"])
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

def create_agentic_pipeline(task, available_tools, model="gpt-4.1", planning_model="o4-mini"):
    # 1. Planificar con modelo de razonamiento
    planning_llm = ChatOpenAI(model=planning_model, temperature=1)
    plan_prompt = [
        SystemMessage(content='Responde SOLO con JSON: {"steps": ["paso1", "paso2", ...]}'),
        HumanMessage(content=f"Planifica esta tarea: {task}")
    ]
    plan_json = planning_llm.invoke(plan_prompt).content
    plan = json.loads(plan_json)
    
    # 2. Ejecutar cada paso
    exec_llm = ChatOpenAI(model=model, temperature=0)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(exec_llm, available_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=available_tools, max_iterations=5)
    
    steps_results = []
    for step in plan["steps"]:
        result = executor.invoke({"input": step})
        steps_results.append(result["output"])
    
    # 3. Síntesis final
    final = exec_llm.invoke([
        SystemMessage(content="Sintetiza los resultados en una respuesta final"),
        HumanMessage(content=f"Tarea: {task}\nResultados: {steps_results}")
    ])
    
    return {"plan": plan["steps"], "steps_results": steps_results, "output": final.content}
```

</details>

---

## Pista 11: create_multi_agent_crew()

<details>
<summary>Ver Pista Nivel 1</summary>

CrewAI facilita crear equipos de agentes con roles:
- Cada `Agent` tiene `role`, `goal` y `backstory`
- Cada `Task` tiene `description` y `agent` asignado
- `Crew` orquesta la ejecución secuencial o paralela

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Investigador",
    goal="Encontrar información precisa y actualizada",
    backstory="Experto en investigación con acceso a internet",
    tools=researcher_tools
)

task = Task(description=task, agent=researcher)
crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

def create_multi_agent_crew(task, researcher_tools, writer_tools, model="gpt-4.1"):
    llm = ChatOpenAI(model=model, temperature=0.3)
    
    researcher = Agent(
        role="Investigador Senior",
        goal="Investigar en profundidad el tema dado",
        backstory="Analista experto con acceso a fuentes de información.",
        tools=researcher_tools,
        llm=llm
    )
    
    writer = Agent(
        role="Escritor",
        goal="Redactar contenido claro y atractivo",
        backstory="Comunicador experto que transforma información en texto.",
        tools=writer_tools,
        llm=llm
    )
    
    research_task = Task(description=f"Investiga sobre: {task}", agent=researcher)
    write_task = Task(
        description=f"Escribe un informe sobre: {task}. Usa la investigación previo.",
        agent=writer
    )
    
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task]
    )
    
    return crew.kickoff()
```

</details>

---

## Pista 12: create_human_in_the_loop_agent()

<details>
<summary>Ver Pista Nivel 1</summary>

LangGraph soporta HITL con `interrupt()`:
- El nodo llama a `interrupt(data)` para pausar la ejecución
- Se necesita un `checkpointer` para guardar el estado
- Reanudar con `app.invoke(Command(resume=human_response), config)`

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

def action_node(state):
    # Pausa y pide aprobación humana
    approval = interrupt({"message": "¿Aprobar?", "action": state["action"]})
    if approval.get("approved"):
        return {"result": execute()}
    return {"result": "Cancelado"}

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
from typing import TypedDict, Annotated, Callable, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

def create_human_in_the_loop_agent(tools, model="gpt-4.1", approval_callback=None):
    class HITLState(TypedDict):
        messages: Annotated[list, add_messages]
        pending_tool_call: Optional[dict]
    
    llm = ChatOpenAI(model=model, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    def llm_node(state):
        response = llm_with_tools.invoke(state["messages"])
        if response.tool_calls:
            return {"messages": [response], "pending_tool_call": response.tool_calls[0]}
        return {"messages": [response], "pending_tool_call": None}
    
    def human_approval_node(state):
        pending = state["pending_tool_call"]
        if pending:
            # Pausa aquí y espera respuesta humana
            approval = interrupt({"tool_call": pending, "message": f"¿Ejecutar {pending['name']}?"})
            if not approval.get("approved", True):
                return {"messages": [{"role": "tool", "content": "Acción cancelada", "tool_call_id": pending["id"]}]}
        return {}
    
    def router(state):
        if state.get("pending_tool_call"):
            return "human_approval"
        return END
    
    workflow = StateGraph(HITLState)
    workflow.add_node("agent", llm_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", router,
                                   {"human_approval": "human_approval", END: END})
    workflow.add_edge("human_approval", "tools")
    workflow.add_edge("tools", "agent")
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
```

</details>

---

## Pista 13: setup_mcp_agent()

<details>
<summary>Ver Pista Nivel 1</summary>

MCP (Model Context Protocol) permite conectar agentes con servidores de herramientas:
- Usa `langchain-mcp-adapters` para integrar MCP con LangChain
- `MultiServerMCPClient` se conecta a múltiples servidores MCP a la vez
- Los servidores pueden ser stdio (proceso local) o sse (HTTP)

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

async with MultiServerMCPClient({
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        "transport": "stdio"
    }
}) as client:
    tools = client.get_tools()
    agent = create_react_agent(llm, tools)
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

def setup_mcp_agent(mcp_server_urls, model="gpt-4.1"):
    llm = ChatOpenAI(model=model, temperature=0)
    prompt = hub.pull("hwchase17/react")
    
    # Construir configuración MCP para cada servidor
    servers_config = {}
    for i, url in enumerate(mcp_server_urls):
        if url.startswith("http"):
            servers_config[f"server_{i}"] = {"url": url, "transport": "sse"}
        else:
            # Comando stdio
            servers_config[f"server_{i}"] = {
                "command": "npx", "args": ["-y", url], "transport": "stdio"
            }
    
    async def _run_agent(query):
        async with MultiServerMCPClient(servers_config) as client:
            tools = client.get_tools()
            agent = create_react_agent(llm, tools, prompt)
            executor = AgentExecutor(agent=agent, tools=tools, max_iterations=10)
            result = await executor.ainvoke({"input": query})
            return result["output"]
    
    return _run_agent
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
- **LangChain**: Framework base con AgentExecutor
- **LangGraph**: Agentes con estado como grafos (moderno, recomendado)
- **CrewAI**: Equipos de agentes con roles
- **AutoGen**: Conversaciones multi-agente (Microsoft)

### Agentic Mode en 2026
- `create_langgraph_agent()` — LangGraph StateGraph (funciones 9)
- `create_agentic_pipeline()` — Plan-Execute pattern (función 10)
- `create_multi_agent_crew()` — CrewAI roles (función 11)
- `create_human_in_the_loop_agent()` — LangGraph interrupt() (función 12)
- `setup_mcp_agent()` — Model Context Protocol (función 13)
- `create_parallel_agents()` — asyncio.gather + Semaphore (función 14)

---

## Pista 14: create_parallel_agents()

<details>
<summary>Ver Pista Nivel 1</summary>

El truco es usar el método **async** de los LLMs — todos los clientes de
LangChain exponen `ainvoke()` además de `invoke()`.

- `asyncio.gather(*coroutines)` lanza todas las corrutinas a la vez y espera
  a que terminen todas.
- Para no superar el rate-limit de la API, usa `asyncio.Semaphore(n)` como
  gestor de contexto antes de cada llamada.

</details>

<details>
<summary>Ver Pista Nivel 2</summary>

```python
import asyncio
from langchain_openai import ChatOpenAI

async def _run_all(tasks, model, max_concurrent):
    llm = ChatOpenAI(model=model)
    sem = asyncio.Semaphore(max_concurrent)

    async def call_one(task):
        async with sem:
            response = await llm.ainvoke(task)
            return response.content

    return await asyncio.gather(*[call_one(t) for t in tasks],
                                 return_exceptions=True)

# Lanzar desde código síncrono
results = asyncio.run(_run_all(tasks, model, max_concurrent))
```

</details>

<details>
<summary>Ver Pista Nivel 3 (Solución)</summary>

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

def create_parallel_agents(tasks, tools=None, model="gpt-4.1-mini", max_concurrent=5):
    llm = ChatOpenAI(model=model, temperature=0)
    tools = tools or []

    async def _run_all():
        sem = asyncio.Semaphore(max_concurrent)

        if tools:
            prompt = hub.pull("hwchase17/react")
            agent = create_react_agent(llm, tools, prompt)
            executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5)

            async def call_one(task):
                async with sem:
                    result = await executor.ainvoke({"input": task})
                    return result["output"]
        else:
            async def call_one(task):
                async with sem:
                    response = await llm.ainvoke(task)
                    return response.content

        return await asyncio.gather(
            *[call_one(t) for t in tasks],
            return_exceptions=True
        )

    return asyncio.run(_run_all())
```

</details>

## Recursos
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [LangGraph Send API / Map-Reduce](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)
- [CrewAI Docs](https://docs.crewai.com/)
- [MCP Specification](https://modelcontextprotocol.io/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
