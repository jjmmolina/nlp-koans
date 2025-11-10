"""
Koan 11: AI Agents - Agentes Autónomos con LLMs

Este koan explora cómo crear agentes de IA que pueden:
- Usar herramientas (tools)
- Razonar y planificar (ReAct)
- Mantener memoria de conversaciones
- Tomar decisiones autónomas

Librerías:
- langchain
- langchain-openai
- langchain-community
"""

from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass


@dataclass
class Tool:
    """Definición de una herramienta que el agente puede usar"""

    name: str
    description: str
    function: Callable


def create_simple_agent(
    tools: List[Tool], llm_provider: str = "openai", model: str = "gpt-4o-mini"
) -> Any:
    """
    Crea un agente simple con herramientas.

    Ejemplo:
        >>> tools = [
        ...     Tool("calculator", "Calcula operaciones", calculate),
        ...     Tool("search", "Busca información", search_web)
        ... ]
        >>> agent = create_simple_agent(tools)

    Args:
        tools: Lista de herramientas disponibles
        llm_provider: Proveedor del LLM
        model: Modelo a usar

    Returns:
        Agente configurado
    """
    # TODO: Implementa creación de agente con LangChain
    # Pistas:
    # 1. from langchain.agents import create_tool_calling_agent, AgentExecutor
    # 2. from langchain_openai import ChatOpenAI
    # 3. llm = ChatOpenAI(model=model)
    # 4. agent = create_tool_calling_agent(llm, tools, prompt)
    # 5. return AgentExecutor(agent=agent, tools=tools)
    pass


def run_agent(agent: Any, query: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Ejecuta el agente con una consulta.

    Ejemplo:
        >>> result = run_agent(agent, "¿Cuál es la raíz cuadrada de 144?")
        >>> print(result["output"])

    Args:
        agent: Agente a ejecutar
        query: Consulta del usuario
        verbose: Mostrar pasos intermedios

    Returns:
        Dict con output e información de ejecución
    """
    # TODO: Ejecuta el agente
    # Pistas:
    # 1. result = agent.invoke({"input": query})
    # 2. return {"output": result["output"], "steps": result.get("intermediate_steps", [])}
    pass


def create_react_agent(tools: List[Tool], model: str = "gpt-4o-mini") -> Any:
    """
    Crea un agente ReAct (Reasoning + Acting).

    ReAct alterna entre razonamiento y acción:
    1. Piensa (Thought)
    2. Actúa (Action)
    3. Observa (Observation)
    4. Repite hasta resolver

    Ejemplo:
        >>> agent = create_react_agent(tools)
        >>> result = run_agent(agent, "Investiga sobre transformers en NLP")

    Args:
        tools: Herramientas disponibles
        model: Modelo LLM

    Returns:
        Agente ReAct configurado
    """
    # TODO: Implementa agente ReAct
    # Pistas:
    # 1. from langchain.agents import create_react_agent
    # 2. Usa prompt de ReAct específico
    # 3. return AgentExecutor(agent=agent, tools=tools, verbose=True)
    pass


def create_conversational_agent(
    tools: List[Tool], memory_type: str = "buffer", model: str = "gpt-4o-mini"
) -> Any:
    """
    Crea un agente con memoria conversacional.

    Tipos de memoria:
    - buffer: Guarda últimos N mensajes
    - summary: Resume conversación
    - knowledge_graph: Extrae relaciones

    Ejemplo:
        >>> agent = create_conversational_agent(tools, memory_type="buffer")
        >>> run_agent(agent, "Mi nombre es Ana")
        >>> run_agent(agent, "¿Cómo me llamo?")  # Recuerda!

    Args:
        tools: Herramientas disponibles
        memory_type: Tipo de memoria
        model: Modelo LLM

    Returns:
        Agente con memoria
    """
    # TODO: Implementa agente con memoria
    # Pistas:
    # 1. from langchain.memory import ConversationBufferMemory
    # 2. memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # 3. agent = create_tool_calling_agent(llm, tools, prompt)
    # 4. return AgentExecutor(agent=agent, tools=tools, memory=memory)
    pass


def create_calculator_tool() -> Tool:
    """
    Crea una herramienta de calculadora para el agente.

    Ejemplo:
        >>> calc = create_calculator_tool()
        >>> result = calc.function("5 + 3 * 2")
        >>> print(result)  # 11

    Returns:
        Tool de calculadora
    """
    # TODO: Implementa herramienta de calculadora
    # Pistas:
    # 1. from langchain.tools import Tool
    # 2. def calculate(expression: str) -> str: return str(eval(expression))
    # 3. return Tool(name="calculator", description="...", func=calculate)
    pass


def create_search_tool() -> Tool:
    """
    Crea una herramienta de búsqueda web.

    Usa DuckDuckGo para búsquedas (no requiere API key).

    Ejemplo:
        >>> search = create_search_tool()
        >>> result = search.function("Python programming")

    Returns:
        Tool de búsqueda
    """
    # TODO: Implementa herramienta de búsqueda
    # Pistas:
    # 1. from langchain_community.tools import DuckDuckGoSearchRun
    # 2. search = DuckDuckGoSearchRun()
    # 3. return Tool(name="search", description="...", func=search.run)
    pass


def create_custom_tool(
    name: str, description: str, function: Callable[[str], str]
) -> Tool:
    """
    Crea una herramienta personalizada.

    Ejemplo:
        >>> def get_time(query: str) -> str:
        ...     from datetime import datetime
        ...     return datetime.now().strftime("%H:%M:%S")
        >>>
        >>> time_tool = create_custom_tool(
        ...     "current_time",
        ...     "Obtiene la hora actual",
        ...     get_time
        ... )

    Args:
        name: Nombre de la herramienta
        description: Descripción (ayuda al agente a decidir cuándo usarla)
        function: Función a ejecutar

    Returns:
        Tool personalizada
    """
    # TODO: Implementa herramienta personalizada
    # Pistas:
    # 1. from langchain.tools import Tool
    # 2. return Tool(name=name, description=description, func=function)
    pass


def agent_with_callbacks(agent: Any, query: str) -> Dict[str, Any]:
    """
    Ejecuta un agente con callbacks para monitorear ejecución.

    Los callbacks permiten:
    - Ver cada paso
    - Contar tokens usados
    - Medir tiempos
    - Debug

    Ejemplo:
        >>> result = agent_with_callbacks(agent, "Calculate 5 * 7")
        >>> print(f"Steps: {len(result['steps'])}")
        >>> print(f"Tokens: {result['total_tokens']}")

    Args:
        agent: Agente a ejecutar
        query: Consulta

    Returns:
        Dict con resultados y métricas
    """
    # TODO: Implementa callbacks
    # Pistas:
    # 1. from langchain.callbacks import get_openai_callback
    # 2. with get_openai_callback() as cb:
    # 3.     result = agent.invoke({"input": query})
    # 4. return {"output": result["output"], "total_tokens": cb.total_tokens}
    pass


def multi_agent_collaboration(
    researcher_tools: List[Tool], writer_tools: List[Tool], query: str
) -> str:
    """
    Crea sistema multi-agente donde cada agente tiene rol específico.

    Ejemplo:
        Researcher: Busca información
        Writer: Escribe artículo basado en investigación

    Args:
        researcher_tools: Herramientas del investigador
        writer_tools: Herramientas del escritor
        query: Tarea a realizar

    Returns:
        Resultado final
    """
    # TODO: Implementa multi-agent system
    # Pistas:
    # 1. researcher = create_simple_agent(researcher_tools)
    # 2. research_result = run_agent(researcher, f"Research: {query}")
    # 3. writer = create_simple_agent(writer_tools)
    # 4. final = run_agent(writer, f"Write article about: {research_result}")
    pass
