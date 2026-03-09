"""
Koan 11: AI Agents - Agentes Autónomos con LLMs y Agentic Mode

Este koan explora cómo crear agentes de IA modernos:
- Tool use (herramientas)
- ReAct (Reasoning + Acting)
- Memoria conversacional
- LangGraph (state machines agénticos)
- Multi-agente con LangGraph / CrewAI
- Model Context Protocol (MCP)
- Agentic workflows (plan-execute)
- Human-in-the-loop

Librerías:
- langchain
- langchain-openai
- langchain-community
- langgraph
- crewai (opcional)
"""

from typing import List, Dict, Any, Callable, Optional, TypedDict, Annotated
from dataclasses import dataclass


@dataclass
class Tool:
    """Definición de una herramienta que el agente puede usar"""

    name: str
    description: str
    function: Callable


def create_simple_agent(
    tools: List[Tool], llm_provider: str = "openai", model: str = "gpt-4.1-mini"
) -> Any:
    """
    Crea un agente simple con herramientas.

    Un agente de IA es un sistema que puede:
    - Analizar una petición del usuario
    - Decidir qué herramientas necesita usar
    - Ejecutar esas herramientas
    - Combinar los resultados en una respuesta coherente

    LangChain proporciona diferentes tipos de agentes. El más simple es
    el "tool calling agent" que usa las capacidades de function calling
    del LLM subyacente.

    Ejemplo:
        >>> tools = [
        ...     Tool("calculator", "Calcula operaciones", calculate),
        ...     Tool("search", "Busca información", search_web)
        ... ]
        >>> agent = create_simple_agent(tools)

    Args:
        tools: Lista de herramientas disponibles para el agente
        llm_provider: Proveedor del LLM ("openai", "anthropic", etc.)
        model: Modelo específico a usar

    Returns:
        Agente configurado listo para recibir consultas

    Nota:
        El agente necesita un "prompt" que le explique cómo usar las herramientas.
        LangChain proporciona prompts pre-configurados o puedes crear el tuyo.
        Ver THEORY.md para entender la arquitectura de agentes.
    """
    # TODO: Implementa creación de agente con LangChain
    # Pista: Necesitas inicializar un LLM y crear un AgentExecutor
    # Consulta HINTS.md para la estructura completa
    pass


def run_agent(agent: Any, query: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Ejecuta el agente con una consulta.

    Cuando ejecutas un agente, este realiza un loop de razonamiento:
    1. Analiza la consulta
    2. Decide si necesita usar herramientas
    3. Ejecuta herramientas si es necesario
    4. Observa los resultados
    5. Genera la respuesta final

    Este proceso puede tomar varios pasos ("intermediate steps") antes
    de llegar a la respuesta final.

    Ejemplo:
        >>> result = run_agent(agent, "¿Cuál es la raíz cuadrada de 144?")
        >>> print(result["output"])  # "La raíz cuadrada de 144 es 12"

    Args:
        agent: Agente (AgentExecutor) a ejecutar
        query: Consulta o pregunta del usuario
        verbose: Si True, muestra los pasos intermedios del razonamiento

    Returns:
        Dict con 'output' (respuesta final) y 'steps' (pasos intermedios)

    Nota:
        Los pasos intermedios son útiles para debugging y entender cómo
        el agente llegó a su respuesta. Cada paso incluye el pensamiento
        del agente, la acción tomada, y la observación resultante.
    """
    # TODO: Ejecuta el agente con la consulta
    # Pista: El agente espera un dict con "input"
    pass


def create_react_agent(tools: List[Tool], model: str = "gpt-4.1-mini") -> Any:
    """
    Crea un agente ReAct (Reasoning + Acting).

    ReAct es un patrón de prompting donde el agente alterna entre:
    - **Thought** (Pensamiento): Razona sobre qué hacer
    - **Action**: Elige y ejecuta una herramienta
    - **Observation**: Analiza el resultado
    - Repite hasta resolver la tarea

    Este patrón fue introducido en el paper "ReAct: Synergizing Reasoning and
    Acting in Language Models" y mejora significativamente la capacidad de
    los agentes para resolver tareas complejas de forma transparente.

    Ejemplo de ejecución:
        Thought: Necesito buscar información sobre transformers
        Action: search["transformers in NLP"]
        Observation: Los transformers son una arquitectura...
        Thought: Ahora tengo la información, puedo responder
        Final Answer: Los transformers son...

    Ejemplo:
        >>> agent = create_react_agent(tools)
        >>> result = run_agent(agent, "Investiga sobre transformers en NLP")

    Args:
        tools: Herramientas disponibles para el agente
        model: Modelo LLM a usar (GPT-4 recomendado para mejor razonamiento)

    Returns:
        Agente ReAct configurado con el prompt específico

    Nota:
        El prompt de ReAct es crucial para su funcionamiento. LangChain
        proporciona templates de ReAct pre-optimizados. Ver THEORY.md
        para ejemplos del formato de pensamiento del agente.
    """
    # TODO: Implementa agente ReAct
    # Pista: Busca el prompt template específico de ReAct
    pass


def create_conversational_agent(
    tools: List[Tool], memory_type: str = "buffer", model: str = "gpt-4.1-mini"
) -> Any:
    """
    Crea un agente con memoria conversacional.

    La memoria permite al agente recordar interacciones previas en la conversación.
    Esto es esencial para mantener contexto y construir sobre respuestas anteriores.

    **Tipos de memoria:**

    - **buffer**: Almacena los últimos N mensajes completos
      - Ventaja: Contexto completo
      - Desventaja: Consume muchos tokens con conversaciones largas

    - **summary**: Resume la conversación periódicamente
      - Ventaja: Escala mejor para conversaciones largas
      - Desventaja: Puede perder detalles

    - **knowledge_graph**: Extrae y almacena relaciones entre entidades
      - Ventaja: Retiene información estructurada
      - Desventaja: Más complejo de configurar

    Ejemplo:
        >>> agent = create_conversational_agent(tools, memory_type="buffer")
        >>> run_agent(agent, "Mi nombre es Ana")
        >>> run_agent(agent, "¿Cómo me llamo?")  # "Te llamas Ana"

    Args:
        tools: Herramientas disponibles
        memory_type: Tipo de memoria ("buffer", "summary", "knowledge_graph")
        model: Modelo LLM

    Returns:
        Agente con capacidad de recordar conversaciones previas

    Nota:
        La memoria debe integrarse con el AgentExecutor. La key de memoria
        (memory_key) debe coincidir con la variable en el prompt del agente.
        Ver THEORY.md para comparar diferentes estrategias de memoria.
    """
    # TODO: Implementa agente con memoria
    # Pista: Crea el objeto de memoria apropiado según el tipo
    pass


def create_calculator_tool() -> Tool:
    """
    Crea una herramienta de calculadora para el agente.

    Las herramientas (tools) son funciones que el agente puede usar para
    interactuar con el mundo externo. Una herramienta necesita:
    - **name**: Nombre descriptivo (usado por el LLM para identificarla)
    - **description**: Explicación clara de qué hace (crucial para que el LLM la use correctamente)
    - **function**: La función Python que se ejecuta

    La calculadora es una herramienta simple pero útil para operaciones
    matemáticas que los LLMs no manejan bien de forma nativa.

    Ejemplo:
        >>> calc = create_calculator_tool()
        >>> result = calc.function("5 + 3 * 2")
        >>> print(result)  # "11"

    Returns:
        Tool de calculadora lista para usar con un agente

    Nota:
        La función debe aceptar un string como input y retornar un string.
        Esto mantiene consistencia con otras herramientas y permite al
        agente procesar el resultado fácilmente. Ver THEORY.md para
        mejores prácticas de diseño de herramientas.
    """
    # TODO: Implementa herramienta de calculadora
    # Pista: Crea una función que evalúe expresiones matemáticas de forma segura
    pass


def create_search_tool() -> Tool:
    """
    Crea una herramienta de búsqueda web.

    La búsqueda web es una de las herramientas más poderosas para agentes,
    permitiéndoles acceder a información actualizada que no estaba en sus
    datos de entrenamiento.

    Esta implementación usa DuckDuckGo, que:
    - No requiere API key (gratis y fácil)
    - Respeta la privacidad del usuario
    - Proporciona resultados de calidad decente

    Para casos de uso más avanzados, considera Google Custom Search API,
    Bing Search API, o Serp API.

    Ejemplo:
        >>> search = create_search_tool()
        >>> result = search.function("Python programming tutorial")
        >>> print(result)  # Resultados de búsqueda relevantes

    Returns:
        Tool de búsqueda web

    Nota:
        La herramienta retorna un resumen de los resultados de búsqueda,
        no URLs completas. El agente debe interpretar esta información
        y extraer lo relevante para responder al usuario. Ver THEORY.md
        para otras opciones de búsqueda.
    """
    # TODO: Implementa herramienta de búsqueda
    # Pista: LangChain tiene integraciones para DuckDuckGo
    pass


def create_custom_tool(
    name: str, description: str, function: Callable[[str], str]
) -> Tool:
    """
    Crea una herramienta personalizada.

    Crear herramientas personalizadas te permite extender las capacidades
    de tu agente con cualquier funcionalidad que necesites:
    - Consultar bases de datos
    - Llamar APIs propias
    - Ejecutar scripts
    - Interactuar con hardware
    - Y mucho más

    La clave está en proporcionar una **description** clara y precisa.
    El LLM usa esta descripción para decidir CUÁNDO usar la herramienta,
    así que debe explicar claramente qué hace y cuándo es apropiada.

    Ejemplo:
        >>> def get_time(query: str) -> str:
        ...     from datetime import datetime
        ...     return datetime.now().strftime("%H:%M:%S")
        >>>
        >>> time_tool = create_custom_tool(
        ...     "current_time",
        ...     "Obtiene la hora actual en formato HH:MM:SS. Usa esta herramienta cuando el usuario pregunte qué hora es.",
        ...     get_time
        ... )

    Args:
        name: Nombre único de la herramienta (snake_case recomendado)
        description: Descripción detallada de qué hace y cuándo usarla
        function: Función que toma string y retorna string

    Returns:
        Tool personalizada lista para usar

    Nota:
        La función DEBE aceptar un argumento string (aunque no lo use) y
        DEBE retornar un string. Esto mantiene consistencia con el sistema
        de herramientas de LangChain. Ver THEORY.md para patrones avanzados.
    """
    # TODO: Implementa herramienta personalizada
    # Pista: Envuelve la función en un objeto Tool de LangChain
    pass


def agent_with_callbacks(agent: Any, query: str) -> Dict[str, Any]:
    """
    Ejecuta un agente con callbacks para monitorear ejecución.

    Los callbacks son hooks que se ejecutan en diferentes etapas del
    proceso del agente, permitiéndote:
    - **Observar** cada paso del razonamiento
    - **Contar tokens** y calcular costos
    - **Medir tiempos** de cada operación
    - **Registrar** información para debugging
    - **Interceptar** y modificar el comportamiento

    Son esenciales para:
    - Debugging de agentes complejos
    - Optimización de costos en producción
    - Auditoría y compliance
    - Monitoring de performance

    Ejemplo:
        >>> result = agent_with_callbacks(agent, "Calculate 5 * 7")
        >>> print(f"Respuesta: {result['output']}")
        >>> print(f"Steps tomados: {len(result['steps'])}")
        >>> print(f"Tokens usados: {result['total_tokens']}")
        >>> print(f"Costo: ${result.get('cost', 0):.4f}")

    Args:
        agent: Agente a ejecutar (AgentExecutor)
        query: Consulta del usuario

    Returns:
        Dict con resultados y métricas completas:
        - output: Respuesta final
        - steps: Pasos intermedios
        - total_tokens: Total de tokens usados
        - cost: Costo estimado en USD

    Nota:
        LangChain proporciona varios callbacks pre-hechos (OpenAI, Anthropic, etc.)
        También puedes crear callbacks personalizados heredando de BaseCallbackHandler.
        Ver THEORY.md para ejemplos de callbacks avanzados.
    """
    # TODO: Implementa ejecución con callbacks
    # Pista: Usa context manager para capturar métricas
    pass


def multi_agent_collaboration(
    researcher_tools: List[Tool], writer_tools: List[Tool], query: str
) -> str:
    """
    Crea sistema multi-agente donde cada agente tiene un rol específico.

    En sistemas complejos, un solo agente puede no ser suficiente. Los
    sistemas multi-agente permiten:
    - **Especialización**: Cada agente es experto en su dominio
    - **Paralelización**: Múltiples agentes trabajan simultáneamente
    - **Modularidad**: Más fácil de mantener y mejorar
    - **Escalabilidad**: Añade agentes según necesites

    Ejemplo de flujo:
    1. **Researcher Agent**: Busca información en internet
    2. **Writer Agent**: Recibe la investigación y escribe un artículo
    3. **Editor Agent** (opcional): Revisa y mejora el artículo

    Frameworks avanzados como **LangGraph** o **CrewAI** facilitan
    la orquestación de múltiples agentes con roles, jerarquías, y
    comunicación compleja.

    Ejemplo:
        >>> research_tools = [create_search_tool()]
        >>> writer_tools = []  # Solo usa el LLM
        >>> result = multi_agent_collaboration(
        ...     research_tools,
        ...     writer_tools,
        ...     "Escribe un artículo sobre transformers en NLP"
        ... )
        >>> print(result)  # Artículo completo basado en investigación web

    Args:
        researcher_tools: Herramientas del agente investigador (búsqueda, APIs)
        writer_tools: Herramientas del agente escritor (usualmente ninguna)
        query: Tarea a realizar

    Returns:
        Resultado final después de la colaboración de ambos agentes

    Nota:
        Esta es una implementación simple secuencial. Para sistemas más
        complejos, considera usar LangGraph que permite flujos condicionales,
        loops, y comunicación bidireccional. Ver THEORY.md para arquitecturas
        avanzadas multi-agente.
    """
    # TODO: Implementa sistema multi-agente
    # Pista: Crea dos agentes y coordina su ejecución secuencial
    pass


# =============================================================================
# AGENTIC MODE - SECCIÓN NUEVA (2025-2026)
# =============================================================================


class AgentState(TypedDict):
    """Estado del agente LangGraph. Define la memoria compartida del grafo."""

    messages: Annotated[List[Dict[str, Any]], "Lista de mensajes de la conversación"]
    current_step: str
    tool_results: List[str]


def create_langgraph_agent(
    tools: List[Tool],
    model: str = "gpt-4.1-mini",
    max_iterations: int = 10,
) -> Any:
    """
    Crea un agente usando LangGraph — el estándar moderno para agentes en producción.

    **¿Por qué LangGraph?**
    LangGraph representa la evolución de los agentes: en lugar de un "AgentExecutor"
    monolítico, modela el agente como un **grafo de estados** donde cada nodo es
    una función y las aristas son transiciones condicionales. Esto permite:
    - Ciclos explícitos (el agente puede iterar)
    - Bifurcaciones (si/si-no)
    - Estado persistente entre pasos
    - Human-in-the-loop integrado
    - Mejor control y debugging

    **Arquitectura del grafo:**
    ```
    START → [llm_node] → ¿tool_call?
                ↓ No          ↓ Sí
              END       [tool_node]
                              ↓
                         [llm_node] → ...
    ```

    Ejemplo:
        >>> tools = [create_calculator_tool(), create_search_tool()]
        >>> agent = create_langgraph_agent(tools)
        >>> result = agent.invoke({"messages": [{"role": "user", "content": "¿Cuánto es raíz de 144?"}]})
        >>> print(result["messages"][-1]["content"])

    Args:
        tools: Lista de herramientas disponibles
        model: Modelo LLM a usar
        max_iterations: Límite de iteraciones para evitar bucles infinitos

    Returns:
        Grafo LangGraph compilado listo para invocar

    Nota:
        El grafo se invoca con un dict que contiene "messages".
        La respuesta final está en result["messages"][-1]["content"].
        Ver THEORY.md para entender la arquitectura completa.
    """
    # TODO: Implementa agente LangGraph
    # Pista: Crea un StateGraph con AgentState, añade nodos y aristas condicionales
    # Usa create_react_agent de langchain o implementa los nodos manualmente
    pass


def create_agentic_pipeline(
    task: str,
    available_tools: List[Tool],
    model: str = "gpt-4.1",
    planning_model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Implementa un pipeline agéntico Plan-Execute para tareas complejas.

    El patrón **Plan-Execute** es una mejora sobre ReAct para tareas largas:
    1. **Plan**: Un modelo de razonamiento (o4-mini) crea un plan step-by-step
    2. **Execute**: Un modelo rápido ejecuta cada paso del plan
    3. **Replan**: Si algo falla, se replantea

    Este patrón es más eficiente que ReAct para tareas predecibles porque:
    - Separa planificación (cara pero solo 1 vez) de ejecución (barata y repetida)
    - El plan es visible y editable antes de ejecutar
    - Más predecible y controlable

    **Flujo:**
    ```
    Task → [Planning LLM (o4-mini)] → Plan [{step1}, {step2}, ...]
                                          ↓
                                    [Execute step1]
                                          ↓
                                    [Execute step2]
                                          ↓ (si falla)
                                    [Replan]
                                          ↓
                                    [Final Answer]
    ```

    Ejemplo:
        >>> tools = [create_search_tool(), create_calculator_tool()]
        >>> result = create_agentic_pipeline(
        ...     "Busca el precio del iPhone 16 y calcula el precio con 21% de IVA",
        ...     tools
        ... )
        >>> print(result["plan"])     # Lista de pasos
        >>> print(result["output"])   # Respuesta final

    Args:
        task: Tarea compleja a realizar
        available_tools: Herramientas disponibles
        model: Modelo para la ejecución de pasos
        planning_model: Modelo para la planificación (preferiblemente de razonamiento)

    Returns:
        Dict con 'plan' (lista de pasos), 'steps_results' y 'output' (respuesta final)

    Nota:
        Usar un modelo de razonamiento (o4-mini) para planificar pero un modelo
        rápido (gpt-4.1-mini) para ejecutar cada paso optimiza costo y velocidad.
        Ver THEORY.md para comparar Plan-Execute vs ReAct.
    """
    # TODO: Implementa pipeline Plan-Execute
    # Pista 1: Usa planning_model para generar un plan estructurado (JSON)
    # Pista 2: Itera sobre los pasos del plan ejecutando con el agente
    # Pista 3: Maneja fallos con replanificación si es necesario
    pass


def create_multi_agent_crew(
    task: str,
    researcher_tools: List[Tool],
    writer_tools: List[Tool],
    model: str = "gpt-4.1",
) -> str:
    """
    Crea un sistema multi-agente usando CrewAI o composición manual.

    Los sistemas **multi-agente** dividen tareas complejas entre agentes
    especializados que colaboran. Cada agente tiene:
    - **Role**: Su función en el equipo (investigador, escritor, revisor)
    - **Goal**: Su objetivo específico
    - **Backstory**: Contexto que configura su comportamiento
    - **Tools**: Las herramientas que puede usar

    **Arquitectura del Crew:**
    ```
    Task
      ↓
    [Researcher Agent] → investiga y recopila información
      ↓ (pasa contexto)
    [Writer Agent] → escribe el contenido
      ↓
    Output
    ```

    **Frameworks disponibles:**
    - **CrewAI**: API declarativa de alto nivel, fácil de usar
    - **LangGraph**: Más control, mejor para producción
    - **AutoGen**: Microsoft, excelente para agentes que se comunican entre sí
    - **Smolagents**: HuggingFace, minimalista y flexible

    Ejemplo:
        >>> research_tools = [create_search_tool()]
        >>> writer_tools = []
        >>> result = create_multi_agent_crew(
        ...     "Escribe un artículo sobre los últimos avances en NLP",
        ...     research_tools, writer_tools
        ... )
        >>> print(result)  # Artículo completo

    Args:
        task: Tarea compleja que requiere colaboración
        researcher_tools: Herramientas del agente investigador
        writer_tools: Herramientas del agente escritor
        model: Modelo LLM para todos los agentes

    Returns:
        Resultado final después de la colaboración multi-agente

    Nota:
        Para una implementación más sofisticada, usa CrewAI directamente.
        Ver THEORY.md para arquitecturas avanzadas de multi-agente.
    """
    # TODO: Implementa sistema multi-agente
    # Opción A: Usa CrewAI (pip install crewai)
    # Opción B: Coordina dos agentes LangChain secuencialmente
    pass


def create_human_in_the_loop_agent(
    tools: List[Tool],
    model: str = "gpt-4.1",
    approval_callback: Optional[Callable[[str, str], bool]] = None,
) -> Any:
    """
    Agente con Human-in-the-Loop (HITL): requiere aprobación humana para ciertas acciones.

    En producción, los agentes autónomos pueden tomar acciones destructivas o
    costosas. El patrón **Human-in-the-Loop** permite que el agente pause y
    espere aprobación humana antes de ejecutar acciones críticas.

    **Cuándo usar HITL:**
    - Acciones irreversibles (borrar archivos, enviar emails, hacer pagos)
    - Acciones costosas (llamadas API caras, consultas a bases de datos grandes)
    - Decisiones de alto riesgo (cambios en producción, comunicaciones externas)
    - Durante desarrollo y testing de agentes nuevos

    **Implementación con LangGraph:**
    LangGraph tiene soporte nativo para HITL mediante `interrupt()`:
    ```python
    # El grafo pausa aquí esperando input humano
    action = interrupt({"message": "¿Aprobar esta acción?", "action": proposed_action})
    if action["approved"]:
        execute_action()
    ```

    Ejemplo:
        >>> def my_approval(action: str, reason: str) -> bool:
        ...     print(f"Acción propuesta: {action}")
        ...     return input("¿Aprobar? (s/n): ").lower() == 's'
        >>>
        >>> agent = create_human_in_the_loop_agent(tools, approval_callback=my_approval)
        >>> result = agent.invoke({"input": "Envía un email de bienvenida"})

    Args:
        tools: Lista de herramientas (algunas marcarás como "requieren aprobación")
        model: Modelo LLM
        approval_callback: Función que recibe (action, reason) y retorna bool.
                          Si es None, aprueba automáticamente (modo auto).

    Returns:
        Agente con capacidad de solicitar aprobación humana

    Nota:
        LangGraph's `interrupt()` es la forma recomendada para HITL en producción.
        Ver THEORY.md para patrones avanzados de HITL con LangGraph.
    """
    # TODO: Implementa agente con Human-in-the-Loop
    # Pista: Usa interrupt() de LangGraph o wrappea las tools para pedir aprobación
    pass


def setup_mcp_agent(
    mcp_server_urls: List[str],
    model: str = "gpt-4.1",
) -> Any:
    """
    Configura un agente que usa herramientas del Model Context Protocol (MCP).

    **Model Context Protocol (MCP)** es un estándar open source creado por
    Anthropic (2024) para conectar agentes con fuentes de datos y herramientas
    de forma estandarizada. Es como "USB para AI agents".

    **¿Por qué MCP?**
    Sin MCP:
    - Cada herramienta tiene su propia integración custom
    - Mucho código repetitivo
    - Difícil de compartir herramientas entre proyectos

    Con MCP:
    - Un protocolo único para todas las herramientas
    - Servidores MCP reutilizables (filesystem, databases, APIs)
    - Compatible con Claude, OpenAI, y otros LLMs

    **Servidores MCP Comunes:**
    - `@modelcontextprotocol/server-filesystem`: Leer/escribir archivos
    - `@modelcontextprotocol/server-github`: API de GitHub
    - `@modelcontextprotocol/server-sqlite`: Base de datos SQLite
    - `@modelcontextprotocol/server-web-search`: Búsqueda web
    - `mcp-server-fetch`: HTTP requests

    **Arquitectura MCP:**
    ```
    [AI Agent] ←→ [MCP Client] ←→ [MCP Server] ←→ [Resource/Tool]
    ```

    Ejemplo:
        >>> # Requiere servidores MCP corriendo
        >>> agent = setup_mcp_agent(
        ...     mcp_server_urls=["stdio://mcp-server-filesystem"]
        ... )
        >>> result = agent.invoke("Lista los archivos en /tmp/")

    Args:
        mcp_server_urls: URLs/comandos de los servidores MCP a conectar
        model: Modelo LLM a usar con las herramientas MCP

    Returns:
        Agente configurado con herramientas MCP

    Nota:
        MCP requiere servidores MCP corriendo (Node.js o Python).
        langchain-mcp-adapters facilita la integración con LangChain.
        Ver THEORY.md para la arquitectura completa de MCP.
    """
    # TODO: Implementa integración MCP
    # Pista: Usa langchain-mcp-adapters o mcp-use para conectar con servidores MCP
    # La librería MultiServerMCPClient facilita conectarse a múltiples servidores
    pass
