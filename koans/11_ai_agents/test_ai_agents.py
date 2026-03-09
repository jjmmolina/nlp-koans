"""Tests para Koan 11: AI Agents - Requiere API keys configuradas"""

import pytest
import os
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="Requiere OPENAI_API_KEY configurada"
)


# ---------------------------------------------------------------------------
# Imports de las funciones del koan
# ---------------------------------------------------------------------------
from ai_agents import (
    Tool,
    create_simple_agent,
    run_agent,
    create_react_agent,
    create_conversational_agent,
    create_calculator_tool,
    create_search_tool,
    create_custom_tool,
    agent_with_callbacks,
    multi_agent_collaboration,
    AgentState,
    create_langgraph_agent,
    create_agentic_pipeline,
    create_multi_agent_crew,
    create_human_in_the_loop_agent,
    setup_mcp_agent,
)


# ===========================================================================
# 1. Tests de herramientas básicas
# ===========================================================================


class TestToolCreation:
    """Tests that tools are created with the right structure."""

    def test_create_calculator_tool_returns_tool(self):
        tool = create_calculator_tool()
        assert tool is not None

    def test_calculator_tool_has_name(self):
        tool = create_calculator_tool()
        assert hasattr(tool, "name") or (
            hasattr(tool, "func") or callable(tool)
        ), "Tool debe tener nombre"

    def test_create_search_tool_returns_tool(self):
        tool = create_search_tool()
        assert tool is not None

    def test_create_custom_tool(self):
        def dummy(s: str) -> str:
            return s.upper()

        tool = create_custom_tool("upper_case", "Convierte texto a mayúsculas", dummy)
        assert tool is not None


# ===========================================================================
# 2. Tests de creación de agentes básicos
# ===========================================================================


class TestAgentCreation:
    """Tests that agents are created without errors."""

    def test_create_simple_agent_not_none(self):
        tools = [create_calculator_tool()]
        agent = create_simple_agent(tools)
        assert agent is not None

    def test_create_react_agent_not_none(self):
        tools = [create_calculator_tool()]
        agent = create_react_agent(tools)
        assert agent is not None

    def test_create_conversational_agent_not_none(self):
        tools = [create_calculator_tool()]
        agent = create_conversational_agent(tools)
        assert agent is not None


# ===========================================================================
# 3. Tests de ejecución de agentes
# ===========================================================================


class TestAgentExecution:
    """Tests that agents can run simple queries."""

    def test_run_agent_returns_dict(self):
        tools = [create_calculator_tool()]
        agent = create_simple_agent(tools)
        if agent is None:
            pytest.skip("create_simple_agent no implementado aún")
        result = run_agent(agent, "¿Cuánto es 2 + 2?")
        assert isinstance(result, dict)

    def test_run_agent_has_output_key(self):
        tools = [create_calculator_tool()]
        agent = create_simple_agent(tools)
        if agent is None:
            pytest.skip("create_simple_agent no implementado aún")
        result = run_agent(agent, "¿Cuánto es 2 + 2?")
        assert "output" in result

    def test_calculator_agent_gives_correct_answer(self):
        tools = [create_calculator_tool()]
        agent = create_simple_agent(tools)
        if agent is None:
            pytest.skip("create_simple_agent no implementado aún")
        result = run_agent(agent, "What is 10 * 5?")
        assert "50" in result.get("output", "")

    def test_callbacks_agent_returns_dict(self):
        tools = [create_calculator_tool()]
        agent = create_simple_agent(tools)
        if agent is None:
            pytest.skip("create_simple_agent no implementado aún")
        result = agent_with_callbacks(agent, "¿Cuánto es 3 * 3?")
        assert isinstance(result, dict)
        assert "output" in result


# ===========================================================================
# 4. Tests de memoria conversacional
# ===========================================================================


class TestConversationalMemory:
    """Tests that conversational memory works across turns."""

    def test_agent_remembers_context(self):
        tools = []
        agent = create_conversational_agent(tools)
        if agent is None:
            pytest.skip("create_conversational_agent no implementado aún")
        run_agent(agent, "Mi nombre es María")
        result = run_agent(agent, "¿Cómo me llamo?")
        assert (
            "María" in result.get("output", "")
            or "maria" in result.get("output", "").lower()
        )


# ===========================================================================
# 5. Tests de multi-agente básico
# ===========================================================================


class TestMultiAgentCollaboration:
    """Tests for multi-agent coordination."""

    def test_multi_agent_returns_string(self):
        research_tools = [create_search_tool()]
        writer_tools = []
        result = multi_agent_collaboration(
            research_tools, writer_tools, "Escribe un breve resumen sobre Python"
        )
        if result is None:
            pytest.skip("multi_agent_collaboration no implementado aún")
        assert isinstance(result, str)
        assert len(result) > 0


# ===========================================================================
# 6. Tests de LangGraph Agent (Agentic Mode)
# ===========================================================================


class TestLangGraphAgent:
    """Tests for the modern LangGraph-based agent."""

    def test_langgraph_agent_not_none(self):
        tools = [create_calculator_tool()]
        agent = create_langgraph_agent(tools)
        if agent is None:
            pytest.skip("create_langgraph_agent no implementado aún")
        assert agent is not None

    def test_langgraph_agent_is_compilable(self):
        """The agent should be a compiled LangGraph app (has .invoke)."""
        tools = [create_calculator_tool()]
        agent = create_langgraph_agent(tools)
        if agent is None:
            pytest.skip("create_langgraph_agent no implementado aún")
        assert hasattr(agent, "invoke"), "El agente LangGraph debe tener método .invoke"

    def test_langgraph_agent_invoke_returns_messages(self):
        tools = [create_calculator_tool()]
        agent = create_langgraph_agent(tools)
        if agent is None:
            pytest.skip("create_langgraph_agent no implementado aún")
        result = agent.invoke({"messages": [("user", "Hola, ¿cómo estás?")]})
        assert "messages" in result
        assert len(result["messages"]) > 0

    def test_langgraph_agent_uses_tool(self):
        tools = [create_calculator_tool()]
        agent = create_langgraph_agent(tools)
        if agent is None:
            pytest.skip("create_langgraph_agent no implementado aún")
        result = agent.invoke({"messages": [("user", "¿Cuánto es 12 al cuadrado?")]})
        last_message = result["messages"][-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )
        assert "144" in content


# ===========================================================================
# 7. Tests de AgentState TypedDict
# ===========================================================================


class TestAgentState:
    """Tests that AgentState has the expected structure."""

    def test_agent_state_is_typeddict(self):
        from typing import get_type_hints

        assert AgentState is not None

    def test_agent_state_has_messages_field(self):
        # AgentState should be usable as a dict with messages key
        state: AgentState = {
            "messages": [],
            "current_step": "start",
            "tool_results": [],
        }
        assert "messages" in state

    def test_agent_state_has_required_fields(self):
        state: AgentState = {
            "messages": [{"role": "user", "content": "hello"}],
            "current_step": "planning",
            "tool_results": ["result1"],
        }
        assert state["current_step"] == "planning"
        assert len(state["tool_results"]) == 1


# ===========================================================================
# 8. Tests de Agentic Pipeline (Plan-Execute)
# ===========================================================================


class TestAgenticPipeline:
    """Tests for the Plan-Execute agentic pipeline."""

    def test_pipeline_returns_dict(self):
        tools = [create_calculator_tool()]
        result = create_agentic_pipeline(
            "Calcula 5 al cubo",
            tools,
            model="gpt-4.1-mini",
            planning_model="gpt-4.1-mini",
        )
        if result is None:
            pytest.skip("create_agentic_pipeline no implementado aún")
        assert isinstance(result, dict)

    def test_pipeline_has_required_keys(self):
        tools = [create_calculator_tool()]
        result = create_agentic_pipeline(
            "Calcula 3 al cuadrado",
            tools,
            model="gpt-4.1-mini",
            planning_model="gpt-4.1-mini",
        )
        if result is None:
            pytest.skip("create_agentic_pipeline no implementado aún")
        assert "plan" in result, "El resultado debe contener 'plan'"
        assert "output" in result, "El resultado debe contener 'output'"

    def test_pipeline_plan_is_list(self):
        tools = [create_calculator_tool()]
        result = create_agentic_pipeline(
            "Calcula 2 + 2", tools, model="gpt-4.1-mini", planning_model="gpt-4.1-mini"
        )
        if result is None:
            pytest.skip("create_agentic_pipeline no implementado aún")
        assert isinstance(result["plan"], list), "plan debe ser una lista de pasos"

    def test_pipeline_output_is_string(self):
        tools = [create_calculator_tool()]
        result = create_agentic_pipeline(
            "Calcula 2 + 2", tools, model="gpt-4.1-mini", planning_model="gpt-4.1-mini"
        )
        if result is None:
            pytest.skip("create_agentic_pipeline no implementado aún")
        assert isinstance(result["output"], str)


# ===========================================================================
# 9. Tests de Multi-Agent Crew (CrewAI)
# ===========================================================================


class TestMultiAgentCrew:
    """Tests for the CrewAI multi-agent system."""

    def test_crew_returns_non_empty_string(self):
        research_tools = [create_search_tool()]
        writer_tools = []
        result = create_multi_agent_crew(
            "Escribe un párrafo breve sobre Python",
            research_tools,
            writer_tools,
            model="gpt-4.1-mini",
        )
        if result is None:
            pytest.skip("create_multi_agent_crew no implementado aún")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_crew_produces_relevant_content(self):
        research_tools = [create_search_tool()]
        writer_tools = []
        result = create_multi_agent_crew(
            "Explica qué es Python en una oración",
            research_tools,
            writer_tools,
            model="gpt-4.1-mini",
        )
        if result is None:
            pytest.skip("create_multi_agent_crew no implementado aún")
        assert "python" in result.lower() or "programación" in result.lower()


# ===========================================================================
# 10. Tests de Human-in-the-Loop Agent
# ===========================================================================


class TestHumanInTheLoopAgent:
    """Tests for the Human-in-the-Loop agent."""

    def test_hitl_agent_not_none(self):
        tools = [create_calculator_tool()]
        agent = create_human_in_the_loop_agent(tools)
        if agent is None:
            pytest.skip("create_human_in_the_loop_agent no implementado aún")
        assert agent is not None

    def test_hitl_agent_has_invoke(self):
        tools = [create_calculator_tool()]
        agent = create_human_in_the_loop_agent(tools)
        if agent is None:
            pytest.skip("create_human_in_the_loop_agent no implementado aún")
        assert hasattr(agent, "invoke"), "HITL agent debe tener método .invoke"

    def test_hitl_agent_with_auto_approval(self):
        """With auto-approval callback (returns True always), agent should complete."""
        tools = [create_calculator_tool()]

        def auto_approve(action: str, reason: str) -> bool:
            return True

        agent = create_human_in_the_loop_agent(tools, approval_callback=auto_approve)
        if agent is None:
            pytest.skip("create_human_in_the_loop_agent no implementado aún")
        # Should not raise an exception
        assert agent is not None


# ===========================================================================
# 11. Tests de MCP Agent
# ===========================================================================


class TestMCPAgent:
    """Tests for the MCP-based agent."""

    @pytest.mark.skipif(
        not os.getenv("MCP_AVAILABLE"),
        reason="MCP servers not available in CI. Set MCP_AVAILABLE=1 to run.",
    )
    def test_mcp_agent_creation(self):
        agent = setup_mcp_agent(["stdio://mcp-server-filesystem"])
        assert agent is not None

    def test_mcp_agent_returns_callable(self):
        """setup_mcp_agent should return a callable (async function)."""
        agent = setup_mcp_agent(["http://localhost:3001/sse"])
        if agent is None:
            pytest.skip("setup_mcp_agent no implementado aún")
        assert callable(agent), "setup_mcp_agent debe devolver una función invocable"
