"""
Tests para Koan 10: Modern LLMs & APIs

IMPORTANTE: Para ejecutar estos tests necesitas:
1. API keys configuradas como variables de entorno:
   - OPENAI_API_KEY
   - ANTHROPIC_API_KEY
   - GOOGLE_API_KEY

2. Instalar las librerías:
   pip install openai anthropic google-generativeai

Ejecuta con:
    pytest koans/10_modern_llms/test_modern_llms.py -v
    pytest koans/10_modern_llms/test_modern_llms.py -v -m "not expensive"
"""

import pytest
import os
from modern_llms import (
    call_openai_chat,
    call_openai_streaming,
    call_anthropic_claude,
    call_google_gemini,
    openai_function_calling,
    calculate_token_cost,
    compare_llm_outputs,
    safe_llm_call,
)


# Marca tests que requieren API keys y cuestan dinero
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="Requiere OPENAI_API_KEY configurada"
)


class TestOpenAIBasics:
    """Tests básicos de OpenAI API"""

    @pytest.mark.expensive
    def test_call_openai_chat(self):
        """Test: Llamada básica a OpenAI"""
        messages = [
            {"role": "system", "content": "Eres un asistente conciso."},
            {"role": "user", "content": "Di 'hola' y nada más."},
        ]
        response = call_openai_chat(messages, model="gpt-4o-mini", max_tokens=10)

        assert isinstance(response, str)
        assert len(response) > 0
        assert "hola" in response.lower()

    @pytest.mark.expensive
    def test_openai_temperature(self):
        """Test: Temperature afecta creatividad"""
        messages = [{"role": "user", "content": "Di un número del 1 al 10"}]

        # Temperature baja = más determinista
        response1 = call_openai_chat(messages, temperature=0.1)
        response2 = call_openai_chat(messages, temperature=0.1)

        assert isinstance(response1, str)
        assert isinstance(response2, str)

    @pytest.mark.expensive
    def test_call_openai_streaming(self):
        """Test: Streaming funciona"""
        messages = [{"role": "user", "content": "Cuenta del 1 al 3, solo números"}]

        chunks = list(call_openai_streaming(messages))

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0


class TestAnthropicClaude:
    """Tests de Anthropic Claude API"""

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"), reason="Requiere ANTHROPIC_API_KEY"
    )
    @pytest.mark.expensive
    def test_call_anthropic_claude(self):
        """Test: Llamada básica a Claude"""
        messages = [{"role": "user", "content": "Di 'hola' y nada más."}]
        response = call_anthropic_claude(messages, max_tokens=20)

        assert isinstance(response, str)
        assert len(response) > 0
        assert "hola" in response.lower()


class TestGoogleGemini:
    """Tests de Google Gemini API"""

    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY"), reason="Requiere GOOGLE_API_KEY"
    )
    @pytest.mark.expensive
    def test_call_google_gemini(self):
        """Test: Llamada básica a Gemini"""
        response = call_google_gemini("Di 'hola' y nada más.")

        assert isinstance(response, str)
        assert len(response) > 0
        assert "hola" in response.lower()


class TestFunctionCalling:
    """Tests de function calling"""

    @pytest.mark.expensive
    def test_openai_function_calling(self):
        """Test: Function calling básico"""
        functions = [
            {
                "name": "get_weather",
                "description": "Obtiene el clima de una ciudad",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Nombre de la ciudad",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            }
        ]

        messages = [{"role": "user", "content": "¿Qué tiempo hace en Madrid?"}]

        result = openai_function_calling(messages, functions)

        assert isinstance(result, dict)
        assert "name" in result
        assert result["name"] == "get_weather"
        assert "arguments" in result

        # Los argumentos deben mencionar Madrid
        import json

        args = json.loads(result["arguments"])
        assert "madrid" in args["city"].lower()


class TestTokenCost:
    """Tests de cálculo de costos"""

    def test_calculate_token_cost_gpt4o_mini(self):
        """Test: Costo de GPT-4o-mini"""
        cost = calculate_token_cost(1000, 500, "gpt-4o-mini")

        assert isinstance(cost, float)
        assert cost > 0
        assert cost < 0.01  # Debe ser barato para estos tokens

    def test_calculate_token_cost_gpt4o(self):
        """Test: GPT-4o es más caro que mini"""
        cost_mini = calculate_token_cost(1000, 500, "gpt-4o-mini")
        cost_full = calculate_token_cost(1000, 500, "gpt-4o")

        assert cost_full > cost_mini

    def test_calculate_token_cost_different_ratios(self):
        """Test: Completion tokens cuestan más"""
        # Más tokens de entrada
        cost1 = calculate_token_cost(10000, 100, "gpt-4o-mini")
        # Más tokens de salida
        cost2 = calculate_token_cost(100, 10000, "gpt-4o-mini")

        assert cost2 > cost1  # Output es más caro


class TestLLMComparison:
    """Tests de comparación de modelos"""

    @pytest.mark.expensive
    def test_compare_llm_outputs(self):
        """Test: Comparación de múltiples LLMs"""
        # Solo usa modelos disponibles
        available_models = ["gpt-4o-mini"]
        if os.getenv("ANTHROPIC_API_KEY"):
            available_models.append("claude-3-5-sonnet-20241022")
        if os.getenv("GOOGLE_API_KEY"):
            available_models.append("gemini-1.5-flash")

        results = compare_llm_outputs(
            "Di solo la palabra 'test'", models=available_models
        )

        assert isinstance(results, dict)
        assert len(results) > 0

        for model, response in results.items():
            assert isinstance(response, str)
            assert len(response) > 0


class TestErrorHandling:
    """Tests de manejo de errores"""

    def test_safe_llm_call_success(self):
        """Test: Llamada exitosa"""
        response = safe_llm_call("Di 'test'", max_retries=1)

        if response is not None:  # Solo si la API está disponible
            assert isinstance(response, str)

    def test_safe_llm_call_invalid_model(self):
        """Test: Modelo inválido retorna None o error manejado"""
        response = safe_llm_call(
            "test", model="modelo-que-no-existe-xyz", max_retries=1
        )

        # Debe manejar el error gracefully
        assert response is None or isinstance(response, str)


class TestRealWorldScenarios:
    """Tests con escenarios del mundo real"""

    @pytest.mark.expensive
    def test_multi_turn_conversation(self):
        """Test: Conversación de múltiples turnos"""
        messages = [
            {"role": "system", "content": "Eres un asistente matemático."},
            {"role": "user", "content": "¿Cuánto es 5 + 3?"},
        ]

        response1 = call_openai_chat(messages, max_tokens=50)
        assert "8" in response1

        # Agregar respuesta del asistente y nueva pregunta
        messages.append({"role": "assistant", "content": response1})
        messages.append({"role": "user", "content": "¿Y si le sumo 2?"})

        response2 = call_openai_chat(messages, max_tokens=50)
        assert "10" in response2

    @pytest.mark.expensive
    def test_system_prompt_effectiveness(self):
        """Test: System prompt afecta comportamiento"""
        user_message = "¿Qué es Python?"

        # Con system prompt técnico
        messages1 = [
            {
                "role": "system",
                "content": "Eres un experto técnico. Responde con precisión técnica.",
            },
            {"role": "user", "content": user_message},
        ]
        response1 = call_openai_chat(messages1, max_tokens=100)

        # Con system prompt simple
        messages2 = [
            {"role": "system", "content": "Explica todo como si tuvieras 5 años."},
            {"role": "user", "content": user_message},
        ]
        response2 = call_openai_chat(messages2, max_tokens=100)

        assert response1 != response2
        assert len(response1) > 0
        assert len(response2) > 0


# Fixtures para tests
@pytest.fixture
def sample_messages():
    """Mensajes de ejemplo para tests"""
    return [
        {"role": "system", "content": "Eres un asistente útil."},
        {"role": "user", "content": "Hola"},
    ]


@pytest.fixture
def sample_function_definition():
    """Definición de función de ejemplo"""
    return {
        "name": "calculate",
        "description": "Realiza cálculos matemáticos",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                },
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        },
    }
