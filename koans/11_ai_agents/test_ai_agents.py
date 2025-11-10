"""Tests para Koan 11: AI Agents - Requiere API keys configuradas"""

import pytest
import os

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="Requiere OPENAI_API_KEY configurada"
)

# Tests aquí cuando implementes las funciones
# pytest koans/11_ai_agents/test_ai_agents.py -v
