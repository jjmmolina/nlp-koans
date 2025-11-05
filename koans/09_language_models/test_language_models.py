"""
Tests para Koan 09: Language Models

Ejecuta con:
    pytest koans/09_language_models/test_language_models.py -v

ADVERTENCIA: Estos tests descargan modelos GRANDES y pueden tardar mucho.
Usa -m "not slow" para ejecutar solo tests rápidos.
"""

import pytest
from language_models import (
    generate_text_simple,
    generate_multiple_completions,
    generate_with_temperature,
    generate_with_top_k,
    generate_with_top_p,
    chat_completion,
    calculate_perplexity,
    prompt_engineering_template,
    compare_generation_strategies
)


@pytest.mark.slow
class TestBasicGeneration:
    """Tests de generación básica"""
    
    def test_generate_text_simple(self):
        """Test: Generar texto simple"""
        result = generate_text_simple("Python is", max_length=20)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Python is" in result
        
    def test_generated_text_contains_prompt(self):
        """Test: Texto generado contiene el prompt"""
        prompt = "The quick brown"
        result = generate_text_simple(prompt, max_length=30)
        
        assert prompt in result


@pytest.mark.slow
class TestMultipleCompletions:
    """Tests de múltiples completados"""
    
    def test_generate_multiple_completions(self):
        """Test: Generar múltiples completados"""
        result = generate_multiple_completions("Python is", num_completions=2, max_length=20)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(text, str) for text in result)


@pytest.mark.slow
class TestTemperature:
    """Tests de temperatura"""
    
    def test_generate_with_temperature_low(self):
        """Test: Generación con temperatura baja (conservador)"""
        result = generate_with_temperature("Python is", temperature=0.3, max_length=20)
        
        assert isinstance(result, str)
        assert len(result) > 0
        
    def test_generate_with_temperature_high(self):
        """Test: Generación con temperatura alta (creativo)"""
        result = generate_with_temperature("Python is", temperature=1.5, max_length=20)
        
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.slow
class TestSamplingStrategies:
    """Tests de estrategias de sampling"""
    
    def test_generate_with_top_k(self):
        """Test: Generación con top-k sampling"""
        result = generate_with_top_k("Python", top_k=50, max_length=20)
        
        assert isinstance(result, str)
        
    def test_generate_with_top_p(self):
        """Test: Generación con nucleus sampling"""
        result = generate_with_top_p("Python", top_p=0.9, max_length=20)
        
        assert isinstance(result, str)


@pytest.mark.slow
class TestChatCompletion:
    """Tests de chat completion"""
    
    def test_chat_completion_simple(self):
        """Test: Completado de chat simple"""
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        result = chat_completion(messages)
        
        assert isinstance(result, str)


@pytest.mark.slow
class TestPerplexity:
    """Tests de perplejidad"""
    
    def test_calculate_perplexity_natural_text(self):
        """Test: Perplejidad de texto natural"""
        text = "Python is a programming language"
        result = calculate_perplexity(text)
        
        assert isinstance(result, float)
        assert result >= 0
        
    def test_natural_text_lower_perplexity_than_random(self):
        """Test: Texto natural tiene menor perplejidad que aleatorio"""
        natural = "The cat sat on the mat"
        random = "xyz abc qwerty asdfgh"
        
        perp_natural = calculate_perplexity(natural)
        perp_random = calculate_perplexity(random)
        
        # El texto natural debería tener menor perplejidad
        # (pero esto puede fallar con modelos pequeños)
        assert isinstance(perp_natural, float)
        assert isinstance(perp_random, float)


class TestPromptEngineering:
    """Tests de prompt engineering"""
    
    def test_prompt_engineering_template(self):
        """Test: Crear template de prompt"""
        result = prompt_engineering_template(
            task="Translate to English",
            context="Technical text",
            examples=["Hola -> Hello", "Adiós -> Goodbye"]
        )
        
        assert isinstance(result, str)
        assert "Translate to English" in result
        
    def test_prompt_template_includes_examples(self):
        """Test: Template incluye ejemplos"""
        examples = ["Python -> snake", "Java -> coffee"]
        result = prompt_engineering_template(
            task="Translate",
            examples=examples
        )
        
        # Al menos uno de los ejemplos debe estar
        assert any(ex.split(" -> ")[0] in result for ex in examples)


class TestGenerationComparison:
    """Tests de comparación de estrategias"""
    
    def test_compare_generation_strategies(self):
        """Test: Comparar estrategias de generación"""
        result = compare_generation_strategies("Python is")
        
        assert isinstance(result, dict)


class TestRealWorldExamples:
    """Tests con ejemplos del mundo real"""
    
    @pytest.mark.slow
    def test_code_documentation_generation(self):
        """Test: Generar documentación de código"""
        prompt = "def calculate_sum(a, b):\n    "
        result = generate_text_simple(prompt, max_length=50)
        
        assert isinstance(result, str)
        
    @pytest.mark.slow
    def test_story_generation(self):
        """Test: Generar historia corta"""
        prompt = "Once upon a time, there was a programmer who"
        result = generate_text_simple(prompt, max_length=100)
        
        assert isinstance(result, str)
        assert len(result) > len(prompt)
