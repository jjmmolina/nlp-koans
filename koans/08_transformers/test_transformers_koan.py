"""
Tests para Koan 08: Transformers

Ejecuta con:
    pytest koans/08_transformers/test_transformers_koan.py -v

NOTA: Estos tests descargan modelos grandes. Pueden tardar la primera vez.
Marca algunos como @pytest.mark.slow
"""

import pytest
import torch
from transformers_koan import (
    load_pretrained_pipeline,
    extract_features_bert,
    question_answering,
    fill_mask,
    zero_shot_classification,
    summarize_text,
    translate_text,
    compare_models_performance
)


@pytest.mark.slow
class TestPipelineLoading:
    """Tests de carga de pipelines"""
    
    def test_load_pretrained_pipeline(self):
        """Test: Cargar pipeline pre-entrenado"""
        pipe = load_pretrained_pipeline("sentiment-analysis")
        
        assert pipe is not None


@pytest.mark.slow
class TestBERT:
    """Tests de BERT"""
    
    def test_extract_features_bert(self):
        """Test: Extraer características con BERT"""
        features = extract_features_bert("Python es genial")
        
        assert isinstance(features, torch.Tensor)
        assert len(features.shape) == 3  # [batch, seq_len, hidden_size]


@pytest.mark.slow
class TestQuestionAnswering:
    """Tests de QA"""
    
    def test_question_answering_simple(self):
        """Test: Responder pregunta simple"""
        context = "Python fue creado por Guido van Rossum en 1991"
        question = "¿Quién creó Python?"
        
        result = question_answering(context, question)
        
        assert isinstance(result, dict)
        if 'answer' in result:
            assert isinstance(result['answer'], str)


@pytest.mark.slow
class TestFillMask:
    """Tests de fill-mask"""
    
    def test_fill_mask(self):
        """Test: Rellenar palabra enmascarada"""
        text = "Python es un [MASK] de programación"
        result = fill_mask(text)
        
        assert isinstance(result, list)


@pytest.mark.slow
class TestZeroShot:
    """Tests de zero-shot classification"""
    
    def test_zero_shot_classification(self):
        """Test: Clasificación zero-shot"""
        text = "Este código tiene un bug"
        labels = ["problema", "éxito", "neutral"]
        
        result = zero_shot_classification(text, labels)
        
        assert isinstance(result, dict)


@pytest.mark.slow
class TestSummarization:
    """Tests de resumen de texto"""
    
    def test_summarize_text(self):
        """Test: Resumir texto"""
        text = """
        Python is a high-level, interpreted programming language.
        It was created by Guido van Rossum and first released in 1991.
        Python's design philosophy emphasizes code readability.
        """
        
        summary = summarize_text(text, lang="en")
        
        assert isinstance(summary, str)
        if len(summary) > 0:
            assert len(summary) < len(text)


@pytest.mark.slow
class TestTranslation:
    """Tests de traducción"""
    
    def test_translate_text(self):
        """Test: Traducir texto"""
        result = translate_text("Hola mundo", source_lang="es", target_lang="en")
        
        assert isinstance(result, str)


class TestModelComparison:
    """Tests de comparación de modelos"""
    
    def test_compare_models_performance(self):
        """Test: Comparar rendimiento de modelos"""
        text = "Me encanta Python"
        models = []  # Dejar vacío para evitar descargas en tests rápidos
        
        result = compare_models_performance(text, models)
        
        assert isinstance(result, dict)


class TestRealWorldExamples:
    """Tests con ejemplos del mundo real"""
    
    @pytest.mark.slow
    def test_code_documentation_qa(self):
        """Test: QA sobre documentación de código"""
        context = """
        La función map() aplica una función a cada elemento de un iterable.
        Retorna un iterador que contiene los resultados.
        """
        question = "¿Qué retorna map()?"
        
        result = question_answering(context, question, lang="es")
        assert isinstance(result, dict)
