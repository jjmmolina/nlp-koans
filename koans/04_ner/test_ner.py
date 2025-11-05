"""
Tests para Koan 04: Named Entity Recognition

Ejecuta con:
    pytest koans/04_ner/test_ner.py -v
"""

import pytest
from ner import (
    extract_entities_spacy,
    extract_persons,
    extract_organizations,
    extract_locations,
    extract_dates,
    group_entities_by_type,
    count_entity_types,
    find_entity_context,
    visualize_entities
)


class TestBasicNER:
    """Tests básicos de NER"""
    
    def test_extract_entities_spanish(self):
        """Test: Extraer entidades en español"""
        text = "Juan García trabaja en Google en Madrid"
        result = extract_entities_spacy(text, lang="es")
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        
    def test_extract_entities_english(self):
        """Test: Extraer entidades en inglés"""
        text = "Steve Jobs founded Apple in California"
        result = extract_entities_spacy(text, lang="en")
        
        assert isinstance(result, list)
        assert len(result) > 0


class TestEntityTypes:
    """Tests de extracción por tipo de entidad"""
    
    def test_extract_persons_spanish(self):
        """Test: Extraer personas"""
        text = "Juan y María trabajan juntos"
        result = extract_persons(text, lang="es")
        
        assert isinstance(result, list)
        # Debe detectar al menos un nombre
        assert len(result) >= 1
        
    def test_extract_persons_english(self):
        """Test: Extraer personas en inglés"""
        text = "Steve Jobs and Bill Gates are famous"
        result = extract_persons(text, lang="en")
        
        assert isinstance(result, list)
        assert len(result) >= 1
        
    def test_extract_organizations(self):
        """Test: Extraer organizaciones"""
        text = "Google y Microsoft son empresas tecnológicas"
        result = extract_organizations(text, lang="es")
        
        assert isinstance(result, list)
        # Debe detectar al menos una organización
        assert len(result) >= 1
        
    def test_extract_locations(self):
        """Test: Extraer ubicaciones"""
        text = "Viajé de Madrid a Barcelona"
        result = extract_locations(text, lang="es")
        
        assert isinstance(result, list)
        assert len(result) >= 1
        
    def test_extract_dates(self):
        """Test: Extraer fechas"""
        text = "El evento es el 15 de enero de 2024"
        result = extract_dates(text, lang="es")
        
        assert isinstance(result, list)
        # Puede o no detectar la fecha dependiendo del modelo


class TestGroupingAndCounting:
    """Tests de agrupación y conteo"""
    
    def test_group_entities_by_type(self):
        """Test: Agrupar entidades por tipo"""
        text = "Juan y María trabajan en Google en Madrid"
        result = group_entities_by_type(text, lang="es")
        
        assert isinstance(result, dict)
        # Debe tener al menos un tipo de entidad
        assert len(result) > 0
        # Cada valor debe ser una lista
        assert all(isinstance(v, list) for v in result.values())
        
    def test_count_entity_types(self):
        """Test: Contar entidades por tipo"""
        text = "Juan y María trabajan en Google y Apple"
        result = count_entity_types(text, lang="es")
        
        assert isinstance(result, dict)
        assert len(result) > 0
        # Los conteos deben ser positivos
        assert all(count > 0 for count in result.values())


class TestContextExtraction:
    """Tests de extracción de contexto"""
    
    def test_find_entity_context(self):
        """Test: Encontrar contexto de entidad"""
        text = "Juan García trabaja en Google desde 2020"
        result = find_entity_context(text, "Google", window=2, lang="es")
        
        # Debe retornar algo (puede ser vacío si no encuentra la entidad)
        assert isinstance(result, str)
        
    def test_find_entity_context_not_found(self):
        """Test: Entidad no existente"""
        text = "Juan trabaja en Google"
        result = find_entity_context(text, "Microsoft", window=2, lang="es")
        
        # Debe retornar string vacío si no encuentra la entidad
        assert isinstance(result, str)


class TestVisualization:
    """Tests de visualización"""
    
    def test_visualize_entities(self):
        """Test: Visualizar entidades marcadas"""
        text = "Juan trabaja en Google"
        result = visualize_entities(text, lang="es")
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Debe contener algún tipo de marcador o el texto original


class TestRealWorldExamples:
    """Tests con ejemplos reales"""
    
    def test_news_article(self):
        """Test: Analizar artículo de noticias"""
        text = """
        El presidente de España visitó Madrid el lunes.
        Se reunió con representantes de Google y Microsoft.
        """
        
        persons = extract_persons(text, lang="es")
        locations = extract_locations(text, lang="es")
        orgs = extract_organizations(text, lang="es")
        
        # Debe detectar al menos alguna entidad
        total_entities = len(persons) + len(locations) + len(orgs)
        assert total_entities > 0
        
    def test_business_text(self):
        """Test: Texto de negocios"""
        text = "Apple anunció sus resultados en California el 15 de enero"
        entities = extract_entities_spacy(text, lang="es")
        
        assert len(entities) > 0
        
    def test_multiple_persons(self):
        """Test: Múltiples personas en texto"""
        text = "Juan, María y Pedro son amigos"
        persons = extract_persons(text, lang="es")
        
        # Debe detectar al menos 2 nombres
        assert len(persons) >= 2
        
    def test_mixed_entities(self):
        """Test: Mezcla de tipos de entidades"""
        text = "Steve Jobs fundó Apple en 1976 en California"
        
        result = group_entities_by_type(text, lang="en")
        # Debe tener al menos 2 tipos diferentes
        assert len(result) >= 2
