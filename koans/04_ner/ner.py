"""
Koan 04: Named Entity Recognition (NER) - Reconocimiento de Entidades Nombradas

NER identifica y clasifica entidades en el texto:
- Personas (PER): "Juan García"
- Organizaciones (ORG): "Google", "ONU"
- Lugares (LOC): "Madrid", "España"
- Fechas/Tiempos (DATE/TIME)
- Cantidades de dinero (MONEY)
- etc.

Librería principal: spaCy (excelente para NER multiidioma)
"""

import spacy
from typing import List, Dict, Tuple


def extract_entities_spacy(text: str, lang: str = "es") -> List[Tuple[str, str]]:
    """
    Extrae todas las entidades nombradas de un texto.

    Ejemplo:
        >>> extract_entities_spacy("Steve Jobs fundó Apple en California")
        [('Steve Jobs', 'PER'), ('Apple', 'ORG'), ('California', 'LOC')]

    Args:
        text: Texto a analizar
        lang: Idioma ('es' o 'en')

    Returns:
        Lista de tuplas (entidad, tipo)
    """
    # TODO: Implementa extracción de entidades con spaCy
    # Pistas:
    # 1. Carga el modelo spaCy apropiado
    # 2. Procesa el texto
    # 3. Los documentos procesados tienen un atributo .ents con las entidades
    # Consulta HINTS.md para más ayuda
    pass


def extract_persons(text: str, lang: str = "es") -> List[str]:
    """
    Extrae solo nombres de personas.

    Ejemplo:
        >>> extract_persons("Juan y María trabajan en Google")
        ['Juan', 'María']

    Args:
        text: Texto a analizar
        lang: Idioma

    Returns:
        Lista de nombres de personas
    """
    # TODO: Filtra entidades de tipo PERSONA
    # Pista: Primero extrae todas las entidades, luego filtra por tipo
    pass


def extract_organizations(text: str, lang: str = "es") -> List[str]:
    """
    Extrae nombres de organizaciones.

    Ejemplo:
        >>> extract_organizations("Google y Microsoft son grandes empresas")
        ['Google', 'Microsoft']

    Args:
        text: Texto a analizar
        lang: Idioma

    Returns:
        Lista de organizaciones
    """
    # TODO: Filtra entidades de tipo ORGANIZACIÓN
    pass


def extract_locations(text: str, lang: str = "es") -> List[str]:
    """
    Extrae nombres de lugares.

    Ejemplo:
        >>> extract_locations("Viajé de Madrid a Barcelona")
        ['Madrid', 'Barcelona']

    Args:
        text: Texto a analizar
        lang: Idioma

    Returns:
        Lista de lugares
    """
    # TODO: Filtra entidades de tipo LUGAR
    # Nota: Puede ser 'LOC' o 'GPE' dependiendo del modelo
    pass


def extract_dates(text: str, lang: str = "es") -> List[str]:
    """
    Extrae expresiones de fecha y tiempo.

    Ejemplo:
        >>> extract_dates("El 15 de enero de 2024 fue un día importante")
        ['15 de enero de 2024']

    Args:
        text: Texto a analizar
        lang: Idioma

    Returns:
        Lista de expresiones temporales
    """
    # TODO: Filtra entidades de tipo FECHA/TIEMPO
    pass


def group_entities_by_type(text: str, lang: str = "es") -> Dict[str, List[str]]:
    """
    Agrupa las entidades por tipo.

    Ejemplo:
        >>> group_entities_by_type("Juan trabaja en Google en Madrid")
        {
            'PER': ['Juan'],
            'ORG': ['Google'],
            'LOC': ['Madrid']
        }

    Args:
        text: Texto a analizar
        lang: Idioma

    Returns:
        Diccionario con entidades agrupadas por tipo
    """
    # TODO: Implementa agrupación de entidades
    # Pista: Recorre las entidades y agrúpalas por su label_
    pass


def count_entity_types(text: str, lang: str = "es") -> Dict[str, int]:
    """
    Cuenta cuántas entidades de cada tipo hay.

    Ejemplo:
        >>> count_entity_types("Juan y María trabajan en Google y Apple")
        {'PER': 2, 'ORG': 2}

    Args:
        text: Texto a analizar
        lang: Idioma

    Returns:
        Diccionario con conteo por tipo
    """
    # TODO: Cuenta las entidades por tipo
    # Pista: Similar a group_entities_by_type pero solo cuentas
    pass


def find_entity_context(
    text: str, entity: str, window: int = 5, lang: str = "es"
) -> str:
    """
    Encuentra el contexto alrededor de una entidad específica.

    Retorna las N palabras antes y después de la entidad.

    Ejemplo:
        >>> find_entity_context("Juan García trabaja en Google desde 2020", "Google", window=2)
        'en Google desde'

    Args:
        text: Texto a analizar
        entity: Entidad a buscar
        window: Número de palabras de contexto (antes y después)
        lang: Idioma

    Returns:
        Contexto de la entidad
    """
    # TODO: Implementa extracción de contexto
    # Pistas:
    # 1. Procesa el texto con spaCy
    # 2. Busca la entidad específica en doc.ents
    # 3. Obtén los tokens alrededor usando índices de span
    pass


def visualize_entities(text: str, lang: str = "es") -> str:
    """
    Crea una representación visual de las entidades (simplificada).

    Marca las entidades con corchetes y su tipo.

    Ejemplo:
        >>> visualize_entities("Juan trabaja en Google")
        '[PER: Juan] trabaja en [ORG: Google]'

    Args:
        text: Texto a analizar
        lang: Idioma

    Returns:
        Texto con entidades marcadas
    """
    # TODO: Implementa visualización simple
    # Pista: Construye un nuevo string insertando marcadores alrededor de las entidades
    pass
