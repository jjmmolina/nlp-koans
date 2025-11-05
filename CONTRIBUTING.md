# 🤝 Contribuyendo a NLP Koans

¡Gracias por tu interés en contribuir! Este proyecto es de código abierto y las contribuciones son bienvenidas.

## 🎯 Tipos de Contribuciones

### 1. Reportar Bugs
- Usa los Issues de GitHub
- Describe el problema claramente
- Incluye pasos para reproducirlo
- Especifica tu entorno (Python version, OS, etc.)

### 2. Sugerir Mejoras
- Nuevos koans o ejercicios
- Mejoras en la documentación
- Correcciones de errores tipográficos

### 3. Enviar Pull Requests
- Arreglos de bugs
- Nuevos tests
- Mejoras de código
- Traducciones

## 🔧 Proceso de Desarrollo

### 1. Fork y Clone

```bash
# Fork el repositorio en GitHub
# Luego clona tu fork
git clone https://github.com/TU-USUARIO/NLP-Koan.git
cd NLP-Koan
```

### 2. Crear Entorno

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
.\venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 3. Crear Rama

```bash
git checkout -b feature/nueva-funcionalidad
# o
git checkout -b fix/arreglo-bug
```

### 4. Hacer Cambios

- Sigue el estilo de código existente
- Agrega tests para nuevo código
- Actualiza documentación si es necesario

### 5. Ejecutar Tests

```bash
# Ejecutar todos los tests
pytest

# Ejecutar tests específicos
pytest koans/01_tokenization/

# Con cobertura
pytest --cov=koans
```

### 6. Commit

```bash
git add .
git commit -m "feat: descripción clara del cambio"
```

**Formato de commits:**
- `feat:` nueva funcionalidad
- `fix:` corrección de bug
- `docs:` cambios en documentación
- `test:` agregar o modificar tests
- `refactor:` refactorización de código

### 7. Push y Pull Request

```bash
git push origin feature/nueva-funcionalidad
```

Luego crea un Pull Request en GitHub.

## 📝 Guías de Estilo

### Python

- Sigue PEP 8
- Usa type hints
- Documenta funciones con docstrings
- Máximo 100 caracteres por línea

**Ejemplo:**

```python
def tokenize_words_nltk(text: str) -> List[str]:
    """
    Tokeniza un texto en palabras usando NLTK.
    
    Args:
        text: Texto a tokenizar
        
    Returns:
        Lista de tokens
    """
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)
```

### Tests

- Nombres descriptivos
- Un concepto por test
- Usa clases para agrupar tests relacionados
- Incluye docstrings explicando qué se prueba

**Ejemplo:**

```python
class TestTokenization:
    """Tests de tokenización básica"""
    
    def test_tokenize_spanish_text(self):
        """Test: Tokenizar texto en español"""
        result = tokenize_words_nltk("Hola mundo")
        assert "Hola" in result
        assert "mundo" in result
```

## 🎓 Agregar Nuevos Koans

Si quieres agregar un nuevo koan:

### Estructura

```
koans/
  XX_nombre_koan/
    __init__.py
    nombre_koan.py      # Implementación con TODOs
    test_nombre_koan.py # Tests completos
```

### Contenido del Koan

1. **Docstring descriptivo** explicando el concepto
2. **Funciones con TODOs** para que los estudiantes implementen
3. **Docstrings en funciones** con ejemplos
4. **Comentarios útiles** como pistas
5. **Tests completos** que fallen inicialmente

### Ejemplo de Función de Koan

```python
def nueva_funcion(text: str) -> List[str]:
    """
    Descripción clara de qué hace.
    
    Ejemplo:
        >>> nueva_funcion("ejemplo")
        ['e', 'j', 'e', 'm', 'p', 'l', 'o']
    
    Args:
        text: Descripción del parámetro
        
    Returns:
        Descripción del retorno
    """
    # TODO: Implementa aquí
    # Pista: puedes usar list(text)
    return []
```

## ✅ Checklist Antes de Enviar PR

- [ ] Los tests pasan: `pytest`
- [ ] El código sigue PEP 8
- [ ] Agregaste docstrings
- [ ] Actualizaste README.md si es necesario
- [ ] Agregaste tests para nuevo código
- [ ] Probaste manualmente los cambios
- [ ] El commit tiene mensaje descriptivo

## 📖 Recursos

- [PEP 8 Style Guide](https://pep8.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [spaCy Documentation](https://spacy.io/)
- [NLTK Documentation](https://www.nltk.org/)

## 💬 Comunicación

- Issues de GitHub para bugs y features
- Pull Requests para contribuciones de código
- Discussions para preguntas generales

## 📄 Licencia

Al contribuir, aceptas que tus contribuciones sean licenciadas bajo la MIT License.

---

**¡Gracias por contribuir al aprendizaje de NLP! 🎉**
