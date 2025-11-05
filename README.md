# 🧠 NLP Koans - Aprende Procesamiento de Lenguaje Natural con TDD

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![pytest](https://img.shields.io/badge/tested%20with-pytest-orange.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Un proyecto tutorial tipo **Koan** para aprender **Procesamiento de Lenguaje Natural (NLP)** usando **Test-Driven Development (TDD)** en Python.

## 🎯 ¿Qué son los NLP Koans?

Los **Koans** son ejercicios de aprendizaje donde:
1. ✅ Los tests **fallan inicialmente** 
2. 🔧 Tú **arreglas el código** para hacerlos pasar
3. 🎓 **Aprendes** los conceptos de NLP progresivamente

## 🚀 Inicio Rápido

```bash
# 1. Clonar el repositorio
git clone <tu-repo>
cd NLP-Koan

# 2. Crear entorno virtual
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar modelos de spaCy y NLTK
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# 5. Ejecutar todos los tests
pytest

# 6. Ejecutar un koan específico
pytest koans/01_tokenization/test_tokenization.py -v
```

## 📚 Estructura de Koans

| Koan | Tema | Librerías | Conceptos |
|------|------|-----------|-----------|
| **01** | Tokenización | NLTK, spaCy | Separación de texto en palabras/oraciones |
| **02** | Stemming & Lemmatization | NLTK, spaCy | Normalización de palabras |
| **03** | POS Tagging | spaCy, NLTK | Etiquetado gramatical |
| **04** | Named Entity Recognition | spaCy | Extracción de entidades |
| **05** | Text Classification | scikit-learn | Clasificación de textos |
| **06** | Sentiment Analysis | transformers | Análisis de sentimientos |
| **07** | Word Embeddings | spaCy, gensim | Representaciones vectoriales |
| **08** | Transformers | transformers (Hugging Face) | Modelos preentrenados |
| **09** | Language Models | transformers | Generación de texto |

## 🎓 Cómo Usar Este Tutorial

### Paso 1: Empieza con el Primer Koan
```bash
cd koans/01_tokenization
pytest test_tokenization.py -v
```

### Paso 2: Lee los Errores
Los tests te dirán **exactamente** qué falta. Ejemplo:
```
FAILED - assert actual == expected
AssertionError: Tu implementación debe tokenizar el texto
```

### Paso 3: Arregla el Código
Abre `tokenization.py` y completa las funciones marcadas con `# TODO`

### Paso 4: Repite hasta que Pasen Todos los Tests ✅

### Paso 5: ¡Siguiente Koan! 🎉

## 🛠️ Tecnologías y Librerías

- **🐍 Python 3.8+**: Lenguaje base
- **✅ pytest**: Framework de testing
- **🦅 spaCy**: Procesamiento industrial de NLP
- **📚 NLTK**: Natural Language Toolkit clásico
- **🤗 transformers**: Modelos de Hugging Face
- **📊 scikit-learn**: Machine Learning tradicional
- **🎯 gensim**: Topic modeling y embeddings

## 📖 Documentación Adicional

- 📘 [**GUIA.md**](GUIA.md) - Guía detallada paso a paso
- 🤝 [**CONTRIBUTING.md**](CONTRIBUTING.md) - Cómo contribuir
- 📄 [**LICENSE**](LICENSE) - Licencia MIT

## 🌟 Orden Recomendado

Se recomienda seguir el orden de los koans (01 → 09) ya que cada uno construye sobre conceptos anteriores.

**Prerrequisitos**:
- ✅ Python básico (variables, funciones, clases)
- ✅ Comprensión básica de testing (opcional pero útil)

**No necesitas saber**:
- ❌ NLP previo
- ❌ Matemáticas avanzadas
- ❌ Deep Learning

## 💡 Consejos

1. **No te saltes koans**: Cada uno enseña conceptos fundamentales
2. **Lee la documentación**: Cada koan tiene comentarios explicativos
3. **Experimenta**: Prueba con tus propios textos
4. **Usa VS Code**: Configurado con tareas y debugging

## 🎯 VS Code Integration

Este proyecto está optimizado para VS Code con:
- ✅ Configuración de testing automática
- ✅ Debugging integrado
- ✅ Tasks para ejecutar koans individuales

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Ve [CONTRIBUTING.md](CONTRIBUTING.md) para más detalles.

## 📝 Licencia

MIT License - ve [LICENSE](LICENSE) para más detalles.

## 🙏 Inspiración

Proyecto inspirado en:
- Ruby Koans
- Go Koans
- El poder del aprendizaje mediante práctica deliberada

---

**¡Disfruta aprendiendo NLP! 🚀🧠**
