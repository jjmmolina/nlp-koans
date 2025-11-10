# 🧠 NLP Koans - Aprende Procesamiento de Lenguaje Natural con TDD

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![pytest](https://img.shields.io/badge/tested%20with-pytest-orange.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![spaCy](https://img.shields.io/badge/spaCy-3.7%2B-09a3d5.svg)](https://spacy.io/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

Un proyecto tutorial tipo **Koan** para aprender **Procesamiento de Lenguaje Natural (NLP)** usando **Test-Driven Development (TDD)** en Python.

## 🎯 ¿Qué son los NLP Koans?

Los **Koans** son ejercicios de aprendizaje donde:
1. ✅ Los tests **fallan inicialmente** 
2. 🔧 Tú **arreglas el código** para hacerlos pasar
3. 🎓 **Aprendes** los conceptos de NLP progresivamente

## 🚀 Inicio Rápido

### ⚡ Quick Start (5 minutos)

```bash
# 1. Clonar el repositorio
git clone https://github.com/jjmmolina/nlp-koans.git
cd nlp-koans

# 2. Crear entorno virtual
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias básicas (instala lo mínimo para empezar)
pip install pytest nltk

# 4. ¡Empezar con el primer koan!
pytest koans/01_tokenization/test_tokenization.py -v
# Verás tests fallando - ¡es lo esperado! 🎯
```

### 📦 Instalación Completa

Para usar TODOS los koans (incluyendo los avanzados):

```bash
# Instalar todas las dependencias (puede tardar)
pip install -r requirements.txt

# Descargar modelos de spaCy
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm

# Descargar recursos de NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt_tab')"

# Ejecutar todos los tests
pytest
```

> 💡 **Consejo**: Empieza con el Quick Start. Instala el resto cuando llegues a koans avanzados.

## 📚 Estructura de Koans

### 🎯 Nivel Básico (Koans 1-4)
| Koan | Tema | Librerías | Conceptos |
|------|------|-----------|-----------|
| **01** | Tokenización | NLTK, spaCy | Separación de texto en palabras/oraciones |
| **02** | Stemming & Lemmatization | NLTK, spaCy | Normalización de palabras |
| **03** | POS Tagging | spaCy, NLTK | Etiquetado gramatical |
| **04** | Named Entity Recognition | spaCy | Extracción de entidades |

### 🚀 Nivel Intermedio (Koans 5-7)
| Koan | Tema | Librerías | Conceptos |
|------|------|-----------|-----------|
| **05** | Text Classification | scikit-learn | Clasificación de textos |
| **06** | Sentiment Analysis | transformers | Análisis de sentimientos |
| **07** | Word Embeddings | spaCy, gensim | Representaciones vectoriales |

### 🧠 Nivel Avanzado (Koans 8-9)
| Koan | Tema | Librerías | Conceptos |
|------|------|-----------|-----------|
| **08** | Transformers | transformers (Hugging Face) | Modelos preentrenados |
| **09** | Language Models | transformers | Generación de texto |

### 🔮 Nivel Experto - LLMs Modernos (Koans 10-13)
| Koan | Tema | Librerías | Conceptos |
|------|------|-----------|-----------|
| **10** | Modern LLMs & APIs | OpenAI, Anthropic, **Ollama** | GPT-4, Claude, Gemini, local LLMs, streaming, function calling, **structured outputs** |
| **11** | AI Agents | LangChain, LangGraph | ReAct pattern, herramientas, memoria, callbacks, **DSPy** |
| **12** | Semantic Search | sentence-transformers, ChromaDB, FAISS | Embeddings, vector databases, búsqueda semántica, **híbrida** |
| **13** | RAG | LangChain, ChromaDB, **Instructor** | Retrieval-Augmented Generation, chunking, **evaluation**, **observabilidad** |

> 🆕 **Novedades 2025**: Ollama para LLMs locales (sin API keys), Instructor para outputs estructurados, DSPy para optimización automática, Guardrails AI para seguridad, LangSmith para observabilidad.

## 🎓 Cómo Usar Este Tutorial

### 🎯 Tu Primer Koan en 3 Pasos

**Paso 1: Ejecuta el test (verás que falla)**
```bash
cd koans/01_tokenization
pytest test_tokenization.py::TestTokenizationBasics::test_tokenize_words_nltk_spanish -v
```

Verás:
```
FAILED - AssertionError: La lista no debe estar vacía
```

**Paso 2: Abre `tokenization.py` y encuentra:**
```python
def tokenize_words_nltk(text: str) -> List[str]:
    # TODO: Implementa la tokenización de palabras con nltk.word_tokenize()
    # Pista: from nltk.tokenize import word_tokenize
    return []  # ← Esto está mal, retorna lista vacía
```

**Paso 3: Implementa la solución:**
```python
def tokenize_words_nltk(text: str) -> List[str]:
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)  # ← ¡Así está bien!
```

**Verifica:**
```bash
pytest test_tokenization.py::TestTokenizationBasics::test_tokenize_words_nltk_spanish -v
# ✅ PASSED - ¡Felicidades!
```

**¡Repite este proceso con todas las funciones!** 🔄

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
- 🗺️ [**LEARNING_PATH.md**](LEARNING_PATH.md) - Ruta de aprendizaje optimizada con tiempos estimados
- ❓ [**FAQ.md**](FAQ.md) - Preguntas frecuentes y troubleshooting
- 🤝 [**CONTRIBUTING.md**](CONTRIBUTING.md) - Cómo contribuir al proyecto
- 📄 [**LICENSE**](LICENSE) - Licencia MIT
- 📊 [**PROJECT_SUMMARY.md**](PROJECT_SUMMARY.md) - Resumen técnico del proyecto

## 🌟 Orden Recomendado

Se recomienda seguir el orden de los koans (01 → 13) ya que cada uno construye sobre conceptos anteriores.

**Niveles de Aprendizaje**:
- 🎯 **Básico (Koans 1-4)**: Fundamentos de NLP - 6-8 horas
- 🚀 **Intermedio (Koans 5-7)**: ML aplicado a NLP - 8-10 horas  
- 🧠 **Avanzado (Koans 8-9)**: Transformers y LLMs - 8-10 horas
- 🔮 **Experto (Koans 10-13)**: APIs modernas, Agentes, RAG - 10-15 horas

> 💡 **Los koans 10-13 ahora incluyen alternativas locales con Ollama** (sin API keys necesarias). Las API keys comerciales (OpenAI, Anthropic) son opcionales para comparar modelos.

> 🔬 **Tech Radar 2025**: El curso incorpora técnicas del Thoughtworks Technology Radar Vol. 33: DSPy (programming over prompting), Instructor (structured outputs), Guardrails AI (safety), LangSmith (observabilidad), y Mem0 (memoria personalizada).

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

## � VS Code Integration

Este proyecto está optimizado para VS Code con:
- ✅ Configuración de testing automática
- ✅ Debugging integrado
- ✅ Tasks para ejecutar koans individuales

## 🏆 Quick Wins - Tus Primeros 30 Minutos

¿Quieres ver resultados inmediatos? Sigue esto:

### 1️⃣ Setup Rápido (5 min)
```bash
git clone https://github.com/jjmmolina/nlp-koans.git
cd nlp-koans
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install pytest nltk
```

### 2️⃣ Tu Primera Victoria (10 min)
```bash
cd koans/01_tokenization
pytest test_tokenization.py::TestCustomTokenization::test_custom_tokenize_spaces -v
```

Abre `tokenization.py` y cambia:
```python
def custom_tokenize(text: str, delimiter: str = " ") -> List[str]:
    return []  # ❌ MAL
```

Por:
```python
def custom_tokenize(text: str, delimiter: str = " ") -> List[str]:
    return text.split(delimiter)  # ✅ BIEN
```

Ejecuta el test de nuevo:
```bash
pytest test_tokenization.py::TestCustomTokenization::test_custom_tokenize_spaces -v
# ✅ PASSED!
```

**🎉 ¡Felicidades! Completaste tu primer koan.**

### 3️⃣ Siguiente Nivel (15 min)

Ahora implementa `tokenize_words_nltk()`:
1. Lee el archivo `HINTS.md`
2. Sigue las pistas nivel por nivel
3. Haz pasar el test

```bash
pytest test_tokenization.py::TestTokenizationBasics::test_tokenize_words_nltk_spanish -v
```

**💪 ¡Ya dominas tokenización básica!**

---

**Continúa con el resto del Koan 01 y estarás oficialmente en camino al dominio de NLP.** 🚀

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
