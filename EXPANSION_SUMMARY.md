# 🎉 NLP Koans - Expansión Completada

## ✅ Resumen de lo Agregado

Se han añadido **4 nuevos koans avanzados** (10-13) que cubren las tecnologías más modernas de NLP y LLMs:

### 🔮 Koan 10: Modern LLMs & APIs
**Archivos creados:**
- `modern_llms.py` - 7 funciones para trabajar con APIs de LLMs
- `test_modern_llms.py` - Tests completos para cada función
- `HINTS.md` - Guía progresiva con 3 niveles de pistas
- `__init__.py`

**Aprenderás:**
- Llamar a OpenAI GPT-4, GPT-4o, y o1
- Usar Anthropic Claude (claude-3-5-sonnet)
- Integrar Google Gemini API
- Streaming de respuestas
- Function calling con LLMs
- Comparar outputs de múltiples proveedores
- Manejo de errores y rate limits

**APIs necesarias:** OpenAI, Anthropic, Google (todas tienen créditos gratis de prueba)

---

### 🤖 Koan 11: AI Agents
**Archivos creados:**
- `ai_agents.py` - 8 funciones para construir agentes autónomos
- `test_ai_agents.py` - Tests para ReAct, tools, memoria
- `HINTS.md` - Guía completa con ejemplos de LangChain
- `__init__.py`

**Aprenderás:**
- Patrón ReAct (Reasoning + Acting)
- Crear agentes con LangChain
- Usar herramientas: calculadora, búsqueda web
- Crear herramientas personalizadas
- Memoria conversacional
- Callbacks para monitoreo
- Colaboración multi-agente

**Tecnologías:** LangChain, LangChain Tools, DuckDuckGo Search

---

### 🔍 Koan 12: Semantic Search & Vector Databases
**Archivos creados:**
- `semantic_search.py` - 8 funciones para búsqueda semántica
- `test_semantic_search.py` - Tests para embeddings y búsqueda
- `HINTS.md` - Guía de embeddings y vector DBs
- `__init__.py`

**Aprenderás:**
- Crear embeddings con OpenAI
- Usar Sentence Transformers (local, gratis)
- Búsqueda por similitud coseno
- ChromaDB para vector search
- FAISS para búsqueda rápida
- Reranking con cross-encoders
- Comparar estrategias de búsqueda

**Tecnologías:** OpenAI Embeddings, sentence-transformers, ChromaDB, FAISS

---

### 📚 Koan 13: RAG (Retrieval-Augmented Generation)
**Archivos creados:**
- `rag.py` - 9 funciones para RAG
- `test_rag.py` - Tests para pipelines RAG
- `HINTS.md` - Guía completa de RAG patterns
- `__init__.py`

**Aprenderás:**
- Document chunking inteligente
- Vector stores con LangChain
- Retrievers (similarity, MMR)
- RAG básico con LangChain
- RAG con citas y fuentes
- Multi-query RAG
- RAG Fusion (múltiples estrategias)
- RAG conversacional
- Métricas de evaluación (faithfulness, relevancy)

**Tecnologías:** LangChain, ChromaDB, OpenAI

---

## 📝 Archivos Actualizados

### `requirements.txt`
Agregadas dependencias:
```
# LLM APIs (Koans 10-13)
openai>=1.0.0
anthropic>=0.18.0
google-generativeai>=0.3.0

# LangChain & Agents (Koans 11, 13)
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.20
duckduckgo-search>=4.0.0

# Semantic Search & Vector Databases (Koans 12-13)
sentence-transformers>=2.2.0
chromadb>=0.4.0
faiss-cpu>=1.7.4
```

### `README.md`
- ✅ Agregada tabla con 4 niveles de dificultad
- ✅ Incluidos koans 10-13 en estructura
- ✅ Tiempos estimados por nivel
- ✅ Nota sobre requerimiento de API keys

---

## 📊 Estadísticas del Proyecto

| Métrica | Valor |
|---------|-------|
| **Total de Koans** | 13 |
| **Archivos Python creados** | 8 (4 koans × 2) |
| **Tests creados** | 8 archivos de test |
| **HINTS.md escritos** | 4 (uno por koan) |
| **Líneas de código** | ~3,500+ |
| **Funciones totales** | 32 funciones nuevas |
| **Tests totales** | ~30 test classes nuevas |

---

## 🚀 Próximos Pasos para el Usuario

### 1. Instalar Nuevas Dependencias
```bash
pip install -r requirements.txt
```

### 2. Configurar API Keys
Crea un archivo `.env` o configura variables de entorno:
```bash
export OPENAI_API_KEY="tu-key-aqui"
export ANTHROPIC_API_KEY="tu-key-aqui"
export GOOGLE_API_KEY="tu-key-aqui"
```

**Obtener API Keys gratis:**
- OpenAI: https://platform.openai.com/ ($5 crédito gratis)
- Anthropic: https://console.anthropic.com/ ($5 crédito gratis)
- Google AI: https://makersuite.google.com/app/apikey (gratis)

### 3. Empezar con Koan 10
```bash
cd koans/10_modern_llms
pytest test_modern_llms.py -v
```

### 4. Seguir la Ruta de Aprendizaje
- Koan 10: Modern LLMs & APIs (2-3 horas)
- Koan 11: AI Agents (3-4 horas)
- Koan 12: Semantic Search (2-3 horas)
- Koan 13: RAG (3-5 horas)

---

## 💡 Características de los Nuevos Koans

### ✨ Todos incluyen:
- ✅ Implementaciones con `pass` (para que el estudiante complete)
- ✅ Tests exhaustivos con `@pytest.mark.skipif` para API keys opcionales
- ✅ HINTS.md con 3 niveles de ayuda
- ✅ Ejemplos reales y prácticos
- ✅ Comentarios explicativos en español
- ✅ Seguimiento del patrón TDD

### 🎯 Patrón de Aprendizaje:
1. **Ejecutar test** → Falla (expected)
2. **Leer HINTS.md** → 3 niveles de ayuda
3. **Implementar función** → Arreglar código
4. **Ejecutar test** → Pasa ✅
5. **Siguiente función** → Repetir

---

## 🌟 Tecnologías Modernas Cubiertas

### APIs de LLMs
- OpenAI GPT-4, GPT-4o, o1
- Anthropic Claude 3.5 Sonnet
- Google Gemini 1.5 Pro

### Frameworks
- LangChain (Agents & RAG)
- Hugging Face sentence-transformers

### Vector Databases
- ChromaDB (simple, local)
- FAISS (rápido, escalable)

### Patrones Avanzados
- ReAct (Reasoning + Acting)
- RAG (Retrieval-Augmented Generation)
- Multi-agent collaboration
- Semantic search con reranking

---

## 📚 Recursos Adicionales en HINTS.md

Cada HINTS.md incluye:
- 📖 Conceptos clave explicados
- 🔧 Mejores prácticas
- 📊 Tablas comparativas
- 🔗 Links a documentación oficial
- 💡 Tips y trucos
- ⚠️ Problemas comunes y soluciones

---

## 🎓 Nivel de Dificultad

| Nivel | Koans | Requisitos | Tiempo |
|-------|-------|------------|--------|
| 🎯 Básico | 1-4 | Python básico | 6-8h |
| 🚀 Intermedio | 5-7 | ML básico | 8-10h |
| 🧠 Avanzado | 8-9 | Transformers | 8-10h |
| 🔮 Experto | 10-13 | API keys, $ | 10-15h |

**Total: ~35-45 horas** de aprendizaje práctico

---

## ✅ Checklist de Completitud

- ✅ Koan 10: Modern LLMs & APIs
  - ✅ modern_llms.py (7 funciones)
  - ✅ test_modern_llms.py (7 test classes)
  - ✅ HINTS.md (completo)
  - ✅ __init__.py

- ✅ Koan 11: AI Agents  
  - ✅ ai_agents.py (8 funciones)
  - ✅ test_ai_agents.py (7 test classes)
  - ✅ HINTS.md (completo)
  - ✅ __init__.py

- ✅ Koan 12: Semantic Search
  - ✅ semantic_search.py (8 funciones)
  - ✅ test_semantic_search.py (7 test classes)
  - ✅ HINTS.md (completo)
  - ✅ __init__.py

- ✅ Koan 13: RAG
  - ✅ rag.py (9 funciones)
  - ✅ test_rag.py (7 test classes)
  - ✅ HINTS.md (completo)
  - ✅ __init__.py

- ✅ Documentación actualizada
  - ✅ README.md (tabla de 4 niveles)
  - ✅ requirements.txt (nuevas deps)

- ✅ Git
  - ✅ Commit con mensaje descriptivo
  - ✅ Push a GitHub

---

## 🎉 ¡Proyecto Expandido Exitosamente!

El proyecto NLP Koans ahora cubre **desde tokenización básica hasta RAG con LLMs modernos**, proporcionando una ruta de aprendizaje completa de 35-45 horas para dominar NLP y LLMs.

**Repositorio:** https://github.com/jjmmolina/nlp-koans
**Commit:** d7b4c37 - "feat: Add 4 new advanced koans (10-13) for modern LLMs, agents, semantic search, and RAG"

---

## 📬 Siguiente Paso Recomendado

```bash
# Actualizar el repositorio local
git pull

# Instalar nuevas dependencias
pip install -r requirements.txt

# Probar el primer koan avanzado
cd koans/10_modern_llms
pytest test_modern_llms.py -v
```

**¡Disfruta explorando el mundo de los LLMs modernos! 🚀🤖**
