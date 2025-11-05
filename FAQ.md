# ❓ Preguntas Frecuentes (FAQ)

## 🚀 Instalación y Setup

### ¿Tengo que instalar TODO requirements.txt desde el inicio?

**No**. Puedes empezar solo con lo básico:

```bash
# Para Koans 1-3 (solo necesitas NLTK y spaCy)
pip install pytest nltk spacy
python -m spacy download es_core_news_sm
```

Instala el resto cuando llegues a koans avanzados (06-09).

### ¿Qué versión de Python necesito?

Python **3.8 o superior**. Verifica con:
```bash
python --version
```

### Los modelos de spaCy/NLTK no se descargan

**Solución**:
```python
# Ejecuta esto en Python interactivo
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Para spaCy, usa el comando directo
python -m spacy download es_core_news_sm
```

### Error: "No module named 'transformers'"

Solo necesitas transformers para Koans 06-09. Instálalo cuando llegues allí:
```bash
pip install transformers torch
```

---

## 🧪 Tests y Ejecución

### ¿Por qué todos los tests fallan al inicio?

**¡Es lo esperado!** Los koans están diseñados así:
- Tests rojos = concepto por aprender
- Tests verdes = concepto dominado

### ¿Cómo ejecuto un solo test?

```bash
# Un test específico
pytest koans/01_tokenization/test_tokenization.py::TestTokenizationBasics::test_tokenize_words_nltk_spanish -v

# Todos los tests de una clase
pytest koans/01_tokenization/test_tokenization.py::TestTokenizationBasics -v

# Todos los tests de un archivo
pytest koans/01_tokenization/test_tokenization.py -v
```

### ¿Qué significa @pytest.mark.slow?

Tests que requieren descargar/ejecutar modelos grandes (transformers, GPT).

```bash
# Omitir tests lentos
pytest -m "not slow"

# Ejecutar SOLO tests lentos
pytest -m "slow"
```

### Error: "AssertionError: La lista no debe estar vacía"

Esto significa que tu función retorna `[]` en lugar del resultado correcto. Es el error más común cuando empiezas. **Implementa la función**.

---

## 💻 Programación

### ¿Tengo que implementar TODO en cada función?

**Sí**. Cada `# TODO` marca código que DEBES escribir. Los tests te guían.

### ¿Puedo ver las soluciones?

**Sí**, pero intenta primero:
1. Lee las pistas en `HINTS.md` (si existe)
2. Intenta resolver 10-15 minutos
3. Consulta la solución en `HINTS.md` como último recurso

### ¿Puedo usar otras librerías?

Para aprender, usa las que se mencionan. Después experimenta libremente.

### Mi código funciona pero los tests fallan

Verifica:
1. **Tipo de retorno**: ¿Retornas `list`, `dict`, `str` como se espera?
2. **Formato exacto**: Lee el docstring del test
3. **Ejecuta con -vv**: `pytest -vv` muestra más detalles

---

## 🎓 Aprendizaje

### ¿Cuánto tiempo toma completar todo?

**Estimado**: 30-40 horas totales
- **Intensivo**: 1-2 semanas (3-4h/día)
- **Normal**: 3-4 semanas (1-2h/día)
- **Relajado**: 6-8 semanas (1h/día, 3-4 días/semana)

Ver `LEARNING_PATH.md` para más detalles.

### ¿Puedo saltar koans?

**No recomendado**. Cada koan construye sobre el anterior. Si algo es muy difícil, revisa koans anteriores.

### ¿Necesito saber matemáticas avanzadas?

**No para koans 1-5** (básicos e intermedios).  
**Ayuda para koans 6-9** (embeddings, transformers) entender:
- Vectores y matrices (básico)
- Probabilidades (básico)
- No necesitas cálculo ni álgebra lineal avanzada

### ¿Necesito experiencia previa en NLP?

**No**. Este tutorial asume:
- ✅ Sabes Python básico (funciones, listas, diccionarios)
- ✅ Entiendes qué son los tests (o lo aprenderás rápido)
- ❌ NO necesitas NLP previo
- ❌ NO necesitas ML previo

### ¿Los koans están en español o inglés?

- **Código y comentarios**: Español
- **Nombres de funciones**: Inglés (convención)
- **Ejemplos**: Ambos idiomas
- **Tests**: Español

---

## 🐛 Problemas Comunes

### ImportError: cannot import name 'word_tokenize'

**Causa**: NLTK no ha descargado el recurso `punkt`.

**Solución**:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### OSError: [E050] Can't find model 'es_core_news_sm'

**Causa**: Modelo de spaCy no instalado.

**Solución**:
```bash
python -m spacy download es_core_news_sm
```

### Tests de Transformers muy lentos

**Es normal**. La primera vez descarga modelos (GB de datos).

**Soluciones**:
- Ejecuta en momento con buena conexión
- Usa `-m "not slow"` para omitirlos temporalmente
- Sé paciente (solo pasa una vez)

### RuntimeError: Torch not compiled with CUDA

**No es un error**. Significa que PyTorch usará CPU en lugar de GPU.  
Para koans, CPU es suficiente (solo será más lento).

### Memory Error al ejecutar tests

**Causa**: Modelos grandes (GPT, BERT) requieren RAM.

**Soluciones**:
- Cierra otras aplicaciones
- Ejecuta tests de uno en uno
- Usa modelos más pequeños si es posible

---

## 📚 Librerías Específicas

### ¿Cuándo uso NLTK vs spaCy?

**NLTK**: 
- ✅ Aprendizaje (más explícito)
- ✅ Tareas simples
- ✅ Control fino

**spaCy**:
- ✅ Producción
- ✅ Rendimiento
- ✅ Pipeline completo

**En los koans**: Usarás ambos para comparar.

### ¿Por qué Transformers es tan pesado?

Modelos pre-entrenados (BERT, GPT) son redes neuronales GRANDES:
- BERT-base: ~110M parámetros
- GPT-2: ~1.5B parámetros

Es normal que ocupen GB y tarden en descargar.

### ¿Necesito GPU?

**No**. Todos los koans funcionan en CPU.  
GPU acelera, pero no es necesaria para aprender.

---

## 🔧 Troubleshooting Avanzado

### pytest no encuentra los módulos

**Asegúrate de estar en el directorio correcto**:
```bash
cd NLP-Koan  # Raíz del proyecto
pytest koans/01_tokenization/test_tokenization.py
```

### Conflictos de dependencias

**Usa entorno virtual**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Tests pasan localmente pero fallan en CI

Puede deberse a:
- Versiones diferentes de librerías
- Modelos no descargados en CI
- Diferencias de sistema operativo

Para koans locales, no te preocupes por CI.

---

## 🎯 Después de Completar

### ¿Qué hago después de terminar todos los koans?

1. **Proyectos propios**: Aplica lo aprendido
2. **Contribuye**: Mejora este proyecto
3. **Profundiza**: 
   - Curso de Deep Learning
   - Papers de investigación
   - Kaggle competitions

### ¿Hay koans más avanzados?

**Actualmente no**, pero podrías:
- Proponer nuevos koans (ver CONTRIBUTING.md)
- Hacer fork y crear tus propios
- Compartir tus proyectos

### ¿Cómo puedo practicar más?

**Proyectos sugeridos**:
1. Clasificador de noticias
2. Analizador de sentimientos de Twitter
3. Chatbot simple
4. Extractor de información de CVs
5. Sistema de Q&A sobre documentos

---

## 🤝 Comunidad y Contribución

### ¿Cómo reporto un bug?

1. Abre un Issue en GitHub
2. Describe el problema
3. Incluye:
   - Versión de Python
   - Output del error
   - Pasos para reproducir

### ¿Puedo contribuir?

¡Sí! Ver `CONTRIBUTING.md` para detalles.

**Ideas de contribución**:
- Nuevos tests
- Más pistas (HINTS.md)
- Correcciones
- Traducción al inglés
- Nuevos koans

### ¿Hay un chat/foro?

Usa **GitHub Discussions** para:
- Preguntas
- Compartir proyectos
- Discutir mejoras

---

## 📖 Recursos Adicionales

### Documentación oficial

- **NLTK**: https://www.nltk.org/
- **spaCy**: https://spacy.io/
- **Transformers**: https://huggingface.co/docs/transformers/
- **scikit-learn**: https://scikit-learn.org/

### Cursos recomendados

- **Fast.ai NLP**: Gratuito, práctico
- **CS224n (Stanford)**: Profundo, teórico
- **Coursera NLP Specialization**: Completo

### Libros

- "Speech and Language Processing" (Jurafsky & Martin) - Gratuito online
- "Natural Language Processing with Python" (NLTK Book)

---

**¿No encuentras tu pregunta? Abre un Issue en GitHub!** 🙋
