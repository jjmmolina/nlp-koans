# 📚 NLP Koans - Resumen del Proyecto

## 🎯 Descripción General

**NLP Koans** es un proyecto tutorial interactivo para aprender **Procesamiento de Lenguaje Natural (NLP)** usando la metodología **Test-Driven Development (TDD)** con un enfoque tipo **Koan**.

## 🏗️ Estructura del Proyecto

```
NLP-Koan/
├── .github/
│   └── copilot-instructions.md    # Instrucciones para Copilot
├── koans/                          # 9 koans progresivos
│   ├── 01_tokenization/
│   ├── 02_stemming_lemmatization/
│   ├── 03_pos_tagging/
│   ├── 04_ner/
│   ├── 05_text_classification/
│   ├── 06_sentiment_analysis/
│   ├── 07_word_embeddings/
│   ├── 08_transformers/
│   └── 09_language_models/
├── README.md                       # Documentación principal
├── GUIA.md                         # Guía paso a paso
├── CONTRIBUTING.md                 # Guía para contribuir
├── requirements.txt                # Dependencias Python
├── pytest.ini                      # Configuración de pytest
├── LICENSE                         # Licencia MIT
├── .gitignore                      # Archivos a ignorar en Git
├── check_progress.ps1              # Script de progreso (Windows)
└── check_progress.sh               # Script de progreso (Linux/Mac)
```

## 📚 Koans Incluidos

### Koan 01: Tokenización
- **Conceptos**: División de texto en tokens
- **Librerías**: NLTK, spaCy
- **Funciones**: 7 funciones con TODOs
- **Tests**: 6 clases de test, 15+ tests

### Koan 02: Stemming y Lemmatization
- **Conceptos**: Normalización de palabras
- **Librerías**: NLTK (Porter, Snowball), spaCy
- **Funciones**: 8 funciones con TODOs
- **Tests**: 5 clases de test, 12+ tests

### Koan 03: POS Tagging
- **Conceptos**: Etiquetado gramatical
- **Librerías**: spaCy, NLTK
- **Funciones**: 8 funciones con TODOs
- **Tests**: 6 clases de test, 14+ tests

### Koan 04: Named Entity Recognition (NER)
- **Conceptos**: Reconocimiento de entidades nombradas
- **Librerías**: spaCy
- **Funciones**: 9 funciones con TODOs
- **Tests**: 7 clases de test, 16+ tests

### Koan 05: Text Classification
- **Conceptos**: Clasificación de textos con ML
- **Librerías**: scikit-learn
- **Funciones**: 9 funciones con TODOs
- **Tests**: 6 clases de test, 13+ tests

### Koan 06: Sentiment Analysis
- **Conceptos**: Análisis de sentimientos
- **Librerías**: transformers (Hugging Face)
- **Funciones**: 7 funciones con TODOs
- **Tests**: 6 clases de test (algunos marcados como @slow)

### Koan 07: Word Embeddings
- **Conceptos**: Representaciones vectoriales
- **Librerías**: spaCy, numpy, scipy
- **Funciones**: 8 funciones con TODOs
- **Tests**: 6 clases de test, 10+ tests

### Koan 08: Transformers
- **Conceptos**: BERT, GPT, modelos pre-entrenados
- **Librerías**: transformers, torch
- **Funciones**: 9 funciones con TODOs
- **Tests**: 7 clases de test (mayoría marcados como @slow)

### Koan 09: Language Models
- **Conceptos**: Generación de texto, LLMs
- **Librerías**: transformers, torch
- **Funciones**: 10 funciones con TODOs
- **Tests**: 7 clases de test (mayoría marcados como @slow)

## 🛠️ Tecnologías y Dependencias

### Librerías Principales
- **spaCy 3.7+**: Procesamiento industrial de NLP
- **NLTK 3.8+**: Toolkit clásico de NLP
- **transformers 4.35+**: Modelos de Hugging Face
- **scikit-learn 1.3+**: Machine Learning tradicional
- **torch 2.1+**: Backend para transformers
- **gensim 4.3+**: Word embeddings y topic modeling

### Testing
- **pytest**: Framework de testing
- **pytest-cov**: Cobertura de código
- **pytest-xdist**: Ejecución paralela

### Modelos Requeridos
```bash
# spaCy
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm

# NLTK
punkt, stopwords, averaged_perceptron_tagger,
wordnet, omw-1.4, punkt_tab
```

## 📊 Estadísticas del Proyecto

- **Total de Koans**: 9
- **Total de Funciones**: ~75 funciones con TODOs
- **Total de Tests**: ~120+ tests
- **Líneas de Código**: ~3,500+ líneas
- **Clases de Test**: ~55 clases
- **Idiomas Soportados**: Español e Inglés

## 🎓 Metodología de Aprendizaje

### Filosofía Koan
1. **Red**: Los tests fallan inicialmente
2. **Green**: Implementas el código para hacerlos pasar
3. **Refactor**: Mejoras el código (opcional)
4. **Reflexión**: Entiendes el concepto

### Progresión
- **Básico → Intermedio → Avanzado**
- **Clásico (NLTK) → Moderno (spaCy) → Estado del Arte (Transformers)**
- **Teoría → Práctica → Aplicación Real**

## 💡 Características Únicas

### 1. Dual-Language Support
- Ejemplos en español e inglés
- Comentarios en español
- Soporte para modelos multiidioma

### 2. Tests Completos
- Tests descriptivos con docstrings
- Ejemplos del mundo real
- Marcadores para tests lentos (@slow)

### 3. Documentación Exhaustiva
- Docstrings con ejemplos en todas las funciones
- Pistas (hints) en los TODOs
- README, GUIA y CONTRIBUTING detallados

### 4. Herramientas de Progreso
- Scripts de verificación automática
- Soporte para Windows y Linux/Mac
- Informes visuales de progreso

## 🚀 Casos de Uso

### Para Estudiantes
- Aprender NLP desde cero
- Preparación para proyectos de NLP
- Entender librerías modernas

### Para Profesores
- Material de curso listo para usar
- Tests automáticos para evaluación
- Progresión estructurada

### Para Desarrolladores
- Referencia rápida de NLP
- Ejemplos prácticos
- Comparación de técnicas

## 📈 Roadmap Futuro (Posibles Mejoras)

1. **Koans Adicionales**:
   - Topic Modeling (LDA, NMF)
   - Text Summarization
   - Machine Translation avanzada
   - Speech Recognition

2. **Mejoras Técnicas**:
   - Notebooks Jupyter interactivos
   - Visualizaciones de resultados
   - Datasets de ejemplo incluidos
   - Docker container para fácil setup

3. **Internacionalización**:
   - Versión completa en inglés
   - Soporte para más idiomas
   - Modelos específicos por idioma

4. **Integraciones**:
   - GitHub Codespaces ready
   - VS Code extension
   - Integración con plataformas de aprendizaje

## 🤝 Contribuciones

El proyecto está abierto a contribuciones:
- Nuevos koans
- Mejoras en tests
- Correcciones de bugs
- Traducciones
- Documentación

## 📄 Licencia

MIT License - Uso libre para educación y proyectos comerciales

## 🙏 Agradecimientos

Inspirado por:
- Ruby Koans
- Go Koans
- La comunidad de NLP en Python

## 📞 Contacto y Soporte

- **Issues**: Para reportar bugs o sugerir mejoras
- **Discussions**: Para preguntas generales
- **Pull Requests**: Para contribuciones

---

**Versión**: 1.0.0  
**Fecha de Creación**: Noviembre 2025  
**Última Actualización**: Noviembre 2025

¡Feliz aprendizaje de NLP! 🚀🧠
