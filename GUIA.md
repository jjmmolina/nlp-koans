# 📘 Guía Completa de NLP Koans

## 🎯 Cómo Usar Este Tutorial

### Filosofía de los Koans

Los **Koans** son ejercicios de aprendizaje progresivo donde:

1. **Lees** el código y los comentarios
2. **Ejecutas** los tests (que fallan inicialmente)
3. **Implementas** el código faltante
4. **Verificas** que los tests pasen
5. **Reflexionas** sobre lo aprendido

### Orden Recomendado

Sigue este orden para máximo aprendizaje:

```
01. Tokenización           → Fundamentos: dividir texto
02. Stemming/Lemmatization → Normalización de palabras
03. POS Tagging            → Etiquetado gramatical
04. NER                    → Reconocimiento de entidades
05. Text Classification    → Clasificación con ML tradicional
06. Sentiment Analysis     → Análisis de sentimientos
07. Word Embeddings        → Representaciones vectoriales
08. Transformers           → Modelos modernos (BERT, GPT)
09. Language Models        → Generación de texto
```

## 🚀 Instalación Paso a Paso

### 1. Clonar el Repositorio

```bash
git clone <tu-repo>
cd NLP-Koan
```

### 2. Crear Entorno Virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar Modelos de NLP

**spaCy (español e inglés):**
```bash
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

**NLTK:**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt_tab')"
```

## 📖 Ejemplo Práctico: Koan 01

### Paso 1: Ubicación

```bash
cd koans/01_tokenization
```

### Paso 2: Ejecutar Tests

```bash
pytest test_tokenization.py -v
```

Verás algo como:
```
FAILED test_tokenization.py::TestTokenizationBasics::test_tokenize_words_nltk_spanish
AssertionError: La lista no debe estar vacía
```

### Paso 3: Abrir el Código

Abre `tokenization.py` y encuentra:

```python
def tokenize_words_nltk(text: str) -> List[str]:
    # TODO: Implementa la tokenización de palabras con nltk.word_tokenize()
    return []
```

### Paso 4: Implementar

```python
from nltk.tokenize import word_tokenize

def tokenize_words_nltk(text: str) -> List[str]:
    return word_tokenize(text)
```

### Paso 5: Verificar

```bash
pytest test_tokenization.py::TestTokenizationBasics::test_tokenize_words_nltk_spanish -v
```

Si ves `PASSED`, ¡lo lograste! ✅

### Paso 6: Siguiente Función

Repite el proceso con la siguiente función marcada con `# TODO`.

## 🎓 Consejos para Programadores Python

### Diferencias Clave con Otras Librerías

**1. spaCy vs NLTK**

```python
# NLTK: más manual, más control
from nltk.tokenize import word_tokenize
tokens = word_tokenize("Hola mundo")

# spaCy: más automático, más integrado
import spacy
nlp = spacy.load("es_core_news_sm")
doc = nlp("Hola mundo")
tokens = [token.text for token in doc]
```

**2. TF-IDF vs Embeddings**

```python
# TF-IDF: basado en frecuencia (clásico)
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(texts)

# Embeddings: basado en contexto (moderno)
import spacy
nlp = spacy.load("es_core_news_sm")
doc = nlp("Python es genial")
vector = doc.vector  # Vector de 96 dimensiones
```

**3. ML Clásico vs Transformers**

```python
# Clásico: scikit-learn (rápido, menos datos)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Moderno: Transformers (potente, más datos)
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("Me encanta Python!")
```

## 💡 Trucos y Atajos

### Ejecutar Solo Tests que Fallan

```bash
pytest --lf  # last-failed
```

### Ejecutar con Más Detalle

```bash
pytest -vv --tb=short
```

### Ejecutar Un Solo Test

```bash
pytest koans/01_tokenization/test_tokenization.py::TestTokenizationBasics::test_tokenize_words_nltk_spanish -v
```

### Medir Cobertura

```bash
pytest --cov=koans --cov-report=html
```

## 🔍 Debugging

### Usar Print Debugging

```python
def tokenize_words_nltk(text: str) -> List[str]:
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    print(f"DEBUG: tokens = {tokens}")  # 👈 Agrega esto
    return tokens
```

### Usar el Debugger de VS Code

1. Coloca un breakpoint (F9) en la línea que quieres inspeccionar
2. Ejecuta con Debug (F5)
3. Inspecciona variables en el panel lateral

### Usar pytest con pdb

```bash
pytest --pdb  # Se detiene en el primer error
```

## 📊 Progreso

Para ver tu progreso:

```bash
# Ejecutar todos los tests
pytest

# Ver resumen
pytest --tb=no -q
```

## 🎯 Objetivos de Aprendizaje por Koan

### Koan 01: Tokenización
- ✅ Entender qué es la tokenización
- ✅ Usar NLTK para tokenizar
- ✅ Usar spaCy para tokenizar
- ✅ Diferencias entre tokenización de palabras y oraciones

### Koan 02: Stemming/Lemmatization
- ✅ Diferencias entre stemming y lemmatization
- ✅ Cuándo usar cada técnica
- ✅ Implementar con NLTK y spaCy

### Koan 03: POS Tagging
- ✅ Identificar categorías gramaticales
- ✅ Extraer sustantivos, verbos, adjetivos
- ✅ Usar POS tags para análisis

### Koan 04: NER
- ✅ Reconocer entidades nombradas
- ✅ Extraer personas, lugares, organizaciones
- ✅ Aplicaciones prácticas de NER

### Koan 05: Text Classification
- ✅ Características TF-IDF y BoW
- ✅ Entrenar clasificadores
- ✅ Evaluar modelos

### Koan 06: Sentiment Analysis
- ✅ Análisis de sentimientos con Transformers
- ✅ Modelos preentrenados
- ✅ Fine-tuning (opcional)

### Koan 07: Word Embeddings
- ✅ Representaciones vectoriales
- ✅ Similitud semántica
- ✅ spaCy vectors y word2vec

### Koan 08: Transformers
- ✅ BERT, GPT y otros modelos
- ✅ Hugging Face Transformers
- ✅ Pipelines predefinidos

### Koan 09: Language Models
- ✅ Generación de texto
- ✅ Completado automático
- ✅ Modelos generativos

## 🤝 Pedir Ayuda

Si te quedas atascado:

1. **Lee los comentarios** en el código
2. **Consulta la documentación** oficial de cada librería
3. **Ejecuta los tests con -vv** para ver más detalles
4. **Busca ejemplos** en la documentación de spaCy/NLTK
5. **Abre un issue** en el repositorio

## 🎓 Recursos Adicionales

- **spaCy**: https://spacy.io/
- **NLTK**: https://www.nltk.org/
- **Hugging Face**: https://huggingface.co/
- **scikit-learn**: https://scikit-learn.org/

¡Disfruta del aprendizaje! 🚀
