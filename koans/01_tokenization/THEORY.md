# Teoría: Tokenización

## 📚 Tabla de Contenidos
1. [Introducción a la Tokenización](#introducción)
2. [Tipos de Tokenización](#tipos)
3. [Tokenización en Diferentes Idiomas](#idiomas)
4. [Herramientas y Bibliotecas](#herramientas)
5. [Casos Especiales](#casos-especiales)
6. [Tokenización Moderna](#moderna)
7. [Casos de Uso](#casos-uso)

---

## 🎯 Introducción a la Tokenización {#introducción}

### ¿Qué es la Tokenización?

La **tokenización** es el proceso de dividir texto en unidades más pequeñas llamadas **tokens**. Es el primer paso fundamental en prácticamente cualquier pipeline de procesamiento de lenguaje natural (NLP).

```
Texto: "Hello, world! How are you?"

Tokens: ["Hello", ",", "world", "!", "How", "are", "you", "?"]
```

### ¿Por qué es Importante?

**1. Unidad Básica de Procesamiento**
```python
# Las computadoras necesitan unidades discretas para trabajar
texto = "I love Python"

# ❌ Difícil de procesar como string completo
# ✅ Fácil de procesar como lista de palabras
tokens = ["I", "love", "Python"]
```

**2. Base para Análisis Posterior**
```
Tokenización → POS Tagging → NER → Parsing → ...
     ↑
  Primer paso esencial
```

**3. Impacto en Calidad**
```python
# Mala tokenización
"don't" → ["don", "'", "t"]  # ❌ Pierde significado

# Buena tokenización
"don't" → ["do", "n't"]  # ✅ Preserva estructura gramatical
# O alternativamente:
"don't" → ["don't"]  # ✅ Mantiene como unidad
```

### Desafíos

**Ambigüedad de Límites:**
```
"New York" → ¿["New", "York"] o ["New York"]?
"Ph.D." → ¿["Ph", ".", "D", "."] o ["Ph.D."]?
"rock'n'roll" → ¿["rock", "'", "n", "'", "roll"] o ["rock'n'roll"]?
```

**Diferencias Entre Idiomas:**
```
Inglés: "I love NLP" → ["I", "love", "NLP"] ✅ (separados por espacios)
Chino: "我爱自然语言处理" → ¿? (sin espacios explícitos)
Alemán: "Donaudampfschifffahrtsgesellschaft" → ¿una palabra o varias?
```

---

## 📝 Tipos de Tokenización {#tipos}

### 1. Word Tokenization (Tokenización por Palabras)

La forma más común: dividir texto en palabras.

**Método Ingenuo:**
```python
# ❌ Demasiado simple
text = "Hello, world!"
tokens = text.split()  # ["Hello,", "world!"]
# Problema: puntuación pegada a palabras
```

**Método con Regex:**
```python
import re

text = "Hello, world! How are you?"
tokens = re.findall(r'\w+|[^\w\s]', text)
# ["Hello", ",", "world", "!", "How", "are", "you", "?"]
```

**NLTK Word Tokenizer:**
```python
from nltk.tokenize import word_tokenize

text = "Hello, world! Don't worry."
tokens = word_tokenize(text)
# ["Hello", ",", "world", "!", "Do", "n't", "worry", "."]
```

**spaCy Tokenizer:**
```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Hello, world! Don't worry.")
tokens = [token.text for token in doc]
# ["Hello", ",", "world", "!", "Do", "n't", "worry", "."]
```

### 2. Sentence Tokenization (Tokenización por Oraciones)

Dividir texto en oraciones.

**Desafío:**
```python
text = "Dr. Smith works at U.S.A. Inc. He loves NLP."
# ¿Dónde terminan las oraciones?
# "Dr." no es fin de oración
# "U.S.A." tampoco
# "Inc." tampoco
# Solo después de "NLP." es fin de oración
```

**NLTK Sentence Tokenizer:**
```python
from nltk.tokenize import sent_tokenize

text = "Dr. Smith works at U.S.A. Inc. He loves NLP."
sentences = sent_tokenize(text)
# ["Dr. Smith works at U.S.A. Inc.", "He loves NLP."]
```

**spaCy Sentence Segmentation:**
```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Dr. Smith works at U.S.A. Inc. He loves NLP.")
sentences = [sent.text for sent in doc.sents]
# ["Dr. Smith works at U.S.A. Inc.", "He loves NLP."]
```

### 3. Character Tokenization

Dividir en caracteres individuales.

```python
text = "Hello"
tokens = list(text)
# ["H", "e", "l", "l", "o"]
```

**Cuándo usar:**
- Modelos de generación de texto
- OCR (reconocimiento óptico de caracteres)
- Análisis morfológico detallado

### 4. Subword Tokenization

Dividir en subpalabras (entre caracteres y palabras completas).

**Problema que Resuelve:**
```python
# Vocabulario limitado con palabras completas
vocab = {"cat", "dog", "run", "running"}
# ¿Qué hacer con "cats", "dogs", "runner"? ❌ No están en vocabulario

# Con subword tokenization
vocab = {"cat", "dog", "run", "ning", "s", "er"}
"cats" → ["cat", "s"] ✅
"running" → ["run", "ning"] ✅
"runner" → ["run", "er"] ✅
```

**BPE (Byte-Pair Encoding):**
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE())

# Ejemplo de tokens
"unhappiness" → ["un", "happiness"]
"unbelievable" → ["un", "believ", "able"]
```

**WordPiece (BERT):**
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize("unhappiness")
# ["un", "##happi", "##ness"]
# "##" indica continuación de palabra
```

**SentencePiece:**
```python
import sentencepiece as spm

# Usado por modelos como T5, XLNet
sp = spm.SentencePieceProcessor()
sp.load('model.model')
tokens = sp.encode_as_pieces('unhappiness')
# ["▁un", "happiness"]
# "▁" indica inicio de palabra
```

### Comparativa de Métodos

| Método | Granularidad | Vocabulario | Uso Principal | Ventajas |
|--------|--------------|-------------|---------------|----------|
| **Word** | Palabras completas | Grande | NLP clásico | Interpretable |
| **Character** | Caracteres | Pequeño (~100) | Generación | Sin OOV |
| **Subword** | Fragmentos | Medio (10k-50k) | Transformers | Balance |

**OOV = Out Of Vocabulary (palabras desconocidas)**

---

## 🌍 Tokenización en Diferentes Idiomas {#idiomas}

### Inglés

**Características:**
- ✅ Espacios separan palabras claramente
- ⚠️ Contracciones: "don't", "I'm", "we'll"
- ⚠️ Compuestos con guión: "state-of-the-art"
- ⚠️ Abreviaturas: "Dr.", "U.S.A."

**Ejemplo:**
```python
text = "I'm learning state-of-the-art NLP at Dr. Smith's lab."
tokens = word_tokenize(text)
# ["I", "'m", "learning", "state-of-the-art", "NLP", "at", 
#  "Dr.", "Smith", "'s", "lab", "."]
```

### Español

**Características:**
- ✅ Similar al inglés (espacios como separadores)
- ⚠️ Contracciones: "del" (de+el), "al" (a+el)
- ⚠️ Acentos: "están", "número", "día"
- ⚠️ Interrogación/Exclamación: "¿Cómo estás?"

**Ejemplo:**
```python
import spacy

nlp = spacy.load("es_core_news_sm")
doc = nlp("¿Cómo estás? Voy al mercado.")
tokens = [token.text for token in doc]
# ["¿", "Cómo", "estás", "?", "Voy", "al", "mercado", "."]
```

**Nota:** `"al"` puede mantenerse como un token o dividirse en `["a", "el"]` dependiendo del objetivo.

### Chino

**Características:**
- ❌ Sin espacios entre palabras
- ⚠️ Cada carácter puede ser una palabra o parte de una
- ⚠️ Requiere diccionarios o modelos ML

**Ejemplo:**
```python
import jieba  # Biblioteca popular para chino

text = "我爱自然语言处理"
tokens = jieba.cut(text)
# ["我", "爱", "自然语言", "处理"]
# "我" = yo
# "爱" = amo
# "自然语言" = lenguaje natural
# "处理" = procesamiento
```

### Alemán

**Características:**
- ⚠️ Palabras compuestas largas
- ⚠️ "Fusswegpulverisierer" = pisador-de-caminos-pulverizador

**Ejemplo:**
```python
text = "Donaudampfschifffahrtsgesellschaft"
# Palabra compuesta: Danubio-vapor-navegación-compañía

# Tokenización simple
tokens = [text]  # ["Donaudampfschifffahrtsgesellschaft"]

# Tokenización con descomposición
tokens = ["Donau", "dampf", "schiff", "fahrt", "gesellschaft"]
```

### Japonés

**Características:**
- ❌ Sin espacios
- ⚠️ Mezcla de 3 sistemas: Hiragana, Katakana, Kanji

**Ejemplo:**
```python
import fugashi  # Tokenizer japonés

text = "私は日本語を勉強します"
tagger = fugashi.Tagger()
tokens = [word.surface for word in tagger(text)]
# ["私", "は", "日本語", "を", "勉強", "します"]
```

---

## 🛠️ Herramientas y Bibliotecas {#herramientas}

### 1. NLTK (Natural Language Toolkit)

**Características:**
- 📚 Educacional y completo
- 🐢 Más lento
- 🎯 Bueno para aprendizaje

**Word Tokenization:**
```python
from nltk.tokenize import word_tokenize

text = "Hello, world!"
tokens = word_tokenize(text)
```

**Sentence Tokenization:**
```python
from nltk.tokenize import sent_tokenize

text = "Hello! How are you? I'm fine."
sentences = sent_tokenize(text)
# ["Hello!", "How are you?", "I'm fine."]
```

**Otros Tokenizers:**
```python
from nltk.tokenize import (
    WordPunctTokenizer,
    TweetTokenizer,
    MWETokenizer
)

# WordPunctTokenizer: separa toda puntuación
tokenizer = WordPunctTokenizer()
tokenizer.tokenize("Don't worry!")
# ["Don", "'", "t", "worry", "!"]

# TweetTokenizer: para redes sociales
tokenizer = TweetTokenizer()
tokenizer.tokenize("@user Love #NLP! 😊")
# ["@user", "Love", "#NLP", "!", "😊"]

# MWETokenizer: multi-word expressions
tokenizer = MWETokenizer([("New", "York"), ("San", "Francisco")])
tokenizer.tokenize(["I", "live", "in", "New", "York"])
# ["I", "live", "in", "New_York"]
```

### 2. spaCy

**Características:**
- ⚡ Muy rápido (Cython)
- 🏭 Orientado a producción
- 🧠 Incluye modelos pre-entrenados

**Tokenización Básica:**
```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.is_stop)
```

**Ventajas:**
```python
# Maneja casos especiales automáticamente
doc = nlp("We're here at 9 a.m. in the U.S.A.")
# "We", "'re", "here", "at", "9", "a.m.", "in", "the", "U.S.A.", "."
```

**Personalización:**
```python
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex

# Añadir reglas personalizadas
def custom_tokenizer(nlp):
    inf = list(nlp.Defaults.infixes)
    inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")
    infix_re = compile_infix_regex(inf)
    return Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)

nlp.tokenizer = custom_tokenizer(nlp)
```

### 3. Transformers (Hugging Face)

**Para modelos modernos:**
```python
from transformers import AutoTokenizer

# BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("Hello, world!")
# ['hello', ',', 'world', '!']

# GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.tokenize("Hello, world!")
# ['Hello', ',', 'Ġworld', '!']
# 'Ġ' representa espacio

# Encoding completo (tokens → IDs)
encoded = tokenizer("Hello, world!", return_tensors="pt")
# {'input_ids': tensor([[...]])}
```

### 4. Stanza (Stanford NLP)

**Características:**
- 🌍 70+ idiomas
- 🎓 Académico (Stanford)
- 🔬 Alta precisión

```python
import stanza

stanza.download('en')
nlp = stanza.Pipeline('en')
doc = nlp("Barack Obama was born in Hawaii.")

for sentence in doc.sentences:
    for token in sentence.tokens:
        print(token.text)
```

### Comparativa de Performance

| Biblioteca | Velocidad | Precisión | Idiomas | Uso |
|------------|-----------|-----------|---------|-----|
| **NLTK** | 🐢 Lento | ⭐⭐⭐ | ~40 | Educación |
| **spaCy** | ⚡⚡⚡ Rápido | ⭐⭐⭐⭐ | ~60 | Producción |
| **Transformers** | ⚡⚡ Medio | ⭐⭐⭐⭐⭐ | 100+ | Deep Learning |
| **Stanza** | ⚡ Medio | ⭐⭐⭐⭐⭐ | 70+ | Academia |

---

## 🔧 Casos Especiales {#casos-especiales}

### 1. Contracciones

**Inglés:**
```python
from nltk.tokenize import word_tokenize

contractions = [
    "don't",    # do + n't
    "I'm",      # I + 'm (am)
    "we'll",    # we + 'll (will)
    "wouldn't", # would + n't
    "it's"      # it + 's (is/has)
]

for word in contractions:
    print(word, "→", word_tokenize(word))

# don't → ['do', "n't"]
# I'm → ['I', "'m"]
# we'll → ['we', "'ll"]
```

**Español:**
```python
# "del" = de + el
# "al" = a + el
text = "Voy al mercado del pueblo"
# Opción 1: mantener como tokens
# ["Voy", "al", "mercado", "del", "pueblo"]
# Opción 2: expandir
# ["Voy", "a", "el", "mercado", "de", "el", "pueblo"]
```

### 2. Números y Fechas

```python
examples = [
    "3.14",           # número decimal
    "1,000",          # mil con coma
    "01/15/2024",     # fecha
    "$100.50",        # dinero
    "10:30",          # hora
    "+1-555-1234",    # teléfono
]

for ex in examples:
    tokens = word_tokenize(ex)
    print(f"{ex} → {tokens}")

# 3.14 → ['3.14']
# 1,000 → ['1,000']
# 01/15/2024 → ['01/15/2024']
# $100.50 → ['$', '100.50']
```

### 3. URLs y Emails

```python
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()

text = "Visit https://example.com or email user@example.com"
tokens = tokenizer.tokenize(text)
# ['Visit', 'https://example.com', 'or', 'email', 'user@example.com']
```

### 4. Hashtags y Mentions

```python
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()

text = "@user1 Check out #NLP and #DeepLearning! 🚀"
tokens = tokenizer.tokenize(text)
# ['@user1', 'Check', 'out', '#NLP', 'and', '#DeepLearning', '!', '🚀']
```

### 5. Emojis

```python
import emoji

text = "I love Python! 😍🐍"

# Opción 1: Mantener emojis
tokens = word_tokenize(text)
# ['I', 'love', 'Python', '!', '😍', '🐍']

# Opción 2: Convertir a texto
text_with_emoji = emoji.demojize(text)
# "I love Python! :smiling_face_with_heart-eyes::snake:"
```

### 6. Abreviaturas

```python
text = "Dr. Smith works at NASA, U.S.A."

# NLTK maneja bien
tokens = word_tokenize(text)
# ['Dr.', 'Smith', 'works', 'at', 'NASA', ',', 'U.S.A', '.']
```

---

## 🚀 Tokenización Moderna {#moderna}

### Subword Tokenization en Transformers

**¿Por qué Subword?**

```python
# Problema: Vocabulario infinito
palabras_posibles = infinitas  # "run", "running", "runner", "runs", ...

# Solución: Subword units
subwords = {"run", "ning", "er", "s"}
"running" → ["run", "ning"]
"runner" → ["run", "er"]
"runs" → ["run", "s"]
```

**Ventajas:**
1. ✅ Vocabulario finito pero flexible
2. ✅ Maneja palabras desconocidas (OOV)
3. ✅ Captura morfología

### BPE (Byte-Pair Encoding)

Usado por: GPT-2, GPT-3, RoBERTa

**Algoritmo:**
```
1. Empezar con caracteres individuales
2. Encontrar el par más frecuente
3. Fusionar ese par en un nuevo símbolo
4. Repetir hasta llegar al tamaño de vocabulario deseado
```

**Ejemplo:**
```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "unhappiness"
tokens = tokenizer.tokenize(text)
# ['un', 'happiness']

text = "unbelievable"
tokens = tokenizer.tokenize(text)
# ['un', 'bel', 'iev', 'able']
```

### WordPiece

Usado por: BERT, DistilBERT

**Diferencia con BPE:**
- BPE: fusiona el par más frecuente
- WordPiece: fusiona el par que maximiza likelihood del corpus

**Ejemplo:**
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "unhappiness"
tokens = tokenizer.tokenize(text)
# ['un', '##hap', '##pi', '##ness']
# "##" indica continuación de palabra

text = "playing"
tokens = tokenizer.tokenize(text)
# ['playing']  # En vocabulario como palabra completa
```

### SentencePiece

Usado por: T5, ALBERT, XLNet

**Características:**
- ✅ No requiere pre-tokenización
- ✅ Funciona directamente en texto raw
- ✅ Ideal para idiomas sin espacios

**Ejemplo:**
```python
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")

text = "Hello world"
tokens = tokenizer.tokenize(text)
# ['▁Hello', '▁world']
# "▁" representa espacio
```

### Tokens Especiales

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokens especiales
print(tokenizer.cls_token)      # [CLS] - clasificación
print(tokenizer.sep_token)      # [SEP] - separador
print(tokenizer.pad_token)      # [PAD] - padding
print(tokenizer.mask_token)     # [MASK] - masked LM
print(tokenizer.unk_token)      # [UNK] - unknown

# Encoding con tokens especiales
encoded = tokenizer("Hello", "World")
print(tokenizer.convert_ids_to_tokens(encoded['input_ids']))
# ['[CLS]', 'hello', '[SEP]', 'world', '[SEP]']
```

---

## 💼 Casos de Uso {#casos-uso}

### 1. Análisis de Sentimientos

```python
# Tokenización → Análisis
text = "I absolutely love this product! It's amazing!"
tokens = word_tokenize(text)

# Contar palabras positivas
positive_words = {"love", "amazing", "excellent"}
sentiment_score = sum(1 for token in tokens if token.lower() in positive_words)
```

### 2. Búsqueda y Recuperación de Información

```python
# Tokenizar documentos y queries
documents = [
    "Python is a programming language",
    "Natural Language Processing with Python",
]

query = "Python programming"

# Tokenizar todo
doc_tokens = [word_tokenize(doc.lower()) for doc in documents]
query_tokens = word_tokenize(query.lower())

# Calcular similitud (simplificado)
```

### 3. Machine Translation

```python
# Tokenización bilingüe
en_text = "Hello world"
es_text = "Hola mundo"

en_tokens = word_tokenize(en_text)  # ['Hello', 'world']
es_tokens = word_tokenize(es_text)  # ['Hola', 'mundo']

# Alineamiento: Hello ↔ Hola, world ↔ mundo
```

### 4. Text Normalization

```python
import string

text = "Hello, WORLD! This is AMAZING!!!"

# Tokenizar
tokens = word_tokenize(text)

# Normalizar
tokens = [
    token.lower() 
    for token in tokens 
    if token not in string.punctuation
]
# ['hello', 'world', 'this', 'is', 'amazing']
```

### 5. Feature Extraction para ML

```python
from sklearn.feature_extraction.text import CountVectorizer

# El vectorizer usa tokenización internamente
vectorizer = CountVectorizer(tokenizer=word_tokenize)

corpus = [
    "I love Python",
    "Python is great",
    "I love programming"
]

X = vectorizer.fit_transform(corpus)
# Matriz de features basada en tokens
```

---

## 📊 Best Practices

### 1. Elegir el Tokenizer Apropiado

```python
# Para NLP clásico (análisis, clasificación)
from nltk.tokenize import word_tokenize  # ✅

# Para producción (velocidad importante)
import spacy  # ✅

# Para modelos Transformer
from transformers import AutoTokenizer  # ✅

# Para redes sociales
from nltk.tokenize import TweetTokenizer  # ✅
```

### 2. Consistencia

```python
# ✅ Usar el mismo tokenizer en train y test
tokenizer = word_tokenize

train_tokens = [tokenizer(text) for text in train_data]
test_tokens = [tokenizer(text) for text in test_data]

# ❌ NO mezclar tokenizers
train_tokens = [word_tokenize(text) for text in train_data]
test_tokens = [spacy_tokenize(text) for text in test_data]  # ❌
```

### 3. Normalización

```python
def preprocess(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Tokenizar
    tokens = word_tokenize(text)
    
    # 3. Remover puntuación (opcional)
    tokens = [t for t in tokens if t.isalnum()]
    
    # 4. Remover stopwords (opcional)
    from nltk.corpus import stopwords
    stops = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stops]
    
    return tokens
```

### 4. Manejo de Casos Especiales

```python
# URLs
text = "Visit https://example.com for more info"
# Opción 1: Tokenizar normalmente
# Opción 2: Reemplazar URLs con token especial
text = text.replace(r'https?://\S+', '[URL]')

# Números
text = "I have 123 apples"
# Opción 1: Mantener "123"
# Opción 2: Reemplazar con "[NUM]"
```

---

## 🎓 Resumen

**Conceptos Clave:**
- Tokenización es el primer paso en NLP
- Diferentes niveles: carácter, palabra, subpalabra, oración
- Herramientas: NLTK (educación), spaCy (producción), Transformers (DL)
- Subword tokenization (BPE, WordPiece) es estándar en modelos modernos

**Decisiones Importantes:**
1. ¿Qué nivel de granularidad? (palabra, subpalabra, carácter)
2. ¿Qué hacer con puntuación?
3. ¿Cómo manejar casos especiales? (URLs, emojis, etc.)
4. ¿Normalizar o no? (lowercase, eliminar acentos)

**Próximos Pasos:**
- **Koan 2**: Stemming y Lemmatization (normalización avanzada)
- **Koan 3**: POS Tagging (análisis gramatical)
- **Koan 7**: Word Embeddings (representaciones vectoriales)

¡La tokenización es la base de todo en NLP! 🚀
