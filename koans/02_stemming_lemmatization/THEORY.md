# Teoría: Stemming & Lemmatization

## 📚 Tabla de Contenidos
1. [Introducción a la Normalización de Texto](#introducción)
2. [Stemming](#stemming)
3. [Lemmatization](#lemmatization)
4. [Comparación Stemming vs Lemmatization](#comparación)
5. [Algoritmos y Técnicas](#algoritmos)
6. [Herramientas](#herramientas)
7. [Casos de Uso](#casos-uso)

---

## 🎯 Introducción a la Normalización de Texto {#introducción}

### ¿Por qué Normalizar?

**Problema:**
```python
palabras = ["run", "runs", "running", "ran", "runner"]
# ¿Son todas diferentes? Para una computadora, SÍ.
# Para un humano, todas se relacionan con "correr"
```

**Solución: Reducir a una forma canónica**
```python
# Después de normalización
todas → "run"
```

### Variaciones Morfológicas

**Inflexión** (cambios gramaticales):
```
Verbos: walk → walks, walked, walking
Sustantivos: cat → cats
Adjetivos: good → better, best
```

**Derivación** (nuevas palabras):
```
happy → happiness, unhappy, happily
nation → national, nationality, nationalize
```

### Beneficios

**1. Reducción de Vocabulario:**
```python
# Antes
vocab = {"run", "runs", "running", "ran", "runner"}  # 5 palabras

# Después
vocab = {"run"}  # 1 palabra
```

**2. Mejora en Búsqueda:**
```python
query = "running shoes"
documento = "Best shoes for runners"

# Sin normalización: NO match ❌
# Con normalización: "run" match "run" ✅
```

**3. Mejora en ML:**
```python
# Menos features = modelo más simple y robusto
# "running" y "run" ahora son el mismo feature
```

---

## ✂️ Stemming {#stemming}

### Concepto

**Stemming** es el proceso de reducir palabras a su raíz (stem) mediante reglas heurísticas, generalmente cortando sufijos.

```
Palabra → Stem (raíz aproximada)

running → run
happiness → happi
studies → studi
```

**Características:**
- ⚡ Rápido (basado en reglas)
- ⚠️ No siempre produce palabras reales
- 🎯 Objetivo: velocidad sobre precisión

### Algoritmos de Stemming

#### 1. Porter Stemmer (1980)

El más popular y usado.

**Funcionamiento:**
```
Aplica 5 fases de reglas:
Fase 1: Plurales y -ed, -ing
Fase 2: -ational → -ate, -ization → -ize
Fase 3: -icate → -ic, -ative → [nada]
Fase 4: -al, -ance, -ence, -er, -ic, -able, -ible, -ant, -ment
Fase 5: -e, -ll → -l
```

**Ejemplos:**
```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

palabras = [
    "running",     # → run
    "runner",      # → runner (¡no cambia!)
    "easily",      # → easili
    "happiness",   # → happi
    "connection",  # → connect
    "conditional", # → condit
]

for palabra in palabras:
    print(f"{palabra:15} → {stemmer.stem(palabra)}")
```

**Resultados:**
```
running         → run
runner          → runner  # ⚠️ no reduce a "run"
easily          → easili  # ⚠️ no es palabra real
happiness       → happi   # ⚠️ no es palabra real
connection      → connect ✅
conditional     → condit  # ⚠️ no es palabra real
```

#### 2. Lancaster Stemmer (Paice-Husk, 1990)

Más agresivo que Porter.

**Ejemplos:**
```python
from nltk.stem import LancasterStemmer

stemmer = LancasterStemmer()

palabras = [
    "running",     # → run
    "runner",      # → run
    "easily",      # → easy
    "happiness",   # → happy
    "connection",  # → connect
    "maximum",     # → maxim
]

for palabra in palabras:
    print(f"{palabra:15} → {stemmer.stem(palabra)}")
```

**Resultados:**
```
running         → run
runner          → run     # ✅ más agresivo
easily          → easy    # ✅ mejor que Porter
happiness       → happy   # ✅ reconoce la raíz
connection      → connect
maximum         → maxim
```

**Características:**
- ✅ Más agresivo
- ✅ Reduce más variaciones
- ⚠️ Mayor riesgo de sobre-stemming

#### 3. Snowball Stemmer (Porter2, 2001)

Mejora de Porter, con soporte multilingüe.

**Ejemplos:**
```python
from nltk.stem import SnowballStemmer

# Inglés
stemmer = SnowballStemmer("english")

palabras = [
    "running",
    "easily", 
    "happiness",
    "generously"
]

for palabra in palabras:
    print(f"{palabra:15} → {stemmer.stem(palabra)}")
```

**Español:**
```python
stemmer_es = SnowballStemmer("spanish")

palabras_es = [
    "corriendo",   # → corr
    "corredor",    # → corr
    "felizmente",  # → feliz
    "cantando",    # → cant
]

for palabra in palabras_es:
    print(f"{palabra:15} → {stemmer_es.stem(palabra)}")
```

### Problemas del Stemming

#### 1. Over-stemming

Reduce demasiado, conflando palabras no relacionadas.

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# Ejemplo 1: "universal" y "university"
print(stemmer.stem("universal"))   # → univers
print(stemmer.stem("university"))  # → univers
# ⚠️ Palabras diferentes reducidas a lo mismo

# Ejemplo 2: "organization" y "organ"
print(stemmer.stem("organization")) # → organ
print(stemmer.stem("organ"))        # → organ
# ⚠️ Significados muy diferentes
```

#### 2. Under-stemming

No reduce suficiente, dejando variaciones separadas.

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

print(stemmer.stem("data"))        # → data
print(stemmer.stem("datum"))       # → datum
# ⚠️ Misma palabra (data es plural de datum) pero stems diferentes

print(stemmer.stem("aluminum"))    # → aluminum
print(stemmer.stem("aluminium"))   # → aluminium
# ⚠️ Misma palabra, ortografías diferentes
```

#### 3. No Produce Palabras Reales

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

palabras = ["happiness", "easily", "conditional"]

for palabra in palabras:
    stem = stemmer.stem(palabra)
    print(f"{palabra:15} → {stem:10} {'❌ No es palabra real' if stem not in ['happy', 'easy', 'condition'] else ''}")

# happiness       → happi      ❌ No es palabra real
# easily          → easili     ❌ No es palabra real  
# conditional     → condit     ❌ No es palabra real
```

---

## 📖 Lemmatization {#lemmatization}

### Concepto

**Lemmatization** reduce palabras a su forma base (lema) usando análisis morfológico y diccionarios.

```
Palabra → Lemma (forma base real en diccionario)

running → run
better → good
am/is/are/was/were → be
mice → mouse
```

**Características:**
- 🐢 Más lento (usa diccionarios y reglas)
- ✅ Siempre produce palabras reales
- 🎯 Objetivo: precisión sobre velocidad

### WordNet Lemmatizer (NLTK)

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

palabras = [
    "running",
    "ran",
    "better",
    "mice",
    "geese",
    "cacti"
]

for palabra in palabras:
    lemma = lemmatizer.lemmatize(palabra)
    print(f"{palabra:15} → {lemma}")
```

**Resultados:**
```
running         → running  # ⚠️ Necesita POS tag
ran             → ran      # ⚠️ Necesita POS tag
better          → better   # ⚠️ Necesita POS tag
mice            → mouse    ✅
geese           → goose    ✅
cacti           → cactus   ✅
```

### Part-of-Speech (POS) Tags

**Problema:**
```python
lemmatizer.lemmatize("running")  # → running (sin cambio)
```

**Solución:** Especificar la categoría gramatical

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Sin POS tag (asume sustantivo por defecto)
print(lemmatizer.lemmatize("running"))  # → running

# Con POS tag: verbo
print(lemmatizer.lemmatize("running", pos='v'))  # → run

# Con POS tag: adjetivo
print(lemmatizer.lemmatize("better", pos='a'))  # → good

# Con POS tag: verbo
print(lemmatizer.lemmatize("was", pos='v'))  # → be
```

**POS Tags en WordNet:**
```python
# 'n' = noun (sustantivo)
# 'v' = verb (verbo)
# 'a' = adjective (adjetivo)
# 'r' = adverb (adverbio)
```

### Lemmatization con POS Tagging Automático

```python
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):
    """Convierte Penn Treebank tags a WordNet POS tags"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default

def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    
    # Tokenizar y POS tag
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    
    # Lemmatizar con POS correcto
    lemmas = []
    for word, tag in pos_tags:
        wn_pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=wn_pos)
        lemmas.append(lemma)
    
    return lemmas

# Ejemplo
sentence = "The striped bats are hanging on their feet for best"
print(lemmatize_sentence(sentence))
# ['The', 'strip', 'bat', 'be', 'hang', 'on', 'their', 'foot', 'for', 'good']
```

### spaCy Lemmatization

spaCy hace lemmatization automáticamente con POS tagging integrado.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("The striped bats are hanging on their feet for best")

for token in doc:
    print(f"{token.text:15} → {token.lemma_:15} ({token.pos_})")
```

**Resultado:**
```
The             → the             (DET)
striped         → strip           (VERB)
bats            → bat             (NOUN)
are             → be              (AUX)
hanging         → hang            (VERB)
on              → on              (ADP)
their           → their           (PRON)
feet            → foot            (NOUN)
for             → for             (ADP)
best            → good            (ADJ)
```

---

## ⚖️ Comparación Stemming vs Lemmatization {#comparación}

### Comparativa Directa

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

palabras = [
    ("running", "v"),
    ("better", "a"),
    ("studies", "n"),
    ("feet", "n"),
    ("geese", "n"),
    ("easily", "r"),
]

print(f"{'Word':<15} {'Stem':<15} {'NLTK Lemma':<15} {'spaCy Lemma':<15}")
print("-" * 60)

for word, pos in palabras:
    stem = stemmer.stem(word)
    lemma_nltk = lemmatizer.lemmatize(word, pos=pos)
    lemma_spacy = nlp(word)[0].lemma_
    
    print(f"{word:<15} {stem:<15} {lemma_nltk:<15} {lemma_spacy:<15}")
```

**Resultado:**
```
Word            Stem            NLTK Lemma      spaCy Lemma    
------------------------------------------------------------
running         run             run             run            
better          better          good            well           
studies         studi           study           study          
feet            feet            foot            foot           
geese           gees            goose           goose          
easily          easili          easily          easily         
```

### Tabla Comparativa

| Aspecto | Stemming | Lemmatization |
|---------|----------|---------------|
| **Velocidad** | ⚡⚡⚡ Muy rápido | 🐢 Más lento |
| **Precisión** | ⚠️ Aproximada | ✅ Alta |
| **Resultado** | Raíz (puede no ser palabra real) | Lema (palabra válida) |
| **Método** | Reglas heurísticas | Análisis morfológico + diccionario |
| **Requiere POS** | ❌ No | ✅ Sí (para mejor resultado) |
| **Ejemplos** | running → run<br>easily → easili | running → run<br>easily → easy |
| **Uso Típico** | Búsqueda de texto<br>IR simple | NLP avanzado<br>Análisis semántico |

### Cuándo Usar Cada Uno

**Usar Stemming cuando:**
- ⚡ Velocidad es crítica
- 📊 Trabajas con grandes volúmenes
- 🔍 Búsqueda y recuperación de información
- 📈 Features para ML donde precisión no es crítica

**Usar Lemmatization cuando:**
- 🎯 Precisión es importante
- 📖 Análisis semántico
- 🗣️ Sistemas de diálogo
- 🔬 Investigación lingüística
- 🎓 Aplicaciones educativas

---

## 🔧 Algoritmos y Técnicas {#algoritmos}

### Porter Stemmer en Detalle

**5 Fases de Reglas:**

**Fase 1: Sufijos comunes**
```
SSES → SS          caresses → caress
IES  → I           ponies → poni
SS   → SS          caress → caress
S    →             cats → cat
```

**Fase 2: Sufijos derivacionales**
```
(m>0) ATIONAL → ATE    relational → relate
(m>0) TIONAL  → TION   conditional → condition
(m>0) ENCI    → ENCE   valenci → valence
```

**Fase 3: Más derivacionales**
```
(m>0) ICATE → IC       triplicate → triplic
(m>0) ATIVE →          formative → form
```

**Fase 4: Sufijos más comunes**
```
(m>1) AL    →          revival → reviv
(m>1) ANCE  →          allowance → allow
(m>1) ENCE  →          inference → infer
```

**Fase 5: Limpieza final**
```
(m>1) E     →          probate → probat
(m=1 and not *o) E →   rate → rate
```

**Métrica m (measure):**
```
m = número de secuencias [consonante(s)][vocal(es)]

tree:     (VC)  → m=1
trees:    (VC)s → m=1
trouble:  (VC)(VC) → m=2
```

### Lancaster Stemmer en Detalle

**Características:**
- Usa tabla de ~120 reglas
- Más agresivo que Porter
- Aplica reglas en orden de especificidad

**Ejemplos de reglas:**
```
SSES    → SS    (como Porter)
IES     → Y     (más agresivo que Porter)
ATIONAL → ATE   
TIONAL  → TION  
```

### Snowball (Porter2)

**Mejoras sobre Porter:**
- ✅ Mejor manejo de excepciones
- ✅ Soporte multilingüe (15+ idiomas)
- ✅ Más eficiente
- ✅ Mejor documentación

**Idiomas soportados:**
```python
from nltk.stem import SnowballStemmer

# Ver idiomas disponibles
print(SnowballStemmer.languages)
# ('arabic', 'danish', 'dutch', 'english', 'finnish', 'french', 
#  'german', 'hungarian', 'italian', 'norwegian', 'porter', 
#  'portuguese', 'romanian', 'russian', 'spanish', 'swedish')
```

---

## 🛠️ Herramientas {#herramientas}

### NLTK

**Stemmers:**
```python
from nltk.stem import (
    PorterStemmer,
    LancasterStemmer,
    SnowballStemmer,
    RegexpStemmer
)

# Porter
porter = PorterStemmer()
porter.stem("running")  # → run

# Lancaster
lancaster = LancasterStemmer()
lancaster.stem("running")  # → run

# Snowball (multilingüe)
snowball = SnowballStemmer("english")
snowball.stem("running")  # → run

# Custom Regexp Stemmer
regexp = RegexpStemmer('ing$|s$|e$', min=4)
regexp.stem("running")  # → runn
```

**Lemmatizers:**
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Con POS tag
lemmatizer.lemmatize("running", pos='v')  # → run
lemmatizer.lemmatize("better", pos='a')   # → good
```

### spaCy

**Lemmatization Automática:**
```python
import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("running dogs are better")

for token in doc:
    print(token.text, "→", token.lemma_)

# running → run
# dogs → dog
# are → be
# better → well
```

**Ventajas de spaCy:**
- ✅ POS tagging automático
- ✅ Muy rápido
- ✅ Preciso
- ✅ Multilingüe

### Stanza (Stanford NLP)

```python
import stanza

nlp = stanza.Pipeline('en', processors='tokenize,lemma')

doc = nlp("The quick brown foxes are jumping")

for sentence in doc.sentences:
    for word in sentence.words:
        print(f"{word.text} → {word.lemma}")

# The → the
# quick → quick
# brown → brown
# foxes → fox
# are → be
# jumping → jump
```

### Comparativa de Performance

| Herramienta | Velocidad | Precisión | Facilidad |
|-------------|-----------|-----------|-----------|
| **Porter (NLTK)** | ⚡⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Lancaster (NLTK)** | ⚡⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **NLTK Lemmatizer** | ⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ |
| **spaCy** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Stanza** | ⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 💼 Casos de Uso {#casos-uso}

### 1. Búsqueda de Información

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# Query del usuario
query = "running shoes"
query_stems = [stemmer.stem(word) for word in query.split()]
# ["run", "shoe"]

# Documentos
docs = [
    "Best shoes for runners",
    "Running shoe reviews",
    "Marathon running tips"
]

# Buscar matches
for doc in docs:
    doc_stems = [stemmer.stem(word) for word in doc.lower().split()]
    if any(stem in doc_stems for stem in query_stems):
        print(f"✅ Match: {doc}")

# ✅ Match: Best shoes for runners (shoe, run)
# ✅ Match: Running shoe reviews (run, shoe)
# ✅ Match: Marathon running tips (run)
```

### 2. Análisis de Sentimientos

```python
import spacy

nlp = spacy.load("en_core_web_sm")

# Reviews con variaciones
reviews = [
    "I love this product! It's amazing!",
    "Loved it! Amazingly good!",
    "Loving every moment. Amazing quality."
]

# Palabras positivas (en lema)
positive_lemmas = {"love", "amazing", "good", "excellent"}

for review in reviews:
    doc = nlp(review.lower())
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    
    sentiment_score = sum(1 for lemma in lemmas if lemma in positive_lemmas)
    print(f"{review[:30]:30} → Score: {sentiment_score}")

# I love this product! It's am → Score: 2 (love, amazing)
# Loved it! Amazingly good!    → Score: 3 (love, amazing, good)
# Loving every moment. Amazi  → Score: 2 (love, amazing)
```

### 3. Text Classification

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def stem_tokenizer(text):
    tokens = text.lower().split()
    return [stemmer.stem(token) for token in tokens]

# Vectorizer con stemming
vectorizer = TfidfVectorizer(tokenizer=stem_tokenizer)

corpus = [
    "Python programming tutorial",
    "Programming in Python for beginners",
    "Learn to program with Python"
]

X = vectorizer.fit_transform(corpus)

# Features son stems: ["python", "program", "tutori", "begin", "learn"]
# "programming", "programmer", "programs" → todos son "program"
```

### 4. Chatbots

```python
import spacy

nlp = spacy.load("en_core_web_sm")

# Base de conocimiento con variaciones
faq = {
    "reset password": "To reset your password, go to...",
    "change password": "To change your password, go to...",
    "update password": "To update your password, go to...",
}

def find_answer(user_query):
    # Lemmatizar query del usuario
    doc_query = nlp(user_query.lower())
    query_lemmas = {token.lemma_ for token in doc_query if not token.is_stop}
    
    # Buscar mejor match
    best_match = None
    best_score = 0
    
    for faq_key, answer in faq.items():
        doc_faq = nlp(faq_key)
        faq_lemmas = {token.lemma_ for token in doc_faq}
        
        # Similitud simple: intersección de lemmas
        score = len(query_lemmas & faq_lemmas)
        
        if score > best_score:
            best_score = score
            best_match = answer
    
    return best_match if best_score > 0 else "I don't understand"

# Usuario puede preguntar de diferentes formas
print(find_answer("How do I reset my password?"))
# → "To reset your password, go to..."

print(find_answer("I want to change my password"))
# → "To change your password, go to..."

print(find_answer("updating password"))
# → "To update your password, go to..."
```

### 5. Reducción de Features para ML

```python
from collections import Counter
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

corpus = [
    "machine learning is great",
    "learning machines are smart",
    "I love machine learning"
]

# Sin stemming
words_no_stem = []
for doc in corpus:
    words_no_stem.extend(doc.lower().split())

vocab_no_stem = Counter(words_no_stem)
print(f"Vocabulario sin stemming: {len(vocab_no_stem)} palabras")
print(vocab_no_stem)
# Vocabulario sin stemming: 9 palabras
# {'machine': 2, 'learning': 3, 'is': 1, 'great': 1, 
#  'machines': 1, 'are': 1, 'smart': 1, 'i': 1, 'love': 1}

# Con stemming
words_stem = []
for doc in corpus:
    tokens = doc.lower().split()
    words_stem.extend([stemmer.stem(t) for t in tokens])

vocab_stem = Counter(words_stem)
print(f"\nVocabulario con stemming: {len(vocab_stem)} palabras")
print(vocab_stem)
# Vocabulario con stemming: 7 palabras
# {'machin': 3, 'learn': 3, 'is': 1, 'great': 1, 
#  'are': 1, 'smart': 1, 'i': 1, 'love': 1}
```

---

## 📊 Best Practices

### 1. Elegir la Herramienta Correcta

```python
# Para búsqueda simple y velocidad
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()  # ✅

# Para análisis semántico preciso
import spacy
nlp = spacy.load("en_core_web_sm")  # ✅

# Para multilingüe
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("spanish")  # ✅
```

### 2. Consistencia

```python
# ✅ Usar la misma técnica en todo el pipeline
def preprocess(text, method='lemma'):
    if method == 'stem':
        stemmer = PorterStemmer()
        tokens = text.split()
        return [stemmer.stem(t) for t in tokens]
    elif method == 'lemma':
        doc = nlp(text)
        return [token.lemma_ for token in doc]

# Aplicar consistentemente
train_processed = [preprocess(text, 'lemma') for text in train]
test_processed = [preprocess(text, 'lemma') for text in test]
```

### 3. Combinar con Otras Técnicas

```python
import spacy
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def advanced_preprocess(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Tokenizar y lemmatizar
    doc = nlp(text)
    
    # 3. Filtrar
    tokens = [
        token.lemma_ 
        for token in doc 
        if token.is_alpha  # Solo palabras
        and not token.is_stop  # Sin stopwords
        and len(token) > 2  # Longitud mínima
    ]
    
    return tokens

text = "The running dogs are jumping over the fence"
print(advanced_preprocess(text))
# ['run', 'dog', 'jump', 'fence']
```

### 4. Manejo de Excepciones

```python
import spacy

nlp = spacy.load("en_core_web_sm")

# Palabras especiales que no deben lemmatizarse
special_words = {"COVID-19", "iPhone", "NASA"}

def lemmatize_with_exceptions(text, exceptions=special_words):
    doc = nlp(text)
    
    lemmas = []
    for token in doc:
        if token.text in exceptions:
            lemmas.append(token.text)  # Mantener original
        else:
            lemmas.append(token.lemma_)
    
    return lemmas

text = "NASA announced iPhone support for COVID-19 tracking"
print(lemmatize_with_exceptions(text))
# ['NASA', 'announce', 'iPhone', 'support', 'for', 'COVID-19', 'track']
```

---

## 🎓 Resumen

**Conceptos Clave:**
- **Stemming**: Reducción a raíz mediante reglas (rápido, aproximado)
- **Lemmatization**: Reducción a lema mediante análisis (preciso, lento)
- Stemming para velocidad, lemmatization para precisión
- spaCy es la mejor opción para producción

**Algoritmos Principales:**
- **Porter**: Balance, más usado
- **Lancaster**: Más agresivo
- **Snowball**: Porter mejorado, multilingüe
- **WordNet**: Lemmatization con diccionario

**Decisiones Importantes:**
1. ¿Velocidad o precisión?
2. ¿Palabras reales importan?
3. ¿Multilingüe?
4. ¿Integrar con POS tagging?

**Próximos Pasos:**
- **Koan 3**: POS Tagging (necesario para lemmatization óptima)
- **Koan 5**: Text Classification (usando normalización)
- **Koan 6**: Sentiment Analysis (beneficiándose de normalización)

¡La normalización mejora todos los pipelines de NLP! 🚀
