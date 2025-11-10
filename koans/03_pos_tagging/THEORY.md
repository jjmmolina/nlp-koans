# Teoría: Part-of-Speech (POS) Tagging

## 📚 Tabla de Contenidos
1. [Introducción al POS Tagging](#introducción)
2. [Tagsets y Etiquetas](#tagsets)
3. [Algoritmos y Modelos](#algoritmos)
4. [Herramientas](#herramientas)
5. [Aplicaciones](#aplicaciones)

---

## 🎯 Introducción al POS Tagging {#introducción}

### ¿Qué es POS Tagging?

**POS (Part-of-Speech) Tagging** es el proceso de etiquetar cada palabra en un texto con su categoría gramatical.

```python
Texto: "The quick brown fox jumps"
POS:   DET  ADJ   ADJ   NOUN VERB

# Cada palabra recibe una etiqueta gramatical
```

### ¿Por qué es Importante?

**1. Desambiguación:**
```python
# "book" puede ser verbo o sustantivo
"I read a book"  → book/NOUN
"I book a flight" → book/VERB

# "fly" puede ser verbo o sustantivo  
"Birds fly"      → fly/VERB
"Catch the fly"  → fly/NOUN
```

**2. Base para Análisis Más Profundo:**
```
POS Tagging → Chunking → Parsing → NER → Relaciones Semánticas
```

**3. Mejora Lemmatization:**
```python
from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()

# Sin POS: asume sustantivo
lem.lemmatize("better")  # → better

# Con POS: adjetivo
lem.lemmatize("better", pos='a')  # → good ✅
```

---

## 🏷️ Tagsets y Etiquetas {#tagsets}

### Penn Treebank Tagset

El más usado en inglés (45 etiquetas).

**Sustantivos:**
```
NN    → Noun, singular (dog, car)
NNS   → Noun, plural (dogs, cars)
NNP   → Proper noun, singular (John, London)
NNPS  → Proper noun, plural (Americans, Beatles)
```

**Verbos:**
```
VB    → Verb, base form (run, eat)
VBD   → Verb, past tense (ran, ate)
VBG   → Verb, gerund/present participle (running, eating)
VBN   → Verb, past participle (run, eaten)
VBP   → Verb, non-3rd person present (I/you/we run)
VBZ   → Verb, 3rd person present (he/she runs)
```

**Adjetivos y Adverbios:**
```
JJ    → Adjective (big, old, green)
JJR   → Adjective, comparative (bigger, older)
JJS   → Adjective, superlative (biggest, oldest)
RB    → Adverb (quickly, silently)
RBR   → Adverb, comparative (faster)
RBS   → Adverb, superlative (fastest)
```

**Otros:**
```
DT    → Determiner (the, a, this)
IN    → Preposition/conjunction (in, of, on)
CC    → Coordinating conjunction (and, or, but)
PRP   → Personal pronoun (I, you, he)
PRP$  → Possessive pronoun (my, your, his)
TO    → "to"
```

### Universal Dependencies Tagset

Tagset universal más simple (17 etiquetas).

```
NOUN  → Sustantivo
VERB  → Verbo
ADJ   → Adjetivo
ADV   → Adverbio
PRON  → Pronombre
DET   → Determinante
ADP   → Adposición (preposición)
NUM   → Número
CONJ  → Conjunción
PRT   → Partícula
.     → Puntuación
X     → Otro
```

**Ejemplo Comparativo:**
```python
import spacy
import nltk

# spaCy usa Universal Dependencies
nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps")

for token in doc:
    print(f"{token.text:10} {token.pos_:6} {token.tag_:6}")

# The        DET    DT    
# quick      ADJ    JJ    
# brown      ADJ    JJ    
# fox        NOUN   NN    
# jumps      VERB   VBZ   
```

---

## 🤖 Algoritmos y Modelos {#algoritmos}

### 1. Rule-Based Tagging

Reglas manuales basadas en patrones.

```python
# Reglas simples
if word.endswith('ing'):
    tag = 'VBG'  # Gerundio
elif word.endswith('ed'):
    tag = 'VBD'  # Pasado
elif word in ['the', 'a', 'an']:
    tag = 'DT'   # Determinante
```

**Ventajas:** Simple, interpretable
**Desventajas:** No escala, muchas excepciones

### 2. Hidden Markov Models (HMM)

Modelo probabilístico que considera:
- **Emisión**: P(palabra|etiqueta)
- **Transición**: P(etiqueta_siguiente|etiqueta_actual)

```python
from nltk.tag import hmm

# Entrenar HMM tagger
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train(training_data)

# Usar
tagger.tag(['The', 'dog', 'runs'])
# [('The', 'DT'), ('dog', 'NN'), ('runs', 'VBZ')]
```

**Ventajas:** Considera contexto, probabilístico
**Desventajas:** Requiere datos etiquetados

### 3. Maximum Entropy (MaxEnt)

Modelo discriminativo que usa features.

```python
# Features usadas:
# - Palabra actual
# - Sufijo (-ing, -ed, -ly)
# - Prefijo (un-, re-)
# - Palabra anterior
# - Etiqueta anterior
# - Es mayúscula?
# - Es número?
```

**Ventajas:** Flexible, muchas features
**Desventajas:** Más lento que HMM

### 4. Conditional Random Fields (CRF)

Similar a MaxEnt pero considera toda la secuencia.

```python
# Considera:
# - Palabra i-2, i-1, i, i+1, i+2
# - Etiquetas i-2, i-1
# - Características morfológicas
```

**Ventajas:** Estado del arte en ML clásico
**Desventajas:** Complejo, requiere features engineerin g

### 5. Deep Learning (RNN, LSTM, Transformers)

Modelos neuronales que aprenden representaciones.

```python
# Arquitectura típica:
# Embeddings → BiLSTM → CRF → Tags

# spaCy usa esto por defecto
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox")
# Usa modelo neural entrenado
```

**Ventajas:** Máxima precisión, aprende features automáticamente
**Desventajas:** Requiere mucho entrenamiento, recursos

---

## 🛠️ Herramientas {#herramientas}

### NLTK

```python
import nltk
from nltk import pos_tag, word_tokenize

text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)
tags = pos_tag(tokens)

print(tags)
# [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), 
#  ('fox', 'NN'), ('jumps', 'VBZ'), ...]
```

**Tagger por Defecto:**
- MaxEnt tagger entrenado en Penn Treebank
- ~3-5% error rate

**Otros Taggers:**
```python
# HMM Tagger
from nltk.tag import hmm

# Brill Tagger (basado en transformaciones)
from nltk.tag import brill

# Tagger personalizado
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger
```

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The quick brown fox jumps")

for token in doc:
    print(f"{token.text:10} POS: {token.pos_:6} Tag: {token.tag_:6}")

# The        POS: DET    Tag: DT    
# quick      POS: ADJ    Tag: JJ    
# brown      POS: ADJ    Tag: JJ    
# fox        POS: NOUN   Tag: NN    
# jumps      POS: VERB   Tag: VBZ   
```

**Características:**
- ⚡ Muy rápido
- ⭐ Alta precisión (~97%)
- 🧠 Modelos pre-entrenados
- 🌍 Multilingüe

### Stanza

```python
import stanza

stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,pos')

doc = nlp("Barack Obama was born in Hawaii")

for sentence in doc.sentences:
    for word in sentence.words:
        print(f"{word.text:10} {word.pos:6} {word.xpos:6}")

# Barack     PROPN  NNP   
# Obama      PROPN  NNP   
# was        AUX    VBD   
# born       VERB   VBN   
```

**Características:**
- 🎓 Academia (Stanford)
- ⭐⭐ Máxima precisión
- 🌍 70+ idiomas

### Comparativa

| Herramienta | Velocidad | Precisión | Uso |
|-------------|-----------|-----------|-----|
| **NLTK** | 🐢 | ⭐⭐⭐ | Educación |
| **spaCy** | ⚡⚡⚡ | ⭐⭐⭐⭐ | Producción |
| **Stanza** | ⚡⚡ | ⭐⭐⭐⭐⭐ | Academia |

---

## 💼 Aplicaciones {#aplicaciones}

### 1. Mejora de Lemmatization

```python
import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("better flies flying")

for token in doc:
    print(f"{token.text:10} {token.pos_:6} → {token.lemma_}")

# better     ADJ    → well
# flies      VERB   → fly
# flying     VERB   → fly
```

### 2. Extracción de Información

```python
# Extraer sustantivos y verbos
doc = nlp("Apple announced a new iPhone at the conference")

nouns = [token.text for token in doc if token.pos_ == "NOUN"]
verbs = [token.text for token in doc if token.pos_ == "VERB"]

print("Nouns:", nouns)     # ['iPhone', 'conference']
print("Verbs:", verbs)     # ['announced']
```

### 3. Named Entity Recognition

POS tagging es un paso previo para NER.

```python
# Nombres propios (PROPN) son candidatos para entidades
doc = nlp("Barack Obama visited Paris")

proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]
print(proper_nouns)  # ['Barack', 'Obama', 'Paris']
```

### 4. Text Simplification

```python
# Identificar y simplificar adjetivos complejos
doc = nlp("The extraordinarily beautiful landscape")

for token in doc:
    if token.pos_ == "ADJ" and len(token.text) > 8:
        print(f"Simplify: {token.text}")
# Simplify: extraordinarily
```

### 5. Question Answering

```python
# Identificar tipo de pregunta basado en POS
questions = [
    "Who is the president?",      # PRON (who) → PERSON
    "Where is Paris?",            # ADV (where) → LOCATION
    "When was he born?",          # ADV (when) → DATE
]

for q in questions:
    doc = nlp(q)
    if doc[0].pos_ == "PRON":
        print(f"{q} → Expecting PERSON")
    elif doc[0].pos_ == "ADV":
        if doc[0].text.lower() == "where":
            print(f"{q} → Expecting LOCATION")
        elif doc[0].text.lower() == "when":
            print(f"{q} → Expecting DATE")
```

---

## 🎓 Resumen

**Conceptos Clave:**
- POS Tagging asigna categorías gramaticales a palabras
- Tagsets: Penn Treebank (45 tags), Universal Dependencies (17 tags)
- Algoritmos: HMM, MaxEnt, CRF, Deep Learning
- Esencial para lemmatization, NER, parsing

**Próximos Pasos:**
- **Koan 4**: NER (usa POS tagging)
- **Koan 7**: Word Embeddings (contexto gramatical)

¡POS tagging es fundamental para NLP avanzado! 🚀
