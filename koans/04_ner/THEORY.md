# Teoría: Named Entity Recognition (NER)

## 📚 Tabla de Contenidos
1. [Introducción a NER](#introducción)
2. [Tipos de Entidades](#tipos)
3. [Enfoques y Algoritmos](#enfoques)
4. [Herramientas](#herramientas)
5. [Evaluación](#evaluación)
6. [Aplicaciones](#aplicaciones)

---

## 🎯 Introducción a NER {#introducción}

### ¿Qué es NER?

**Named Entity Recognition** identifica y clasifica entidades nombradas en texto (personas, lugares, organizaciones, fechas, etc.).

```python
Texto: "Barack Obama visited Paris in 2024"

Entidades:
- "Barack Obama" → PERSON
- "Paris"        → LOCATION (GPE)
- "2024"         → DATE
```

### ¿Por qué es Importante?

**Extracción de Información Estructurada:**
```python
# De texto no estructurado...
"Apple CEO Tim Cook announced the new iPhone 15 in September 2023"

# ...a información estructurada
{
    "ORG": ["Apple"],
    "PERSON": ["Tim Cook"],
    "PRODUCT": ["iPhone 15"],
    "DATE": ["September 2023"]
}
```

---

## 🏷️ Tipos de Entidades {#tipos}

### Entidades Comunes

**OntoNotes 5 (spaCy):**
```
PERSON      → Personas (John, Mary)
ORG         → Organizaciones (Apple, UN)
GPE         → Geo-Political Entity (París, España)
LOC         → Locaciones no-GPE (Mount Everest)
PRODUCT     → Productos (iPhone, Windows)
EVENT       → Eventos (Olympics, World War II)
DATE        → Fechas absolutas/relativas
TIME        → Horas
MONEY       → Cantidades monetarias
PERCENT     → Porcentajes
QUANTITY    → Medidas (10 kg)
ORDINAL     → Números ordinales (first, second)
CARDINAL    → Números cardinales (one, two, 1, 2)
```

**CoNLL 2003 (más simple):**
```
PER  → Personas
ORG  → Organizaciones
LOC  → Locaciones
MISC → Misceláneo
```

### Ejemplo Completo

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = """
Apple Inc. was founded by Steve Jobs in Cupertino in 1976.
The company released the iPhone on June 29, 2007, priced at $499.
"""

doc = nlp(text)

for ent in doc.ents:
    print(f"{ent.text:20} → {ent.label_:15} ({ent.start_char}:{ent.end_char})")

# Apple Inc.           → ORG             (1:11)
# Steve Jobs           → PERSON          (29:39)
# Cupertino            → GPE             (43:52)
# 1976                 → DATE            (56:60)
# iPhone               → PRODUCT         (85:91)
# June 29, 2007        → DATE            (95:108)
# $499                 → MONEY           (120:124)
```

---

## 🤖 Enfoques y Algoritmos {#enfoques}

### 1. Rule-Based

Reglas y diccionarios manuales.

```python
# Reglas simples
if word.istitle() and word_before in ["Mr.", "Dr."]:
    entity = "PERSON"
    
if word in cities_dict:
    entity = "LOCATION"
    
if re.match(r'\d{4}-\d{2}-\d{2}', word):
    entity = "DATE"
```

**Ventajas:** Precisión alta en dominios específicos
**Desventajas:** No escala, no generaliza

### 2. Machine Learning Clásico

**CRF (Conditional Random Fields):**
- Features: palabras vecinas, POS tags, capitalización, prefijos/sufijos
- Considera secuencia completa

```python
# Features típicas para CRF
features = {
    'word': current_word,
    'word.lower()': current_word.lower(),
    'word.isupper()': current_word.isupper(),
    'word.istitle()': current_word.istitle(),
    'word.isdigit()': current_word.isdigit(),
    'postag': pos_tag,
    'postag[:2]': pos_tag[:2],
    'prev_word': prev_word,
    'next_word': next_word,
    'suffix_2': current_word[-2:],
    'suffix_3': current_word[-3:],
}
```

### 3. Deep Learning

**BiLSTM-CRF:**
```
Palabras → Embeddings → BiLSTM → CRF → Etiquetas

"Barack"  → [0.1, ...] → LSTM → CRF → B-PERSON
"Obama"   → [0.2, ...] → LSTM → CRF → I-PERSON
"visited" → [0.3, ...] → LSTM → CRF → O
"Paris"   → [0.4, ...] → LSTM → CRF → B-LOC
```

**Transformers (BERT-based):**
```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER")

text = "Barack Obama visited Paris"
entities = ner(text)

# [{'entity': 'B-PER', 'word': 'Barack'}, 
#  {'entity': 'I-PER', 'word': 'Obama'}, 
#  {'entity': 'B-LOC', 'word': 'Paris'}]
```

### BIO Tagging Scheme

```
B-TAG  → Beginning (inicio de entidad)
I-TAG  → Inside (continuación de entidad)
O      → Outside (no es entidad)

Ejemplo:
"Barack Obama visited Paris"

Barack  → B-PER
Obama   → I-PER
visited → O
Paris   → B-LOC
```

---

## 🛠️ Herramientas {#herramientas}

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple CEO Tim Cook announced iPhone 15")

# Entidades detectadas
for ent in doc.ents:
    print(f"{ent.text:15} → {ent.label_}")

# Apple           → ORG
# Tim Cook        → PERSON
# iPhone 15       → PRODUCT

# Visualizar
from spacy import displacy
displacy.render(doc, style="ent")
```

**Modelos disponibles:**
- `en_core_web_sm` → Pequeño, rápido
- `en_core_web_md` → Mediano, balance
- `en_core_web_lg` → Grande, preciso

### Stanford NER

```python
from nltk.tag import StanfordNERTagger

tagger = StanfordNERTagger(
    'stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
    'stanford-ner/stanford-ner.jar'
)

text = "Barack Obama visited Paris".split()
tags = tagger.tag(text)

# [('Barack', 'PERSON'), ('Obama', 'PERSON'), 
#  ('visited', 'O'), ('Paris', 'LOCATION')]
```

### Transformers (Hugging Face)

```python
from transformers import pipeline

# Modelo fine-tuned para NER
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

text = "Elon Musk founded SpaceX"
entities = ner(text)

for entity in entities:
    print(f"{entity['word']:15} → {entity['entity']}")

# Elon            → B-PER
# Musk            → I-PER
# SpaceX          → B-ORG
```

### Flair

```python
from flair.data import Sentence
from flair.models import SequenceTagger

tagger = SequenceTagger.load("ner")

sentence = Sentence("George Washington went to Washington")
tagger.predict(sentence)

for entity in sentence.get_spans('ner'):
    print(f"{entity.text:20} → {entity.tag}")

# George Washington    → PER
# Washington           → LOC
```

### Comparativa

| Herramienta | Velocidad | Precisión | Facilidad |
|-------------|-----------|-----------|-----------|
| **spaCy** | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Stanford NER** | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Transformers** | ⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Flair** | ⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 📊 Evaluación {#evaluación}

### Métricas

**Precision (Precisión):**
```
Precision = TP / (TP + FP)

De las entidades predichas, ¿cuántas son correctas?
```

**Recall (Cobertura):**
```
Recall = TP / (TP + FN)

De las entidades reales, ¿cuántas detectamos?
```

**F1-Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Balance entre precisión y recall
```

### Ejemplo de Evaluación

```python
# Ground Truth
true_entities = [
    ("Barack Obama", "PERSON", 0, 12),
    ("Paris", "LOC", 21, 26)
]

# Predicciones
pred_entities = [
    ("Barack Obama", "PERSON", 0, 12),  # ✅ TP
    ("Paris", "ORG", 21, 26),           # ❌ FP (tipo incorrecto)
]

# Resultados:
# TP = 1 (Barack Obama correcto)
# FP = 1 (Paris con tipo incorrecto)
# FN = 1 (Paris LOCATION no detectado)

# Precision = 1 / (1 + 1) = 0.5
# Recall = 1 / (1 + 1) = 0.5
# F1 = 0.5
```

### Evaluación Estricta vs Relajada

**Estricta:** Entidad y tipo deben coincidir exactamente
**Relajada:** Solo debe detectar la entidad (tipo puede variar)

---

## 💼 Aplicaciones {#aplicaciones}

### 1. Extracción de Información

```python
import spacy

nlp = spacy.load("en_core_web_sm")

article = """
Microsoft CEO Satya Nadella announced quarterly earnings of $52.7 billion 
in Redmond on October 24, 2023. The stock price rose 3.5%.
"""

doc = nlp(article)

# Extraer información estructurada
info = {
    "companies": [],
    "people": [],
    "locations": [],
    "dates": [],
    "money": []
}

for ent in doc.ents:
    if ent.label_ == "ORG":
        info["companies"].append(ent.text)
    elif ent.label_ == "PERSON":
        info["people"].append(ent.text)
    elif ent.label_ == "GPE":
        info["locations"].append(ent.text)
    elif ent.label_ == "DATE":
        info["dates"].append(ent.text)
    elif ent.label_ == "MONEY":
        info["money"].append(ent.text)

print(info)
# {'companies': ['Microsoft'], 'people': ['Satya Nadella'], 
#  'locations': ['Redmond'], 'dates': ['October 24, 2023'], 
#  'money': ['$52.7 billion']}
```

### 2. Question Answering

```python
# Pregunta: "Who is the CEO of Microsoft?"
# Respuesta esperada: PERSON

def answer_question(question, context):
    doc = nlp(context)
    
    # Identificar tipo de entidad esperada
    if question.lower().startswith("who"):
        entity_type = "PERSON"
    elif question.lower().startswith("where"):
        entity_type = "GPE"
    elif question.lower().startswith("when"):
        entity_type = "DATE"
    
    # Extraer entidades del tipo correcto
    answers = [ent.text for ent in doc.ents if ent.label_ == entity_type]
    
    return answers[0] if answers else "No answer found"

context = "Microsoft CEO is Satya Nadella based in Redmond"
print(answer_question("Who is the CEO?", context))
# → "Satya Nadella"
```

### 3. Anonimización de Datos

```python
def anonymize_text(text):
    doc = nlp(text)
    
    anonymized = text
    for ent in reversed(doc.ents):  # reversed para mantener offsets
        if ent.label_ == "PERSON":
            anonymized = (anonymized[:ent.start_char] + 
                         "[PERSON]" + 
                         anonymized[ent.end_char:])
        elif ent.label_ == "GPE":
            anonymized = (anonymized[:ent.start_char] + 
                         "[LOCATION]" + 
                         anonymized[ent.end_char:])
    
    return anonymized

text = "John Smith lives in New York and works at Google"
print(anonymize_text(text))
# "[PERSON] lives in [LOCATION] and works at Google"
```

### 4. Resumen de Documentos

```python
# Extraer entidades principales para resumen
def extract_key_entities(text, top_n=5):
    doc = nlp(text)
    
    # Contar frecuencia de entidades
    from collections import Counter
    entities = Counter([ent.text for ent in doc.ents 
                       if ent.label_ in ["PERSON", "ORG", "GPE"]])
    
    return entities.most_common(top_n)

article = """
Apple Inc. CEO Tim Cook announced the new iPhone 15. 
Apple stock rose after Tim Cook's announcement. 
The iPhone 15 will be available in September.
"""

print(extract_key_entities(article))
# [('Apple', 2), ('Tim Cook', 2), ('iPhone 15', 2)]
```

---

## 🎓 Resumen

**Conceptos Clave:**
- NER identifica y clasifica entidades en texto
- Tipos comunes: PERSON, ORG, LOC, DATE, MONEY
- Enfoques: Rule-based, ML (CRF), DL (LSTM, Transformers)
- BIO tagging: B-TAG, I-TAG, O

**Mejores Herramientas:**
- spaCy → Producción (balance velocidad/precisión)
- Transformers → Máxima precisión
- Stanford NER → Academia

**Próximos Pasos:**
- **Koan 13**: RAG (usar NER para extracción estructurada)

¡NER es esencial para extracción de información! 🚀
