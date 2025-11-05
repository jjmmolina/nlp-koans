# 💡 Pistas para Koan 04: Named Entity Recognition (NER)

## 🎯 Objetivo del Koan

Aprender a **identificar entidades nombradas** en texto:
- Personas (PER)
- Organizaciones (ORG)
- Lugares (LOC)
- Fechas, dinero, etc.

---

## 📝 Función 1: `extract_entities_spacy()`

### Nivel 1: Concepto
spaCy detecta entidades automáticamente cuando procesas texto.

### Nivel 2: Implementación
```python
import spacy
nlp = spacy.load("es_core_news_sm")  # o en_core_web_sm
doc = nlp(text)
# doc.ents contiene las entidades
# Cada entidad tiene .text y .label_
```

### Nivel 3: Casi la solución
```python
doc = nlp(text)
return [(ent.text, ent.label_) for ent in doc.ents]
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def extract_entities_spacy(text: str, lang: str = "es") -> List[Tuple[str, str]]:
    import spacy
    model = "es_core_news_sm" if lang == "es" else "en_core_web_sm"
    nlp = spacy.load(model)
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
```
</details>

---

## 📝 Función 2: `get_entities_by_type()`

### Nivel 1: Concepto
Filtra entidades por un tipo específico (PER, ORG, LOC, etc.)

### Nivel 2: Pasos
1. Extrae todas las entidades con `extract_entities_spacy()`
2. Filtra solo las que tienen `label == entity_type`
3. Retorna solo los textos (sin etiquetas)

### Nivel 3: Casi la solución
```python
entities = extract_entities_spacy(text, lang)
return [text for text, label in entities if label == entity_type]
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def get_entities_by_type(text: str, entity_type: str, lang: str = "es") -> List[str]:
    entities = extract_entities_spacy(text, lang)
    return [ent_text for ent_text, ent_label in entities if ent_label == entity_type]
```
</details>

---

## 📝 Función 3: `get_person_names()`

### Nivel 1: Concepto
Usa `get_entities_by_type()` para extraer solo personas.

### Nivel 2: Tipos de entidad
- Español: `"PER"`
- Inglés: `"PERSON"`

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def get_person_names(text: str, lang: str = "es") -> List[str]:
    entity_type = "PER" if lang == "es" else "PERSON"
    return get_entities_by_type(text, entity_type, lang)
```
</details>

---

## 📝 Función 4: `get_locations()`

### Nivel 1: Concepto
Similar a personas, pero para lugares.

### Nivel 2: Tipos
- Español: `"LOC"`
- Inglés: `"GPE"` (Geo-Political Entity) o `"LOC"`

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def get_locations(text: str, lang: str = "es") -> List[str]:
    entity_type = "LOC" if lang == "es" else "GPE"
    return get_entities_by_type(text, entity_type, lang)
```
</details>

---

## 📝 Función 5: `get_organizations()`

### Nivel 1: Concepto
Extrae nombres de organizaciones, empresas, instituciones.

### Nivel 2: Tipos
- Español: `"ORG"`
- Inglés: `"ORG"`

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def get_organizations(text: str, lang: str = "es") -> List[str]:
    return get_entities_by_type(text, "ORG", lang)
```
</details>

---

## 📝 Función 6: `count_entity_types()`

### Nivel 1: Concepto
Cuenta cuántas entidades hay de cada tipo.

### Nivel 2: Implementación
```python
from collections import Counter
entities = extract_entities_spacy(text, lang)
labels = [label for _, label in entities]
return dict(Counter(labels))
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def count_entity_types(text: str, lang: str = "es") -> dict:
    from collections import Counter
    entities = extract_entities_spacy(text, lang)
    entity_labels = [label for _, label in entities]
    return dict(Counter(entity_labels))
```
</details>

---

## 📝 Función 7: `extract_entities_with_context()`

### Nivel 1: Concepto
Extrae entidades con información adicional: texto, etiqueta, posición inicial/final.

### Nivel 2: Implementación
```python
doc = nlp(text)
return [(ent.text, ent.label_, ent.start_char, ent.end_char) 
        for ent in doc.ents]
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def extract_entities_with_context(text: str, lang: str = "es") -> List[Tuple[str, str, int, int]]:
    import spacy
    model = "es_core_news_sm" if lang == "es" else "en_core_web_sm"
    nlp = spacy.load(model)
    doc = nlp(text)
    return [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
```
</details>

---

## 🎯 Tipos de Entidades

### Español (spaCy es_core_news_sm)

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| **PER** | Persona | María García, Einstein |
| **LOC** | Lugar | Madrid, Amazonas |
| **ORG** | Organización | Google, ONU, Real Madrid |
| **MISC** | Misceláneo | Nobel, Oscar |

### Inglés (spaCy en_core_web_sm)

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| **PERSON** | Persona | John Smith, Einstein |
| **GPE** | Entidad geopolítica | Madrid, USA, London |
| **LOC** | Lugar no-GPE | Mount Everest, Amazon River |
| **ORG** | Organización | Google, UN, Real Madrid |
| **DATE** | Fecha | yesterday, 2024, March |
| **MONEY** | Dinero | $100, 50 euros |
| **PRODUCT** | Producto | iPhone, Windows |

## 💡 Tips de NER

### 1. Modelos más grandes = mejor precisión
```python
# Modelo pequeño (rápido)
nlp = spacy.load("es_core_news_sm")

# Modelo mediano (balanceado)
nlp = spacy.load("es_core_news_md")

# Modelo grande (mejor precisión)
nlp = spacy.load("es_core_news_lg")
```

### 2. Capitalización importa
```python
# Mejor reconocimiento
"Elon Musk trabaja en Tesla"

# Peor reconocimiento
"elon musk trabaja en tesla"
```

### 3. Contexto ayuda
```python
# Ambiguo
"Apple es roja"  # ¿Empresa o fruta?

# Claro
"Apple lanzó el iPhone"  # Empresa
```

### 4. Entidades compuestas
```python
entities = extract_entities_spacy("Real Madrid ganó la Champions")
# [("Real Madrid", "ORG"), ("Champions", "MISC")]
```

## 🚀 Casos de Uso

### Extracción de información
```python
text = "María García trabaja en Google en Madrid"
persons = get_person_names(text)      # ["María García"]
orgs = get_organizations(text)         # ["Google"]
locs = get_locations(text)             # ["Madrid"]
```

### Anonimización
```python
def anonymize_text(text):
    doc = nlp(text)
    result = text
    for ent in reversed(doc.ents):  # reversed para mantener índices
        if ent.label_ == "PER":
            result = result[:ent.start_char] + "[REDACTED]" + result[ent.end_char:]
    return result
```

### Análisis de menciones
```python
counts = count_entity_types(article)
# {"PER": 15, "ORG": 8, "LOC": 12}
print(f"El artículo menciona {counts['PER']} personas")
```

## 🔧 Troubleshooting

### Problema: No detecta entidades
**Solución**: Verifica que el texto tenga mayúsculas apropiadas

### Problema: Entidades incorrectas
**Solución**: Usa un modelo más grande o entrena uno personalizado

### Problema: Rendimiento lento
**Solución**: Usa modelos pequeños o procesa en lotes

## 🚀 Siguiente Paso

Una vez completo, ve al **Koan 05: Text Classification**!
