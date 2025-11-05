# 💡 Pistas para Koan 08: Transformers

## 🎯 Objetivo del Koan

Aprender a usar **modelos Transformer** de Hugging Face:
- Pipelines para tareas comunes
- Modelos pre-entrenados (BERT, GPT, etc.)
- Tokenización y embeddings contextuales
- Fine-tuning básico

---

## 📝 Función 1: `load_pipeline()`

### Nivel 1: Concepto
Pipelines son la forma más fácil de usar modelos de Hugging Face para tareas estándar.

### Nivel 2: Tareas disponibles
```python
from transformers import pipeline

# Tareas comunes:
# - "sentiment-analysis"
# - "text-generation"
# - "fill-mask"
# - "ner" (Named Entity Recognition)
# - "question-answering"
# - "summarization"
# - "translation"
```

### Nivel 3: Implementación
```python
pipe = pipeline(task_name, model=model_name if model_name else None)
return pipe
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def load_pipeline(task_name: str, model_name: str = None):
    from transformers import pipeline
    
    if model_name:
        pipe = pipeline(task_name, model=model_name)
    else:
        # Usa modelo por defecto para la tarea
        pipe = pipeline(task_name)
    
    return pipe
```
</details>

---

## 📝 Función 2: `generate_text()`

### Nivel 1: Concepto
Genera texto continuando un prompt dado usando modelos generativos (GPT-2, GPT-3).

### Nivel 2: Implementación
```python
from transformers import pipeline
generator = pipeline("text-generation", model=model_name)
result = generator(prompt, max_length=max_length, num_return_sequences=1)
return result[0]["generated_text"]
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def generate_text(prompt: str, model_name: str = "gpt2", 
                  max_length: int = 50) -> str:
    from transformers import pipeline
    
    generator = pipeline("text-generation", model=model_name)
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    
    return result[0]["generated_text"]
```
</details>

---

## 📝 Función 3: `fill_mask()`

### Nivel 1: Concepto
Modelos BERT predicen palabras enmascaradas (missing words) en una oración.

### Nivel 2: Uso
```python
text = "Paris is the [MASK] of France."
# Modelo predice: "capital"
```

### Nivel 3: Implementación
```python
from transformers import pipeline
unmasker = pipeline("fill-mask", model=model_name)
results = unmasker(text)
# Retorna lista ordenada por score
return [(r["token_str"], r["score"]) for r in results[:top_k]]
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def fill_mask(text: str, model_name: str = "bert-base-uncased", 
              top_k: int = 5) -> List[Tuple[str, float]]:
    from transformers import pipeline
    
    unmasker = pipeline("fill-mask", model=model_name)
    results = unmasker(text)
    
    # Retorna top_k predicciones con sus scores
    return [(result["token_str"].strip(), result["score"]) 
            for result in results[:top_k]]
```
</details>

---

## 📝 Función 4: `extract_features_bert()`

### Nivel 1: Concepto
Extrae embeddings contextuales de BERT para cada token.

### Nivel 2: Diferencia con Word2Vec
- Word2Vec: mismo vector para "bank" (río) y "bank" (dinero)
- BERT: vectores diferentes según contexto

### Nivel 3: Implementación
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
# outputs.last_hidden_state contiene embeddings
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def extract_features_bert(text: str, 
                          model_name: str = "bert-base-uncased") -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Tokeniza
    inputs = tokenizer(text, return_tensors="pt")
    
    # Pasa por modelo
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extrae embeddings (promedio de todos los tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    
    return embeddings.numpy()
```
</details>

---

## 📝 Función 5: `answer_question()`

### Nivel 1: Concepto
Question Answering: extrae respuesta de un contexto dado.

### Nivel 2: Formato
```python
context = "Paris is the capital of France. It has 2.2M inhabitants."
question = "What is the capital of France?"
# Respuesta: "Paris"
```

### Nivel 3: Implementación
```python
from transformers import pipeline
qa_pipeline = pipeline("question-answering", model=model_name)
result = qa_pipeline(question=question, context=context)
return result["answer"]
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def answer_question(question: str, context: str, 
                    model_name: str = "distilbert-base-cased-distilled-squad") -> dict:
    from transformers import pipeline
    
    qa_pipeline = pipeline("question-answering", model=model_name)
    result = qa_pipeline(question=question, context=context)
    
    return {
        "answer": result["answer"],
        "score": result["score"],
        "start": result["start"],
        "end": result["end"]
    }
```
</details>

---

## 📝 Función 6: `summarize_text()`

### Nivel 1: Concepto
Resume textos largos automáticamente usando modelos como BART o T5.

### Nivel 2: Parámetros importantes
```python
max_length: longitud máxima del resumen
min_length: longitud mínima del resumen
do_sample: usar sampling (más creativo) vs greedy (más conservador)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def summarize_text(text: str, 
                   model_name: str = "facebook/bart-large-cnn",
                   max_length: int = 130,
                   min_length: int = 30) -> str:
    from transformers import pipeline
    
    summarizer = pipeline("summarization", model=model_name)
    
    result = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )
    
    return result[0]["summary_text"]
```
</details>

---

## 📝 Función 7: `compare_transformer_embeddings()`

### Nivel 1: Concepto
Compara embeddings de la misma palabra en diferentes contextos para mostrar que BERT es contextual.

### Nivel 2: Ejemplo
```python
text1 = "I went to the bank to deposit money"
text2 = "I sat by the river bank"
# "bank" tiene embeddings diferentes en cada oración
```

### Nivel 3: Pasos
1. Extrae embeddings de ambos textos
2. Calcula similitud coseno
3. Retorna comparación

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def compare_transformer_embeddings(text1: str, text2: str,
                                    model_name: str = "bert-base-uncased") -> dict:
    import numpy as np
    
    # Extrae embeddings de ambos textos
    emb1 = extract_features_bert(text1, model_name)
    emb2 = extract_features_bert(text2, model_name)
    
    # Calcula similitud coseno
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    similarity = dot_product / (norm1 * norm2)
    
    return {
        "text1": text1,
        "text2": text2,
        "embedding1_shape": emb1.shape,
        "embedding2_shape": emb2.shape,
        "cosine_similarity": float(similarity)
    }
```
</details>

---

## 🎯 Conceptos Clave

### Arquitectura Transformer

**Componentes principales**:
1. **Self-Attention**: Cada palabra "atiende" a todas las demás
2. **Multi-Head Attention**: Múltiples mecanismos de atención en paralelo
3. **Feed-Forward Networks**: Transformaciones no lineales
4. **Positional Encoding**: Codifica posición de tokens

### Modelos Principales

| Modelo | Tipo | Uso Principal |
|--------|------|---------------|
| **BERT** | Encoder | Clasificación, NER, Q&A |
| **GPT** | Decoder | Generación de texto |
| **T5** | Encoder-Decoder | Traducción, resumen |
| **BART** | Encoder-Decoder | Resumen, paráfrasis |
| **RoBERTa** | Encoder | BERT mejorado |
| **DistilBERT** | Encoder | BERT más rápido/pequeño |

### Pipelines Disponibles

```python
# Análisis
pipeline("sentiment-analysis")
pipeline("ner")
pipeline("zero-shot-classification")

# Generación
pipeline("text-generation")
pipeline("summarization")
pipeline("translation_en_to_fr")

# Comprensión
pipeline("question-answering")
pipeline("fill-mask")

# Audio/Visión
pipeline("automatic-speech-recognition")
pipeline("image-classification")
```

### Embeddings Contextuales vs Estáticos

**Word2Vec (Estático)**:
```python
"bank" → [0.1, 0.5, ...] (siempre el mismo vector)
```

**BERT (Contextual)**:
```python
"I went to the bank" → "bank" = [0.2, 0.7, ...]
"River bank"         → "bank" = [0.8, 0.1, ...]
                               (vectores diferentes!)
```

## 💡 Tips Prácticos

### 1. Elige el modelo correcto para la tarea

```python
# Para generación de texto
model = "gpt2"  # o "gpt2-medium", "gpt2-large"

# Para clasificación/NER
model = "bert-base-uncased"  # o "roberta-base"

# Para resumen
model = "facebook/bart-large-cnn"  # o "t5-small"

# Para Q&A
model = "distilbert-base-cased-distilled-squad"
```

### 2. Modelos distilled son más rápidos

```python
# Grande pero preciso
"bert-base-uncased"  # 110M parámetros

# Pequeño pero rápido (60% más rápido, 95% precisión)
"distilbert-base-uncased"  # 66M parámetros
```

### 3. Usa GPU cuando sea posible

```python
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-generation", model="gpt2", device=device)
```

### 4. Controla longitud de generación

```python
# Texto corto
generate_text(prompt, max_length=50)

# Texto largo
generate_text(prompt, max_length=200)
```

## 🚀 Casos de Uso

### Chatbot simple
```python
generator = pipeline("text-generation", model="gpt2")
while True:
    user_input = input("You: ")
    response = generator(user_input, max_length=50)[0]["generated_text"]
    print(f"Bot: {response}")
```

### Sistema de Q&A sobre documentos
```python
doc = load_document("company_handbook.txt")
qa = pipeline("question-answering")

questions = [
    "What is the vacation policy?",
    "How many sick days do I get?"
]

for q in questions:
    answer = qa(question=q, context=doc)
    print(f"Q: {q}\nA: {answer['answer']}\n")
```

### Resumen automático de artículos
```python
article = fetch_news_article()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(article, max_length=100, min_length=30)
print(summary[0]["summary_text"])
```

### Auto-completar código/texto
```python
code_prompt = "def fibonacci(n):\n    if n <= 1:"
generator = pipeline("text-generation", model="gpt2")
completed = generator(code_prompt, max_length=100)
```

## 🔧 Troubleshooting

### Problema: Out of memory (GPU/CPU)
**Solución**:
```python
# Usa modelos distilled más pequeños
"distilbert-base-uncased" en vez de "bert-base-uncased"

# O procesa textos más cortos
text = text[:512]  # BERT tiene límite de 512 tokens
```

### Problema: Generación muy lenta
**Solución**:
```python
# Reduce max_length
generate_text(prompt, max_length=30)

# Usa GPU
pipe = pipeline("text-generation", device=0)
```

### Problema: Token [MASK] no funciona
**Solución**:
```python
# Cada modelo usa token diferente
# BERT: "[MASK]"
# RoBERTa: "<mask>"
# ALBERT: "[MASK]"

# Verifica el tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
print(tokenizer.mask_token)  # "<mask>"
```

### Problema: Respuestas no coherentes
**Solución**:
```python
# Ajusta temperatura para generación
generator(
    prompt,
    max_length=50,
    temperature=0.7,  # Menos aleatorio
    top_p=0.9,
    do_sample=True
)
```

## 📚 Recursos

### Hugging Face Hub
- 100,000+ modelos pre-entrenados
- https://huggingface.co/models

### Modelos populares español
```python
"dccuchile/bert-base-spanish-wwm-uncased"
"PlanTL-GOB-ES/roberta-base-bne"
"mrm8488/electricidad-base-discriminator"
```

## 🚀 Siguiente Paso

Una vez completo, ve al **Koan 09: Language Models**!
