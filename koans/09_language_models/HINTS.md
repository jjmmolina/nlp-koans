# 💡 Pistas para Koan 09: Language Models

## 🎯 Objetivo del Koan

Aprender sobre **modelos de lenguaje modernos**:
- Generación de texto con control
- Prompting efectivo
- Evaluación de modelos
- Uso de LLMs para tareas complejas

---

## 📝 Función 1: `calculate_perplexity()`

### Nivel 1: Concepto
Perplexity mide qué tan "sorprendido" está un modelo por una secuencia. Menor perplexity = mejor modelo.

### Nivel 2: Fórmula
```
Perplexity = exp(average_negative_log_likelihood)
```

### Nivel 3: Implementación
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss

perplexity = torch.exp(loss).item()
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def calculate_perplexity(text: str, model_name: str = "gpt2") -> float:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Tokeniza
    inputs = tokenizer(text, return_tensors="pt")
    
    # Calcula loss (cross-entropy)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    
    # Perplexity = exp(loss)
    perplexity = torch.exp(loss).item()
    
    return perplexity
```
</details>

---

## 📝 Función 2: `generate_with_parameters()`

### Nivel 1: Concepto
Controla la generación de texto con parámetros como temperatura, top_k, top_p.

### Nivel 2: Parámetros clave
```python
temperature: 0.1-2.0 (más alto = más creativo/aleatorio)
top_k: Considera solo top K tokens (ej: 50)
top_p: Nucleus sampling, considera tokens hasta probabilidad p (ej: 0.9)
num_return_sequences: Cuántas versiones generar
```

### Nivel 3: Implementación
```python
from transformers import pipeline

generator = pipeline("text-generation", model=model_name)
results = generator(
    prompt,
    max_length=max_length,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    num_return_sequences=num_sequences
)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def generate_with_parameters(prompt: str,
                              model_name: str = "gpt2",
                              max_length: int = 50,
                              temperature: float = 1.0,
                              top_k: int = 50,
                              top_p: float = 0.95,
                              num_sequences: int = 1) -> List[str]:
    from transformers import pipeline
    
    generator = pipeline("text-generation", model=model_name)
    
    results = generator(
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_sequences,
        do_sample=True
    )
    
    return [result["generated_text"] for result in results]
```
</details>

---

## 📝 Función 3: `prompt_engineering()`

### Nivel 1: Concepto
Diseña prompts efectivos para obtener respuestas específicas de LLMs.

### Nivel 2: Estrategias
```python
# Few-shot learning: dar ejemplos
prompt = """
Traduce al español:
English: Hello
Spanish: Hola
English: Thank you
Spanish: Gracias
English: Goodbye
Spanish:"""

# Zero-shot: instrucción directa
prompt = "Traduce 'Goodbye' al español:"
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def prompt_engineering(task: str, 
                       examples: List[Tuple[str, str]] = None,
                       query: str = "",
                       model_name: str = "gpt2") -> str:
    from transformers import pipeline
    
    # Construye prompt
    if examples:
        # Few-shot learning
        prompt = f"{task}:\n\n"
        for input_ex, output_ex in examples:
            prompt += f"Input: {input_ex}\nOutput: {output_ex}\n\n"
        prompt += f"Input: {query}\nOutput:"
    else:
        # Zero-shot
        prompt = f"{task}: {query}"
    
    # Genera respuesta
    generator = pipeline("text-generation", model=model_name)
    result = generator(prompt, max_length=len(prompt.split()) + 50)[0]["generated_text"]
    
    # Extrae solo la respuesta nueva
    response = result[len(prompt):].strip()
    
    return response
```
</details>

---

## 📝 Función 4: `get_token_probabilities()`

### Nivel 1: Concepto
Obtiene las probabilidades de los siguientes tokens según el modelo.

### Nivel 2: Uso
```python
text = "The capital of France is"
# Modelo da alta probabilidad a "Paris", "Paris.", etc.
```

### Nivel 3: Implementación
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # Último token
    probs = torch.softmax(logits, dim=-1)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def get_token_probabilities(text: str, 
                             model_name: str = "gpt2",
                             top_k: int = 5) -> List[Tuple[str, float]]:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Tokeniza
    inputs = tokenizer(text, return_tensors="pt")
    
    # Obtiene logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Logits del último token
    
    # Convierte a probabilidades
    probs = torch.softmax(logits, dim=-1)
    
    # Obtiene top_k tokens
    top_probs, top_indices = torch.topk(probs, top_k)
    
    # Convierte índices a tokens
    results = []
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.decode([idx])
        results.append((token, prob.item()))
    
    return results
```
</details>

---

## 📝 Función 5: `conditional_generation()`

### Nivel 1: Concepto
Genera texto que cumple ciertas condiciones (longitud, estilo, keywords).

### Nivel 2: Estrategias
```python
# Control de longitud
min_length, max_length

# Control de contenido
# Penaliza repeticiones con repetition_penalty
# Evita tokens con bad_words_ids
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def conditional_generation(prompt: str,
                           model_name: str = "gpt2",
                           min_length: int = 20,
                           max_length: int = 100,
                           no_repeat_ngram_size: int = 3,
                           repetition_penalty: float = 1.2) -> str:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Tokeniza prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Genera con restricciones
    outputs = model.generate(
        **inputs,
        min_length=min_length,
        max_length=max_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        top_p=0.92,
        top_k=50
    )
    
    # Decodifica
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text
```
</details>

---

## 📝 Función 6: `compare_model_outputs()`

### Nivel 1: Concepto
Compara cómo diferentes modelos responden al mismo prompt.

### Nivel 2: Modelos a comparar
```python
models = [
    "gpt2",           # 117M params
    "gpt2-medium",    # 345M params
    "gpt2-large",     # 774M params
    "distilgpt2"      # 82M params (más rápido)
]
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def compare_model_outputs(prompt: str, 
                          model_names: List[str],
                          max_length: int = 50) -> dict:
    from transformers import pipeline
    
    results = {}
    
    for model_name in model_names:
        try:
            generator = pipeline("text-generation", model=model_name)
            output = generator(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True
            )[0]["generated_text"]
            
            results[model_name] = output
        except Exception as e:
            results[model_name] = f"Error: {str(e)}"
    
    return results
```
</details>

---

## 📝 Función 7: `evaluate_generation_quality()`

### Nivel 1: Concepto
Evalúa la calidad del texto generado usando métricas automáticas.

### Nivel 2: Métricas
```python
# Perplexity: qué tan "natural" es el texto
# Diversity: cuántos tokens únicos
# Repetition: cuántas n-gramas repetidas
# Coherence: similitud semántica entre oraciones
```

### Nivel 3: Implementación básica
```python
import numpy as np

tokens = text.split()
unique_tokens = len(set(tokens))
total_tokens = len(tokens)

diversity = unique_tokens / total_tokens
perplexity = calculate_perplexity(text, model_name)
```

### ✅ Solución
<details>
<summary>Click para ver</summary>

```python
def evaluate_generation_quality(text: str, 
                                 model_name: str = "gpt2") -> dict:
    import numpy as np
    from collections import Counter
    
    # Tokeniza
    tokens = text.split()
    
    # Diversity: ratio de tokens únicos
    unique_tokens = len(set(tokens))
    total_tokens = len(tokens)
    diversity = unique_tokens / total_tokens if total_tokens > 0 else 0
    
    # Repetition: bigrams repetidos
    bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 1)
    repetition_ratio = repeated_bigrams / len(bigrams) if bigrams else 0
    
    # Perplexity
    try:
        perplexity = calculate_perplexity(text, model_name)
    except:
        perplexity = None
    
    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "diversity": diversity,
        "repetition_ratio": repetition_ratio,
        "perplexity": perplexity
    }
```
</details>

---

## 🎯 Conceptos Clave

### Language Models (LMs)

**Definición**: Modelos que predicen la probabilidad de secuencias de palabras.

```
P("The cat sat on the mat")
P("on mat sat The cat the") # Baja probabilidad
```

### Tipos de Language Models

| Tipo | Arquitectura | Ejemplo | Uso |
|------|--------------|---------|-----|
| **Causal LM** | Decoder | GPT-2/3 | Generación de texto |
| **Masked LM** | Encoder | BERT | Clasificación, NER |
| **Seq2Seq** | Enc-Dec | T5, BART | Traducción, resumen |

### Parámetros de Generación

#### Temperature (0.1 - 2.0)
```python
temperature = 0.1  # Conservador, predecible
temperature = 1.0  # Balanceado (default)
temperature = 2.0  # Creativo, aleatorio
```

#### Top-k Sampling
```python
top_k = 1   # Greedy (siempre el más probable)
top_k = 50  # Considera 50 tokens más probables
```

#### Top-p (Nucleus Sampling)
```python
top_p = 0.9  # Considera tokens hasta 90% probabilidad acumulada
top_p = 0.5  # Más conservador
```

### Prompt Engineering

**Zero-Shot**:
```python
"Traduce al español: Hello"
```

**Few-Shot**:
```python
"""
Traduce al español:
English: Hello → Spanish: Hola
English: Bye → Spanish: Adiós
English: Thanks → Spanish:
"""
```

**Chain-of-Thought**:
```python
"""
P: ¿Cuánto es 15 + 27?
R: Primero sumo 15 + 20 = 35
   Luego sumo 35 + 7 = 42
   Respuesta: 42
"""
```

## 💡 Tips Prácticos

### 1. Ajusta temperatura según necesidad

```python
# Factual/Preciso (código, matemáticas)
temperature = 0.3

# Creativo (historias, poesía)
temperature = 1.5
```

### 2. Combina top_k y top_p

```python
# Buena configuración general
generate_with_parameters(
    prompt,
    temperature=0.8,
    top_k=50,
    top_p=0.95
)
```

### 3. Evita repeticiones

```python
conditional_generation(
    prompt,
    no_repeat_ngram_size=3,      # No repite 3-gramas
    repetition_penalty=1.2       # Penaliza tokens repetidos
)
```

### 4. Controla longitud efectivamente

```python
# Mínimo garantizado
min_length = 50

# Máximo absoluto
max_length = 200

# O usa early stopping
generate(..., early_stopping=True)
```

## 🚀 Casos de Uso

### Autocompletar código
```python
prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:"
code = generate_with_parameters(
    prompt, 
    model_name="gpt2",
    temperature=0.3,  # Conservador para código
    max_length=100
)
```

### Generación creativa
```python
prompt = "Once upon a time in a distant galaxy"
stories = generate_with_parameters(
    prompt,
    temperature=1.2,  # Creativo
    num_sequences=3   # 3 versiones
)
```

### Clasificación con prompts
```python
prompt = """
Clasifica el sentimiento:
"Me encanta este producto" -> Positivo
"Es horrible" -> Negativo
"No está mal" ->"""

result = prompt_engineering(
    task="Sentiment classification",
    examples=[
        ("Me encanta", "Positivo"),
        ("Es horrible", "Negativo")
    ],
    query="No está mal"
)
```

### Generación de resúmenes
```python
article = "Long article text here..."
prompt = f"Resumir en una oración:\n{article}\n\nResumen:"

summary = conditional_generation(
    prompt,
    min_length=10,
    max_length=50
)
```

## 🔧 Troubleshooting

### Problema: Texto muy repetitivo
**Solución**:
```python
conditional_generation(
    prompt,
    no_repeat_ngram_size=3,
    repetition_penalty=1.5
)
```

### Problema: Generación sin sentido
**Solución**:
```python
# Reduce temperatura
generate_with_parameters(prompt, temperature=0.7)

# O usa top_p más bajo
generate_with_parameters(prompt, top_p=0.8)
```

### Problema: Muy lento
**Solución**:
```python
# Usa modelo más pequeño
model_name = "distilgpt2"  # vs "gpt2-large"

# O reduce max_length
max_length = 50  # vs 200
```

### Problema: Perplexity muy alto
**Solución**: 
- El texto no es natural
- Está en idioma diferente al del modelo
- Contiene caracteres especiales/raros

## 📚 Recursos Adicionales

### Modelos recomendados

**Para español**:
```python
"mrm8488/GPT-2-finetuned-SQUAD-spanish"
"DeepESP/gpt2-spanish"
```

**Para inglés**:
```python
"gpt2"           # Base (117M)
"gpt2-medium"    # Mediano (345M)
"gpt2-large"     # Grande (774M)
"distilgpt2"     # Rápido (82M)
```

### Papers importantes
- "Attention Is All You Need" (Transformers)
- "Language Models are Few-Shot Learners" (GPT-3)
- "BERT: Pre-training of Deep Bidirectional Transformers"

## 🎉 ¡Felicidades!

Has completado todos los **9 Koans de NLP**! 🎊

### Lo que has aprendido:
✅ Tokenización (NLTK, spaCy)
✅ Stemming y Lemmatization
✅ POS Tagging
✅ Named Entity Recognition
✅ Text Classification (TF-IDF, ML)
✅ Sentiment Analysis (TextBlob, Transformers)
✅ Word Embeddings (Word2Vec, spaCy)
✅ Transformers (BERT, GPT, pipelines)
✅ Language Models (generación, prompting)

### Próximos pasos:
1. 🔧 **Proyectos propios**: Aplica lo aprendido
2. 📚 **Papers**: Lee investigación reciente
3. 🚀 **Fine-tuning**: Entrena modelos con tus datos
4. 🤝 **Contribuye**: Comparte en GitHub
5. 🎓 **Especialízate**: Elige un área (NER, QA, etc.)

### Recursos para continuar:
- Hugging Face Course: https://huggingface.co/course
- Fast.ai NLP: https://www.fast.ai/
- Papers With Code: https://paperswithcode.com/area/natural-language-processing
- r/LanguageTechnology en Reddit

**¡Mucha suerte en tu viaje de NLP!** 🚀
