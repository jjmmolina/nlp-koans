# Teoría: Transformers

## 📚 Tabla de Contenidos
1. [Introducción](#introducción)
2. [Arquitectura Transformer](#arquitectura)
3. [Attention Mechanism](#attention)
4. [BERT](#bert)
5. [GPT](#gpt)
6. [Otros Modelos](#otros)
7. [Aplicaciones](#aplicaciones)

---

## 🎯 Introducción {#introducción}

### La Revolución de 2017

El paper "Attention is All You Need" (Vaswani et al., 2017) cambió el NLP para siempre.

**Antes (RNN/LSTM):**
```
Procesa secuencial → Lento
Contexto limitado → Olvida información lejana
```

**Después (Transformers):**
```
Procesa paralelo → Rápido
Contexto completo → Self-attention captura todo
```

### ¿Por qué son Revolucionarios?

**1. Paralelización:**
```python
# RNN: procesa palabra por palabra
for word in sentence:  # Secuencial ❌
    hidden = f(word, hidden_prev)

# Transformer: procesa todo a la vez
hidden_all = attention(all_words)  # Paralelo ✅
```

**2. Long-Range Dependencies:**
```python
# "The cat that chased the mouse ran away"
# RNN: difícil conectar "cat" con "ran"
# Transformer: attention conecta directamente
```

**3. Transfer Learning:**
```python
# Pre-entrenar en corpus gigante
BERT → 3.3B palabras (Wikipedia + Books)

# Fine-tune para tarea específica
sentiment_model = fine_tune(BERT, sentiment_data)
```

---

## 🏗️ Arquitectura Transformer {#arquitectura}

### Componentes Principales

```
INPUT
  ↓
Embeddings + Positional Encoding
  ↓
┌─────────────────────────┐
│  ENCODER (N capas)      │
│  - Multi-Head Attention │
│  - Feed Forward         │
│  - Layer Norm           │
│  - Residual Connections │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│  DECODER (N capas)      │
│  - Masked Attention     │
│  - Cross Attention      │
│  - Feed Forward         │
└────────────┬────────────┘
             ↓
          OUTPUT
```

### Encoder

```python
# Pseudo-código
class EncoderLayer:
    def forward(self, x):
        # 1. Multi-Head Self-Attention
        attn_output = self.attention(x, x, x)
        x = x + attn_output  # Residual
        x = self.layer_norm1(x)
        
        # 2. Feed Forward
        ff_output = self.feed_forward(x)
        x = x + ff_output  # Residual
        x = self.layer_norm2(x)
        
        return x
```

### Decoder

```python
class DecoderLayer:
    def forward(self, x, encoder_output):
        # 1. Masked Self-Attention (no ve el futuro)
        attn1 = self.masked_attention(x, x, x)
        x = x + attn1
        x = self.layer_norm1(x)
        
        # 2. Cross-Attention (con encoder)
        attn2 = self.cross_attention(x, encoder_output, encoder_output)
        x = x + attn2
        x = self.layer_norm2(x)
        
        # 3. Feed Forward
        ff = self.feed_forward(x)
        x = x + ff
        x = self.layer_norm3(x)
        
        return x
```

### Positional Encoding

Los Transformers no ven el orden → necesitamos añadirlo.

```python
import numpy as np

def positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

# Ejemplo
pe = positional_encoding(seq_len=10, d_model=512)
# Añade información de posición a embeddings
embeddings_with_position = word_embeddings + pe
```

---

## 🔍 Attention Mechanism {#attention}

### Self-Attention

**Idea:** Cada palabra "atiende" a todas las demás para entender contexto.

```python
# Ejemplo
sentence = "The cat sat on the mat"

# "sat" atiende a:
# - "cat" (quién se sentó) ✅ Alta atención
# - "mat" (dónde) ✅ Alta atención
# - "the" ❌ Baja atención
```

### Cálculo de Attention

```
1. Query (Q), Key (K), Value (V) = transformaciones lineales de input

2. Scores = Q · K^T / √d_k
   (similitud entre queries y keys)

3. Weights = softmax(Scores)
   (probabilidades normalizadas)

4. Output = Weights · V
   (promedio ponderado de values)
```

**Código:**
```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    """
    Q: (batch, seq_len, d_k)
    K: (batch, seq_len, d_k)
    V: (batch, seq_len, d_v)
    """
    d_k = Q.size(-1)
    
    # Scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Weights
    weights = F.softmax(scores, dim=-1)
    
    # Output
    output = torch.matmul(weights, V)
    
    return output, weights
```

### Multi-Head Attention

Múltiples atenciones en paralelo para capturar diferentes aspectos.

```python
# Single head: 1 perspectiva
attention = self_attention(x)

# Multi-head: 8 perspectivas diferentes
head_1 = self_attention_1(x)  # Relaciones sintácticas
head_2 = self_attention_2(x)  # Relaciones semánticas
head_3 = self_attention_3(x)  # Nombres propios
...
head_8 = self_attention_8(x)

# Concatenar y proyectar
output = linear(concat(head_1, ..., head_8))
```

**Implementación:**
```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.W_q(x)  # (batch, seq, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        output, _ = scaled_dot_product_attention(Q, K, V)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Final linear
        output = self.W_o(output)
        
        return output
```

---

## 🤖 BERT {#bert}

### Bidirectional Encoder Representations from Transformers

**Características:**
- Solo ENCODER (no decoder)
- Bidireccional (ve contexto completo)
- Pre-entrenado con Masked Language Modeling

### Arquitectura

```
BERT-Base:
- 12 capas de encoder
- 768 hidden units
- 12 attention heads
- 110M parámetros

BERT-Large:
- 24 capas
- 1024 hidden units
- 16 attention heads
- 340M parámetros
```

### Pre-training Tasks

**1. Masked Language Modeling (MLM):**
```
Input:  "The cat [MASK] on the mat"
Output: "sat"

# Predice palabras enmascaradas
```

**2. Next Sentence Prediction (NSP):**
```
Sentence A: "The cat sat on the mat"
Sentence B: "It was comfortable"
Label: IsNext ✅

Sentence A: "The cat sat on the mat"
Sentence B: "I like pizza"
Label: NotNext ❌
```

### Uso con Transformers

```python
from transformers import BertTokenizer, BertModel
import torch

# Cargar modelo y tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenizar
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors='pt')

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Outputs
last_hidden_state = outputs.last_hidden_state  # (1, seq_len, 768)
pooler_output = outputs.pooler_output          # (1, 768) [CLS] token
```

### Fine-tuning

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Cargar modelo para clasificación
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Entrenar
trainer.train()
```

---

## 🚀 GPT {#gpt}

### Generative Pre-trained Transformer

**Características:**
- Solo DECODER (no encoder)
- Auto-regresivo (genera palabra por palabra)
- Pre-entrenado con Language Modeling

### Evolución

```
GPT-1 (2018):   117M parámetros
GPT-2 (2019):   1.5B parámetros
GPT-3 (2020):   175B parámetros
GPT-4 (2023):   ~1.7T parámetros (estimado)
```

### Pre-training Task

**Causal Language Modeling:**
```
Input:  "The cat sat on"
Output: "the"

Input:  "The cat sat on the"
Output: "mat"

# Predice siguiente palabra
```

### Uso

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Cargar
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generar
input_text = "Once upon a time"
inputs = tokenizer.encode(input_text, return_tensors='pt')

# Generar continuación
outputs = model.generate(
    inputs,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
# "Once upon a time, there was a small village..."
```

---

## 🌟 Otros Modelos {#otros}

### RoBERTa

BERT mejorado (Facebook).

```
Mejoras:
- Sin NSP task
- Más datos
- Entrenamiento más largo
- Dinamic masking
```

### ALBERT

BERT ligero (Google).

```
Reduce parámetros:
- Factorización de embeddings
- Sharing de parámetros entre capas
```

### DistilBERT

BERT destilado (Hugging Face).

```
- 40% menos parámetros
- 60% más rápido
- 97% performance de BERT
```

### T5

Text-to-Text Transfer Transformer (Google).

```
TODO es text-to-text:
"translate English to German: Hello" → "Hallo"
"summarize: [long text]" → "[summary]"
```

### Comparativa

| Modelo | Parámetros | Velocidad | Uso |
|--------|-----------|-----------|-----|
| **BERT-base** | 110M | ⚡⚡ | Clasificación, NER |
| **RoBERTa** | 125M | ⚡⚡ | Mejor BERT |
| **DistilBERT** | 66M | ⚡⚡⚡⚡ | Producción rápida |
| **GPT-2** | 1.5B | ⚡ | Generación |
| **T5** | 220M-11B | ⚡ | Versátil |

---

## 💼 Aplicaciones {#aplicaciones}

### 1. Text Classification

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

result = classifier("I love this product!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.999}]
```

### 2. Named Entity Recognition

```python
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

text = "Apple CEO Tim Cook visited Paris"
entities = ner(text)

for ent in entities:
    print(f"{ent['word']:15} → {ent['entity']}")
```

### 3. Question Answering

```python
qa = pipeline("question-answering")

context = "Python is a programming language created by Guido van Rossum"
question = "Who created Python?"

answer = qa(question=question, context=context)
print(answer['answer'])  # "Guido van Rossum"
```

### 4. Text Generation

```python
generator = pipeline("text-generation", model="gpt2")

prompt = "Artificial intelligence will"
output = generator(prompt, max_length=50, num_return_sequences=1)

print(output[0]['generated_text'])
```

### 5. Embeddings

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Cargar
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Obtener embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings

# Usar
emb1 = get_embedding("I love machine learning")
emb2 = get_embedding("I enjoy deep learning")

# Similitud coseno
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(emb1, emb2)
print(f"Similarity: {similarity[0][0]:.3f}")
```

---

## 🎓 Resumen

**Conceptos Clave:**
- Transformers revolucionaron NLP en 2017
- Basados en attention mechanism (no RNN)
- Encoder-only (BERT), Decoder-only (GPT), Encoder-Decoder (T5)
- Pre-training + Fine-tuning paradigm

**Modelos Principales:**
- BERT → Clasificación, NER, QA
- GPT → Generación de texto
- T5 → Versátil (todo es text-to-text)

**Próximos Pasos:**
- **Koan 10**: Modern LLMs (GPT-4, Claude, etc.)
- **Koan 12**: Semantic Search (con embeddings)

¡Los Transformers son el futuro del NLP! 🚀
