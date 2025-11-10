# Teoría: Language Models

## 📚 Tabla de Contenidos
1. [Introducción](#introducción)
2. [N-gram Language Models](#ngrams)
3. [Evaluación: Perplexity](#perplexity)
4. [Neural Language Models](#neural)
5. [Generación de Texto](#generación)
6. [Aplicaciones](#aplicaciones)

---

## 🎯 Introducción {#introducción}

### ¿Qué es un Language Model?

Un modelo que asigna probabilidades a secuencias de palabras.

```python
P("I love Python") = 0.001  # Probable
P("Python love I") = 0.00001  # Improbable
```

**Objetivo:** Modelar la distribución de probabilidad del lenguaje.

### ¿Para qué sirve?

**1. Generación de Texto:**
```python
# Predecir siguiente palabra
"I love" → P("Python" | "I love") = 0.3
        → P("coffee" | "I love") = 0.2
        → P("you" | "I love") = 0.15
```

**2. Evaluación de Fluidez:**
```python
# ¿Qué oración es más natural?
P("The cat sat on the mat") > P("Mat the on sat cat the")
```

**3. Corrección de Errores:**
```python
# Spelling correction
"I liek Python" → ¿"like" o "lick"?
P("I like Python") > P("I lick Python")
→ Corregir a "like"
```

**4. Speech Recognition:**
```python
# Fonéticamente similar
"recognize speech" vs "wreck a nice beach"
LM ayuda a elegir la correcta
```

---

## 📊 N-gram Language Models {#ngrams}

### Unigram Model

Cada palabra es independiente.

```python
P("I love Python") = P("I") × P("love") × P("Python")
```

**Estimación:**
```python
P("Python") = count("Python") / total_words
```

**Problema:** Asume independencia (no captura contexto).

### Bigram Model

Considera palabra anterior.

```python
P("I love Python") = P("I") × P("love"|"I") × P("Python"|"love")
```

**Estimación:**
```python
P("love" | "I") = count("I love") / count("I")
```

**Implementación:**
```python
from collections import defaultdict, Counter

class BigramModel:
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
    
    def train(self, corpus):
        for sentence in corpus:
            words = ['<s>'] + sentence + ['</s>']  # Start/end tokens
            
            for i in range(len(words) - 1):
                self.unigram_counts[words[i]] += 1
                self.bigram_counts[words[i]][words[i+1]] += 1
    
    def probability(self, word, prev_word):
        if self.unigram_counts[prev_word] == 0:
            return 0
        
        return self.bigram_counts[prev_word][word] / self.unigram_counts[prev_word]
    
    def sentence_probability(self, sentence):
        words = ['<s>'] + sentence + ['</s>']
        prob = 1.0
        
        for i in range(len(words) - 1):
            prob *= self.probability(words[i+1], words[i])
        
        return prob

# Uso
corpus = [
    ['I', 'love', 'Python'],
    ['I', 'love', 'coding'],
    ['Python', 'is', 'great']
]

model = BigramModel()
model.train(corpus)

# Probabilidad
prob = model.sentence_probability(['I', 'love', 'Python'])
print(f"P(I love Python) = {prob:.6f}")
```

### Trigram Model

Considera 2 palabras anteriores.

```python
P("I love Python") = P("I") × P("love"|"I") × P("Python"|"I love")
```

### N-gram General

```python
P(w_i | w_1, ..., w_{i-1}) ≈ P(w_i | w_{i-n+1}, ..., w_{i-1})

# Bigram: n=2
# Trigram: n=3
```

### Smoothing

**Problema: Zero Probabilities**
```python
# Si nunca vimos "love Python" en corpus
P("Python" | "love") = 0 / count("love") = 0
# Mata probabilidad de toda la oración
```

**Solución 1: Add-k Smoothing (Laplace)**
```python
P("Python" | "love") = (count("love Python") + k) / (count("love") + k*V)

# k=1: Laplace smoothing
# V = tamaño del vocabulario
```

**Solución 2: Interpolation**
```python
P(w_i | w_{i-1}) = λ₁P(w_i | w_{i-1}) + λ₂P(w_i) + λ₃(1/V)

# Combina trigram, bigram, unigram
# λ₁ + λ₂ + λ₃ = 1
```

**Solución 3: Backoff**
```python
if trigram_count > 0:
    use trigram
elif bigram_count > 0:
    use bigram
else:
    use unigram
```

---

## 📏 Evaluación: Perplexity {#perplexity}

### Concepto

Perplexity mide qué tan "sorprendido" está el modelo.

```
Perplexity = 2^H

donde H es la entropía:
H = -1/N Σ log₂ P(w_i | context)
```

**Interpretación:**
```
Perplejidad = 10 → En promedio, el modelo duda entre 10 palabras
Perplejidad = 100 → Duda entre 100 palabras

Menor perplejidad = Mejor modelo
```

### Implementación

```python
import numpy as np

def perplexity(model, test_sentences):
    """
    Calcula perplexity de un modelo en un test set
    """
    N = 0  # Total de palabras
    log_prob_sum = 0
    
    for sentence in test_sentences:
        words = ['<s>'] + sentence + ['</s>']
        
        for i in range(1, len(words)):
            prob = model.probability(words[i], words[i-1])
            
            if prob > 0:
                log_prob_sum += np.log2(prob)
            else:
                log_prob_sum += np.log2(1e-10)  # Evitar log(0)
            
            N += 1
    
    entropy = -log_prob_sum / N
    perplexity = 2 ** entropy
    
    return perplexity

# Uso
test_data = [['I', 'love', 'Python'], ['Python', 'is', 'fun']]
ppl = perplexity(model, test_data)
print(f"Perplexity: {ppl:.2f}")
```

### Comparación de Modelos

```python
# Unigram model
ppl_unigram = 247.3  # Peor

# Bigram model
ppl_bigram = 152.8   # Mejor

# Trigram model
ppl_trigram = 98.5   # Aún mejor

# Neural LM
ppl_neural = 45.2    # Mucho mejor
```

---

## 🧠 Neural Language Models {#neural}

### RNN Language Model

```python
import torch
import torch.nn as nn

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # RNN layer
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        
        # Embed
        embedded = self.embedding(x)  # (batch, seq_len, embed_size)
        
        # RNN
        output, hidden = self.rnn(embedded, hidden)  # (batch, seq_len, hidden_size)
        
        # Predict next word
        logits = self.fc(output)  # (batch, seq_len, vocab_size)
        
        return logits, hidden

# Crear modelo
vocab_size = 10000
model = RNNLanguageModel(vocab_size=vocab_size, embed_size=256, hidden_size=512)
```

### Entrenamiento

```python
import torch.nn.functional as F

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # batch: (batch_size, seq_len)
        inputs = batch[:, :-1]   # All except last
        targets = batch[:, 1:]   # All except first
        
        # Forward
        logits, _ = model(inputs)
        
        # Reshape for loss
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        
        # Loss
        loss = criterion(logits, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### LSTM vs Transformer

| Aspecto | LSTM | Transformer |
|---------|------|-------------|
| **Paralelo** | ❌ Secuencial | ✅ Paralelo |
| **Long-range** | ⚠️ Difícil | ✅ Attention |
| **Velocidad** | 🐢 Lento | ⚡ Rápido |
| **Performance** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 📝 Generación de Texto {#generación}

### Estrategias de Sampling

**1. Greedy Decoding:**
```python
# Siempre elige la palabra más probable
def greedy_generate(model, start_tokens, max_length):
    tokens = start_tokens
    
    for _ in range(max_length):
        logits, _ = model(tokens)
        next_token = logits[:, -1, :].argmax(dim=-1)
        tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
    
    return tokens
```

**Problema:** Repetitivo, predecible.

**2. Random Sampling:**
```python
# Samplea según probabilidades
def random_generate(model, start_tokens, max_length):
    tokens = start_tokens
    
    for _ in range(max_length):
        logits, _ = model(tokens)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    
    return tokens
```

**3. Temperature Sampling:**
```python
def temperature_sample(logits, temperature=1.0):
    """
    temperature < 1: Más conservador (picos más altos)
    temperature = 1: Normal
    temperature > 1: Más aleatorio (distribución más plana)
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# Uso
# temperature=0.5: "The cat sat on the mat"  # Conservador
# temperature=1.0: "The cat jumped over a fence"  # Normal
# temperature=1.5: "The purple elephant danced"  # Creativo
```

**4. Top-k Sampling:**
```python
def top_k_sample(logits, k=50):
    """
    Solo considera las k palabras más probables
    """
    # Obtener top-k
    top_k_logits, top_k_indices = torch.topk(logits, k)
    
    # Softmax solo en top-k
    probs = F.softmax(top_k_logits, dim=-1)
    
    # Sample
    next_token_index = torch.multinomial(probs, num_samples=1)
    next_token = top_k_indices.gather(-1, next_token_index)
    
    return next_token
```

**5. Top-p (Nucleus) Sampling:**
```python
def top_p_sample(logits, p=0.9):
    """
    Considera palabras hasta que probabilidad acumulada = p
    """
    # Sort
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remover tokens después de threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Zero out removed indices
    logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')
    
    # Sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## 💼 Aplicaciones {#aplicaciones}

### 1. Autocompletado

```python
# Gmail, Google Search, IDEs
def autocomplete(prefix, model, top_k=5):
    # Predecir siguiente(s) palabra(s)
    logits, _ = model(tokenize(prefix))
    probs = F.softmax(logits[:, -1, :], dim=-1)
    
    top_k_probs, top_k_indices = torch.topk(probs, top_k)
    
    suggestions = [detokenize(idx) for idx in top_k_indices[0]]
    
    return suggestions

# Ejemplo
suggestions = autocomplete("I love", model)
# ['Python', 'you', 'coding', 'it', 'this']
```

### 2. Corrección Ortográfica

```python
def spell_correct(sentence, model):
    words = sentence.split()
    
    for i, word in enumerate(words):
        # Generar candidatos (1 edit distance)
        candidates = generate_candidates(word)
        
        # Evaluar con LM
        best_candidate = word
        best_prob = 0
        
        for candidate in candidates:
            test_sentence = words[:i] + [candidate] + words[i+1:]
            prob = model.sentence_probability(test_sentence)
            
            if prob > best_prob:
                best_prob = prob
                best_candidate = candidate
        
        words[i] = best_candidate
    
    return ' '.join(words)
```

### 3. Generación de Texto

```python
# Generación creativa
prompt = "Once upon a time"
generated = model.generate(
    prompt,
    max_length=100,
    temperature=0.8,
    top_p=0.9
)
print(generated)
```

### 4. Machine Translation

```python
# LM en idioma target para fluency
def translate(source, translation_model, lm_target):
    # Generar múltiples traducciones
    candidates = translation_model.beam_search(source, beam_size=10)
    
    # Rerank con LM
    best_translation = None
    best_score = float('-inf')
    
    for candidate in candidates:
        # Score = translation_prob * lm_prob
        trans_prob = candidate.score
        lm_prob = lm_target.sentence_probability(candidate.text)
        
        score = trans_prob + 0.5 * lm_prob
        
        if score > best_score:
            best_score = score
            best_translation = candidate.text
    
    return best_translation
```

---

## 🎓 Resumen

**Conceptos Clave:**
- Language Models modelan P(secuencia de palabras)
- N-grams: unigram, bigram, trigram
- Perplexity mide calidad (menor = mejor)
- Neural LMs superan a N-grams
- Generación: greedy, sampling, temperature, top-k, top-p

**Evolución:**
```
N-grams (1990s) → RNN LMs (2010s) → Transformer LMs (2017+)
```

**Próximos Pasos:**
- **Koan 10**: Modern LLMs (GPT-4, Claude)
- **Koan 13**: RAG (usando LMs para generación)

¡Los Language Models son el corazón del NLP moderno! 🚀
