# 📚 NLP Koans - Teoría Completa

## 🎯 Introducción

Este documento consolida toda la teoría de los 13 Koans de NLP, desde los fundamentos hasta las técnicas más avanzadas de IA moderna.

**Path de Aprendizaje:**
```
PARTE 1: Fundamentos (Koans 1-4)
  ↓ Tokenization → Normalization → POS Tagging → NER
PARTE 2: Aplicaciones Clásicas (Koans 5-6)
  ↓ Text Classification → Sentiment Analysis
PARTE 3: Representaciones (Koans 7-9)
  ↓ Word Embeddings → Transformers → Language Models
PARTE 4: NLP Moderna (Koans 10-13)
  ↓ Modern LLMs → AI Agents → Semantic Search → RAG
```

---

# 📖 PARTE 1: Fundamentos del NLP

## 1️⃣ Tokenization

### ¿Qué es Tokenization?

Dividir texto en unidades (tokens): palabras, subpalabras, caracteres.

```python
# Word tokenization
"I love Python" → ["I", "love", "Python"]

# Sentence tokenization
"Hello. How are you?" → ["Hello.", "How are you?"]

# Subword tokenization
"unhappiness" → ["un", "happiness"]
```

### Tipos de Tokenización

**1. Word Tokenization:**
```python
import nltk
nltk.download('punkt')

text = "I love Python!"
tokens = nltk.word_tokenize(text)
# ['I', 'love', 'Python', '!']
```

**2. Sentence Tokenization:**
```python
text = "Hello. How are you? I'm fine."
sentences = nltk.sent_tokenize(text)
# ['Hello.', 'How are you?', "I'm fine."]
```

**3. Subword Tokenization (BPE, WordPiece):**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("unhappiness")
# ['un', '##hap', '##pi', '##ness']
```

### Herramientas

| Herramienta | Velocidad | Multilingüe | Subword |
|-------------|-----------|-------------|---------|
| **NLTK** | ⚡ | ⚠️ | ❌ |
| **spaCy** | ⚡⚡⚡ | ✅ | ❌ |
| **Transformers** | ⚡⚡ | ✅ | ✅ |

---

## 2️⃣ Stemming & Lemmatization

### Normalización de Texto

**Stemming:** Corta palabras a raíz (algoritmo rápido, impreciso).
```python
running → run
runs → run
runner → runner  # ¡Error!
```

**Lemmatization:** Reduce a forma base léxica (preciso, requiere diccionario).
```python
running → run
runs → run
runner → runner  # ✓ Correcto (es un sustantivo)
```

### Algoritmos de Stemming

**Porter Stemmer:**
```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["running", "runs", "easily", "fairly"]
stems = [stemmer.stem(word) for word in words]
# ['run', 'run', 'easili', 'fairli']
```

**Lancaster Stemmer (más agresivo):**
```python
from nltk.stem import LancasterStemmer

stemmer = LancasterStemmer()
print(stemmer.stem("running"))  # 'run'
print(stemmer.stem("maximum"))  # 'maxim'
```

### Lemmatization

```python
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Con POS tag
print(lemmatizer.lemmatize("running", pos='v'))  # run
print(lemmatizer.lemmatize("better", pos='a'))   # good
```

### Comparación

| Aspecto | Stemming | Lemmatization |
|---------|----------|---------------|
| **Velocidad** | ⚡⚡⚡ Rápido | ⚡ Lento |
| **Precisión** | ⚠️ Baja | ✅ Alta |
| **Diccionario** | ❌ No | ✅ Sí |
| **Uso** | Search, IR | NLU, QA |

---

## 3️⃣ POS Tagging

### Part-of-Speech Tagging

Asignar categoría gramatical a cada palabra.

```python
"I love Python"
I    → PRP (pronoun)
love → VBP (verb)
Python → NNP (proper noun)
```

### Tagsets

**Penn Treebank (45 tags):**
```
NN:   Noun singular
NNS:  Noun plural
VB:   Verb base form
VBD:  Verb past tense
JJ:   Adjective
...
```

**Universal Dependencies (17 tags):**
```
NOUN, VERB, ADJ, ADV, PRON, DET, ADP, NUM, CONJ, ...
```

### Herramientas

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I love Python")

for token in doc:
    print(f"{token.text:10} → {token.pos_:5} ({token.tag_})")
# I          → PRON  (PRP)
# love       → VERB  (VBP)
# Python     → PROPN (NNP)
```

### Algoritmos

1. **Hidden Markov Models (HMM)**
2. **Maximum Entropy (MaxEnt)**
3. **Conditional Random Fields (CRF)**
4. **Deep Learning (BiLSTM, Transformers)**

---

## 4️⃣ Named Entity Recognition

### ¿Qué es NER?

Identificar y clasificar entidades nombradas en texto.

```python
"Apple CEO Tim Cook visited Paris"

Apple  → ORGANIZATION
Tim Cook → PERSON
Paris   → LOCATION
```

### Tipos de Entidades

**OntoNotes 5.0:**
```
PERSON, ORGANIZATION, GPE (Geo-Political Entity),
DATE, TIME, MONEY, PERCENT, QUANTITY, ...
```

**CoNLL 2003:**
```
PER (Person), ORG (Organization), LOC (Location), MISC
```

### BIO Tagging

```
Apple  → B-ORG  (Begin Organization)
CEO    → O      (Outside)
Tim    → B-PER  (Begin Person)
Cook   → I-PER  (Inside Person)
```

### Implementación

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple CEO Tim Cook visited Paris on June 5th")

for ent in doc.ents:
    print(f"{ent.text:15} → {ent.label_}")

# Apple          → ORG
# Tim Cook       → PERSON
# Paris          → GPE
# June 5th       → DATE
```

### Herramientas

| Herramienta | Precisión | Velocidad | Multilingüe |
|-------------|-----------|-----------|-------------|
| **spaCy** | ⭐⭐⭐⭐ | ⚡⚡⚡ | ✅ |
| **Stanford NER** | ⭐⭐⭐⭐ | ⚡⚡ | ✅ |
| **Transformers** | ⭐⭐⭐⭐⭐ | ⚡ | ✅ |

---

# 📊 PARTE 2: Aplicaciones Clásicas

## 5️⃣ Text Classification

### Clasificación de Documentos

Asignar categoría(s) a un documento.

```python
"This product is amazing!" → POSITIVE
"Spam: Win money now!"     → SPAM
"Python tutorial"          → TECHNOLOGY
```

### Feature Engineering

**1. Bag of Words (BoW):**
```python
from sklearn.feature_extraction.text import CountVectorizer

texts = ["I love Python", "I love Java"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# [[1, 1, 0, 1],   # "I love Python"
#  [1, 0, 1, 1]]   # "I love Java"
#  I  Java love Python
```

**2. TF-IDF:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Da más peso a palabras raras
# "Python" tiene mayor peso que "I"
```

### Modelos Clásicos

**Naive Bayes:**
```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Logistic Regression:**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

**SVM:**
```python
from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(X_train, y_train)
```

### Pipeline Completo

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Evaluación

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

#              precision    recall  f1-score   support
#     POSITIVE       0.88      0.92      0.90       100
#     NEGATIVE       0.91      0.87      0.89       100
```

---

## 6️⃣ Sentiment Analysis

### Análisis de Sentimiento

Determinar emoción/polaridad en texto.

```python
"I love this product!" → POSITIVE (0.95)
"Terrible experience"  → NEGATIVE (0.88)
"It's okay"            → NEUTRAL  (0.60)
```

### Enfoques

**1. Lexicon-based (VADER):**
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores("I love this!")

print(scores)
# {'neg': 0.0, 'neu': 0.323, 'pos': 0.677, 'compound': 0.6369}
```

**2. Machine Learning:**
```python
from sklearn.linear_model import LogisticRegression

# Features: TF-IDF
# Labels: positive/negative
model = LogisticRegression()
model.fit(X_train, y_train)
```

**3. Deep Learning (Transformers):**
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")

print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

### Niveles de Análisis

**1. Document-level:**
```python
"I love this product!" → POSITIVE
```

**2. Sentence-level:**
```python
"The movie was great. But the ending sucked."
→ Sentence 1: POSITIVE
→ Sentence 2: NEGATIVE
```

**3. Aspect-based:**
```python
"The food was great but service was terrible"
→ food: POSITIVE
→ service: NEGATIVE
```

---

# 🧮 PARTE 3: Representaciones Vectoriales

## 7️⃣ Word Embeddings

### Dense Vector Representations

Representar palabras como vectores densos.

```python
# One-hot (sparse)
"cat" → [0, 0, 1, 0, 0, ..., 0]  # 10,000 dims

# Embedding (dense)
"cat" → [0.2, -0.4, 0.1, ...]  # 300 dims
```

### Word2Vec

**CBOW (Continuous Bag of Words):**
```
Context → Target
"I ___ Python" → "love"
```

**Skip-gram:**
```
Target → Context
"love" → "I", "Python"
```

**Implementación:**
```python
from gensim.models import Word2Vec

sentences = [["I", "love", "Python"], ["Python", "is", "great"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Similitud
similarity = model.wv.similarity("Python", "programming")

# Analogía
result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
# queen
```

### GloVe

Global Vectors - basado en co-ocurrencias.

```python
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Convertir GloVe a Word2Vec format
glove2word2vec("glove.6B.100d.txt", "word2vec.txt")

# Cargar
model = KeyedVectors.load_word2vec_format("word2vec.txt")
```

### FastText

Word2Vec + información de subpalabras.

```python
from gensim.models import FastText

model = FastText(sentences, vector_size=100, window=5)

# Puede manejar OOV (Out-of-Vocabulary)
vector = model.wv["unfindableword"]  # ✓ Funciona
```

### Propiedades Mágicas

**1. Similitud Semántica:**
```python
similarity("cat", "dog") = 0.8  # Alta
similarity("cat", "car") = 0.2  # Baja
```

**2. Analogías:**
```python
king - man + woman ≈ queen
Paris - France + Spain ≈ Madrid
```

**3. Clustering:**
```python
# Palabras similares se agrupan
cluster_1: [cat, dog, animal, pet]
cluster_2: [car, vehicle, truck]
```

---

## 8️⃣ Transformers

### Revolución del NLP

**Attention is All You Need** (2017)

**Antes (RNN/LSTM):**
- Procesamiento secuencial → lento
- Difícil capturar dependencias largas

**Después (Transformers):**
- Procesamiento paralelo → rápido
- Self-attention → captura todo el contexto

### Arquitectura

```
INPUT
  ↓
Embeddings + Positional Encoding
  ↓
ENCODER (N capas)
  - Multi-Head Attention
  - Feed Forward
  - Layer Norm
  ↓
DECODER (N capas)
  - Masked Attention
  - Cross Attention
  - Feed Forward
  ↓
OUTPUT
```

### Self-Attention

Cada palabra "atiende" a todas las demás.

```python
"The cat sat on the mat"

# "sat" atiende a:
# - "cat" (quién) ✅ Alta atención
# - "mat" (dónde) ✅ Alta atención
# - "the" ❌ Baja atención
```

**Cálculo:**
```
1. Q, K, V = linear(input)
2. Scores = Q · K^T / √d_k
3. Weights = softmax(Scores)
4. Output = Weights · V
```

### BERT

**Bidirectional Encoder Representations from Transformers**

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, how are you?", return_tensors='pt')
outputs = model(**inputs)

embeddings = outputs.last_hidden_state  # (1, seq_len, 768)
```

**Pre-training:**
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)

### GPT

**Generative Pre-trained Transformer**

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

input_text = "Once upon a time"
inputs = tokenizer.encode(input_text, return_tensors='pt')

outputs = model.generate(inputs, max_length=50)
generated = tokenizer.decode(outputs[0])
print(generated)
```

**Características:**
- Decoder-only
- Auto-regresivo (predice siguiente palabra)
- Causal Language Modeling

### Comparación

| Modelo | Arquitectura | Uso Principal |
|--------|--------------|---------------|
| **BERT** | Encoder-only | Clasificación, NER, QA |
| **GPT** | Decoder-only | Generación de texto |
| **T5** | Encoder-Decoder | Versátil (todo es text-to-text) |

---

## 9️⃣ Language Models

### ¿Qué es un LM?

Modelo que asigna probabilidades a secuencias.

```python
P("I love Python") = 0.001  # Probable
P("Python love I") = 0.00001  # Improbable
```

### N-gram Models

**Unigram:**
```python
P("I love Python") = P("I") × P("love") × P("Python")
```

**Bigram:**
```python
P("I love Python") = P("I") × P("love"|"I") × P("Python"|"love")
```

**Trigram:**
```python
P("I love Python") = P("I") × P("love"|"I") × P("Python"|"I love")
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
            words = ['<s>'] + sentence + ['</s>']
            
            for i in range(len(words) - 1):
                self.unigram_counts[words[i]] += 1
                self.bigram_counts[words[i]][words[i+1]] += 1
    
    def probability(self, word, prev_word):
        return self.bigram_counts[prev_word][word] / self.unigram_counts[prev_word]
```

### Perplexity

Mide qué tan "sorprendido" está el modelo.

```python
Perplexity = 2^H

H = -1/N Σ log₂ P(w_i | context)
```

**Interpretación:**
- Menor perplejidad = Mejor modelo
- Perplejidad = 10 → Duda entre 10 palabras
- Perplejidad = 100 → Duda entre 100 palabras

### Neural Language Models

```python
import torch.nn as nn

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        logits = self.fc(output)
        return logits
```

### Generación de Texto

**Estrategias:**

1. **Greedy:** Siempre la más probable
2. **Random Sampling:** Según probabilidades
3. **Temperature:** Controla aleatoriedad
4. **Top-k:** Solo k más probables
5. **Top-p (Nucleus):** Probabilidad acumulada

```python
# Temperature sampling
def sample(logits, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)

# temperature=0.5: Conservador
# temperature=1.0: Normal
# temperature=1.5: Creativo
```

---

# 🚀 PARTE 4: NLP Moderna

## 🔟 Modern LLMs

### Large Language Models

**Evolución:**
```
GPT-1 (2018):   117M parámetros
GPT-2 (2019):   1.5B parámetros
GPT-3 (2020):   175B parámetros
GPT-4 (2023):   ~1.7T parámetros
```

### Características

**1. Few-Shot Learning:**
```python
# Sin fine-tuning, solo ejemplos en el prompt

prompt = """
Translate to Spanish:
English: Hello → Spanish: Hola
English: Thank you → Spanish: Gracias
English: Good morning → Spanish:
"""

# Modelo completa: "Buenos días"
```

**2. Chain-of-Thought:**
```python
prompt = """
Question: Roger has 5 balls. He buys 2 more cans of 3 balls each. 
How many balls does he have?

Let's think step by step:
1. Roger starts with 5 balls
2. He buys 2 cans of 3 balls each: 2 × 3 = 6 balls
3. Total: 5 + 6 = 11 balls

Answer: 11
"""
```

### APIs

**OpenAI:**
```python
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

print(response.choices[0].message.content)
```

**Anthropic Claude:**
```python
import anthropic

client = anthropic.Client(api_key="...")
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ]
)
```

### Prompting Techniques

**1. Zero-Shot:**
```
"Classify sentiment: I love this product!"
```

**2. Few-Shot:**
```
"I love this → POSITIVE
I hate this → NEGATIVE
It's okay → NEUTRAL
This is amazing →"
```

**3. Chain-of-Thought:**
```
"Let's solve this step by step..."
```

**4. ReAct (Reasoning + Acting):**
```
Thought: I need to search for information
Action: search("quantum computing")
Observation: [results]
Thought: Now I can answer
Answer: ...
```

---

## 1️⃣1️⃣ AI Agents

### Agentes Autónomos

Sistemas que pueden:
1. Razonar sobre tareas
2. Planificar acciones
3. Usar herramientas
4. Ejecutar y verificar

### Arquitectura

```
USER QUERY
    ↓
REASONING ENGINE (LLM)
    ↓
PLANNING
    ↓
TOOL SELECTION
    ↓
[Calculator] [Search] [Code] [Database]
    ↓
EXECUTION
    ↓
VERIFICATION
    ↓
RESPONSE
```

### Ejemplo: LangChain Agent

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

# Definir herramientas
tools = [
    Tool(
        name="Calculator",
        func=lambda x: eval(x),
        description="Para cálculos matemáticos"
    ),
    Tool(
        name="Search",
        func=search_function,
        description="Para buscar información"
    )
]

# Crear agente
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Ejecutar
result = agent.run("What is 25% of 480?")
```

### ReAct Pattern

```
Question: What is the capital of the country where the Eiffel Tower is located?

Thought 1: I need to find where the Eiffel Tower is
Action 1: Search("Eiffel Tower location")
Observation 1: The Eiffel Tower is in Paris, France

Thought 2: Now I need the capital of France
Action 2: Search("capital of France")
Observation 2: Paris is the capital of France

Thought 3: I can now answer
Answer: Paris
```

### Herramientas

**LangChain:**
```python
from langchain.agents import create_react_agent
from langchain.tools import Tool

# Definir tools
tools = [...]

# Crear agente
agent = create_react_agent(llm, tools, prompt)

# Ejecutar
agent.invoke({"input": "user query"})
```

**AutoGPT:**
```python
# Agente autónomo con objetivos de largo plazo
autogpt = AutoGPT(
    goal="Build a website for my business",
    max_iterations=10
)

autogpt.run()
```

---

## 1️⃣2️⃣ Semantic Search

### Búsqueda Semántica

Buscar por significado, no por palabras exactas.

```python
# Búsqueda tradicional (keyword)
Query: "Python programming"
Results: Documentos con "Python" y "programming"

# Búsqueda semántica
Query: "Learn to code in Python"
Results: 
- "Python tutorial for beginners" ✅
- "How to program in Python" ✅
- "Python programming guide" ✅
# Incluso sin palabras exactas
```

### Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Embeddings
query = "Python programming"
query_emb = model.encode(query)

docs = ["Python tutorial", "Java guide", "Machine learning"]
doc_embs = model.encode(docs)

# Similitud
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity([query_emb], doc_embs)[0]

# Ranking
for doc, sim in zip(docs, similarities):
    print(f"{sim:.3f} - {doc}")

# 0.872 - Python tutorial
# 0.543 - Machine learning
# 0.421 - Java guide
```

### Vector Databases

**Pinecone:**
```python
import pinecone

# Conectar
pinecone.init(api_key="...")
index = pinecone.Index("my-index")

# Insertar
index.upsert(vectors=[
    ("id1", embedding1, {"text": "Python tutorial"}),
    ("id2", embedding2, {"text": "Java guide"}),
])

# Buscar
results = index.query(query_embedding, top_k=5)
```

**FAISS:**
```python
import faiss
import numpy as np

# Crear índice
dimension = 768
index = faiss.IndexFlatL2(dimension)

# Añadir vectores
embeddings = np.array([emb1, emb2, emb3])
index.add(embeddings)

# Buscar
distances, indices = index.search(query_embedding, k=5)
```

### Hybrid Search

Combinar keyword + semantic.

```python
# Score final = α × keyword_score + (1-α) × semantic_score

def hybrid_search(query, alpha=0.5):
    # BM25 (keyword)
    keyword_scores = bm25_search(query)
    
    # Vector search (semantic)
    semantic_scores = vector_search(query)
    
    # Combinar
    final_scores = alpha * keyword_scores + (1 - alpha) * semantic_scores
    
    return rank_by_score(final_scores)
```

---

## 1️⃣3️⃣ RAG (Retrieval-Augmented Generation)

### Concepto

Combinar recuperación de información + generación de LLM.

```
USER QUERY
    ↓
RETRIEVE relevant documents
    ↓
AUGMENT prompt with context
    ↓
GENERATE response with LLM
```

### ¿Por qué RAG?

**Problemas de LLMs:**
- Conocimiento desactualizado
- Alucinaciones
- Sin acceso a datos privados

**Solución RAG:**
```python
# Sin RAG
Query: "What is our company's vacation policy?"
LLM: [Alucina respuesta genérica]

# Con RAG
Query: "What is our company's vacation policy?"
1. Retrieve: [Company handbook section on vacations]
2. Augment: "Based on this context: [context], answer: [query]"
3. Generate: [Respuesta basada en documento real]
```

### Implementación Básica

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. Cargar documentos
loader = TextLoader("company_docs.txt")
documents = loader.load()

# 2. Split en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3. Crear embeddings y vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Crear RAG chain
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 5. Query
response = qa_chain.run("What is the vacation policy?")
print(response)
```

### Pipeline Completo

```python
class RAGSystem:
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
    
    def retrieve(self, query, k=5):
        """Recuperar documentos relevantes"""
        return self.vectorstore.similarity_search(query, k=k)
    
    def augment_prompt(self, query, documents):
        """Crear prompt con contexto"""
        context = "\n\n".join([doc.page_content for doc in documents])
        
        prompt = f"""
        Based on the following context, answer the question.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
        
        return prompt
    
    def generate(self, prompt):
        """Generar respuesta"""
        return self.llm(prompt)
    
    def query(self, question):
        """Pipeline completo"""
        # 1. Retrieve
        docs = self.retrieve(question)
        
        # 2. Augment
        prompt = self.augment_prompt(question, docs)
        
        # 3. Generate
        response = self.generate(prompt)
        
        return response, docs  # Incluir fuentes
```

### Técnicas Avanzadas

**1. Re-ranking:**
```python
# Retrieve más documentos, luego re-rankear
docs = retrieve(query, k=20)
reranked = rerank_model(query, docs)
top_docs = reranked[:5]
```

**2. Hypothetical Document Embeddings (HyDE):**
```python
# Generar respuesta hipotética, luego buscar
hypothetical_answer = llm.generate(query)
relevant_docs = search(hypothetical_answer)
```

**3. Multi-Query:**
```python
# Generar múltiples queries, combinar resultados
queries = generate_variants(original_query)
all_docs = [retrieve(q) for q in queries]
combined = deduplicate_and_rank(all_docs)
```

### Evaluación

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Evaluar
results = evaluate(
    dataset=test_dataset,
    metrics=[faithfulness, answer_relevancy]
)

print(results)
# faithfulness: 0.92      (respuesta fiel al contexto)
# answer_relevancy: 0.88  (respuesta relevante a query)
```

---

# 🎓 Resumen Final

## Evolución del NLP

```
1990s: Rule-based + N-grams
2000s: Statistical ML (Naive Bayes, SVM)
2013: Word Embeddings (Word2Vec)
2017: Transformers (BERT, GPT)
2020: Large Language Models (GPT-3)
2023: Multimodal & Agents (GPT-4, Claude)
2024: RAG & Specialized LLMs
```

## Stack Moderno

**Fundamentos:**
```python
spaCy → Tokenization, POS, NER
NLTK → Procesamiento básico
```

**Embeddings:**
```python
Sentence Transformers → Semantic search
```

**LLMs:**
```python
OpenAI API / Anthropic → Generación
Hugging Face Transformers → Fine-tuning
```

**Frameworks:**
```python
LangChain → RAG, Agents
LlamaIndex → RAG avanzado
```

**Vector DBs:**
```python
Pinecone / FAISS / Chroma → Búsqueda semántica
```

## Roadmap de Aprendizaje

**Nivel 1 - Fundamentos (Koans 1-4):**
1. Tokenization
2. Stemming & Lemmatization
3. POS Tagging
4. NER

**Nivel 2 - Aplicaciones (Koans 5-6):**
5. Text Classification
6. Sentiment Analysis

**Nivel 3 - Representaciones (Koans 7-9):**
7. Word Embeddings
8. Transformers
9. Language Models

**Nivel 4 - NLP Moderna (Koans 10-13):**
10. Modern LLMs
11. AI Agents
12. Semantic Search
13. RAG

## Recursos

**Papers Clave:**
- Word2Vec: "Efficient Estimation of Word Representations"
- GloVe: "Global Vectors for Word Representation"
- Transformers: "Attention is All You Need"
- BERT: "Pre-training of Deep Bidirectional Transformers"
- GPT-3: "Language Models are Few-Shot Learners"

**Cursos:**
- Stanford CS224N (NLP with Deep Learning)
- fast.ai (NLP)
- DeepLearning.AI (LLM courses)

**Libros:**
- "Speech and Language Processing" (Jurafsky & Martin)
- "Natural Language Processing with Transformers"
- "Build a Large Language Model (From Scratch)"

---

¡Felicidades por completar el path de NLP Koans! 🎉🚀

Este documento cubre desde los fundamentos hasta las técnicas más avanzadas del NLP moderno. Usa cada Koan como punto de profundización práctica.

**Next Steps:**
1. Practica con cada Koan (tests)
2. Construye proyectos reales
3. Explora papers recientes
4. Contribuye a open source

¡El NLP está en constante evolución - sigue aprendiendo! 📚✨
