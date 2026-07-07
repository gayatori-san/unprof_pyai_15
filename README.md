# 📚 FAISS Semantic Similarity Search
## 📌 Project Overview

This project demonstrates how to build a **Semantic Document Search System** using **Sentence Transformers** and **FAISS (Facebook AI Similarity Search)**.

Instead of searching documents based on exact keyword matches, semantic search understands the **meaning** of a sentence. Even if different words are used, the system can retrieve the most relevant document chunks.

Example:

**Query**

> Tell me about defending networks from hackers.

Instead of looking for the exact words, the model understands that the query relates to **Cybersecurity**, and retrieves the cybersecurity document chunks.

---

# 🎯 Objectives

* Learn semantic search.
* Generate sentence embeddings.
* Store embeddings inside a FAISS vector database.
* Perform fast similarity search.
* Retrieve the most relevant document chunks.

---

# 🛠 Technologies Used

* Python
* FAISS
* Sentence Transformers
* NumPy

---

# 📦 Required Libraries

Install the required packages:

```bash
pip install faiss-cpu
pip install sentence-transformers
pip install numpy
```

---

# 📂 Project Workflow

## Step 1 – Import Libraries

Import:

* faiss
* numpy
* SentenceTransformer

These libraries are used for vector storage and semantic embeddings.

---

## Step 2 – Load Embedding Model

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
```

The **all-MiniLM-L6-v2** model:

* Fast
* Lightweight
* Produces 384-dimensional embeddings
* Ideal for semantic search applications

---

## Step 3 – Create Document Chunks

The project creates **50 document chunks**.

Topics include:

* Vector Databases
* Python
* Cybersecurity
* Cloud Computing
* Genshin Impact

Each chunk contains a small variation so the dataset is not identical.

Example:

```
Document Chunk 3:
Cybersecurity involves protecting systems, networks, and programs from digital attacks.
```

---

## Step 4 – Generate Embeddings

Each document is converted into a numerical vector.

```python
embeddings = model.encode(documents)
```

Sentence Transformers convert text into dense vectors while preserving semantic meaning.

---

## Step 5 – Create FAISS Index

The embedding dimension is obtained automatically.

```python
dimension = embeddings.shape[1]
```

A FAISS **IndexFlatL2** is created.

```python
index = faiss.IndexFlatL2(dimension)
```

This index compares vectors using **Euclidean (L2) Distance**.

---

## Step 6 – Store Embeddings

All document vectors are inserted into the FAISS database.

```python
index.add(embeddings)
```

FAISS now contains all 50 document embeddings.

---

## Step 7 – User Query

Example query:

```
Tell me about defending networks from hackers.
```

The query is also converted into an embedding.

```python
query_embedding = model.encode([query])
```

---

## Step 8 – Similarity Search

Search for the **Top 3** nearest vectors.

```python
distances, indices = index.search(query_embedding, 3)
```

FAISS returns:

* Distances
* Document indices

The smaller the distance, the more similar the document.

---

## Step 9 – Display Results

Example output:

```
Top 3 Most Relevant Document Chunks

Rank 1
Distance: 0.48

Document Chunk 18:
Cybersecurity involves protecting systems,
networks, and programs from digital attacks.
```

---

# 🔄 Project Flow Diagram

```
Documents
     │
     ▼
Sentence Transformer
     │
     ▼
Embeddings (384 Dimensions)
     │
     ▼
FAISS Index
     │
     ▼
User Query
     │
     ▼
Query Embedding
     │
     ▼
FAISS Similarity Search
     │
     ▼
Top 3 Relevant Documents
```

---

# 📊 Example Output

```
Loading sentence-transformer model...

Generating embeddings for 50 chunks...

Successfully added 50 vectors to the FAISS index.

User Query:
Tell me about defending networks from hackers.

Top 3 Most Relevant Document Chunks

Rank 1
Distance: 0.4521

Document Chunk 23:
Cybersecurity involves protecting systems,
networks, and programs from digital attacks.

Rank 2
Distance: 0.4673

Document Chunk 48:
Cybersecurity involves protecting systems,
networks, and programs from digital attacks.

Rank 3
Distance: 0.4825

Document Chunk 13:
Cybersecurity involves protecting systems,
networks, and programs from digital attacks.
```

---

# 💡 Key Concepts Learned

* Semantic Search
* Vector Embeddings
* Sentence Transformers
* Dense Vector Representation
* FAISS Vector Database
* Euclidean (L2) Distance
* Similarity Search
* Information Retrieval
* Document Chunking
* Retrieval-Augmented Generation (RAG)

---

# ✅ Conclusion

This project demonstrates the fundamentals of **semantic document retrieval** using **Sentence Transformers** and **FAISS**. By converting documents into vector embeddings and storing them in a FAISS index, the system retrieves information based on meaning rather than exact keywords. This approach is widely used in modern AI applications, including **Retrieval-Augmented Generation (RAG)**, intelligent search engines, recommendation systems, and AI-powered assistants.
