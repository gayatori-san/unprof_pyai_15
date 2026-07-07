import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Initialize the embedding model
# 'all-MiniLM-L6-v2' is fast, lightweight, and perfect for semantic search tasks
print("Loading sentence-transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Create 50 Document Chunks
# We'll generate chunks based on a few different themes so the search has varied data.
themes = [
    "Vector databases like FAISS are essential for modern AI applications and RAG systems.",
    "Python is widely used in data science, machine learning, and artificial intelligence.",
    "Cybersecurity involves protecting systems, networks, and programs from digital attacks.",
    "Cloud computing platforms like Google Cloud provide scalable infrastructure for developers.",
    "Genshin Impact features a complex elemental combat system and deep regional lore."
]

documents = []
for i in range(50):
    theme = themes[i % len(themes)]
    documents.append(f"Document Chunk {i+1}: {theme} This is variation {i}.")

# 3. Generate embeddings for each chunk
print("Generating embeddings for 50 chunks...")
# model.encode returns a numpy array of embeddings
embeddings = model.encode(documents)

# 4. Store embeddings in a FAISS Index
# Get the dimension of the embeddings (384 for all-MiniLM-L6-v2)
dimension = embeddings.shape[1]

# IndexFlatL2 measures the L2 (Euclidean) distance between vectors
index = faiss.IndexFlatL2(dimension)

# Add the vectors to the FAISS index
index.add(embeddings)
print(f"Successfully added {index.ntotal} vectors to the FAISS index.\n")

# 5. Accept a user query and perform Similarity Search
query = "Tell me about defending networks from hackers."
print(f"🔍 User Query: '{query}'\n")

# Generate an embedding for the query
query_embedding = model.encode([query])

# Perform the search (k = top 3 most relevant results)
k = 3
distances, indices = index.search(query_embedding, k)

# 6. Output the results
print("🎯 Top 3 Most Relevant Document Chunks:")
for i in range(k):
    doc_index = indices[0][i]
    distance = distances[0][i]
    print(f"\nRank {i+1} (Distance: {distance:.4f}):")
    print(f"📄 {documents[doc_index]}")