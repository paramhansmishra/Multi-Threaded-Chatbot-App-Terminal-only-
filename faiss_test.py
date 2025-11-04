from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Load a lightweight embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Create a few sample sentences
sentences = [
    "I love eating apples",
    "Python is a great programming language",
    "Apples are healthy and tasty",
    "I enjoy writing code in Python"
]

# Step 3: Convert sentences to embeddings (vectors)
embeddings = model.encode(sentences)
print("Embeddings shape:", embeddings.shape)

# Step 4: Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Step 5: Add embeddings to index
index.add(np.array(embeddings))

# Step 6: Query the index
query = "I like fruits"
query_emb = model.encode([query])
distances, indices = index.search(np.array(query_emb), k=2)

print("\nQuery:", query)
print("Top 2 similar sentences:")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {sentences[idx]} (distance: {distances[0][i]:.4f})")