import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, k=4):
    if not os.path.exists("vector_store.index"):
        print("❌ No vector store found. Run create_embeddings.py first.")
        return []

    index = faiss.read_index("vector_store.index")
    with open("vector_store.meta", "rb") as f:
        all_chunks = pickle.load(f)

    qvec = embedder.encode([query])
    D, I = index.search(np.array(qvec, dtype="float32"), k)
    return [all_chunks[i] for i in I[0] if i < len(all_chunks)]

