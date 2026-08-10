import json
import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

json_files = os.listdir("jsons")
all_chunks = []

for jf in json_files:
    try:
        with open(f"jsons/{jf}", "r", encoding="utf-8") as f:
            data = json.load(f)
        for chunk in data["chunks"]:
            all_chunks.append(chunk)
    except Exception as e:
        print(f"⚠️ Skipping unreadable file '{jf}': {e}")
        continue

if not all_chunks:
    print("❌ No chunks found. Did Steps 3 and 4 run successfully?")
else:
    texts = [c["text"] for c in all_chunks]
    vectors = embedder.encode(texts, show_progress_bar=True, batch_size=16)

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors, dtype="float32"))

    faiss.write_index(index, "vector_store.index")
    with open("vector_store.meta", "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"✅ Stored {len(all_chunks)} chunks into FAISS.")