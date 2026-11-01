
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load the chunks we saved
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
print("Loaded", len(chunks), "chunks")

# 2. Embed every chunk once (this is our in-memory "index")
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [c["text"] for c in chunks]
vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
print("Index shape:", vecs.shape)   # (number_of_chunks, 384)


# 3. Search = embed query, dot product with all chunks, take top k
def search(query, k=3):
    q = model.encode(query, normalize_embeddings=True)
    scores = vecs @ q                         # one score per chunk
    for i in np.argsort(scores)[::-1][:k]:
        print(f"\n  {scores[i]:.3f} | page {chunks[i]['page']}")
        print(f"  {chunks[i]['text'][:200]}...")


# 4. Question loop - model loads once, answers instantly
while True:
    query = input("\nAsk (or 'quit'): ")
    if query.lower() == "quit":
        break
    search(query)