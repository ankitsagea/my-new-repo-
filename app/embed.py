from sentence_transformers import SentenceTransformer
import numpy as np
from ingest import get_all_chunks

# This loads a small, free model that turns text into vectors.
# The first time you run this, it downloads the model (~80MB) — that's normal.
_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text):
    """Turns a single piece of text into a vector (list of numbers)."""
    return _model.encode(text)

def build_embedded_chunks():
    """Takes every chunk from ingest.py and attaches its embedding vector."""
    chunks = get_all_chunks()
    for chunk in chunks:
        chunk["embedding"] = embed_text(chunk["text"])
    return chunks

if __name__ == "__main__":
    chunks = build_embedded_chunks()
    print(f"Embedded {len(chunks)} chunks.\n")
    first = chunks[0]
    print(f"Example — source: {first['source']}")
    print(f"Text: {first['text']}")
    print(f"Embedding vector length: {len(first['embedding'])}")
    print(f"First 5 numbers: {first['embedding'][:5]}")
    