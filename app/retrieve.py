import numpy as np
from embed import build_embedded_chunks, embed_text

def cosine_similarity(vec_a, vec_b):
    """Measures how similar two vectors are, from -1 (opposite) to 1 (identical).
    This is the standard way to compare embedding vectors."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)

def retrieve_top_chunks(question, chunks, top_k=3):
    """Given a question and a list of embedded chunks, returns the
    top_k chunks most similar in meaning to the question."""
    question_vector = embed_text(question)

    scored = []
    for chunk in chunks:
        score = cosine_similarity(question_vector, chunk["embedding"])
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored[:top_k]

    return [{"score": float(score), **chunk} for score, chunk in top_matches]

if __name__ == "__main__":
    print("Loading and embedding all chunks (may take a few seconds)...")
    chunks = build_embedded_chunks()

    test_question = "How do I book a service on Fixora?"
    print(f"\nQuestion: {test_question}\n")

    results = retrieve_top_chunks(test_question, chunks, top_k=3)
    for r in results:
        print(f"Score: {r['score']:.3f} | Source: {r['source']}")
        print(f"Text: {r['text']}\n")
        