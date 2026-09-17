import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "LiquidAI/lfm2.5-1.2b-instruct"

def generate_answer(question, retrieved_chunks):
    """Builds a prompt from the question + retrieved context, sends it
    to the local Ollama model, and returns the answer text."""

    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
    )

    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so clearly. After your answer, list which source file(s) you used.

Context:
{context_text}

Question: {question}

Answer:"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()
    result = response.json()
    return result["response"]

if __name__ == "__main__":
    from retrieve import retrieve_top_chunks
    from embed import build_embedded_chunks

    print("Loading and embedding chunks...")
    chunks = build_embedded_chunks()

    test_question = "How do I book a service on Fixora?"
    print(f"\nQuestion: {test_question}\n")

    top_chunks = retrieve_top_chunks(test_question, chunks, top_k=3)
    answer = generate_answer(test_question, top_chunks)

    print("Answer:")
    print(answer)