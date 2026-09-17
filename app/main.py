from ingest import get_all_chunks
from embed import build_embedded_chunks
from retrieve import retrieve_top_chunks
from generate import generate_answer

def main():
    print("Loading and embedding documents (one-time setup)...")
    chunks = build_embedded_chunks()
    print(f"Ready. {len(chunks)} chunks loaded from your knowledge base.\n")
    print("Type a question and press Enter. Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if not question:
            continue

        top_chunks = retrieve_top_chunks(question, chunks, top_k=3)
        answer = generate_answer(question, top_chunks)

        print("\nAnswer:")
        print(answer)
        print("\nRetrieved from:", ", ".join(sorted(set(c["source"] for c in top_chunks))))
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()