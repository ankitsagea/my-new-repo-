import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def load_documents():
    """Reads every .txt file in the data folder and returns a list of
    (filename, full_text) pairs."""
    documents = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            documents.append((filename, text))
    return documents

def chunk_text(text, max_words=40):
    """Splits text into chunks of roughly max_words words each,
    breaking on sentence boundaries where possible."""
    sentences = text.replace("\n", " ").split(". ")
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        word_count = len(sentence.split())
        if current_word_count + word_count > max_words and current_chunk:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = []
            current_word_count = 0
        current_chunk.append(sentence.rstrip("."))
        current_word_count += word_count

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks

def get_all_chunks():
    """Returns a list of dicts, each with the chunk text and which
    file it came from — this is what embed.py and retrieve.py will use."""
    all_chunks = []
    for filename, text in load_documents():
        for chunk in chunk_text(text):
            all_chunks.append({"source": filename, "text": chunk})
    return all_chunks

if __name__ == "__main__":
    chunks = get_all_chunks()
    print(f"Loaded {len(chunks)} chunks from {len(load_documents())} documents.\n")
    for i, c in enumerate(chunks):
        print(f"[{i}] ({c['source']}): {c['text']}\n")