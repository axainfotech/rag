import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Config ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DOCS_DIR    = BASE_DIR / "documents"
CHUNK_SIZE  = 200                      # words per chunk
TOP_K       = 3                        # how many chunks to retrieve
EMBED_MODEL = "text-embedding-3-small" # OpenAI embedding model
CHAT_MODEL  = "gpt-4o-mini"            # OpenAI chat model (cheap + fast)
# ────────────────────────────────────────────────────────

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── 1. Load & chunk documents ────────────────────────────
def extract_text_from_pdf(path):
    import fitz  # PyMuPDF
    doc  = fitz.open(str(path))
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def load_documents(folder):
    chunks = []
    folder = Path(folder)
    text_files = sorted(folder.glob("*.txt"))
    pdf_files = sorted(folder.glob("*.pdf"))

    for path in text_files:
        text = path.read_text(encoding="utf-8")
        words = text.split()
        for i in range(0, len(words), CHUNK_SIZE):
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            chunks.append({"text": chunk, "source": path.name})

    for path in pdf_files:
        text = extract_text_from_pdf(path)
        words = text.split()
        for i in range(0, len(words), CHUNK_SIZE):
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            chunks.append({"text": chunk, "source": path.name})

    print(
        f"  Loaded {len(chunks)} chunks from {folder}/ "
        f"({len(text_files)} txt, {len(pdf_files)} pdf)"
    )
    return chunks

# ── 2. Build in-memory vector store ─────────────────────
def get_embedding(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return np.array(response.data[0].embedding)

def build_vector_store(chunks):
    print("  Embedding chunks via OpenAI API...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        embeddings.append(get_embedding(chunk["text"]))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(chunks)} chunks embedded...")
    print(f"  Done — {len(embeddings)} embeddings built.")
    return embeddings

# ── 3. Cosine similarity search ──────────────────────────
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, chunks, embeddings, top_k=TOP_K):
    q_emb   = get_embedding(query)
    scores  = [cosine_similarity(q_emb, e) for e in embeddings]
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for i in top_idx:
        results.append({
            "text":       chunks[i]["text"],
            "source":     chunks[i]["source"],
            "similarity": round(float(scores[i]), 4)
        })
    return results

# ── 4. Generate answer with OpenAI ──────────────────────
def generate_answer(query, retrieved_chunks):
    context = "\n\n---\n\n".join(
        [f"[Source: {r['source']} | Similarity: {r['similarity']}]\n{r['text']}"
         for r in retrieved_chunks]
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Recruiter assistant. Use ONLY the context provided "
                    "to answer the question. If the answer is not in the context, "
                    "say: 'I don't have enough information in my knowledge base.'"
                )
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer clearly and concisely:"
            }
        ]
    )
    return response.choices[0].message.content

def build_index():
    print("Loading documents...")
    chunks = load_documents(DOCS_DIR)
    if not chunks:
        return [], []

    print("Building embeddings via OpenAI API...")
    embeddings = build_vector_store(chunks)
    return chunks, embeddings

# ── 5. Main loop ─────────────────────────────────────────
def main():
    print("\n=== Simple RAG — Recruiter Assistant (OpenAI) ===\n")
    print("Commands: 'reload' = re-index documents | 'quit' = exit\n")
    chunks, embeddings = build_index()
    if not chunks:
        return
    print("\nReady! Type your question (or 'quit' to exit)\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower() == "reload":
            print("\nReloading documents and rebuilding embeddings...")
            chunks, embeddings = build_index()
            if not chunks:
                print("\nNo documents found. Add .txt or .pdf files to documents/ and reload.\n")
                continue
            print("\nReload complete.\n")
            continue
        if not query:
            continue

        print("\nSearching knowledge base...")
        results = retrieve(query, chunks, embeddings)

        print("\n--- Retrieved chunks ---")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['source']}] similarity={r['similarity']} — {r['text'][:80]}...")

        print("\n--- Answer ---")
        answer = generate_answer(query, results)
        print(answer)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()