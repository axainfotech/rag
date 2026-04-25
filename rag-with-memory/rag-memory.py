import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

load_dotenv()

# ── Config ───────────────────────────────────────────────
DOCS_DIR        = "documents"
CHUNK_SIZE      = 200
TOP_K           = 3
EMBED_MODEL     = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o-mini"
MAX_MEMORY      = 6          # remember last 6 exchanges (question + answer pairs)
# ─────────────────────────────────────────────────────────

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Memory store ─────────────────────────────────────────
# Each entry: {"role": "user"|"assistant", "content": "..."}
conversation_history = []

# ── 1. PDF loader ─────────────────────────────────────────
def extract_text_from_pdf(path):
    import fitz
    doc  = fitz.open(str(path))
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def load_documents(folder):
    chunks = []
    pdfs   = list(Path(folder).glob("*.pdf"))
    if not pdfs:
        print("  No PDFs found in documents/ — add a PDF and restart.")
        return chunks
    for path in pdfs:
        print(f"  Parsing {path.name}...")
        text  = extract_text_from_pdf(path)
        words = text.split()
        for i in range(0, len(words), CHUNK_SIZE):
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            chunks.append({"text": chunk, "source": path.name})
    print(f"  Loaded {len(chunks)} chunks from {len(pdfs)} PDF(s)")
    return chunks

# ── 2. Embeddings ─────────────────────────────────────────
def get_embedding(text):
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(response.data[0].embedding)

def build_vector_store(chunks):
    print("  Embedding chunks via OpenAI API...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        embeddings.append(get_embedding(chunk["text"]))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(chunks)} chunks embedded...")
    print(f"  Done — {len(embeddings)} embeddings built.")
    return embeddings

# ── 3. Memory helpers ─────────────────────────────────────
def add_to_memory(role, content):
    """Add a message to conversation history, keep only last MAX_MEMORY pairs."""
    conversation_history.append({"role": role, "content": content})
    # Keep only last MAX_MEMORY * 2 messages (each pair = user + assistant)
    if len(conversation_history) > MAX_MEMORY * 2:
        del conversation_history[0:2]   # remove oldest pair

def build_memory_context():
    """Format recent conversation history as a readable string for the prompt."""
    if not conversation_history:
        return "No previous conversation."
    lines = []
    for msg in conversation_history:
        prefix = "Student" if msg["role"] == "user" else "Assistant"
        lines.append(f"{prefix}: {msg['content']}")
    return "\n".join(lines)

def build_contextual_query(current_query):
    """
    Combine the current query with recent history to create a richer
    search query for the vector store. This helps resolve references like
    'its', 'that topic', 'tell me more' etc.
    """
    if not conversation_history:
        return current_query   # no history yet, use query as-is

    # Take last 2 exchanges max to build context
    recent = conversation_history[-4:]
    history_text = " ".join([m["content"] for m in recent])
    combined     = f"{history_text} {current_query}"
    return combined

# ── 4. Retrieval ──────────────────────────────────────────
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, chunks, embeddings, top_k=TOP_K):
    """Retrieve using a context-aware query (combines history + current query)."""
    contextual_query = build_contextual_query(query)
    q_emb   = get_embedding(contextual_query)
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

# ── 5. Answer generation with memory ─────────────────────
def generate_answer(query, retrieved_chunks):
    """
    Send full conversation history + retrieved context to GPT.
    GPT sees both what was said before AND what the documents say.
    """
    # Format retrieved chunks as context
    doc_context = "\n\n---\n\n".join(
        [f"[Source: {r['source']} | Similarity: {r['similarity']}]\n{r['text']}"
         for r in retrieved_chunks]
    )

    # Format conversation memory
    memory_context = build_memory_context()

    # System prompt — tells GPT about its role + how to use memory
    system_prompt = """You are a helpful KCET exam assistant with memory of our conversation.

Your behaviour:
- Use the DOCUMENT CONTEXT to answer factual questions
- Use CONVERSATION HISTORY to understand follow-up questions and references like 'it', 'that', 'tell me more', 'explain further'
- If a question refers to something from earlier in the conversation, connect the dots
- If the answer is not in the documents, say: "I don't have that in my knowledge base."
- Keep answers clear, concise, and exam-focused"""

    # Build the messages array: system + full history + new question with context
    messages = [{"role": "system", "content": system_prompt}]

    # Add previous conversation turns so GPT has full memory
    messages.extend(conversation_history)

    # Add the current question with document context injected
    current_message = f"""DOCUMENT CONTEXT (retrieved for this question):
{doc_context}

CONVERSATION SO FAR:
{memory_context}

NEW QUESTION: {query}"""

    messages.append({"role": "user", "content": current_message})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=600,
        messages=messages
    )
    return response.choices[0].message.content

# ── 6. Main loop ──────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("   Simple RAG with Memory — KCET Assistant")
    print("="*55)
    print("Commands:  'history' = show memory  |  'clear' = reset memory  |  'quit' = exit")
    print("="*55 + "\n")

    print("Loading documents...")
    chunks = load_documents(DOCS_DIR)
    if not chunks:
        return

    print("Building embeddings...")
    embeddings = build_vector_store(chunks)
    print(f"\nReady! Memory holds last {MAX_MEMORY} exchanges.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        # ── Special commands ──
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if query.lower() == "history":
            print("\n── Conversation memory ──")
            if not conversation_history:
                print("  (empty — no exchanges yet)")
            for i, msg in enumerate(conversation_history):
                prefix = "You" if msg["role"] == "user" else "Bot"
                print(f"  [{i+1}] {prefix}: {msg['content'][:100]}{'...' if len(msg['content'])>100 else ''}")
            print()
            continue

        if query.lower() == "clear":
            conversation_history.clear()
            print("  Memory cleared.\n")
            continue

        # ── Normal RAG + memory flow ──
        print("\nSearching knowledge base...")
        results = retrieve(query, chunks, embeddings)

        print("── Retrieved chunks ──")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['source']}] sim={r['similarity']} — {r['text'][:70]}...")

        print("\nAssistant: ", end="", flush=True)
        answer = generate_answer(query, results)
        print(answer)

        # ── Save this exchange to memory ──
        add_to_memory("user",      query)
        add_to_memory("assistant", answer)

        mem_count = len(conversation_history) // 2
        print(f"\n  [Memory: {mem_count}/{MAX_MEMORY} exchanges stored]\n")

if __name__ == "__main__":
    main()