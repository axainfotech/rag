import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import random

load_dotenv()   # ← must be before OpenAI()

client = OpenAI()
EMBED_MODEL = "text-embedding-3-small"

# ── shared embedding helper ───────────────────────────────
def get_embedding(text):
    r = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(r.data[0].embedding)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ── Tool 1: search your PDF knowledge base ────────────────
def search_docs(query: str, chunks: list, embeddings: list) -> str:
    """Searches PDF documents for relevant content."""
    if not chunks:
        return "No documents loaded."
    q_emb  = get_embedding(query)
    scores = [cosine(q_emb, e) for e in embeddings]
    top3   = np.argsort(scores)[::-1][:3]
    result = []
    for i in top3:
        result.append(f"[sim={scores[i]:.2f} | {chunks[i]['source']}]\n{chunks[i]['text'][:400]}")
    return "\n---\n".join(result)

# ── Tool 2: web search simulation ────────────────────────
def web_search(query: str) -> str:
    """Simulates a web search. Replace with real API (SerpAPI, Tavily) if needed."""
    fake_results = {
        "python":        "Python is a high-level, interpreted programming language known for simplicity.",
        "openai":        "OpenAI is an AI research company. GPT-4 is their latest large language model.",
        "rag":           "RAG (Retrieval-Augmented Generation) combines search with LLM generation.",
        "machine learning": "ML is a subset of AI where systems learn patterns from data.",
        "newton":        "Newton's laws describe motion: 1) inertia, 2) F=ma, 3) action-reaction.",
    }
    query_lower = query.lower()
    for key, val in fake_results.items():
        if key in query_lower:
            return f"Web result for '{query}':\n{val}"
    return f"Web result for '{query}':\nNo specific result found. Try rephrasing."

# ── Tool 3: calculator ────────────────────────────────────
def calculate(expression: str) -> str:
    """Safely evaluates a math expression. E.g. '2 * 9.8 * 5'"""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: only basic math operators allowed."
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"

# ── Tool 4: summariser ────────────────────────────────────
def summarise_findings(findings: str) -> str:
    """Asks GPT to summarise accumulated findings into a clear answer."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=400,
        messages=[
            {"role": "system", "content": "Summarise the following research findings into a clear, concise answer."},
            {"role": "user",   "content": findings}
        ]
    )
    return response.choices[0].message.content

# ── Tool registry ─────────────────────────────────────────
TOOLS = {
    "search_docs": {
        "fn":   search_docs,
        "desc": "Search the PDF knowledge base for information.",
        "args": "query (str)"
    },
    "web_search": {
        "fn":   web_search,
        "desc": "Search the web for general knowledge.",
        "args": "query (str)"
    },
    "calculate": {
        "fn":   calculate,
        "desc": "Evaluate a math expression. E.g. '5 * 9.8'",
        "args": "expression (str)"
    },
    "summarise_findings": {
        "fn":   summarise_findings,
        "desc": "Summarise all findings gathered so far into a final answer.",
        "args": "findings (str)"
    },
}