import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tools import TOOLS, get_embedding

load_dotenv()

client     = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 200
MAX_STEPS  = 5

# ── PDF loader ────────────────────────────────────────────
def load_documents(folder):
    chunks = []
    pdfs   = list(Path(folder).glob("*.pdf"))
    if not pdfs:
        print("  No PDFs found — knowledge base empty. Web search + calculator still work.")
        return chunks
    for path in pdfs:
        import fitz
        doc  = fitz.open(str(path))
        text = "".join(p.get_text() for p in doc)
        doc.close()
        words = text.split()
        for i in range(0, len(words), CHUNK_SIZE):
            chunk = " ".join(words[i:i + CHUNK_SIZE])
            chunks.append({"text": chunk, "source": path.name})
        print(f"  Loaded {path.name} → {len(chunks)} chunks")
    return chunks

def build_embeddings(chunks):
    if not chunks:
        return []
    print(f"  Embedding {len(chunks)} chunks...")
    embs = [get_embedding(c["text"]) for c in chunks]
    print("  Done.")
    return embs

# ── Agent brain: one reasoning step ──────────────────────
def think(question: str, scratchpad: str) -> dict:
    tool_list = "\n".join(
        [f"- {n}({v['args']}): {v['desc']}" for n, v in TOOLS.items()]
    )
    prompt = f"""You are a research agent. Answer the question by calling tools step by step.

TOOLS AVAILABLE:
{tool_list}

QUESTION: {question}

RESEARCH SO FAR:
{scratchpad or "Nothing yet."}

Reply with ONLY valid JSON — either:
  {{"action": "tool",   "tool": "", "args": {{...}}}}
  {{"action": "answer", "answer": ""}}

Be strategic. Use search_docs for PDF questions, web_search for general knowledge,
calculate for math. Call summarise_findings when you have enough info."""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        max_tokens=300,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(raw)
    except:
        return {"action": "answer", "answer": raw}

# ── Main agent loop ───────────────────────────────────────
def run_agent(question: str, chunks: list, embeddings: list) -> str:
    print(f"\n{'─'*50}")
    print(f"Question: {question}")
    print(f"{'─'*50}")

    scratchpad = ""

    for step in range(1, MAX_STEPS + 1):
        decision = think(question, scratchpad)

        if decision.get("action") == "answer":
            print(f"[Step {step}] Final answer ready.")
            return decision.get("answer", "")

        tool_name = decision.get("tool", "")
        tool_args = decision.get("args", {})
        print(f"[Step {step}] Tool: {tool_name}({tool_args})")

        if tool_name not in TOOLS:
            scratchpad += f"\nStep {step}: Tool '{tool_name}' not found.\n"
            continue

        # inject chunks + embeddings for search_docs
        if tool_name == "search_docs":
            result = TOOLS[tool_name]["fn"](
                tool_args.get("query", question), chunks, embeddings
            )
        elif tool_name == "summarise_findings":
            result = TOOLS[tool_name]["fn"](scratchpad)
        else:
            fn_args = {k: v for k, v in tool_args.items()}
            result  = TOOLS[tool_name]["fn"](**fn_args)

        print(f"         → {result[:100]}...")
        scratchpad += f"\nStep {step} | {tool_name}({tool_args}):\n{result}\n"

    # fallback: summarise everything found
    return TOOLS["summarise_findings"]["fn"](scratchpad)

# ── Entry point ───────────────────────────────────────────
def main():
    print("\n" + "="*50)
    print("  Agentic RAG — Multi-tool Research Agent")
    print("="*50)
    print("The agent plans, picks tools, and reasons step")
    print("by step until it has a complete answer.\n")
    print("Try asking:")
    print("  What is Newton's second law?")
    print("  If mass is 5kg and force is 20N, what is acceleration?")
    print("  What is RAG in AI?")
    print("  Search my documents about electric charges")
    print("\nType 'quit' to exit\n")

    chunks     = load_documents("documents")
    embeddings = build_embeddings(chunks)
    print("\nAgent ready.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break
        answer = run_agent(q, chunks, embeddings)
        print(f"\nAnswer:\n{answer}\n")

if __name__ == "__main__":
    main()
