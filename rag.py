"""
RAG Chat — fastembed (local ONNX) + ChromaDB + Ollama Cloud LLM

Usage:
    python rag.py <folder_with_docx_files>

Example:
    python rag.py data

Type your question and press Enter. Type $quit to exit.

Environment Variables (.env):
    OLLAMA_API_KEY   - Your Ollama Cloud API key (required)
"""

import sys
import os
from pathlib import Path

from dotenv import load_dotenv
from docx import Document
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import ollama

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

OLLAMA_CLOUD_URL = "https://ollama.com"
LLM_MODEL        = "gpt-oss:120b-cloud"
EMBED_MODEL      = "all-MiniLM-L6-v2"  # chromadb default ONNX model, downloads once
CHROMA_DIR       = "./chroma_db"
CHUNK_SIZE       = 500    # words
CHUNK_OVERLAP    = 50     # words
TOP_K            = 5

PROMPT = """\
You are a helpful assistant. Use ONLY the context below to answer.
If the context does not contain enough information, reply with exactly: IRRELEVANT

Context:
{context}

Question: {question}

Before writing your answer, reason through it step by step.
For every fact or phrase you use, state exactly where it comes from:
  - quote the relevant part of the context and name the source file, OR
  - mark it as [language/grammar] if it is just a connecting word with no factual content.

Respond in this exact format:

THINKING:
- [fact or phrase you will use] -> [source: filename.docx, chunk excerpt] or [language/grammar]
- ...

ANSWER:
[your final answer]"""

# ─── Text splitting (no external deps) ───────────────────────────────────────

def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    step = size - overlap
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += step
    return chunks

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    key = os.getenv("OLLAMA_API_KEY", "").strip()
    if not key:
        print("[ERROR] OLLAMA_API_KEY is not set in .env")
        sys.exit(1)
    return key


def load_and_index(folder: str) -> chromadb.Collection:
    ef = DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection_name = Path(folder).resolve().name.replace(" ", "_")[:50] or "docs"

    # Return existing collection if already indexed
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        print(f"[INFO] Loading existing index '{collection_name}'...")
        return client.get_collection(name=collection_name, embedding_function=ef)

    docx_files = sorted(Path(folder).glob("*.docx"))
    if not docx_files:
        print(f"[ERROR] No .docx files found in '{folder}'")
        sys.exit(1)

    print(f"[INFO] Indexing {len(docx_files)} file(s)...")
    collection = client.create_collection(name=collection_name, embedding_function=ef)

    all_texts, all_ids, all_metas = [], [], []
    idx = 0
    for path in docx_files:
        doc = Document(str(path))
        text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip()).lower()
        if not text.strip():
            print(f"  [SKIP] {path.name} — no text")
            continue
        chunks = split_text(text)
        for chunk in chunks:
            all_texts.append(chunk)
            all_ids.append(f"doc_{idx}")
            all_metas.append({"source": path.name})
            idx += 1
        print(f"  [OK]   {path.name} — {len(chunks)} chunks")

    if not all_texts:
        print("[ERROR] No text extracted.")
        sys.exit(1)

    # Add in batches of 100
    batch = 100
    for i in range(0, len(all_texts), batch):
        collection.add(
            documents=all_texts[i : i + batch],
            ids=all_ids[i : i + batch],
            metadatas=all_metas[i : i + batch],
        )
    print(f"[INFO] Done. {len(all_texts)} chunks stored.\n")
    return collection


def ask(question: str, collection: chromadb.Collection, api_key: str) -> str:
    results = collection.query(query_texts=[question], n_results=TOP_K)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    if not docs:
        return "[No relevant context found]"

    context = "\n\n---\n\n".join(
        f"[{m.get('source', '')}]\n{d}" for d, m in zip(docs, metas)
    )
    prompt = PROMPT.format(context=context, question=question)

    client = ollama.Client(
        host=OLLAMA_CLOUD_URL,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response = client.generate(model=LLM_MODEL, prompt=prompt)
    raw = response["response"].strip()

    if raw.upper() == "IRRELEVANT":
        return "[No relevant answer found]"

    # Parse THINKING / ANSWER blocks
    thinking, answer = "", raw
    if "THINKING:" in raw and "ANSWER:" in raw:
        parts = raw.split("ANSWER:", 1)
        thinking_block = parts[0].replace("THINKING:", "").strip()
        answer = parts[1].strip()

        print("\n  Brain line:")
        print("  " + "-" * 51)
        for line in thinking_block.splitlines():
            line = line.strip()
            if line:
                print(f"  {line}")
        print("  " + "-" * 51 + "\n")

    return answer

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"[ERROR] Not a directory: '{folder}'")
        sys.exit(1)

    api_key = get_api_key()
    collection = load_and_index(folder)

    docx_files = sorted(Path(folder).glob("*.docx"))
    print("=" * 55)
    print(f"  {len(docx_files)} document(s) ready:")
    for f in docx_files:
        print(f"    • {f.name}")
    print("  Type $quit to exit.")
    print("=" * 55 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not question:
            continue
        if question == "$quit":
            print("Bye.")
            break
        answer = ask(question, collection, api_key)
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()
