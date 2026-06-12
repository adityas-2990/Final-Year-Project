"""
rag.py
──────
Local RAG (Retrieval-Augmented Generation) pipeline.

Responsibilities
────────────────
- Parse uploaded files into plain text (PDF, CSV, DOCX, TXT, JSON, HTML).
- Split text into overlapping chunks.
- Embed chunks with a local Ollama embedding model.
- Store / retrieve chunks using ChromaDB (in-memory, no server needed).
- Answer user questions by retrieving relevant chunks and passing them
    to an Ollama chat model.

No Streamlit UI code lives here — rendering is handled in rag_tab.py.

Dependencies (add to requirements.txt)
───────────────────────────────────────
    chromadb>=0.5
    pymupdf>=1.24          # fitz  — PDF parsing
    python-docx>=1.1       # docx  — Word parsing
    beautifulsoup4>=4.12   # HTML  parsing
    ollama>=0.2
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import chromadb
import ollama
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE: int = 500          # default characters per chunk
CHUNK_OVERLAP: int = 100       # default overlap between consecutive chunks
TOP_K: int = 10                # default number of chunks to retrieve per query
EMBED_MODEL: str = "nomic-embed-text"   # Ollama embedding model
DEFAULT_CHAT_MODEL: str = "llama3.2:latest"


# ──────────────────────────────────────────────────────────────────────────────
#  File → plain text parsers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_pdf(path: Path) -> str:
    import fitz  # pymupdf
    doc = fitz.open(str(path))
    return "\n".join(page.get_text() for page in doc)


def _parse_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _parse_csv(path: Path) -> str:
    df = pd.read_csv(path)
    return df.to_string(index=False)


def _parse_json(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=2)


def _parse_html(path: Path) -> str:
    from bs4 import BeautifulSoup
    with open(path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text(separator="\n")


def _parse_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


PARSERS: dict[str, Any] = {
    ".pdf":  _parse_pdf,
    ".docx": _parse_docx,
    ".doc":  _parse_docx,
    ".csv":  _parse_csv,
    ".json": _parse_json,
    ".html": _parse_html,
    ".htm":  _parse_html,
    ".txt":  _parse_txt,
    ".md":   _parse_txt,
}


def extract_text(uploaded_file) -> str:
    """
    Accept a Streamlit UploadedFile, write it to a temp file,
    and return extracted plain text.

    Raises ValueError for unsupported extensions.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in PARSERS:
        raise ValueError(
            f"Unsupported file type: '{suffix}'. "
            f"Supported: {', '.join(PARSERS)}"
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    try:
        return PARSERS[suffix](tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
#  Text chunker
# ──────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split `text` into overlapping character-level chunks.
    Tries to break on sentence boundaries ('. ') where possible.
    """
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Try to break at the last sentence boundary within the window
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + overlap:
                end = boundary + 1   # include the period

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
#  ChromaDB collection manager
# ──────────────────────────────────────────────────────────────────────────────

# One in-memory client for the lifetime of the Streamlit process.
_chroma_client = chromadb.Client()   # ephemeral / in-memory


def _collection_name(file_name: str) -> str:
    """
    Derive a ChromaDB-safe collection name from the file name.
    """
    stem = Path(file_name).stem
    safe = re.sub(r"[^a-zA-Z0-9]", ".", stem)
    safe = re.sub(r"\.{2,}", ".", safe)
    safe = safe.strip(".")
    safe = safe[:40] if safe else "doc"
    if not safe[0].isalpha():
        safe = "doc." + safe
    short_hash = hashlib.md5(file_name.encode()).hexdigest()[:6]
    return f"{safe}.{short_hash}"


def build_index(
    uploaded_file,
    embed_model: str = EMBED_MODEL,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> tuple[Any, int]:
    """
    Parse the file, chunk it, embed every chunk with Ollama, and
    store everything in a fresh ChromaDB collection.

    Returns (collection, chunk_count).
    """
    text = extract_text(uploaded_file)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        raise ValueError("No text could be extracted from the uploaded file.")

    col_name = _collection_name(uploaded_file.name)

    # Delete previous version of this collection if it exists
    try:
        _chroma_client.delete_collection(col_name)
    except Exception:
        pass

    collection = _chroma_client.create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Embed and add in batches to avoid huge single requests
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = [
            ollama.embeddings(model=embed_model, prompt=c)["embedding"]
            for c in batch
        ]
        ids = [
            hashlib.md5(f"{col_name}_{i+j}".encode()).hexdigest()
            for j in range(len(batch))
        ]
        collection.add(
            documents=batch,
            embeddings=embeddings,
            ids=ids,
        )

    return collection, len(chunks)


# ──────────────────────────────────────────────────────────────────────────────
#  Retrieval
# ──────────────────────────────────────────────────────────────────────────────

def retrieve(
    collection,
    query: str,
    top_k: int = TOP_K,
    embed_model: str = EMBED_MODEL,
) -> list[str]:
    """
    Embed `query` and return the top-k most relevant chunks.
    """
    query_embedding = ollama.embeddings(model=embed_model, prompt=query)["embedding"]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )
    return results["documents"][0]   # list[str]


# ──────────────────────────────────────────────────────────────────────────────
#  RAG chat
# ──────────────────────────────────────────────────────────────────────────────

def rag_chat(
    collection,
    user_question: str,
    history: list[dict],
    chat_model: str = DEFAULT_CHAT_MODEL,
    embed_model: str = EMBED_MODEL,
    top_k: int = TOP_K,
    temperature: float = 0.0,
) -> tuple[str, list[str]]:
    """
    Retrieve relevant chunks for `user_question`, then answer using
    the Ollama chat model with the chunks as grounding context.

    Returns (assistant_reply, retrieved_chunks).
    """
    chunks = retrieve(collection, user_question, top_k=top_k, embed_model=embed_model)
    context = "\n\n---\n\n".join(chunks)

    system_prompt = (
        "You are an expert data analyst assistant. "
        "Answer the user's question using ONLY the context extracted from "
        "their uploaded document. Be concise and specific. "
        "If the answer is not in the context, say so honestly.\n\n"
        f"DOCUMENT CONTEXT:\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_question})

    response = ollama.chat(
        model=chat_model, 
        messages=messages,
        options={"temperature": temperature}
    )
    return response["message"]["content"], chunks


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def available_embed_models() -> list[str]:
    """Return Ollama models suitable for embedding (best-effort filter)."""
    try:
        all_models = [m["name"] for m in ollama.list()["models"]]
        # Prefer known embedding models; fall back to full list
        embed_keywords = ["embed", "nomic", "mxbai", "all-minilm"]
        preferred = [m for m in all_models if any(k in m for k in embed_keywords)]
        return preferred if preferred else all_models
    except Exception:
        return [EMBED_MODEL]


SUPPORTED_EXTENSIONS: list[str] = list(PARSERS.keys())