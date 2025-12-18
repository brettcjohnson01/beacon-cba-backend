"""
Vector search over FAISS index.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------- Config ----------
EMBEDDING_MODEL = "text-embedding-3-small"

INDEX_DIR = Path("data/index")
FAISS_INDEX_FILE = INDEX_DIR / "faiss.index"
METADATA_FILE = INDEX_DIR / "metadata.jsonl"
# ----------------------------


def load_index():
    if not FAISS_INDEX_FILE.exists():
        raise FileNotFoundError("FAISS index not found. Run build_faiss_index.py first.")
    return faiss.read_index(str(FAISS_INDEX_FILE))


def load_metadata():
    metadata = []
    with METADATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata


def embed_query(query: str) -> np.ndarray:
    client = OpenAI()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    vec = np.array(response.data[0].embedding, dtype="float32")
    return vec.reshape(1, -1)


def search(query: str, top_k: int = 5):
    index = load_index()
    metadata = load_metadata()

    query_vec = embed_query(query)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        chunk = metadata[idx]
        results.append(
            {
                "rank": rank + 1,
                "score": float(distances[0][rank]),
                "text": chunk["text"],
                "doc_id": chunk["doc_id"],
                "source_id": chunk["source_id"],
                "chunk_id": chunk["chunk_id"],
                "title": chunk.get("title"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
            }
        )

    return results
