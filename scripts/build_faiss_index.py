"""
Build a FAISS vector index from chunked documents.

Input:  data/chunks/chunks.jsonl
Output: data/index/faiss.index
        data/index/metadata.jsonl
"""

import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import faiss
import numpy as np
from openai import OpenAI


import json
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

# ---------- Config ----------
EMBEDDING_MODEL = "text-embedding-3-small"

CHUNKS_FILE = Path("data/chunks/chunks.jsonl")
INDEX_DIR = Path("data/index")
FAISS_INDEX_FILE = INDEX_DIR / "faiss.index"
METADATA_FILE = INDEX_DIR / "metadata.jsonl"
# ----------------------------


def load_chunks():
    chunks = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def embed_texts(client, texts):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    vectors = [d.embedding for d in response.data]
    return np.array(vectors, dtype="float32")


def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("chunks.jsonl not found. Run build_chunks.py first.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("[faiss] Loading chunks...")
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    print(f"[faiss] Embedding {len(texts)} chunks...")
    client = OpenAI()
    embeddings = embed_texts(client, texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("[faiss] Writing index...")
    faiss.write_index(index, str(FAISS_INDEX_FILE))

    print("[faiss] Writing metadata...")
    with METADATA_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("[faiss] Done.")
    print(f"Index:    {FAISS_INDEX_FILE}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
