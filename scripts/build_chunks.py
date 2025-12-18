"""
Build text chunks from extracted documents.

Input: data/processed_text/*.txt
Output: data/chunks/chunks.jsonl
"""

import json
import os
from pathlib import Path
from uuid import uuid4

MAX_CHARS = 2000

PROCESSED_DIR = Path("data/processed_text")
OUTPUT_DIR = Path("data/chunks")
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"


def main():
    cwd = Path(os.getcwd())
    processed_abs = (cwd / PROCESSED_DIR).resolve()
    output_abs = (cwd / OUTPUT_FILE).resolve()

    print(f"[build_chunks] CWD: {cwd}")
    print(f"[build_chunks] Looking for .txt in: {processed_abs}")
    print(f"[build_chunks] Will write to: {output_abs}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = list(PROCESSED_DIR.glob("*.txt"))
    print(f"[build_chunks] Found {len(txt_files)} .txt files")

    # Always create/overwrite output file
    wrote = 0
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as out:
        for path in txt_files:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue

            doc_id = path.stem
            chunk = {
                "doc_id": doc_id,
                "source_id": doc_id,
                "chunk_id": str(uuid4()),
                "text": text[:MAX_CHARS],
                "page_start": None,
                "page_end": None,
                "title": path.name,
                "state": None,
                "city": None,
                "project_type": None,
                "year": None,
            }
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            wrote += 1

        out.flush()

    print(f"[build_chunks] Wrote {wrote} chunks to: {output_abs}")
    print()  # ensures prompt starts on a new line


if __name__ == "__main__":
    main()
