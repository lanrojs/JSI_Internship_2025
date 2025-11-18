"""
chunk_and_embed.py — Clean text, chunk by BGE-small token counts with strict boundary rules,
embed each chunk, and store results in both JSON and SQLite.

Usage (CLI):
    python chunk_and_embed.py INPUT_PATH

Usage (imported):
    from chunk_and_embed import chunk_and_embed_path
    chunk_and_embed_path("docs/gdpr.txt")

Outputs:
    <base>_chunks.json      # list of {"doc","id","chunk"}
    <base>_raw.sqlite       # table raw_chunks(doc,id,chunk,emb)
"""

import sys
import re
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Union, Tuple

import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from process_text import clean_text


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_ID = "BAAI/bge-small-en-v1.5"

MAX_TOKENS_PER_CHUNK = 100
TOKEN_OVERLAP = 20
BATCH_SIZE = 64
NORMALIZE_EMBEDDINGS = True

TABLE_NAME = "raw_chunks"


# ---------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------

def ensure_table(conn: sqlite3.Connection) -> None:
    """Ensure the table for raw chunks exists."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            doc   TEXT NOT NULL,
            id    TEXT PRIMARY KEY,
            chunk TEXT NOT NULL,
            emb   BLOB NOT NULL
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_doc ON {TABLE_NAME}(doc)")
    conn.commit()


def to_blob(vec: np.ndarray) -> bytes:
    """Convert NumPy vector to float32 bytes for SQLite storage."""
    if vec.dtype != np.float32:
        vec = vec.astype(np.float32, copy=False)
    return vec.tobytes(order="C")


def insert_rows(conn: sqlite3.Connection, rows: List[Dict[str, Union[str, np.ndarray]]]) -> None:
    """Bulk insert rows into the SQLite table."""
    payload = [(r["doc"], r["id"], r["chunk"], to_blob(r["emb"])) for r in rows]
    conn.executemany(
        f"INSERT OR REPLACE INTO {TABLE_NAME} (doc, id, chunk, emb) VALUES (?, ?, ?, ?)",
        payload,
    )
    conn.commit()


# ---------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------

def is_wordy(tok: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", tok))


def starts_ok(token_text: str) -> bool:
    """
    Return True if the token is a valid chunk start:
      • alphabetic character (normal sentence start)
      • '(' followed by digits/letters and ')'
      • list-style markers like '1.', '1)', '1a)', '(1)', '(a)', etc.
    Rejects bare numbers like '2024' or '12th'.
    """
    if not token_text:
        return False

    # (1), (a)
    if re.match(r"^\([0-9a-zA-Z]+\)$", token_text):
        return True

    # 1., 1), 1a), 2b)
    if re.match(r"^\d+[a-zA-Z]?\)?\.?$", token_text):
        return True

    # normal text start
    if token_text[0].isalpha():
        return True

    return False


def make_chunks(text: str, tokenizer, size: int, overlap: int) -> List[str]:
    """Token-based chunking with boundary and overlap rules."""
    words = text.split()
    n = len(words)
    if n == 0:
        return []

    per_word_len = [len(tokenizer.encode(w, add_special_tokens=False)) for w in words]

    chunks = []
    start = 0
    last_max_j = -1

    while start < n:
        while start < n and not is_wordy(words[start]):
            start += 1
        if start >= n:
            break

        total = 0
        end = start
        while end < n:
            tlen = per_word_len[end]
            if total + tlen > size:
                break
            total += tlen
            end += 1
        if end == start:
            end = start + 1

        # Join to get candidate text
        chunk_text = " ".join(words[start:end]).strip()
        if not chunk_text:
            start = end
            continue

        # Ensure chunk start validity
        first_word = re.sub(r"^\W+", "", words[start])
        if not starts_ok(first_word):
            start = end
            continue

        if end <= last_max_j:
            start = max(end, start + 1)
            continue

        chunks.append(chunk_text)
        last_max_j = end

        if overlap <= 0:
            start = end
            continue

        # compute overlap in tokens
        acc = 0
        k = end
        while k > start and acc < overlap:
            k -= 1
            acc += per_word_len[k]
        new_start = k if acc >= overlap else start
        if new_start <= start:
            new_start = max(end, start + 1)
        start = new_start

    return chunks


# ---------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------

def gather_input_paths(input_path: Path) -> List[Path]:
    """Return a list of .txt files from path or directory."""
    if input_path.is_file() and input_path.suffix.lower() == ".txt":
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt"
        )
    raise ValueError(f"Input must be a .txt file or directory: {input_path}")


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------

def chunk_and_embed_path(input_path: Union[str, Path]) -> Tuple[Path, Path]:
    """Chunk and embed one file or directory, output JSON + SQLite."""
    input_path = Path(input_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    embedder = SentenceTransformer(MODEL_ID)

    paths = gather_input_paths(input_path)

    base = re.sub(r"_cleaned$", "", input_path.stem if input_path.is_file() else input_path.name)
    json_path = input_path.with_name(f"{base}_chunks.json")
    sqlite_path = input_path.with_name(f"{base}_raw.sqlite")

    conn = sqlite3.connect(sqlite_path)
    ensure_table(conn)

    json_results = []

    for path in paths:
        doc_raw = path.stem
        doc = re.sub(r"_cleaned$", "", doc_raw)
        print(f"Processing document: {doc}")

        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = clean_text(raw)

        chunks = make_chunks(text, tokenizer, size=MAX_TOKENS_PER_CHUNK, overlap=TOKEN_OVERLAP)
        if not chunks:
            print(f"  ⚠️ No chunks produced for {doc}")
            continue

        print(f"  {len(chunks)} chunks → embedding ...")

        rows = []
        for start_idx in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start_idx:start_idx + BATCH_SIZE]
            embs = embedder.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )

            if NORMALIZE_EMBEDDINGS:
                norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
                embs = embs / norms

            for offset, chunk_text in enumerate(batch):
                global_idx = start_idx + offset + 1
                rows.append({
                    "doc": doc,
                    "id": f"{doc}_{global_idx}",
                    "chunk": chunk_text,
                    "emb": embs[offset],
                })

        insert_rows(conn, rows)
        print(f"  Inserted {len(rows)} rows into SQLite for {doc}")

        for r in rows:
            json_results.append({"doc": r["doc"], "id": r["id"], "chunk": r["chunk"]})

    conn.close()
    json_path.write_text(json.dumps(json_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ Done. Wrote {len(json_results)} chunks → {json_path}")
    print(f"   Raw embeddings stored in {sqlite_path}")
    return sqlite_path, json_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        prog = Path(sys.argv[0]).name
        print(f"Usage: {prog} INPUT_PATH", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    chunk_and_embed_path(input_path)


if __name__ == "__main__":
    main()
