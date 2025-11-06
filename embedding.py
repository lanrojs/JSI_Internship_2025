"""
embedding.py — Process text -> Chunk -> Embed -> Store in SQLite AND write chunks JSON.

Outputs:
  - <input_basename>_embeddings.sqlite   (table: chunks(doc, id, chunk, emb BLOB))
  - <input_basename>_chunks.json         (list of {"doc","id","chunk"})
"""

import sys
import json
import sqlite3
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union

import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from process_text import clean_text
from chunks import make_chunks

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

# HuggingFace tokenizer
TOKENIZER_MODEL_ID = "BAAI/bge-small-en-v1.5"

# SentenceTransformer embedding model
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"

# Chunk parameters
CHUNK_SIZE_TOKENS = 300     # max tokens per chunk
CHUNK_OVERLAP_TOKENS = 50  # desired overlap in tokens

# SQLite settings
TABLE_NAME = "chunks"       # name of the table inside the SQLite DB

# Embedding settings
BATCH_SIZE = 64             # how many chunks to encode at once
NORMALIZE_EMBEDDINGS = True # L2-normalize embeddings (recommended for cosine/IP)


# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------

def gather_input_paths(input_path: Path) -> List[Path]:
    """
    Given an input path, return a list of .txt files to process.
    """
    if input_path.is_file() and input_path.suffix.lower() == ".txt":
        return [input_path]

    if input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt"
        )

    raise ValueError(f"Input must be a .txt file or a directory of .txt files: {input_path}")


def ensure_table(conn: sqlite3.Connection, table: str) -> None:
    """
    Make sure the SQLite table exists.

    Schema:
        doc   TEXT    (document name, e.g. "gdpr")
        id    TEXT PK (unique chunk id, e.g. "gdpr_1")
        chunk TEXT    (actual text content)
        emb   BLOB    (raw bytes of float32 embedding vector)
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            doc   TEXT NOT NULL,
            id    TEXT PRIMARY KEY,
            chunk TEXT NOT NULL,
            emb   BLOB NOT NULL
        )
    """)
    # Index by document for faster per-doc queries
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_doc ON {table}(doc)")
    conn.commit()


def to_blob(vec: np.ndarray) -> bytes:
    """
    Convert a NumPy float vector into a contiguous float32 byte string.
    This is how we store embeddings as BLOBs in SQLite.
    """
    if vec.dtype != np.float32:
        vec = vec.astype(np.float32, copy=False)
    return vec.tobytes(order="C")


def insert_rows(conn: sqlite3.Connection, table: str, rows: List[Dict]) -> None:
    """
    Insert multiple rows (doc, id, chunk, emb array) into the SQLite table.
    Uses 'INSERT OR REPLACE' so re-running on the same doc/id pair will overwrite.
    """
    payload = [
        (r["doc"], r["id"], r["chunk"], to_blob(r["emb"]))
        for r in rows
    ]
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} (doc, id, chunk, emb) VALUES (?, ?, ?, ?)",
        payload,
    )
    conn.commit()


# calling from a notebook
def embed_path(input_path: Union[str, Path]) -> Tuple[Path, Path]:
    input_path = Path(input_path)

    # -------- Derive base name and output paths --------
    if input_path.is_file():
        base_raw = input_path.stem
    else:
        base_raw = input_path.name

    # Strip a trailing "_cleaned" if present (so gdpr_cleaned.txt → gdpr_*)
    base = re.sub(r"_cleaned$", "", base_raw)

    sqlite_path = input_path.with_name(f"{base}_embeddings.sqlite")
    json_path = input_path.with_name(f"{base}_chunks.json")

    print(f"SQLite DB will be written to: {sqlite_path}")
    print(f"Chunks JSON will be written to: {json_path}")

    # -------- Initialize models --------
    # Tokenizer for BGE token-count-based chunking
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID)

    # SentenceTransformer for embeddings
    embedder = SentenceTransformer(EMBED_MODEL_ID)

    # -------- Collect input files and set up SQLite --------
    paths = gather_input_paths(input_path)
    conn = sqlite3.connect(sqlite_path)
    ensure_table(conn, TABLE_NAME)

    # List of {"doc", "id", "chunk"} for the JSON output
    json_results: List[Dict[str, str]] = []

    # -------- Process each document --------
    for path in paths:
        # doc_raw = filename without extension, possibly with "_cleaned"
        doc_raw = path.stem
        # Remove trailing "_cleaned" from doc identifier
        doc = re.sub(r"_cleaned$", "", doc_raw)

        print(f"Processing document: {path}  (doc id: {doc})")

        # --- Step 1: Read and clean the raw text ---
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = clean_text(raw)  # uses your process_text.clean_text

        # --- Step 2: Token-based chunking using BGE tokenizer ---
        chunks = make_chunks(
            text,
            tokenizer,
            size=CHUNK_SIZE_TOKENS,
            overlap=CHUNK_OVERLAP_TOKENS,
        )

        if not chunks:
            print(f"  ⚠️  No chunks produced for {doc}. Skipping.")
            continue

        # --- Step 3: Embed chunks in batches and prepare rows for DB ---
        rows = []
        total_chunks = len(chunks)
        print(f"  Chunks: {total_chunks}")

        for start_idx in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[start_idx : start_idx + BATCH_SIZE]

            # Compute embeddings for this batch
            embs = embedder.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=False,  # we'll normalize manually (if enabled)
            )

            if NORMALIZE_EMBEDDINGS:
                # L2-normalize each embedding vector: v / ||v||
                norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
                embs = embs / norms

            # For each chunk in the batch, build a row
            for offset, chunk_text in enumerate(batch):
                # Global chunk index (1-based) within this document
                global_index = start_idx + offset + 1
                rows.append(
                    {
                        "doc": doc,
                        "id": f"{doc}_{global_index}",
                        "chunk": chunk_text,
                        "emb": embs[offset],
                    }
                )

        # --- Step 4: Insert into SQLite ---
        insert_rows(conn, TABLE_NAME, rows)
        print(f"  Inserted {len(rows)} rows into SQLite for doc {doc}")

        # --- Step 5: Accumulate JSON entries (without embeddings) ---
        for r in rows:
            json_results.append(
                {
                    "doc": r["doc"],
                    "id": r["id"],
                    "chunk": r["chunk"],
                }
            )

    # -------- Finalize: close DB and write JSON --------
    conn.close()

    # Write pretty-printed JSON with UTF-8 encoding
    json_path.write_text(
        json.dumps(json_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(json_results)} chunks to JSON → {json_path}")
    print("✅ Done.")

    return sqlite_path, json_path


# --------------------------------------------------------------------
# Simple CLI wrapper (no optional arguments, just INPUT_PATH)
# --------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        prog = Path(sys.argv[0]).name
        print(f"Usage: {prog} INPUT_PATH", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    embed_path(input_path)


if __name__ == "__main__":
    main()
