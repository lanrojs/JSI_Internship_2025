#!/usr/bin/env python3
"""
ingest_to_sqlite.py — Clean -> Chunk -> Embed -> Store in SQLite AND write chunks JSON.

Defaults:
  - Chunk size: 300 BGE tokens
  - Overlap:    50 BGE tokens
Outputs:
  - <input_basename>_embeddings.sqlite   (table: chunks(doc, id, chunk, emb BLOB))
  - <input_basename>_chunks.json         (list of {"doc","id","chunk"})
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import List, Dict

import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from process_text import clean_text
from chunking import make_chunks_by_bge_tokens


def gather_input_paths(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".txt":
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
    raise ValueError(f"Input must be a .txt file or a directory: {input_path}")


def ensure_table(conn: sqlite3.Connection, table: str):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            doc   TEXT NOT NULL,
            id    TEXT PRIMARY KEY,
            chunk TEXT NOT NULL,
            emb   BLOB NOT NULL
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_doc ON {table}(doc)")
    conn.commit()


def to_blob(vec: np.ndarray) -> bytes:
    if vec.dtype != np.float32:
        vec = vec.astype(np.float32, copy=False)
    return vec.tobytes(order="C")


def insert_rows(conn: sqlite3.Connection, table: str, rows: List[Dict]):
    payload = [(r["doc"], r["id"], r["chunk"], to_blob(r["emb"])) for r in rows]
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} (doc, id, chunk, emb) VALUES (?, ?, ?, ?)",
        payload
    )
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description="Clean -> Chunk -> Embed (BGE small) -> SQLite + JSON.")
    ap.add_argument("input", type=Path, help="Path to .txt file or directory of .txt files")
    ap.add_argument("--table", type=str, default="chunks", help="SQLite table name")
    ap.add_argument("--size", type=int, default=300, help="Max tokens per chunk (BGE tokenizer)")
    ap.add_argument("--overlap", type=int, default=50, help="Token overlap between chunks (BGE tokenizer)")
    ap.add_argument("--tokenizer", type=str, default="BAAI/bge-small-en-v1.5", help="HF tokenizer for chunking")
    ap.add_argument("--embed-model", type=str, default="BAAI/bge-small-en-v1.5", help="SentenceTransformer model id")
    ap.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize embeddings (recommended for cosine/IP)")
    ap.add_argument("--keep-newlines", action="store_true", help="Cleaner: keep original newlines")
    ap.add_argument("--lowercase", action="store_true", help="Cleaner: lowercase after cleaning")
    ap.add_argument("--no-preclean", action="store_true", help="Skip text cleaning before chunking")
    ap.add_argument("--no-json", action="store_true", help="Skip writing the chunks JSON file")
    args = ap.parse_args()

    # Derive output paths
    if args.input.is_file():
        sqlite_path = args.input.with_name(f"{args.input.stem}_embeddings.sqlite")
        json_path = args.input.with_name(f"{args.input.stem}_chunks.json")
    else:
        sqlite_path = args.input.with_name(f"{args.input.name}_embeddings.sqlite")
        json_path = args.input.with_name(f"{args.input.name}_chunks.json")

    print(f"SQLite → {sqlite_path}")
    if not args.no_json:
        print(f"JSON   → {json_path}")

    # Models
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    embedder = SentenceTransformer(args.embed_model)

    # Inputs & DB
    paths = gather_input_paths(args.input)
    conn = sqlite3.connect(sqlite_path)
    ensure_table(conn, args.table)

    # Collect for JSON output
    json_results: List[Dict[str, str]] = []

    for path in paths:
        doc = path.stem
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = raw if args.no_preclean else clean_text(raw, keep_newlines=args.keep_newlines, lowercase=args.lowercase)

        # Chunk
        chunks = make_chunks_by_bge_tokens(text, tokenizer, size=args.size, overlap=args.overlap)
        if not chunks:
            continue

        rows = []
        for i in range(0, len(chunks), args.batch_size):
            batch = chunks[i : i + args.batch_size]
            embs = embedder.encode(batch, convert_to_numpy=True, normalize_embeddings=False)
            if args.normalize:
                norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
                embs = embs / norms

            for j, chunk_text in enumerate(batch, start=i + 1):
                # 👇 no leading zeroes
                rows.append({
                    "doc": doc,
                    "id": f"{doc}_{j}",
                    "chunk": chunk_text,
                    "emb": embs[j - i - 1]
                })

        insert_rows(conn, args.table, rows)
        print(f"Inserted {len(rows)} rows from {doc}")

        if not args.no_json:
            for r in rows:
                json_results.append({"doc": r["doc"], "id": r["id"], "chunk": r["chunk"]})

    conn.close()

    if not args.no_json:
        json_path.write_text(json.dumps(json_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote chunks JSON → {json_path}")

    print("✅ Done.")


if __name__ == "__main__":
    main()
