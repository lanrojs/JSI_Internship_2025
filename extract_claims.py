"""
extract_claims.py — extract claims from chunks + contextualized chunks (GraphRAG-style)
and store them in both JSONL and SQLite with embeddings.

Expected input JSON structure (list of objects):
[
  {
    "doc": "...",
    "id": "...",
    "chunk": "...",                 # original text from the source
    "contextualized_chunk": "...",  # contextualized or rewritten version
    "merged_chunk": "..."           # (optional, ignored here)
  },
  ...
]

Modes:
  --mode prefix   (default)
      - Claims are grounded in the ORIGINAL raw chunk.
      - contextualized_chunk is used only as auxiliary context to clarify
        references or reduce ambiguity.

  --mode rewrite
      - Claims are grounded in the CONTEXTUALIZED_CHUNK (self-contained version).
      - The raw chunk is available as additional context but the rewritten text
        is the primary basis for claims.

Behavior:
  • For each item, send BOTH `chunk` and `contextualized_chunk` to a local Ollama model.
  • The prompt (Jinja2 template) enforces the correct grounding rules per mode.
  • Ask the model to return a JSON array of claims, each with:
        - claim_text
        - source_quote
  • Write ONE JSON line per claim to an output .jsonl file:
        <base>_claims.jsonl

  • Additionally, embed each claim_text with BGE-small and store them in:
        <base>_claims.sqlite

    SQLite schema (table: claims):
        doc          TEXT NOT NULL
        chunk_id     TEXT NOT NULL
        claim_id     TEXT PRIMARY KEY
        claim_text   TEXT NOT NULL
        source_quote TEXT
        emb          BLOB NOT NULL
"""

import argparse
import json
import re
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Union

import numpy as np
from jinja2 import Environment, FileSystemLoader
from sentence_transformers import SentenceTransformer


# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

# Base URL of your Ollama server
OLLAMA_SERVER = "http://localhost:11434"

# Name/tag of the LLM model to use
OLLAMA_MODEL = "gemma3:4b-it-qat"  # or whatever local model you use

# HTTP timeout for each request (seconds)
REQUEST_TIMEOUT = 120

# Generation temperature
TEMPERATURE = 0.3

# Print progress every N chunks
PROGRESS_EVERY = 1

# Embedding model for claims
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
CLAIM_BATCH_SIZE = 64
NORMALIZE_EMBEDDINGS = True

# SQLite
CLAIMS_TABLE_NAME = "claims"

# Modes
MODE_PREFIX = "prefix"
MODE_REWRITE = "rewrite"

# Prompt templates (Jinja2)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PREFIX_TEMPLATE_NAME = "extract_claims_prefix.j2"
REWRITE_TEMPLATE_NAME = "extract_claims_rewrite.j2"


# --------------------------------------------------------------------
# Jinja2 environment
# --------------------------------------------------------------------

_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_prompt(template_name: str, **kwargs) -> str:
    template = _env.get_template(template_name)
    return template.render(**kwargs)


# --------------------------------------------------------------------
# Low-level HTTP / model-calling helpers
# --------------------------------------------------------------------

def call_ollama(
    server: str,
    model: str,
    prompt: str,
    timeout: int = REQUEST_TIMEOUT,
    temperature: float = TEMPERATURE,
) -> str:
    url = server.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": float(temperature)},
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))

    # Ollama returns a structure where the main content is in message.content
    content = (obj.get("message") or {}).get("content", "") or ""
    return content.strip()


def safe_json_array(s: str) -> List[Dict[str, Any]]:
    """
    Best-effort parsing of a JSON array from the model's reply.

    The model is *supposed* to return a plain JSON array, but in practice
    it might add extra text. This function:

      1. Finds the first '[' and last ']' in the string.
      2. Extracts that substring.
      3. Tries to json.loads it.
      4. Returns a list if successful, otherwise [].
    """
    s = s.strip()
    start = s.find("[")
    end = s.rfind("]")

    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]

    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass

    return []


# --------------------------------------------------------------------
# SQLite + embedding helpers for claims
# --------------------------------------------------------------------

def ensure_claims_table(conn: sqlite3.Connection) -> None:
    """
    Ensure the 'claims' table exists.

    Schema:
        doc          TEXT
        chunk_id     TEXT
        claim_id     TEXT PRIMARY KEY
        claim_text   TEXT
        source_quote TEXT
        emb          BLOB
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {CLAIMS_TABLE_NAME} (
            doc          TEXT NOT NULL,
            chunk_id     TEXT NOT NULL,
            claim_id     TEXT PRIMARY KEY,
            claim_text   TEXT NOT NULL,
            source_quote TEXT,
            emb          BLOB NOT NULL
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{CLAIMS_TABLE_NAME}_doc ON {CLAIMS_TABLE_NAME}(doc)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{CLAIMS_TABLE_NAME}_chunk ON {CLAIMS_TABLE_NAME}(chunk_id)")
    conn.commit()


def to_blob(vec: np.ndarray) -> bytes:
    if vec.dtype != np.float32:
        vec = vec.astype(np.float32, copy=False)
    return vec.tobytes(order="C")


def insert_claim_rows(conn: sqlite3.Connection, rows: List[Dict[str, Any]]) -> None:
    payload = [
        (
            r["doc"],
            r["chunk_id"],
            r["claim_id"],
            r["claim_text"],
            r["source_quote"],
            to_blob(r["emb"]),
        )
        for r in rows
    ]
    conn.executemany(
        f"""
        INSERT OR REPLACE INTO {CLAIMS_TABLE_NAME}
        (doc, chunk_id, claim_id, claim_text, source_quote, emb)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    conn.commit()


# --------------------------------------------------------------------
# Main extraction + embedding logic
# --------------------------------------------------------------------

def extract_claims_from_file(input_path: Union[str, Path], mode: str = MODE_PREFIX) -> Path:
    if mode not in {MODE_PREFIX, MODE_REWRITE}:
        raise ValueError(f"Unknown mode: {mode}")

    input_path = Path(input_path)

    # --- Basic existence check ---
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # --- Derive base name (strip trailing '_contextualized' if present) ---
    base = re.sub(r"_contextualized$", "", input_path.stem)

    # --- Output paths ---
    out_path = input_path.with_name(f"{base}_claims.jsonl")
    sqlite_path = input_path.with_name(f"{base}_claims.sqlite")

    # --- Load input JSON (must be a list) ---
    try:
        items = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}") from e

    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list of objects.")

    total = len(items)
    print(f"Total chunks: {total} (mode={mode})", flush=True)

    written = 0       # how many claims we've written so far
    empty_chunks = 0  # how many chunks had no usable text
    t0 = time.time()  # start time

    # --- Initialize embedder + SQLite for claims ---
    embedder = SentenceTransformer(EMBED_MODEL_ID)
    conn = sqlite3.connect(sqlite_path)
    ensure_claims_table(conn)

    all_claims: List[Dict[str, Any]] = []

    # choose template
    template_name = PREFIX_TEMPLATE_NAME if mode == MODE_PREFIX else REWRITE_TEMPLATE_NAME

    # --- Open output JSONL for writing ---
    with out_path.open("w", encoding="utf-8") as out:
        for i, obj in enumerate(items, start=1):
            # Document ID (e.g. "gdpr2")
            doc = str(obj.get("doc", ""))

            # Chunk identifier (prefer "id", fallback to "chunk_id")
            cid = str(obj.get("id") or obj.get("chunk_id") or "")

            # Original chunk and contextualized chunk from the input
            original_chunk = (obj.get("chunk") or "").strip()
            contextualized_chunk = (obj.get("contextualized_chunk") or "").strip()

            # If we have absolutely no text, count as empty and continue
            if not original_chunk and not contextualized_chunk:
                empty_chunks += 1
                if i % PROGRESS_EVERY == 0 or i == total:
                    print(
                        f"[{i}/{total}] claims (written={written})",
                        flush=True,
                    )
                continue

            # --- Render the prompt with Jinja2 ---
            prompt = render_prompt(
                template_name,
                mode=mode,
                doc=doc,
                chunk_id=cid,
                original_chunk=original_chunk,
                contextualized_chunk=contextualized_chunk,
            )

            # --- Call Ollama model ---
            try:
                reply = call_ollama(
                    server=OLLAMA_SERVER,
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    timeout=REQUEST_TIMEOUT,
                    temperature=TEMPERATURE,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                # Log a warning and skip this chunk
                print(f"[warn] {doc}/{cid} request failed: {e}", flush=True)
                if i % PROGRESS_EVERY == 0 or i == total:
                    print(
                        f"[{i}/{total}] claims (written={written})",
                        flush=True,
                    )
                continue

            # --- Parse the model reply into a JSON array ---
            arr = safe_json_array(reply)

            # --- Write out each valid claim as a JSONL record ---
            idx = 0  # per-chunk claim counter
            for claim in arr:
                idx += 1
                ctext = (claim.get("claim_text") or "").strip()
                if not ctext:
                    # Skip empty or malformed entries
                    continue

                rec = {
                    "doc": doc,
                    "chunk_id": cid,
                    "claim_id": f"{cid}#{idx}",  # unique within this chunk
                    "claim_text": ctext,
                    "source_quote": (claim.get("source_quote") or "").strip(),
                }

                # Write JSONL
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

                # Store in memory for embedding/SQLite
                all_claims.append(rec)

            # Periodic progress logging
            if i % PROGRESS_EVERY == 0 or i == total:
                print(
                    f"[{i}/{total}] claims (written={written})",
                    flush=True,
                )

    # --- Embed all claims and store in SQLite ---
    if all_claims:
        print(f"\nEmbedding {len(all_claims)} claims and writing to SQLite → {sqlite_path}")
        rows_for_db: List[Dict[str, Any]] = []

        for start_idx in range(0, len(all_claims), CLAIM_BATCH_SIZE):
            batch = all_claims[start_idx : start_idx + CLAIM_BATCH_SIZE]
            texts = [c["claim_text"] for c in batch]

            embs = embedder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )

            if NORMALIZE_EMBEDDINGS:
                norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
                embs = embs / norms

            for offset, rec in enumerate(batch):
                rows_for_db.append(
                    {
                        "doc": rec["doc"],
                        "chunk_id": rec["chunk_id"],
                        "claim_id": rec["claim_id"],
                        "claim_text": rec["claim_text"],
                        "source_quote": rec["source_quote"],
                        "emb": embs[offset],
                    }
                )

        insert_claim_rows(conn, rows_for_db)
        print(f"Inserted {len(rows_for_db)} claims into SQLite.")

    conn.close()

    dt = time.time() - t0
    print(f"✓ Done. Wrote {written} claims to: {out_path} in {dt:.1f}s", flush=True)
    print(f"   Claims embeddings stored in: {sqlite_path}")
    return out_path


# --------------------------------------------------------------------
# Simple CLI wrapper with argparse
# --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract atomic factual claims from chunks + contextualized chunks."
    )
    parser.add_argument("input_json", help="Input <base>_contextualized.json")
    parser.add_argument(
        "--mode",
        choices=[MODE_PREFIX, MODE_REWRITE],
        default=MODE_PREFIX,
        help=(
            "Claim extraction mode:\n"
            "  prefix  - claims grounded in the raw chunk (default)\n"
            "  rewrite - claims grounded in contextualized_chunk"
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    try:
        extract_claims_from_file(input_path, mode=args.mode)
    except Exception as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
