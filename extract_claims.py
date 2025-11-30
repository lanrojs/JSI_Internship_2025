"""
extract_claims.py — extract claims from chunks + contextualized chunks (GraphRAG-style)
and store them in both JSONL and SQLite with embeddings.

Expected input JSON structure (list of objects):
[
  {
    "doc": "...",
    "id": "...",
    "chunk": "...",
    "contextualized_chunk": "...",
    "merged_chunk": "..."     # optional, ignored
  },
  ...
]

Modes:
  prefix  → claims grounded in RAW chunk, context_prefix only as auxiliary context
  rewrite → claims grounded in CONTEXTUALIZED_CHUNK (rewritten self-contained text)
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
import yaml


# --------------------------------------------------------------------
# Load configuration YAML
# --------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "extract_claims.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


# --------------------------------------------------------------------
# Config (from YAML)
# --------------------------------------------------------------------

# LLM / Ollama
OLLAMA_SERVER = CFG["llm"]["server"]
OLLAMA_MODEL = CFG["llm"]["model"]
REQUEST_TIMEOUT = CFG["llm"]["request_timeout"]
TEMPERATURE = CFG["llm"]["temperature"]

# Logging
PROGRESS_EVERY = CFG["run"]["progress_every"]
DEBUG = CFG["run"]["debug"]

# Embeddings
EMBED_MODEL_ID = CFG["embeddings"]["model_id"]
CLAIM_BATCH_SIZE = CFG["embeddings"]["batch_size"]
NORMALIZE_EMBEDDINGS = CFG["embeddings"]["normalize"]

# SQLite
CLAIMS_TABLE_NAME = CFG["sqlite"]["claims_table_name"]

# Modes
MODE_PREFIX = CFG["modes"]["prefix"]
MODE_REWRITE = CFG["modes"]["rewrite"]

# Templates
PROMPTS_DIR = Path(__file__).resolve().parent / CFG["templates"]["dir"]
PREFIX_TEMPLATE_NAME = CFG["templates"]["prefix"]
REWRITE_TEMPLATE_NAME = CFG["templates"]["rewrite"]


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
# LLM call
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

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))

    return (obj.get("message") or {}).get("content", "").strip()


def safe_json_array(s: str) -> List[Dict[str, Any]]:
    """Extract a JSON array even if LLM adds junk."""
    s = s.strip()
    start = s.find("[")
    end = s.rfind("]")

    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]

    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass

    return []


# --------------------------------------------------------------------
# SQLite helpers
# --------------------------------------------------------------------

def ensure_claims_table(conn: sqlite3.Connection) -> None:
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
        ) for r in rows
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
# Main logic
# --------------------------------------------------------------------

def extract_claims_from_file(input_path: Union[str, Path], mode: str = MODE_PREFIX) -> Path:
    if mode not in {MODE_PREFIX, MODE_REWRITE}:
        raise ValueError(f"Unknown mode: {mode}")

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    # Derive base name (strip _contextualized)
    base = re.sub(r"_contextualized$", "", input_path.stem)

    out_path = input_path.with_name(f"{base}_claims.jsonl")
    sqlite_path = input_path.with_name(f"{base}_claims.sqlite")

    # Load JSON
    items = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list.")

    print(f"Total chunks: {len(items)} (mode={mode})")

    # Embedder + SQLite
    embedder = SentenceTransformer(EMBED_MODEL_ID)
    conn = sqlite3.connect(sqlite_path)
    ensure_claims_table(conn)

    template_name = PREFIX_TEMPLATE_NAME if mode == MODE_PREFIX else REWRITE_TEMPLATE_NAME

    all_claims = []
    written = 0
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as out:
        for idx, obj in enumerate(items, start=1):

            doc = str(obj.get("doc", ""))
            cid = str(obj.get("id") or obj.get("chunk_id") or "")

            original_chunk = (obj.get("chunk") or "").strip()
            contextualized_chunk = (obj.get("contextualized_chunk") or "").strip()
            context_prefix = (obj.get("context_prefix") or "").strip()

            if not original_chunk and not contextualized_chunk:
                continue

            # Render prompt
            if mode == MODE_PREFIX:
                prompt = render_prompt(
                    template_name,
                    mode=mode,
                    doc=doc,
                    chunk_id=cid,
                    original_chunk=original_chunk,
                    context_prefix=context_prefix,
                )
            else:
                prompt = render_prompt(
                    template_name,
                    mode=mode,
                    doc=doc,
                    chunk_id=cid,
                    original_chunk=original_chunk,
                    contextualized_chunk=contextualized_chunk,
                )

            if DEBUG:
                print("\n========== CLAIMS PROMPT ==========")
                print(prompt)
                print("========== END PROMPT =============\n")

            # Call LLM
            try:
                reply = call_ollama(
                    server=OLLAMA_SERVER,
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    timeout=REQUEST_TIMEOUT,
                    temperature=TEMPERATURE,
                )
            except Exception as e:
                print(f"[warn] Failed request for {cid}: {e}")
                continue

            if DEBUG:
                print("\n===== RAW CLAIMS LLM RESPONSE =====")
                print(reply)
                print("===== END RAW RESPONSE =====\n")

            arr = safe_json_array(reply)

            # Write claims
            per_chunk = 0
            for i2, claim in enumerate(arr, start=1):
                ctext = (claim.get("claim_text") or "").strip()
                if not ctext:
                    continue

                rec = {
                    "doc": doc,
                    "chunk_id": cid,
                    "claim_id": f"{cid}#{i2}",
                    "claim_text": ctext,
                    "source_quote": (claim.get("source_quote") or "").strip(),
                }

                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                per_chunk += 1
                all_claims.append(rec)

            if idx % PROGRESS_EVERY == 0 or idx == len(items):
                print(f"[{idx}/{len(items)}] claims so far: {written}")

    # Embed & store in SQLite
    if all_claims:
        print(f"\nEmbedding {len(all_claims)} claims → {sqlite_path}")
        rows = []

        for start in range(0, len(all_claims), CLAIM_BATCH_SIZE):
            batch = all_claims[start:start+CLAIM_BATCH_SIZE]
            texts = [c["claim_text"] for c in batch]

            embs = embedder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )

            # optional normalization
            if NORMALIZE_EMBEDDINGS:
                norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
                embs = embs / norms

            for off, rec in enumerate(batch):
                rows.append({
                    "doc": rec["doc"],
                    "chunk_id": rec["chunk_id"],
                    "claim_id": rec["claim_id"],
                    "claim_text": rec["claim_text"],
                    "source_quote": rec["source_quote"],
                    "emb": embs[off],
                })

        insert_claim_rows(conn, rows)
        print(f"Inserted {len(rows)} claims.")

    conn.close()

    dt = time.time() - t0
    print(f"\n✓ Done. Wrote {written} claims → {out_path} in {dt:.1f}s")
    print(f"Claims embeddings stored in: {sqlite_path}")

    return out_path


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract factual claims from contextualized chunks.")
    parser.add_argument("input_json", help="Input <base>_contextualized.json")
    parser.add_argument(
        "--mode",
        choices=[MODE_PREFIX, MODE_REWRITE],
        default=MODE_REWRITE,
        help=f"{MODE_PREFIX}=raw grounding, {MODE_REWRITE}=rewrite grounding (default)",
    )

    args = parser.parse_args()

    try:
        extract_claims_from_file(Path(args.input_json), mode=args.mode)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
