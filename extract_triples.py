"""
extract_triples.py — Step 5 of the contextualizing RAG pipeline.
Takes atomic claims (from <base>_claims.jsonl) and extracts structured triples.

Input:
  <base>_claims.jsonl   # one claim per line (from extract_claims.py)

Output:
  <base>_triples.jsonl  # structured triples (JSONL)
  <base>_triples.sqlite # table: claims

SQLite schema:
  claim_id      TEXT PRIMARY KEY
  claim_text    TEXT NOT NULL
  subject       TEXT
  predicate     TEXT
  object        TEXT
  subject_emb   BLOB
  predicate_emb BLOB
  object_emb    BLOB
  claim_emb     BLOB
"""

import json
import sys
import time
import re
import sqlite3
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Union

import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

OLLAMA_SERVER = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b-it-qat"

REQUEST_TIMEOUT = 120
TEMPERATURE = 0.2
MAX_RETRIES = 3
RETRY_WAIT = 2.0

EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64
NORMALIZE_EMBEDDINGS = True

TABLE_NAME = "claims"


# --------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------

TOSTRUCTURE_PROMPT = """You are an information extraction model.

Given a short factual or normative statement (a legal or regulatory claim),
identify its logical components and return them as JSON with the keys:

- subject: the main actor or entity performing or constrained.
- predicate: the main verb or relation phrase expressing the action or relation.
- object: the entity, concept, or outcome affected.

Rules:
- Keep the extracted text spans concise, literal, and faithful to the claim.
- Preserve terminology (do not paraphrase unnecessarily).
- If an element is missing, set it to "" (empty string).
- Output MUST be a single JSON object with exactly these keys and string values.
- Do not output explanations, comments, or any text before or after the JSON.

Example:
Input: Providers of general-purpose AI models must publish summaries of training data.
Output:
{{"subject": "providers of general-purpose AI models",
  "predicate": "must publish",
  "object": "summaries of training data"}}

Now extract for this claim:

"{claim_text}"

Output only the JSON object.
"""


# --------------------------------------------------------------------
# Ollama helper
# --------------------------------------------------------------------

def call_ollama(prompt: str) -> str:
    url = OLLAMA_SERVER.rstrip("/") + "/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": float(TEMPERATURE)},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            return (obj.get("message") or {}).get("content", "").strip()
        except Exception as e:
            wait = RETRY_WAIT * (2 ** attempt)
            print(f"[warn] LLM call failed ({e}) → retry in {wait:.1f}s")
            time.sleep(wait)
    return ""


def safe_json_object(s: str) -> Dict[str, Any]:
    """Extract JSON object {...} from string. Returns {} on failure."""
    s = s.strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        obj = json.loads(s[start:end + 1])
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def clean_slot(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        return str(val)
    if not isinstance(val, str):
        return ""
    return val.strip()


# --------------------------------------------------------------------
# SQLite + embedding helpers
# --------------------------------------------------------------------

def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            claim_id      TEXT PRIMARY KEY,
            claim_text    TEXT NOT NULL,
            subject       TEXT,
            predicate     TEXT,
            object        TEXT,
            subject_emb   BLOB,
            predicate_emb BLOB,
            object_emb    BLOB,
            claim_emb     BLOB
        )
    """)
    conn.commit()


def to_blob(vec: np.ndarray) -> bytes:
    if vec.dtype != np.float32:
        vec = vec.astype(np.float32, copy=False)
    return vec.tobytes(order="C")


def insert_rows(conn: sqlite3.Connection, rows: List[Dict[str, Any]]) -> None:
    payload = [
        (
            r["claim_id"],
            r["claim_text"],
            r.get("subject", ""),
            r.get("predicate", ""),
            r.get("object", ""),
            to_blob(r["subject_emb"]),
            to_blob(r["predicate_emb"]),
            to_blob(r["object_emb"]),
            to_blob(r["claim_emb"]),
        )
        for r in rows
    ]
    conn.executemany(f"""
        INSERT OR REPLACE INTO {TABLE_NAME}
        (claim_id, claim_text, subject, predicate, object,
         subject_emb, predicate_emb, object_emb, claim_emb)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, payload)
    conn.commit()


# --------------------------------------------------------------------
# Core logic
# --------------------------------------------------------------------

def extract_triples_from_file(input_path: Union[str, Path]) -> Path:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    base = re.sub(r"_claims$", "", input_path.stem)
    jsonl_out = input_path.with_name(f"{base}_triples.jsonl")
    sqlite_out = input_path.with_name(f"{base}_triples.sqlite")

    embedder = SentenceTransformer(EMBED_MODEL_ID)
    conn = sqlite3.connect(sqlite_out)
    ensure_table(conn)

    structured_claims: List[Dict[str, Any]] = []
    t0 = time.time()

    # Count total lines for progress logging
    total_lines = sum(1 for _ in input_path.open("r", encoding="utf-8"))

    # --- Read input JSONL and extract structure via LLM ---
    with input_path.open("r", encoding="utf-8") as f, jsonl_out.open("w", encoding="utf-8") as out:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                claim = json.loads(line)
            except Exception:
                print(f"[warn] Skipping malformed JSON line {i}")
                continue

            claim_id = claim.get("claim_id")
            claim_text = clean_slot(claim.get("claim_text"))
            if not claim_id or not claim_text:
                continue

            prompt = TOSTRUCTURE_PROMPT.format(claim_text=claim_text)
            reply = call_ollama(prompt)
            parsed = safe_json_object(reply)

            subject = clean_slot(parsed.get("subject", ""))
            predicate = clean_slot(parsed.get("predicate", ""))
            obj = clean_slot(parsed.get("object", ""))

            rec = {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "subject": subject,
                "predicate": predicate,
                "object": obj,
            }

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            structured_claims.append(rec)

            print(f"[{i}/{total_lines}] processed {claim_id}", flush=True)

    # --- Embedding phase ---
    print(f"\nEmbedding {len(structured_claims)} structured claims...")

    rows_for_db: List[Dict[str, Any]] = []
    for start in range(0, len(structured_claims), BATCH_SIZE):
        batch = structured_claims[start:start + BATCH_SIZE]

        subs = [clean_slot(r.get("subject", "")) for r in batch]
        preds = [clean_slot(r.get("predicate", "")) for r in batch]
        objs = [clean_slot(r.get("object", "")) for r in batch]
        claims = [clean_slot(r.get("claim_text", "")) for r in batch]

        emb_sub = embedder.encode(subs, convert_to_numpy=True, normalize_embeddings=False)
        emb_pred = embedder.encode(preds, convert_to_numpy=True, normalize_embeddings=False)
        emb_obj = embedder.encode(objs, convert_to_numpy=True, normalize_embeddings=False)
        emb_claim = embedder.encode(claims, convert_to_numpy=True, normalize_embeddings=False)

        if NORMALIZE_EMBEDDINGS:
            for arr in (emb_sub, emb_pred, emb_obj, emb_claim):
                norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
                arr /= norms

        for j, r in enumerate(batch):
            rows_for_db.append({
                **r,
                "subject_emb": emb_sub[j],
                "predicate_emb": emb_pred[j],
                "object_emb": emb_obj[j],
                "claim_emb": emb_claim[j],
            })

    if rows_for_db:
        insert_rows(conn, rows_for_db)

    conn.close()

    dt = time.time() - t0
    print(f"\n✓ Done. Extracted and embedded {len(structured_claims)} triples in {dt:.1f}s")
    print(f"   JSONL → {jsonl_out}")
    print(f"   SQLite → {sqlite_out}")
    return jsonl_out


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python extract_triples.py INPUT_CLAIMS_JSONL", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    try:
        extract_triples_from_file(input_path)
    except Exception as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
