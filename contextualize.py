"""
contextualize.py — Sequential contextualization pipeline (middle step).

Input:
  <base>_chunks.json   # produced by chunk_and_embed.py

Outputs:
  <base>_contextualized.json
  <base>_contextualized.sqlite (table: contextualized_chunks)

Modes:
  --mode prefix   (default)
      - LLM writes a short contextual note ("context prefix").
      - contextualized_chunk = context_prefix + " " + chunk

  --mode rewrite
      - LLM REWRITES the passage into a self-contained, compact version,
        integrating necessary context.
      - contextualized_chunk = rewritten_passage
      - context_prefix = rewritten_passage  (no separate note)
"""

import json
import sys
import time
import re
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Union

import urllib.request
import numpy as np
from sentence_transformers import SentenceTransformer
from jinja2 import Environment, FileSystemLoader
import argparse


# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

# LLM setup
OLLAMA_SERVER = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b-it-qat"
REQUEST_TIMEOUT = 120
TEMPERATURE = 0.1
MAX_RETRIES = 3
RETRY_WAIT = 2.0

# Embedding model
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
NORMALIZE_EMBEDDINGS = True

# Local context window
LOCAL_BEFORE = 2
LOCAL_AFTER = 1

# Retrieval config (over *previous* contextualized chunks)
TOP_K_RETRIEVE = 2

TABLE_NAME = "contextualized_chunks"

# Prompt templates (Jinja2)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PREFIX_TEMPLATE_NAME = "contextualize_prefix.j2"
REWRITE_TEMPLATE_NAME = "contextualize_rewrite.j2"

# Modes
MODE_PREFIX = "prefix"
MODE_REWRITE = "rewrite"


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
# Ollama call
# --------------------------------------------------------------------

def call_ollama(server: str, model: str, prompt: str, timeout: int, temperature: float) -> str:
    """Call a local Ollama model via /api/chat and return the reply text."""
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


# --------------------------------------------------------------------
# BM25 helpers
# --------------------------------------------------------------------

def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    return [t for t in re.split(r"[^a-z0-9]+", text) if t]


def bm25_scores(query_tokens, docs_tokens, df, doc_lengths, k1=1.5, b=0.75):
    N = len(docs_tokens)
    if N == 0:
        return []
    avgdl = sum(doc_lengths) / float(N)
    from collections import Counter
    unique_terms = set(query_tokens)
    idf = {
        t: np.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
        for t in unique_terms
    }
    scores = [0.0] * N
    for i, doc in enumerate(docs_tokens):
        freqs = Counter(doc)
        dl = len(doc)
        for t in unique_terms:
            f = freqs.get(t, 0)
            if not f:
                continue
            denom = f + k1 * (1 - b + b * dl / avgdl)
            scores[i] += idf[t] * (f * (k1 + 1)) / denom
    return scores


# --------------------------------------------------------------------
# SQLite helpers
# --------------------------------------------------------------------

def ensure_table(conn: sqlite3.Connection):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            doc            TEXT NOT NULL,
            id             TEXT PRIMARY KEY,
            chunk_raw      TEXT NOT NULL,
            context_prefix TEXT NOT NULL,
            chunk_ctx      TEXT NOT NULL,
            emb_ctx        BLOB NOT NULL
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_doc ON {TABLE_NAME}(doc)")
    conn.commit()


def to_blob(vec: np.ndarray) -> bytes:
    if vec.dtype != np.float32:
        vec = vec.astype(np.float32, copy=False)
    return vec.tobytes(order="C")


def insert_rows(conn: sqlite3.Connection, rows: List[Dict[str, Any]]):
    payload = [
        (
            r["doc"],
            r["id"],
            r["chunk_raw"],
            r["context_prefix"],
            r["chunk_ctx"],
            to_blob(r["emb_ctx"]),
        )
        for r in rows
    ]
    conn.executemany(
        f"""INSERT OR REPLACE INTO {TABLE_NAME}
            (doc, id, chunk_raw, context_prefix, chunk_ctx, emb_ctx)
            VALUES (?, ?, ?, ?, ?, ?)""",
        payload,
    )
    conn.commit()


# --------------------------------------------------------------------
# Core contextualization logic
# --------------------------------------------------------------------

def contextualize_document(
    doc: str,
    chunks: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    mode: str = MODE_PREFIX,
):
    if mode not in {MODE_PREFIX, MODE_REWRITE}:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"  Document '{doc}' — {len(chunks)} chunks (mode={mode})")

    # Retrieval memory over *contextualized* history
    ctx_texts: List[str] = []
    ctx_embs: List[np.ndarray] = []
    ctx_tokens: List[List[str]] = []
    df: Dict[str, int] = {}
    doc_lengths: List[int] = []

    results: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for i, item in enumerate(chunks):
        passage = item["chunk"]
        chunk_id = item["id"]

        # 1) Local context (raw neighbors)
        before = [c["chunk"] for c in chunks[max(0, i - LOCAL_BEFORE):i]]
        after = [c["chunk"] for c in chunks[i + 1:i + 1 + LOCAL_AFTER]]
        local_context = "\n\n".join(before + after) or "(no local context)"

        # 2) Retrieval over previous contextualized chunks
        retrieved_context = "(no retrieved context)"
        if ctx_texts:
            # Query embedding from the current *raw* passage (not stored)
            q_emb = embedder.encode(
                [passage],
                convert_to_numpy=True,
                normalize_embeddings=NORMALIZE_EMBEDDINGS,
            )[0]

            dense_scores = [float(np.dot(e, q_emb)) for e in ctx_embs]

            query_tokens = simple_tokenize(passage)
            bm25_sc = bm25_scores(query_tokens, ctx_tokens, df, doc_lengths)

            dense_top = sorted(
                range(len(dense_scores)), key=lambda idx: dense_scores[idx], reverse=True
            )[:TOP_K_RETRIEVE]
            bm25_top = sorted(
                range(len(bm25_sc)), key=lambda idx: bm25_sc[idx], reverse=True
            )[:TOP_K_RETRIEVE]

            seen, merged = set(), []
            for idx in dense_top + bm25_top:
                if idx not in seen:
                    seen.add(idx)
                    merged.append(idx)

            if merged:
                retrieved_context = "\n\n---\n\n".join(ctx_texts[j] for j in merged)

        # 3) Prompt LLM to produce context (prefix or rewrite)
        if mode == MODE_PREFIX:
            template_name = PREFIX_TEMPLATE_NAME
        else:
            template_name = REWRITE_TEMPLATE_NAME

        prompt = render_prompt(
            template_name,
            current_passage=passage,
            local_context=local_context,
            retrieved_context=retrieved_context,
            doc=doc,
        )

        reply = ""
        for attempt in range(MAX_RETRIES):
            try:
                reply = call_ollama(
                    OLLAMA_SERVER,
                    OLLAMA_MODEL,
                    prompt,
                    REQUEST_TIMEOUT,
                    TEMPERATURE,
                )
                break
            except Exception as e:
                wait = RETRY_WAIT * (2 ** attempt)
                print(f"[warn] {doc}/{chunk_id} attempt {attempt+1}: {e} → wait {wait:.1f}s")
                time.sleep(wait)

        reply = reply.strip()

        if mode == MODE_PREFIX:
            context_prefix = reply or "(No additional context found.)"
            contextualized = f"{context_prefix} {passage}"
        else:
            # REWRITE mode:
            # LLM returns a self-contained rewritten passage.
            # We store it both as context_prefix and chunk_ctx.
            contextualized = reply or passage
            context_prefix = contextualized

        # 4) Embed ONLY the contextualized text
        emb_ctx = embedder.encode(
            [contextualized],
            convert_to_numpy=True,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        )[0]

        # 5) Collect outputs
        results.append({
            "doc": doc,
            "id": chunk_id,
            "chunk": passage,
            "contextualized_chunk": contextualized,
        })
        rows.append({
            "doc": doc,
            "id": chunk_id,
            "chunk_raw": passage,
            "context_prefix": context_prefix,
            "chunk_ctx": contextualized,
            "emb_ctx": emb_ctx,
        })

        # 6) Update retrieval memory
        ctx_texts.append(contextualized)
        ctx_embs.append(emb_ctx)
        toks = simple_tokenize(contextualized)
        ctx_tokens.append(toks)
        doc_lengths.append(len(toks))
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

        print(f"    [{i+1}/{len(chunks)}] contextualized for '{doc}'", flush=True)

    return results, rows


# --------------------------------------------------------------------
# Top-level function
# --------------------------------------------------------------------

def contextualize_file(input_path: Union[str, Path], mode: str = MODE_PREFIX) -> Path:
    input_path = Path(input_path)
    base = re.sub(r"_chunks$", "", input_path.stem)

    embedder = SentenceTransformer(EMBED_MODEL_ID)
    items = json.loads(input_path.read_text(encoding="utf-8"))

    # group by document
    docs: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        docs.setdefault(it["doc"], []).append(it)

    sqlite_path = input_path.with_name(f"{base}_contextualized.sqlite")
    conn = sqlite3.connect(sqlite_path)
    ensure_table(conn)

    all_results: List[Dict[str, Any]] = []
    t0 = time.time()

    def sort_id(o):  # numeric sort
        m = re.search(r"_(\d+)$", o["id"])
        return int(m.group(1)) if m else 10**9

    for doc, lst in docs.items():
        lst = sorted(lst, key=sort_id)
        res, rows = contextualize_document(doc, lst, embedder, mode=mode)
        all_results.extend(res)
        insert_rows(conn, rows)

    conn.close()

    json_path = input_path.with_name(f"{base}_contextualized.json")
    json_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n✓ Done. Contextualized {len(all_results)} chunks → {json_path}")
    print(f"   Contextualized embeddings stored in {sqlite_path}")
    print(f"   Total time: {time.time() - t0:.1f}s")
    return json_path


# --------------------------------------------------------------------
# CLI entry
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sequential contextualization pipeline (prefix or rewrite mode)."
    )
    parser.add_argument("input_chunks_json", help="Input <base>_chunks.json")
    parser.add_argument(
        "--mode",
        choices=[MODE_PREFIX, MODE_REWRITE],
        default=MODE_PREFIX,
        help="Contextualization mode: 'prefix' (default) or 'rewrite'.",
    )

    args = parser.parse_args()
    contextualize_file(args.input_chunks_json, mode=args.mode)


if __name__ == "__main__":
    main()
