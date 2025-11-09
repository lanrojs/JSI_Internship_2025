"""
contextualize.py — Sequential contextualization and embedding pipeline.

Input:
  <base>_chunks.json   # produced by chunks.py
Output:
  <base>_contextualized.json
  <base>_contextualized.sqlite (table: contextualized_chunks)

Each chunk is processed sequentially:
  1. Build local raw context (neighbors).
  2. Retrieve top-N previous contextualized chunks by dense similarity
     and top-M by BM25 lexical similarity (configurable).
  3. Ask LLM for 2–3 sentences of explanatory context.
  4. Append the original chunk.
  5. Embed both raw + contextualized versions and store them.
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


# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

# LLM setup
OLLAMA_SERVER = "http://localhost:11434"
OLLAMA_MODEL = "llama3:8b"        # or "gemma3:4b-it-qat"
REQUEST_TIMEOUT = 120
TEMPERATURE = 0.2
MAX_RETRIES = 3
RETRY_WAIT = 2.0

# Embedding model
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
NORMALIZE_EMBEDDINGS = True

# Local context window
LOCAL_BEFORE = 2
LOCAL_AFTER = 1

# Retrieval config
TOP_K_DENSE = 3       # how many previous chunks to retrieve by dense similarity
TOP_K_BM25 = 2        # how many previous chunks to retrieve by BM25
TABLE_NAME = "contextualized_chunks"


# --------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------

CONTEXTUALIZE_ENRICH_PROMPT = """
You are a precise document contextualizer.

Goal
-----
Given one SHORT passage ("CURRENT PASSAGE") and some surrounding material
("LOCAL RAW CONTEXT" and "RETRIEVED CONTEXT"), write 2–3 sentences of
EXPLANATORY CONTEXT that make the passage self-contained.

The result will later be concatenated with the CURRENT PASSAGE verbatim, so:
- Do NOT quote, paraphrase, or restate the CURRENT PASSAGE.
- Do NOT put quotation marks around it.
- Only write new explanatory sentences.

Grounding rules
---------------
1. Determine the MAIN TOPIC of the passage **only** from CURRENT PASSAGE
   (and any visible section headings) and LOCAL RAW CONTEXT.
2. Use RETRIEVED CONTEXT only to clarify ambiguous references or add small,
   factual details (dates, names, figures) — never to change the topic.
3. If a topic or phrase does not appear in the CURRENT PASSAGE or LOCAL RAW CONTEXT,
   do not mention it.
4. Stay grounded in the text. Never speculate or generalize beyond the context.

Your sentences must:
1. Identify what document this is and roughly where this passage fits
   (e.g., “financial results section of a company report,”
   “legal definitions in a data privacy policy,” etc.).
2. Clarify pronouns and vague references (“the company,” “it,” “previous quarter”).
3. Add concrete facts **only if** they appear explicitly in the context.
4. Avoid repetition of sentences already present in the CURRENT PASSAGE.

Style:
- 2–3 fluent, information-rich sentences in plain English, avoid repeating yourself
- Begin directly with the explanation — no meta phrases like
  “This passage says…” or “Here is the context.”
- Avoid bullet points, lists, or commentary.
- Maintain a neutral, professional tone suitable for regulatory or corporate text.

Common mistakes to avoid
------------------------
- Reusing irrelevant topics from distant sections (e.g., calling a human-capital
   paragraph a “supply chain” discussion).
- If a concept (for example “supply chain”, “components”, “single or limited sources”)
  does not appear in the CURRENT PASSAGE or LOCAL RAW CONTEXT, do NOT mention it.

Good example of contextualization that you should aim for:
--------
CURRENT PASSAGE:
The company's revenue grew by 3% over the previous quarter.

CONTEXTUALIZED PASSAGE:
This chunk is from an SEC filing on ACME Corp’s performance in Q2 2023;
the previous quarter’s revenue was $314 million.
The company’s revenue grew by 3% over the previous quarter.

Notice that the contextualization adds short explanatory sentences 
that clarify *where* the passage comes from and *what* it refers to.

-------------------- LOCAL RAW CONTEXT --------------------
<LOCAL_CONTEXT_START>
{local_context}
<LOCAL_CONTEXT_END>

-------------------- RETRIEVED CONTEXT --------------------
<RETRIEVED_CONTEXT_START>
{retrieved_context}
<RETRIEVED_CONTEXT_END>

-------------------- CURRENT PASSAGE --------------------
<CURRENT_PASSAGE_START>
{current_passage}
<CURRENT_PASSAGE_END>
"""


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
            doc       TEXT NOT NULL,
            id        TEXT PRIMARY KEY,
            chunk_raw TEXT NOT NULL,
            emb_raw   BLOB NOT NULL,
            chunk_ctx TEXT NOT NULL,
            emb_ctx   BLOB NOT NULL
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
        (r["doc"], r["id"], r["chunk_raw"], to_blob(r["emb_raw"]),
         r["chunk_ctx"], to_blob(r["emb_ctx"]))
        for r in rows
    ]
    conn.executemany(
        f"""INSERT OR REPLACE INTO {TABLE_NAME}
            (doc, id, chunk_raw, emb_raw, chunk_ctx, emb_ctx)
            VALUES (?, ?, ?, ?, ?, ?)""",
        payload,
    )
    conn.commit()


# --------------------------------------------------------------------
# Core contextualization logic
# --------------------------------------------------------------------

def contextualize_document(doc: str, chunks: List[Dict[str, Any]], embedder: SentenceTransformer):
    print(f"  Document '{doc}' — {len(chunks)} chunks")

    ctx_texts, ctx_embs, ctx_tokens = [], [], []
    df, doc_lengths = {}, []
    results, rows = [], []

    for i, item in enumerate(chunks):
        passage = item["chunk"]
        chunk_id = item["id"]

        # 0) Embed the raw passage ONCE per chunk and reuse it
        q_emb = embedder.encode(
            [passage],
            convert_to_numpy=True,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        )[0]

        # 1) Local context (raw)
        before = [c["chunk"] for c in chunks[max(0, i - LOCAL_BEFORE):i]]
        after = [c["chunk"] for c in chunks[i + 1:i + 1 + LOCAL_AFTER]]
        local_context = "\n\n".join(before + after) or "(no local context)"

        # 2) Retrieval over previous contextualized chunks
        retrieved_context = "(no retrieved context)"
        if ctx_texts:
            # Dense similarity over contextualized embeddings, no vstack
            dense_scores = [float(np.dot(e, q_emb)) for e in ctx_embs]

            # BM25 similarity
            query_tokens = simple_tokenize(passage)
            bm25_sc = bm25_scores(query_tokens, ctx_tokens, df, doc_lengths)

            # Top-N from each
            dense_top = sorted(
                range(len(dense_scores)), key=lambda i: dense_scores[i], reverse=True
            )[:TOP_K_DENSE]
            bm25_top = sorted(
                range(len(bm25_sc)), key=lambda i: bm25_sc[i], reverse=True
            )[:TOP_K_BM25]

            # Merge unique (preserve order)
            seen, merged = set(), []
            for idx in dense_top + bm25_top:
                if idx not in seen:
                    seen.add(idx)
                    merged.append(idx)

            if merged:
                retrieved_context = "\n\n---\n\n".join(ctx_texts[i] for i in merged)

        # 3) Prompt + LLM
        prompt = CONTEXTUALIZE_ENRICH_PROMPT.format(
            current_passage=passage,
            local_context=local_context,
            retrieved_context=retrieved_context,
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

        prefix = reply.strip()
        if not prefix:
            prefix = "(No additional context found.)"
        contextualized = f"{prefix} {passage}"

        # 4) Embeddings (raw + contextualized)
        #    Reuse q_emb for emb_raw instead of re-encoding
        emb_raw = q_emb
        emb_ctx = embedder.encode(
            [contextualized],
            convert_to_numpy=True,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        )[0]

        # 5) Save outputs
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
            "emb_raw": emb_raw,
            "chunk_ctx": contextualized,
            "emb_ctx": emb_ctx,
        })

        # 6) Update retrieval index
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

def contextualize_file(input_path: Union[str, Path]) -> Path:
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

    all_results = []
    t0 = time.time()

    def sort_id(o):  # numeric sort
        m = re.search(r"_(\d+)$", o["id"])
        return int(m.group(1)) if m else 1e9

    for doc, lst in docs.items():
        lst = sorted(lst, key=sort_id)
        res, rows = contextualize_document(doc, lst, embedder)
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python contextualize.py INPUT_CHUNKS_JSON", file=sys.stderr)
        sys.exit(1)
    contextualize_file(sys.argv[1])
