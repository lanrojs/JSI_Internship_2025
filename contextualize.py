"""
contextualize.py — Contextual retrieval & summarization pipeline.

Input:
  JSON list of chunks, e.g.:
    { "doc": "...", "id": "...", "chunk": "..." }

Output:
  <base>_contextualized.json with:
    {
      "doc": "...",
      "id": "...",
      "chunk": "...",                  # original passage
      "contextualized_chunk": "..."    # short plain-text contextual summary
    }

The contextualized_chunk is *only* a compact description (2–3 sentences),
with no headings, quotes, or repetition of the original text.

Usage from command line:
    python contextualize.py INPUT_CHUNKS_JSON

Usage from notebook (ipynb):
    from contextualize import contextualize_file
    contextualize_file("gdpr2_chunks.json")
"""

import json
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import urllib.request


# Try to reuse embedding model id from embedding.py if available.
# If that import fails (e.g. when running standalone), fall back to a default.
try:
    from embedding import EMBED_MODEL_ID as DEFAULT_EMBED_MODEL_ID
except ImportError:
    DEFAULT_EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"


# --------------------------------------------------------------------
# Configuration (edit these constants instead of adding CLI args)
# --------------------------------------------------------------------

# Ollama server + model for contextualization
OLLAMA_SERVER = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b-it-qat" # gemma3:4b-it-qat

# LLM call behavior
REQUEST_TIMEOUT = 120      # seconds per request
TEMPERATURE = 0.5          # 0.0 = deterministic / stable behavior
MAX_RETRIES = 3            # how many times to retry on network error
RETRY_WAIT = 2.0           # base backoff in seconds

# Progress logging
PROGRESS_EVERY = 1         # print after every N chunks

# Embedding model for dense retrieval over *contextualized* passages
EMBED_MODEL_ID = DEFAULT_EMBED_MODEL_ID

# Local context window size (in number of raw chunks)
LOCAL_BEFORE = 2           # how many chunks before the current one
LOCAL_AFTER = 2            # how many chunks after the current one

# Retrieval parameters for hybrid dense + BM25 over previous contextualizations
TOP_K_DENSE = 3            # top-K by dense similarity
TOP_K_BM25 = 3             # top-K by BM25 score
MAX_RETRIEVED = 3          # max number of contextualized passages to include
MIN_COMBINED_SCORE = 0.1   # discard candidates below this normalized score


# --------------------------------------------------------------------
# Prompt used for contextualization
# --------------------------------------------------------------------

CONTEXTUALIZE_PROMPT = """You are a concise text contextualizer.

Your task is to write a short, self-contained description (2–3 sentences)
that captures the specific meaning and role of ONE passage within a longer document.

Very important constraints:
- BASE YOUR DESCRIPTION ONLY ON THE CURRENT PASSAGE TEXT.
- You may use the DOCUMENT CONTEXT only to disambiguate or locate the passage,
  but you MUST NOT introduce facts that are not clearly present in the CURRENT PASSAGE itself.
- Do NOT invent or import details from earlier or later parts of the document.

Coverage requirement:
- Read the ENTIRE CURRENT PASSAGE carefully.
- If the passage clearly contains MORE THAN ONE distinct idea (e.g. finishes one article
  and begins another, or moves from scope to principles), your summary MUST briefly
  mention each major idea at least once.
- Do NOT focus only on the beginning of the passage; reflect the main content across
  the whole passage.

Focus on:
- What concrete ideas, rules, or facts appear in THIS passage.
- How THIS passage functions in the document (e.g., defines scope, lists principles,
  sets territorial rules, defines legal bases), but only if that function is evident
  from the CURRENT PASSAGE text.
- Any key terms, principles, or distinctions that are explicitly introduced here.

Avoid:
- Restating the general theme of the whole chapter if it is not specifically expressed here.
- Mentioning articles, sections, regulations, exclusions, or actors that do NOT appear
  in the CURRENT PASSAGE text (even if they appeared in earlier context).
- Quoting the passage or paraphrasing it line by line.

Style:
- 2–3 sentences, plain English.
- No headings, no bullet points, no quotes from the passage.
- The result should read like a compact explanation of THIS passage only.

--------------------
DOCUMENT CONTEXT (for reference only – do NOT add new facts from here)
--------------------
\"\"\"{local_context}

{retrieved_context}\"\"\"

--------------------
CURRENT PASSAGE (the only source of facts you may describe)
--------------------
\"\"\"{current_passage}\"\"\""""


# --------------------------------------------------------------------
# Ollama call
# --------------------------------------------------------------------

def call_ollama(server: str, model: str, prompt: str, timeout: int, temperature: float) -> str:
    """
    Call a local Ollama model via /api/chat and return the assistant's reply text.

    This is used to generate the contextualized summary for each passage.
    """
    url = server.rstrip("/") + "/api/chat"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,  # we want a single final response
        "options": {"temperature": float(temperature)},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))

    # Ollama returns {"message": {"content": "..."}}
    return (obj.get("message") or {}).get("content", "").strip()


# --------------------------------------------------------------------
# BM25 helpers (simple implementation, no external dependency)
# --------------------------------------------------------------------

def simple_tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer for BM25:
      - lowercase
      - split on non-alphanumeric characters
      - drop empty tokens
    """
    text = text.lower()
    return [t for t in re.split(r"[^a-z0-9]+", text) if t]


def bm25_scores(query_tokens, docs_tokens, df, doc_lengths, k1: float = 1.5, b: float = 0.75):
    """
    Compute BM25 scores of a query against a list of tokenized documents.

    Parameters
    ----------
    query_tokens : List[str]
        Tokens of the query (current passage).
    docs_tokens : List[List[str]]
        Tokens of each document in the index (contextualized summaries).
    df : Dict[str, int]
        Document frequency for each term (how many docs contain the term).
    doc_lengths : List[int]
        Length of each document in tokens.
    """
    N = len(docs_tokens)
    if N == 0:
        return []

    avgdl = sum(doc_lengths) / float(N)

    from collections import Counter
    unique_terms = set(query_tokens)

    # Precompute IDF values for terms that appear in the query
    idf_cache = {
        t: np.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
        for t in unique_terms
    }

    scores = [0.0] * N

    for i, doc in enumerate(docs_tokens):
        freqs = Counter(doc)  # term frequencies in this doc
        dl = len(doc)

        for t in unique_terms:
            f = freqs.get(t, 0)
            if not f:
                continue
            idf = idf_cache[t]
            denom = f + k1 * (1 - b + b * dl / avgdl)
            scores[i] += idf * (f * (k1 + 1)) / denom

    return scores


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize a list of scores to the [0, 1] range.

    If all scores are the same (or list is empty), returns zeros.
    """
    if not scores:
        return []

    mn, mx = min(scores), max(scores)
    if mx <= mn + 1e-9:
        return [0.0] * len(scores)

    return [(s - mn) / (mx - mn) for s in scores]


# --------------------------------------------------------------------
# Cleaning helper
# --------------------------------------------------------------------

def clean_summary(text: str) -> str:
    """
    Clean the LLM's output for contextualized_chunk.

    We strip whitespace and also remove any accidental prefixes like:
      "[Context]", "Context:", etc., in case the model ignores instructions.
    """
    text = text.strip()
    text = re.sub(r'^\s*\[?context\]?\s*[:\-]*\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


# --------------------------------------------------------------------
# Contextualization per document
# --------------------------------------------------------------------

def contextualize_document(doc: str, chunks: List[Dict[str, Any]], embedder: SentenceTransformer) -> List[Dict[str, Any]]:
    """
    Contextualize all chunks belonging to a single document.

    Steps (per chunk):
      1. Build local raw context from neighboring chunks (before/after).
      2. Retrieve relevant previously-contextualized summaries using
         hybrid dense + BM25 over earlier chunks in this document.
      3. Build an LLM prompt with:
            - local raw context
            - retrieved contextual summaries
            - current raw passage
      4. Ask the LLM for a short, self-contained description.
      5. Store the new contextualized summary in the in-memory index
         so it can be retrieved for later chunks.
    """
    print(f"  Document '{doc}': {len(chunks)} chunks", flush=True)

    # In-memory retrieval index for previously contextualized summaries
    ctx_texts: List[str] = []     # contextualized summaries as text
    ctx_embs: List[np.ndarray] = []   # dense embeddings for summaries
    ctx_tokens: List[List[str]] = []  # tokenized summaries for BM25
    df: Dict[str, int] = {}           # document frequency per term
    doc_lengths: List[int] = []       # summary length in tokens

    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(chunks):
        current_passage = str(item["chunk"])
        chunk_id = str(item["id"])

        # ---------------------------
        # 1) Build local context
        # ---------------------------
        local_parts: List[str] = []

        # Include up to LOCAL_BEFORE raw chunks *before* this one
        for j in range(max(0, idx - LOCAL_BEFORE), idx):
            local_parts.append(chunks[j]["chunk"])

        # Include up to LOCAL_AFTER raw chunks *after* this one
        for j in range(idx + 1, min(len(chunks), idx + LOCAL_AFTER + 1)):
            local_parts.append(chunks[j]["chunk"])

        local_context = "\n\n".join(local_parts) or "(no local context)"

        # ---------------------------
        # 2) Retrieve from previous contextualized summaries (hybrid)
        # ---------------------------
        retrieved_context = "(no retrieved context)"
        if ctx_texts:
            # Dense similarity: encode current passage and compare to all previous summaries
            q_emb = embedder.encode(
                [current_passage],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
            ctx_matrix = np.vstack(ctx_embs)  # shape: (num_prev, dim)
            dense_scores = (ctx_matrix @ q_emb).tolist()

            # BM25: treat current passage as query, previous summaries as docs
            query_tokens = simple_tokenize(current_passage)
            bm25_sc = bm25_scores(query_tokens, ctx_tokens, df, doc_lengths)

            # Normalize both score lists and combine 50/50
            dense_norm = normalize_scores(dense_scores)
            bm25_norm = normalize_scores(bm25_sc)
            combined = [0.5 * d + 0.5 * b for d, b in zip(dense_norm, bm25_norm)]

            # Rank previously contextualized passages by combined score
            idxs = sorted(range(len(ctx_texts)), key=lambda i: combined[i], reverse=True)

            # Keep only those above the relevance threshold, up to MAX_RETRIEVED
            top_idxs = [i for i in idxs if combined[i] >= MIN_COMBINED_SCORE][:MAX_RETRIEVED]
            if top_idxs:
                retrieved_context = "\n\n---\n\n".join(ctx_texts[i] for i in top_idxs)

        # ---------------------------
        # 3) Build prompt for LLM
        # ---------------------------
        prompt = CONTEXTUALIZE_PROMPT.format(
            current_passage=current_passage,
            local_context=local_context,
            retrieved_context=retrieved_context,
        )

        # ---------------------------
        # 4) Call LLM with retries
        # ---------------------------
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
                # Simple exponential backoff on error
                wait = RETRY_WAIT * (2 ** attempt)
                print(f"[warn] {doc}/{chunk_id} attempt {attempt+1}/{MAX_RETRIES} failed: {e}. Waiting {wait:.1f}s", flush=True)
                time.sleep(wait)

        summary = clean_summary(reply)
        if not summary:
            # Fallback: if model returned nothing usable
            summary = "(no context produced)"

        # Save result for this chunk
        results.append({
            "doc": doc,
            "id": chunk_id,
            "chunk": current_passage,
            "contextualized_chunk": summary,
        })

        # ---------------------------
        # 5) Update retrieval index
        # ---------------------------
        ctx_texts.append(summary)

        # Dense embedding for the new summary
        ctx_vec = embedder.encode(
            [summary],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        ctx_embs.append(ctx_vec)

        # BM25 statistics for the new summary
        tokens = simple_tokenize(summary)
        ctx_tokens.append(tokens)
        doc_lengths.append(len(tokens))
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

        # Progress printout
        if (idx + 1) % PROGRESS_EVERY == 0:
            print(f"    [{idx + 1}/{len(chunks)}] contextualized for '{doc}'", flush=True)

    return results


# --------------------------------------------------------------------
# Top-level runner (for CLI and notebooks)
# --------------------------------------------------------------------

def contextualize_file(input_path: Union[str, Path]) -> Path:
    """
    Run the full contextualization pipeline on a JSON chunks file.

    Parameters
    ----------
    input_path : str or Path
        Path to a JSON file containing a list of objects:
          { "doc": "...", "id": "...", "chunk": "..." }

    Returns
    -------
    Path
        Path to the written <base>_contextualized.json file.

    This function is safe to call from a Jupyter notebook, e.g.:

        from contextualize import contextualize_file
        contextualize_file("gdpr2_chunks.json")
    """
    input_path = Path(input_path)

    # Load JSON list of chunks
    items = json.loads(input_path.read_text(encoding="utf-8"))

    # Group chunks by document id
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for obj in items:
        by_doc.setdefault(obj["doc"], []).append(obj)

    # Shared embedder for all documents
    embedder = SentenceTransformer(EMBED_MODEL_ID)

    all_results: List[Dict[str, Any]] = []
    t0 = time.time()

    # Helper to sort chunk ids numerically if they end in _N
    def sort_id(o: Dict[str, Any]) -> int:
        m = re.search(r"_(\d+)$", o["id"])
        return int(m.group(1)) if m else 10**9

    # Process each document independently
    for doc, lst in by_doc.items():
        doc_sorted = sorted(lst, key=sort_id)
        all_results.extend(contextualize_document(doc, doc_sorted, embedder))

    # Derive output filename: <base>_contextualized.json
    out_path = input_path.with_name(
        re.sub(r"_chunks$", "", input_path.stem) + "_contextualized.json"
    )

    # Write all contextualized chunks
    out_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✓ Done. Contextualized {len(all_results)} chunks → {out_path} ({time.time()-t0:.1f}s)")
    return out_path


# --------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------

if __name__ == "__main__":
    # Command-line usage:
    #   python contextualize.py INPUT_CHUNKS_JSON
    if len(sys.argv) != 2:
        print(f"Usage: python contextualize.py INPUT_CHUNKS_JSON", file=sys.stderr)
        sys.exit(1)

    contextualize_file(sys.argv[1])
