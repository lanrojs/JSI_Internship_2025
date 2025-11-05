"""
contextualize.py — Anthropic-style contextual retrieval & contextualization.

Pipeline (per document):
  1. Input: a JSON list of chunks, each like:
       { "doc": "...", "id": "...", "chunk": "..." }

  2. For each document:
       - Process chunks sequentially in order (id 1, 2, 3, ...).
       - For each current chunk:
           * Build LOCAL CONTEXT: up to 2 previous raw chunks + 1 next raw chunk.
           * Build RETRIEVED CONTEXT:
               - dense + BM25 retrieval over *previously contextualized* passages
                 (never the whole doc at once; only what we’ve seen so far).
           * Call a local LLM (Gemma 3 4B IT QAT on Ollama) with:
               Local context + Retrieved context + Current passage
             and ask it to produce a self-contained, compact,
             "contextualized" version of the current passage.
           * Embed the new contextualized passage and add it to the
             in-memory retrieval index (dense + BM25).

       - This index is *temporary* and kept only in memory
         while processing the current document.

  3. Output: a JSON file with the same order and one object per chunk:
       {
         "doc": "...",
         "id": "...",
         "chunk": "...",                  # original passage
         "contextualized_chunk": "...",   # LLM contextualization
         "merged_chunk": "..."            # simple merged view (optional)
       }

This file is then ready for the later stages:
  - extract_claims.py
  - extract_triplets.py

Usage from command line:
    python contextualize.py INPUT_CHUNKS_JSON

Usage from notebook:
    from contextualize import contextualize_file
    out_path = contextualize_file("gdpr_chunks.json")
"""

import json
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import urllib.request
import urllib.error

# Try to reuse your embedding model ID from embedding.py if available.
try:
    from embedding import EMBED_MODEL_ID as DEFAULT_EMBED_MODEL_ID
except ImportError:
    # Fallback if embedding.py is not importable
    DEFAULT_EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"


# --------------------------------------------------------------------
# Configuration constants — tweak these instead of using CLI flags
# --------------------------------------------------------------------

# LLM (contextualizer) settings
OLLAMA_SERVER = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:latest"  # your current choice
REQUEST_TIMEOUT = 120
TEMPERATURE = 0.0
MAX_RETRIES = 3
RETRY_WAIT = 2.0    # seconds (exponential backoff base)
PROGRESS_EVERY = 1  # print progress every N chunks

# Embedding model used for dense retrieval
EMBED_MODEL_ID = DEFAULT_EMBED_MODEL_ID

# Local context window (in number of chunks)
LOCAL_BEFORE = 2   # how many raw chunks before the current one
LOCAL_AFTER = 1    # how many raw chunks after the current one

# Retrieval parameters
TOP_K_DENSE = 3    # top-K for dense similarity
TOP_K_BM25 = 3     # top-K for BM25 scoring
MAX_RETRIEVED = 3  # maximum retrieved contextualized passages we include
MIN_COMBINED_SCORE = 0.05  # threshold to discard very low-relevance passages


# --------------------------------------------------------------------
# Prompt used for contextualization
# --------------------------------------------------------------------

CONTEXTUALIZE_PROMPT = """You are a careful assistant helping to CONTEXTUALIZE a single passage
from a long document (it may be legal, technical, academic, policy, etc.).

You receive three kinds of text:
1) CURRENT PASSAGE            – the passage you MUST rewrite.
2) LOCAL CONTEXT              – nearby raw passages (before/after in the document).
3) PREVIOUSLY CONTEXTUALIZED  – earlier passages that have already been contextualized.

--------------------
CRITICAL INSTRUCTIONS
--------------------
Your primary source is ALWAYS the CURRENT PASSAGE.

Your task is to write ONE self-contained, concise, *contextualized version of the CURRENT PASSAGE* that:

- PRESERVES all important factual and conceptual content from the CURRENT PASSAGE:
  - key statements, conditions, exceptions, definitions,
  - list items (a), (b), (c)… and numbered points (1), (2), (3)… .
- DOES NOT replace the topic of the CURRENT PASSAGE with a more generic or unrelated summary.
  For example, do NOT ignore the main ideas, rules, or definitions in the CURRENT PASSAGE
  and talk only about some generic or previously seen theme.
- DOES NOT add new facts, rules, or conditions that are not implied by the text.
- Uses the same core terminology and concepts as in the original, with minimal paraphrasing.
- May expand ambiguous references (e.g. “it”, “this”, “such processing”, “these systems”)
  using LOCAL CONTEXT and PREVIOUSLY CONTEXTUALIZED PASSAGES.

How to use the context:
- LOCAL CONTEXT:
    Use ONLY to understand the flow of the document and to resolve pronouns
    or vague references in the CURRENT PASSAGE.
- PREVIOUSLY CONTEXTUALIZED PASSAGES:
    Use ONLY when they clearly define or clarify concepts that appear
    in the CURRENT PASSAGE. Do NOT summarize these passages instead of
    focusing on the CURRENT PASSAGE.
- If the context is not clearly relevant, IGNORE IT and just rewrite
  the CURRENT PASSAGE faithfully.

--------------------
OUTPUT FORMAT
--------------------
- Output ONLY the contextualized version of the CURRENT PASSAGE as plain text.
- Do NOT include explanations, bullet lists, or metadata.
- Do NOT mention the terms “current passage” or “context” in your answer.

--------------------
CURRENT PASSAGE
--------------------
\"\"\"{current_passage}\"\"\"

--------------------
LOCAL CONTEXT (raw neighboring passages)
--------------------
\"\"\"{local_context}\"\"\"

--------------------
PREVIOUSLY CONTEXTUALIZED PASSAGES (retrieved for relevance)
--------------------
\"\"\"{retrieved_context}\"\"\""""



# --------------------------------------------------------------------
# Simple HTTP helper to call Ollama's /api/chat endpoint
# --------------------------------------------------------------------

def call_ollama(
    server: str,
    model: str,
    prompt: str,
    timeout: int = REQUEST_TIMEOUT,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Call a local Ollama server with a single user message and return the reply text.

    Parameters
    ----------
    server : str
        Base URL for the Ollama server (e.g. "http://localhost:11434").
    model : str
        Model tag (e.g. "gemma3:4b-it-qat").
    prompt : str
        The full prompt to send as the user message.
    timeout : int
        HTTP timeout in seconds.
    temperature : float
        Sampling temperature.

    Returns
    -------
    str
        The model's reply (message.content), stripped of surrounding whitespace.
    """
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

    content = (obj.get("message") or {}).get("content", "") or ""
    return content.strip()


# --------------------------------------------------------------------
# Tiny tokenizer + incremental BM25 implementation (no external deps)
# --------------------------------------------------------------------

def simple_tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer for BM25.

    - Lowercases the text.
    - Splits on non-alphanumeric characters.
    - Filters out empty tokens.

    Suitable for legal-ish English text; we don't need anything fancy for ranking.
    """
    text = text.lower()
    # Split on anything that's not a letter or digit
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if t]


def bm25_scores(
    query_tokens: List[str],
    docs_tokens: List[List[str]],
    df: Dict[str, int],
    doc_lengths: List[int],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """
    Compute BM25 scores for a query against a list of tokenized documents.

    This is a minimal Okapi BM25 implementation with precomputed:
      - df: document frequency per term
      - doc_lengths: length of each document (number of tokens)

    Parameters
    ----------
    query_tokens : list of str
        Tokenized query.
    docs_tokens : list of list of str
        Tokenized documents.
    df : dict
        Mapping term -> number of documents containing that term.
    doc_lengths : list of int
        Number of tokens in each document.
    k1, b : float
        Standard BM25 hyperparameters.

    Returns
    -------
    list of float
        BM25 score for each document (same length as docs_tokens).
    """
    N = len(docs_tokens)
    if N == 0:
        return []

    avgdl = sum(doc_lengths) / float(N)
    scores = [0.0] * N

    if not query_tokens:
        return scores

    # Unique query terms to avoid redundant work
    unique_terms = set(query_tokens)

    # Precompute idf per query term
    idf_cache: Dict[str, float] = {}
    for term in unique_terms:
        n = df.get(term, 0)
        # Classic BM25 IDF; plus 1 in log argument to avoid negatives for rare cases
        idf = np.log(1.0 + (N - n + 0.5) / (n + 0.5)) if n > 0 else 0.0
        idf_cache[term] = idf

    # Compute scores
    from collections import Counter

    for i, doc_tokens in enumerate(docs_tokens):
        if not doc_tokens:
            continue

        doc_len = doc_lengths[i]
        freqs = Counter(doc_tokens)

        score = 0.0
        for term in unique_terms:
            f = freqs.get(term, 0)
            if f == 0:
                continue
            idf = idf_cache.get(term, 0.0)
            # BM25 term component
            denom = f + k1 * (1 - b + b * doc_len / avgdl)
            score += idf * (f * (k1 + 1)) / denom

        scores[i] = score

    return scores


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize a list of scores to the [0, 1] range.

    If all scores are equal or list is empty, returns zeros.
    """
    if not scores:
        return []

    mn = min(scores)
    mx = max(scores)
    if mx <= mn + 1e-9:
        return [0.0] * len(scores)

    return [(s - mn) / (mx - mn) for s in scores]


# --------------------------------------------------------------------
# Core contextualization logic for one document
# --------------------------------------------------------------------

def contextualize_document(
    doc: str,
    chunks: List[Dict[str, Any]],
    embedder: SentenceTransformer,
) -> List[Dict[str, Any]]:
    """
    Contextualize all chunks belonging to a single document.

    Parameters
    ----------
    doc : str
        Document identifier (e.g. "gdpr").
    chunks : list of dict
        Each dict must have:
            - "doc": document id (same as 'doc' param)
            - "id":  chunk id (e.g. "gdpr_1")
            - "chunk": original passage text
        The list should already be ordered in document order.
    embedder : SentenceTransformer
        Embedding model used for dense retrieval over contextualized passages.

    Returns
    -------
    list of dict
        Same length and order as input chunks. Each dict contains:
            - "doc"
            - "id"
            - "chunk"
            - "contextualized_chunk"
            - "merged_chunk"
    """
    n = len(chunks)
    print(f"  Document '{doc}': {n} chunks", flush=True)

    # In-memory index of *contextualized* passages for this document.
    # We update these as we move forward through the doc.
    ctx_texts: List[str] = []              # raw contextualized text
    ctx_embs: List[np.ndarray] = []        # normalized embeddings (for cosine via dot)
    ctx_tokens: List[List[str]] = []       # tokenized texts for BM25
    df: Dict[str, int] = {}                # document frequency for BM25
    doc_lengths: List[int] = []            # doc lengths for BM25

    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(chunks):
        chunk_id = str(item.get("id", ""))
        current_passage = str(item.get("chunk", ""))

        # ----- Build local context (raw chunks around current index) -----
        local_parts: List[str] = []

        # Up to LOCAL_BEFORE chunks before
        start_before = max(0, idx - LOCAL_BEFORE)
        for j in range(start_before, idx):
            local_parts.append(str(chunks[j]["chunk"]))

        # Up to LOCAL_AFTER chunk(s) after
        end_after = min(n, idx + LOCAL_AFTER + 1)
        for j in range(idx + 1, end_after):
            local_parts.append(str(chunks[j]["chunk"]))

        local_context = "\n\n".join(local_parts).strip()
        if not local_context:
            local_context = "(no local context available)"

        # ----- Build retrieved context from previously contextualized passages -----
        retrieved_context = "(no retrieved context available)"

        if ctx_texts:
            # Embed the *current raw passage* as query vector (normalized)
            q_emb = embedder.encode(
                [current_passage],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]

            # Dense cosine similarity via dot product (embeddings are normalized)
            ctx_matrix = np.vstack(ctx_embs)  # shape (M, D)
            dense_scores = (ctx_matrix @ q_emb).tolist()  # cos similarities

            # BM25 scores over previously contextualized passages
            query_tokens = simple_tokenize(current_passage)
            bm25_sc = bm25_scores(query_tokens, ctx_tokens, df, doc_lengths)

            # Normalize both score vectors to [0, 1]
            dense_norm = normalize_scores(dense_scores)
            bm25_norm = normalize_scores(bm25_sc)

            # Combine scores (simple average of normalized dense + BM25)
            combined_scores: List[float] = []
            for d, b in zip(dense_norm, bm25_norm):
                combined_scores.append(0.5 * d + 0.5 * b)

            # Rank documents by combined score (descending)
            candidates = sorted(
                range(len(ctx_texts)),
                key=lambda i: combined_scores[i],
                reverse=True,
            )

            # Filter and select top retrieved passages
            selected_indices = []
            for i_cand in candidates:
                if combined_scores[i_cand] < MIN_COMBINED_SCORE:
                    # Remaining candidates will also be below threshold (sorted)
                    break
                selected_indices.append(i_cand)
                if len(selected_indices) >= MAX_RETRIEVED:
                    break

            if selected_indices:
                retrieved_parts = [ctx_texts[i_cand] for i_cand in selected_indices]
                retrieved_context = "\n\n---\n\n".join(retrieved_parts)
            else:
                retrieved_context = "(no retrieved context passed relevance threshold)"

        # ----- Build LLM prompt for contextualization -----
        prompt = CONTEXTUALIZE_PROMPT.format(
            current_passage=current_passage,
            local_context=local_context,
            retrieved_context=retrieved_context,
        )

        # ----- Call LLM with retry logic -----
        attempt = 0
        reply = ""
        while attempt < MAX_RETRIES:
            try:
                reply = call_ollama(
                    server=OLLAMA_SERVER,
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    timeout=REQUEST_TIMEOUT,
                    temperature=TEMPERATURE,
                )
                break  # success
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                attempt += 1
                wait = RETRY_WAIT * (2 ** (attempt - 1))
                print(
                    f"[warn] {doc}/{chunk_id} attempt {attempt}/{MAX_RETRIES} error: {e}. "
                    f"Backing off {wait:.1f}s...",
                    flush=True,
                )
                time.sleep(wait)

        # If we never got a reply (e.g., all retries failed), fall back
        contextualized = reply.strip() if reply.strip() else current_passage

        # Simple "merged" view (optional: can be used later if desired)
        merged = contextualized  # kept simple for now

        # Store result for this chunk
        result_obj = {
            "doc": doc,
            "id": chunk_id,
            "chunk": current_passage,
            "contextualized_chunk": contextualized,
            "merged_chunk": merged,
        }
        results.append(result_obj)

        # ----- Update retrieval index with the new contextualized passage -----
        # Embed contextualized passage (normalized) for future dense retrieval
        ctx_vec = embedder.encode(
            [contextualized],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        ctx_embs.append(ctx_vec)
        ctx_texts.append(contextualized)

        # Tokenize for BM25 and update df / doc_lengths
        tokens = simple_tokenize(contextualized)
        ctx_tokens.append(tokens)
        doc_lengths.append(len(tokens))
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

        # Progress logging
        if (idx + 1) % PROGRESS_EVERY == 0 or (idx + 1) == n:
            print(
                f"    [{idx + 1}/{n}] contextualized chunks for '{doc}'",
                flush=True,
            )

    return results


# --------------------------------------------------------------------
# Top-level function (for notebooks and scripts)
# --------------------------------------------------------------------

def contextualize_file(input_path: Union[str, Path]) -> Path:
    """
    Run the entire contextualization pipeline on an input chunks JSON file.

    Parameters
    ----------
    input_path : str or Path
        Path to a JSON file containing a list of objects with:
          - "doc": document id
          - "id":  chunk id (e.g. "gdpr_1")
          - "chunk": passage text

        Typically this is the <base>_chunks.json produced by embedding.py.

    Behavior
    --------
    - Loads all chunks.
    - Groups them by 'doc'.
    - Sorts chunks within each doc by numeric suffix in 'id' (if present).
    - For each doc:
        * Runs Anthropic-style contextualization using local context
          and dense+BM25 retrieval over previously contextualized passages.
    - Writes a new JSON file with the same order and number of items, containing
      "contextualized_chunk" and "merged_chunk".

    Output name
    -----------
    If the input is named:
        gdpr_chunks.json
    The output will be:
        gdpr_contextualized.json

    Returns
    -------
    Path
        Path to the output JSON file.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Derive output path: strip "_chunks" from stem if present, then add "_contextualized"
    base = re.sub(r"_chunks$", "", input_path.stem)
    out_path = input_path.with_name(f"{base}_contextualized.json")

    # Load input JSON list
    try:
        items = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}") from e

    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list of objects.")

    # Group by doc
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for obj in items:
        doc = str(obj.get("doc", ""))
        if not doc:
            # Skip items without doc; could also raise if you prefer strictness.
            continue
        by_doc.setdefault(doc, []).append(obj)

    print(f"Total documents: {len(by_doc)}", flush=True)

    # Initialize embedder once for all docs
    embedder = SentenceTransformer(EMBED_MODEL_ID)

    total_chunks = len(items)
    print(f"Total chunks across all docs: {total_chunks}", flush=True)

    all_results: List[Dict[str, Any]] = []
    t0 = time.time()

    # Helper to extract numeric index from id like "gdpr_42"
    def chunk_index(o: Dict[str, Any]) -> int:
        cid = str(o.get("id", ""))
        m = re.search(r"_(\d+)$", cid)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return 10 ** 9
        return 10 ** 9

    # Process each document independently
    for doc, doc_items in by_doc.items():
        # Sort chunks in document order by numeric suffix
        doc_items_sorted = sorted(doc_items, key=chunk_index)

        doc_results = contextualize_document(doc, doc_items_sorted, embedder)
        all_results.extend(doc_results)

    # Write out all contextualized chunks
    out_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dt = time.time() - t0
    print(
        f"✓ Done. Contextualized {len(all_results)} chunks across {len(by_doc)} docs "
        f"in {dt:.1f}s → {out_path}",
        flush=True,
    )

    return out_path


# --------------------------------------------------------------------
# Simple CLI entry point (single positional argument: INPUT_JSON)
# --------------------------------------------------------------------

def main() -> None:
    """
    Command-line entry point.

    Usage:
        python contextualize.py INPUT_CHUNKS_JSON

    Where INPUT_CHUNKS_JSON is the <base>_chunks.json produced earlier.

    All configuration (LLM model, embedding model, window sizes, etc.)
    is specified at the top of this file as constants.
    """
    if len(sys.argv) != 2:
        prog = Path(sys.argv[0]).name
        print(f"Usage: {prog} INPUT_CHUNKS_JSON", file=sys.stderr)
        sys.exit(1)

    try:
        contextualize_file(sys.argv[1])
    except Exception as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
