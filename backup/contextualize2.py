"""
contextualize.py — Anthropic-style contextual retrieval & contextualization.

Input:
  JSON list of chunks, e.g.:
    { "doc": "...", "id": "...", "chunk": "..." }

Output:
  <base>_contextualized.json with objects:
    {
      "doc": "...",
      "id": "...",
      "chunk": "...",                  # original passage
      "contextualized_chunk": "..."    # [Context] + [Chunk]
    }

Pipeline (per document):
  1. Group chunks by doc and sort them in order (based on numeric suffix in id if present).
  2. Process chunks sequentially:
       - Build LOCAL CONTEXT = a few raw chunks before/after.
       - Build RETRIEVED CONTEXT = dense + BM25 retrieval over
         previously contextualized passages for this document.
       - Call LLM with:
             CURRENT PASSAGE + DOCUMENT-LEVEL CONTEXT
         and ask it to produce a SHORT CONTEXT SUMMARY (1–3 sentences).
       - Construct:
             [Context] <LLM summary>
             [Chunk]   <original chunk, exactly>
       - Store the new contextualized passage in an in-memory index:
             embeddings (dense) + tokens (BM25).
  3. Write all contextualized chunks to JSON.

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
from typing import List, Dict, Any, Union

import numpy as np
from sentence_transformers import SentenceTransformer
import urllib.request
import urllib.error

# Try to reuse the embedding model ID from embedding.py if present
try:
    from embedding import EMBED_MODEL_ID as DEFAULT_EMBED_MODEL_ID
except ImportError:
    DEFAULT_EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"


# --------------------------------------------------------------------
# Configuration constants — tweak these instead of using CLI flags
# --------------------------------------------------------------------

# LLM (contextualizer) settings
OLLAMA_SERVER = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b-it-qat"  # your current model in Ollama
REQUEST_TIMEOUT = 120              # seconds
TEMPERATURE = 0.0
MAX_RETRIES = 3
RETRY_WAIT = 2.0                   # initial backoff (grows exponentially)
PROGRESS_EVERY = 1                 # print progress every N chunks

# Embedding model for dense retrieval over contextualized passages
EMBED_MODEL_ID = DEFAULT_EMBED_MODEL_ID

# Local context window (raw chunks, not contextualized)
LOCAL_BEFORE = 2   # how many raw chunks BEFORE the current one
LOCAL_AFTER = 2    # how many raw chunks AFTER the current one

# Retrieval parameters
TOP_K_DENSE = 3           # top-K by dense similarity to consider
TOP_K_BM25 = 3            # top-K by BM25 score to consider
MAX_RETRIEVED = 3         # maximum retrieved contextualized passages to include
MIN_COMBINED_SCORE = 0.1  # drop candidates below this (0–1 normalized)


# --------------------------------------------------------------------
# Prompt used for contextualization (general-purpose, not just legal)
# --------------------------------------------------------------------

CONTEXTUALIZE_PROMPT = """You are helping build a contextual retrieval system.

Your job is to generate a SHORT, SUCCINCT CONTEXT that situates one chunk of a
long document, so that the chunk will be easier to find and understand later
during search and retrieval.

You will receive:
- Some DOCUMENT-LEVEL CONTEXT (nearby passages and other related passages).
- One CURRENT CHUNK that we want to contextualize.

--------------------
GOAL
--------------------
Write 1–3 sentences that briefly explain:
- What this chunk is about.
- How it fits into the larger document (e.g. section, topic, entity, time period).
- Any especially important concepts, entities, or relationships that appear.

This is NOT a full rewrite of the chunk. It is a compact description that makes
the chunk more self-contained and searchable.

Use DOCUMENT-LEVEL CONTEXT only to better understand the chunk; if the context
is not clearly relevant, ignore it. Do NOT introduce facts that contradict the
chunk.

--------------------
OUTPUT FORMAT (IMPORTANT)
--------------------
- Output ONLY the 1–3 sentence contextual description as plain text.
- Do NOT include headings like [Context] or [Chunk].
- Do NOT repeat or paraphrase the full chunk text; just describe it.

--------------------
DOCUMENT-LEVEL CONTEXT
--------------------
\"\"\"{local_context}

{retrieved_context}\"\"\"


--------------------
CURRENT CHUNK
--------------------
\"\"\"{current_passage}\"\"\""""


# --------------------------------------------------------------------
# HTTP helper to call Ollama /api/chat
# --------------------------------------------------------------------

def call_ollama(
    server: str,
    model: str,
    prompt: str,
    timeout: int = REQUEST_TIMEOUT,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Call a local Ollama model with a single user message and return the reply text.
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
# Tiny tokenizer + BM25 implementation (no extra dependencies)
# --------------------------------------------------------------------

def simple_tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer for BM25:
      - Lowercase
      - Split on non-alphanumeric
      - Remove empty tokens
    """
    text = text.lower()
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
    Minimal Okapi BM25 implementation for ranking documents against a query.

    Parameters
    ----------
    query_tokens : list of str
        Tokenized query.
    docs_tokens : list of list of str
        Tokenized documents.
    df : dict
        Document frequency per term.
    doc_lengths : list of int
        Length of each document (number of tokens).

    Returns
    -------
    list of float
        BM25 scores for each document.
    """
    N = len(docs_tokens)
    if N == 0:
        return []

    avgdl = sum(doc_lengths) / float(N)
    scores = [0.0] * N
    if not query_tokens:
        return scores

    from collections import Counter

    unique_terms = set(query_tokens)

    # Precompute IDF
    idf_cache: Dict[str, float] = {}
    for term in unique_terms:
        n = df.get(term, 0)
        idf = np.log(1.0 + (N - n + 0.5) / (n + 0.5)) if n > 0 else 0.0
        idf_cache[term] = idf

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
            denom = f + k1 * (1 - b + b * doc_len / avgdl)
            score += idf * (f * (k1 + 1)) / denom

        scores[i] = score

    return scores


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize a list of scores to [0, 1].

    If all scores are equal or the list is empty,
    returns all zeros.
    """
    if not scores:
        return []

    mn = min(scores)
    mx = max(scores)
    if mx <= mn + 1e-9:
        return [0.0] * len(scores)

    return [(s - mn) / (mx - mn) for s in scores]


# --------------------------------------------------------------------
# Core contextualization logic for a single document
# --------------------------------------------------------------------

def contextualize_document(
    doc: str,
    chunks: List[Dict[str, Any]],
    embedder: SentenceTransformer,
) -> List[Dict[str, Any]]:
    """
    Contextualize all chunks for a single document.

    Parameters
    ----------
    doc : str
        Document identifier.
    chunks : list of dict
        Objects with fields:
          - "doc"
          - "id"
          - "chunk"
        The list should already be in document order.
    embedder : SentenceTransformer
        Embedding model to use for dense retrieval.

    Returns
    -------
    list of dict
        Objects with fields:
          - "doc"
          - "id"
          - "chunk"
          - "contextualized_chunk"
    """
    n = len(chunks)
    print(f"  Document '{doc}': {n} chunks", flush=True)

    # In-memory retrieval index for *contextualized* passages of this document
    ctx_texts: List[str] = []        # contextualized text ([Context] + [Chunk])
    ctx_embs: List[np.ndarray] = []  # normalized embedding vectors
    ctx_tokens: List[List[str]] = [] # tokenized texts for BM25
    df: Dict[str, int] = {}          # BM25 document frequency
    doc_lengths: List[int] = []      # BM25 document lengths

    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(chunks):
        chunk_id = str(item.get("id", ""))
        current_passage = str(item.get("chunk", ""))

        # ----- Local context: a few raw chunks before and after -----
        local_parts: List[str] = []

        # Raw chunks BEFORE
        start_before = max(0, idx - LOCAL_BEFORE)
        for j in range(start_before, idx):
            local_parts.append(str(chunks[j]["chunk"]))

        # Raw chunks AFTER
        end_after = min(n, idx + LOCAL_AFTER + 1)
        for j in range(idx + 1, end_after):
            local_parts.append(str(chunks[j]["chunk"]))

        local_context = "\n\n".join(local_parts).strip()
        if not local_context:
            local_context = "(no local context available)"

        # ----- Retrieved context: dense + BM25 over previously contextualized -----
        retrieved_context = "(no retrieved context available)"

        if ctx_texts:
            # Current passage as dense query (normalized embedding)
            q_emb = embedder.encode(
                [current_passage],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]

            ctx_count = len(ctx_texts)
            ctx_matrix = np.vstack(ctx_embs)  # (M, D)
            dense_scores = (ctx_matrix @ q_emb).tolist()  # cosine similarity

            # BM25 scores
            query_tokens = simple_tokenize(current_passage)
            bm25_sc = bm25_scores(query_tokens, ctx_tokens, df, doc_lengths)

            # Top-K by each method
            dense_top = sorted(
                range(ctx_count),
                key=lambda i: dense_scores[i],
                reverse=True,
            )[:TOP_K_DENSE]

            bm25_top = sorted(
                range(ctx_count),
                key=lambda i: bm25_sc[i],
                reverse=True,
            )[:TOP_K_BM25]

            # Candidate set = union of dense and BM25 top indices
            candidate_indices = sorted(set(dense_top) | set(bm25_top))

            # Normalize scores for combination
            dense_norm = normalize_scores(dense_scores)
            bm25_norm = normalize_scores(bm25_sc)

            combined_scores = [
                0.5 * dense_norm[i] + 0.5 * bm25_norm[i]
                for i in range(ctx_count)
            ]

            # Rank only the candidate indices by combined score
            candidates_sorted = sorted(
                candidate_indices,
                key=lambda i: combined_scores[i],
                reverse=True,
            )

            selected_indices: List[int] = []
            for i_cand in candidates_sorted:
                if combined_scores[i_cand] < MIN_COMBINED_SCORE:
                    # Candidates are sorted; remaining will also be too low
                    break
                selected_indices.append(i_cand)
                if len(selected_indices) >= MAX_RETRIEVED:
                    break

            if selected_indices:
                retrieved_parts = [ctx_texts[i_cand] for i_cand in selected_indices]
                retrieved_context = "\n\n---\n\n".join(retrieved_parts)
            else:
                retrieved_context = "(no retrieved context passed relevance threshold)"

        # ----- Build prompt for the LLM -----
        prompt = CONTEXTUALIZE_PROMPT.format(
            current_passage=current_passage,
            local_context=local_context,
            retrieved_context=retrieved_context,
        )

        # ----- Call LLM with retry/backoff -----
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
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                attempt += 1
                wait = RETRY_WAIT * (2 ** (attempt - 1))
                print(
                    f"[warn] {doc}/{chunk_id} attempt {attempt}/{MAX_RETRIES} error: {e}. "
                    f"Backing off {wait:.1f}s...",
                    flush=True,
                )
                time.sleep(wait)

        # The model should return only the short context description
        context_text = reply.strip()
        if not context_text:
            context_text = "(no additional context)"

        # 🔒 Construct contextualized chunk **deterministically**:
        # - LLM only contributes [Context] text
        # - [Chunk] is ALWAYS the exact original passage
        contextualized = (
            "[Context]\n"
            f"{context_text}\n\n"
            "[Chunk]\n"
            f"\"\"\"{current_passage}\"\"\""
        )

        # Store output for this chunk
        result_obj = {
            "doc": doc,
            "id": chunk_id,
            "chunk": current_passage,
            "contextualized_chunk": contextualized,
        }
        results.append(result_obj)

        # ----- Update retrieval index with the new contextualized passage -----
        ctx_vec = embedder.encode(
            [contextualized],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        ctx_embs.append(ctx_vec)
        ctx_texts.append(contextualized)

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
    Run contextualization pipeline on an input chunks JSON file.

    Parameters
    ----------
    input_path : str or Path
        Path to a JSON file containing a list of objects:
          - "doc": document id
          - "id":  chunk id
          - "chunk": passage text

    Returns
    -------
    Path
        Path to the <base>_contextualized.json output file.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    base = re.sub(r"_chunks$", "", input_path.stem)
    out_path = input_path.with_name(f"{base}_contextualized.json")

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
            continue
        by_doc.setdefault(doc, []).append(obj)

    print(f"Total documents: {len(by_doc)}", flush=True)
    print(f"Total chunks across all docs: {len(items)}", flush=True)

    # Shared embedder
    embedder = SentenceTransformer(EMBED_MODEL_ID)

    all_results: List[Dict[str, Any]] = []
    t0 = time.time()

    # Helper to sort chunks in numeric order by id if possible
    def chunk_index(o: Dict[str, Any]) -> int:
        cid = str(o.get("id", ""))
        m = re.search(r"_(\d+)$", cid)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return 10**9
        return 10**9

    for doc, doc_items in by_doc.items():
        doc_items_sorted = sorted(doc_items, key=chunk_index)
        doc_results = contextualize_document(doc, doc_items_sorted, embedder)
        all_results.extend(doc_results)

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
# CLI entry point (single positional argument)
# --------------------------------------------------------------------

def main() -> None:
    """
    Command-line entry point.

    Usage:
        python contextualize.py INPUT_CHUNKS_JSON
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
