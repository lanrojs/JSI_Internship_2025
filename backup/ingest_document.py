#!/usr/bin/env python3
"""
ingest_document.py — sequential contextualization with agentic retrieval,
then persistent storage of contextualized passages (dense + bm25 info).

Input:
    --chunks <file>  e.g. gdpr_chunks.json
        [
          { "doc": "gdpr", "id": "gdpr_1", "chunk": "..." },
          { "doc": "gdpr", "id": "gdpr_2", "chunk": "..." },
          ...
        ]

Output side effects:
    1. Append contextualized passages to a global SQLite DB:
          knowledge.db / table contextualized_passages
       Row fields include:
          doc, chunk_id, original_chunk, contextualized_chunk, merged_chunk,
          embedding_contextualized (BLOB), bm25_tokens_json, model_name, timestamp

    2. Also emit <base>_contextualized.json
       (same shape as your original contextualize.py output)
       so extract_claims.py and extract_triplets.py still work unchanged.

High-level:
    - For each doc:
        - walk chunks in order
        - local neighbors: 2 before + 1 after
        - retrieve top-k previous contextualized chunks (hybrid BM25+dense)
        - NEW: agentic relevance+compaction to get a short support block
        - contextualize current chunk using compact support + neighbors
        - store result to DB + keep in per-doc memory for future retrieval
"""

import argparse
import json
import math
import re
import sqlite3
import sys
import time
import urllib.request, urllib.error
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

########################
# === BM25 helpers  ===
########################

TOKEN_SPLIT = re.compile(r"[^\w]+", re.UNICODE)

def tokenize_for_bm25(text: str) -> List[str]:
    return [t for t in TOKEN_SPLIT.split(text.lower()) if t]

class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs_tokens: List[List[str]] = []
        self.df: Dict[str, int] = {}
        self.avgdl: float = 0.0
        self.N: int = 0

    def add_doc(self, tokens: List[str]):
        self.docs_tokens.append(tokens)

    def finalize(self):
        from collections import Counter
        self.N = len(self.docs_tokens)
        total_len = 0
        df_counter = Counter()
        for toks in self.docs_tokens:
            total_len += len(toks)
            for term in set(toks):
                df_counter[term] += 1
        self.df = dict(df_counter)
        self.avgdl = (total_len / self.N) if self.N > 0 else 0.0

    def score_query(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.N
        if self.N == 0:
            return scores

        from collections import Counter
        qtf = Counter(query_tokens)

        for term in qtf:
            df = self.df.get(term)
            if df is None:
                continue
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1e-12)

            for doc_idx, doc_tokens in enumerate(self.docs_tokens):
                # raw term freq
                f = 0
                for tok in doc_tokens:
                    if tok == term:
                        f += 1
                if f == 0:
                    continue

                dl = len(doc_tokens)
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-12))
                score_term = idf * (f * (self.k1 + 1) / denom)
                scores[doc_idx] += score_term
        return scores


########################
# === Embeddings    ===
########################

def embed_texts(embedder: SentenceTransformer,
                texts: List[str],
                normalize: bool = True) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    embs = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=False
    ).astype("float32", copy=False)
    if normalize:
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = (embs / norms).astype("float32", copy=False)
    return embs

def cosine_sim_matrix(query_vec: np.ndarray, index_matrix: np.ndarray) -> np.ndarray:
    if index_matrix.size == 0:
        return np.zeros((0,), dtype="float32")
    return np.dot(index_matrix, query_vec.astype("float32", copy=False))


def hybrid_retrieve_indices(
    query_text: str,
    memory_texts: List[str],
    memory_vecs_norm: np.ndarray,
    bm25_index: BM25Index,
    embedder: SentenceTransformer,
    top_k: int,
) -> List[int]:
    """
    Return indices of up to top_k candidate passages from memory_texts
    using BM25 + dense cosine fusion.
    """
    N = len(memory_texts)
    if N == 0:
        return []

    q_vec = embed_texts(embedder, [query_text], normalize=True)[0]
    dense_sims = cosine_sim_matrix(q_vec, memory_vecs_norm)  # [N]
    dense_order = list(np.argsort(-dense_sims))

    q_tokens = tokenize_for_bm25(query_text)
    bm25_scores = bm25_index.score_query(q_tokens)
    bm25_order = sorted(range(N), key=lambda i: -bm25_scores[i])

    chosen: List[int] = []
    chosen_set = set()

    # BM25 priority (only >0 score)
    for idx in bm25_order:
        if len(chosen) >= top_k:
            break
        if bm25_scores[idx] <= 0:
            break
        if idx not in chosen_set:
            chosen.append(idx)
            chosen_set.add(idx)

    # fill with dense
    for idx in dense_order:
        if len(chosen) >= top_k:
            break
        if idx not in chosen_set:
            chosen.append(idx)
            chosen_set.add(idx)

    return chosen


########################
# === Ollama calls   ===
########################

def ollama_chat(
    server: str,
    model: str,
    system_msg: str,
    user_msg: str,
    timeout: int = 120,
    temperature: float = 0.0,
    num_ctx: Optional[int] = None
) -> str:
    """
    Minimal non-streaming chat call to Ollama.
    """
    url = server.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        "stream": False,
    }
    opts = {"temperature": float(temperature)}
    if num_ctx is not None:
        opts["num_ctx"] = int(num_ctx)
    payload["options"] = opts

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))

    msg = obj.get("message") or {}
    content = msg.get("content", "")
    if not content:
        # compat path if ollama returns openai-ish format
        ch = obj.get("choices")
        if isinstance(ch, list) and ch:
            content = ch[0].get("message", {}).get("content", "") or ""
    return content.strip()


########################
# === Agent prompts ===
########################

RELEVANCE_SYSTEM = (
    "You help narrow context for legal text. "
    "Given candidate passages and a current chunk, "
    "extract ONLY definitions / actors / obligations that clarify the current chunk. "
    "Do NOT invent anything. Output ONLY a concise summary. "
    "If nothing helps, output '(none)'."
)

RELEVANCE_USER_TEMPLATE = """CURRENT CHUNK:
{current_chunk}

CANDIDATE PASSAGES:
{candidates}

TASK:
Summarize ONLY the definitions / actors / obligations from relevant candidates
that clarify CURRENT CHUNK. Keep it short (<200 tokens). If nothing helpful,
respond with "(none)".
"""

def build_relevance_prompt(current_chunk: str, candidate_passages: List[str]) -> str:
    numbered = []
    for idx, p in enumerate(candidate_passages, start=1):
        numbered.append(f"[P{idx}] {p.strip()}")
    return RELEVANCE_USER_TEMPLATE.format(
        current_chunk=current_chunk.strip(),
        candidates="\n\n".join(numbered)
    )


CONTEXTUALIZE_SYSTEM = (
    "You rewrite legal / regulatory text chunks so each chunk is self-contained. "
    "Rules:\n"
    "- Preserve ALL factual content.\n"
    "- Only clarify ambiguous references using provided context if the exact wording appears there.\n"
    "- Don't invent new actors/obligations.\n"
    "- Keep numbering/labels exactly.\n"
    "- Output ONLY the rewritten statute text, no commentary."
)

CONTEXTUALIZE_USER_TEMPLATE = """You will be given:

1. Compact relevant context from earlier in THIS SAME document.
2. Local neighboring raw chunks (before/after).
3. The current raw chunk to rewrite.

Rewrite ONLY the current raw chunk so it's understandable alone.
Use (1) and (2) ONLY to clarify ambiguous references already present in (3).
If something is still unclear, leave it as-is.

=== Compact relevant context
{compact_context}

=== Neighboring raw context
{neighbors}

=== Current raw chunk (to rewrite)
{current_chunk}

=== Output
Return ONLY the rewritten statute text.
"""

def build_contextualize_prompt(compact_context: str,
                               neighbors: str,
                               current_chunk: str) -> str:
    return CONTEXTUALIZE_USER_TEMPLATE.format(
        compact_context = compact_context.strip() if compact_context.strip() else "(none)",
        neighbors       = neighbors.strip()       if neighbors.strip()       else "(none)",
        current_chunk   = current_chunk.strip()
    )


########################
# === SQLite store  ===
########################

def ensure_tables(conn: sqlite3.Connection):
    # contextualized_passages: final storage for RAG
    conn.execute("""
    CREATE TABLE IF NOT EXISTS contextualized_passages (
        chunk_id TEXT PRIMARY KEY,           -- e.g. "gdpr_12"
        doc      TEXT NOT NULL,             -- e.g. "gdpr"
        original_chunk TEXT NOT NULL,
        contextualized_chunk TEXT NOT NULL,
        merged_chunk TEXT NOT NULL,
        model_name TEXT NOT NULL,
        timestamp_utc TEXT NOT NULL,
        bm25_tokens_json TEXT NOT NULL,
        embedding_contextualized BLOB NOT NULL
    )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ctx_doc ON contextualized_passages(doc)")
    conn.commit()


def insert_contextualized_row(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    doc: str,
    original_chunk: str,
    contextualized_chunk: str,
    merged_chunk: str,
    model_name: str,
    bm25_tokens: List[str],
    embedding_vec: np.ndarray
):
    timestamp_utc = datetime.utcnow().isoformat() + "Z"
    tokens_json = json.dumps(bm25_tokens, ensure_ascii=False)

    if embedding_vec.dtype != np.float32:
        embedding_vec = embedding_vec.astype("float32", copy=False)
    emb_blob = embedding_vec.tobytes(order="C")

    conn.execute("""
    INSERT OR REPLACE INTO contextualized_passages (
        chunk_id, doc, original_chunk, contextualized_chunk,
        merged_chunk, model_name, timestamp_utc,
        bm25_tokens_json,
        embedding_contextualized
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        chunk_id,
        doc,
        original_chunk,
        contextualized_chunk,
        merged_chunk,
        model_name,
        timestamp_utc,
        tokens_json,
        emb_blob
    ))
    conn.commit()


########################
# === Helpers       ===
########################

def parse_chunk_index(cid: str, doc: str) -> int:
    prefix = f"{doc}_"
    if cid and cid.startswith(prefix):
        tail = cid[len(prefix):]
        if tail.isdigit():
            return int(tail)
    return math.inf

def group_and_sort_by_doc(items: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    by_doc: Dict[str, List[Dict[str, str]]] = {}
    for it in items:
        d = str(it.get("doc", "_MISC_"))
        by_doc.setdefault(d, []).append(it)
    for d, arr in by_doc.items():
        arr.sort(key=lambda obj: parse_chunk_index(str(obj.get("id", "")), d))
    return by_doc

def get_neighbors(arr: List[Dict[str,str]], idx: int, before: int=2, after: int=1) -> str:
    pieces: List[str] = []
    start_prev = max(0, idx - before)
    for j in range(start_prev, idx):
        ch = (arr[j].get("chunk") or "").strip()
        if ch:
            pieces.append(ch)
    end_next = min(len(arr), idx + 1 + after)
    for j in range(idx+1, end_next):
        ch = (arr[j].get("chunk") or "").strip()
        if ch:
            pieces.append(ch)
    joined = "\n\n".join(pieces).strip()
    return joined if joined else "(none)"

def flatten_newlines(s: str) -> str:
    return s.replace("\n", " ").replace("\r", " ").strip()


########################
# === main loop     ===
########################

def contextualize_document(
    doc_chunks: List[Dict[str,str]],
    *,
    embedder: SentenceTransformer,
    bm25_k1: float,
    bm25_b: float,
    top_k: int,
    server: str,
    model: str,
    timeout: int,
    temperature: float,
    num_ctx: Optional[int],
    conn: sqlite3.Connection,
    source_model_name: str,
    progress_prefix: str
) -> Tuple[List[Dict[str,str]], int, int]:
    """
    Walk chunks for one doc sequentially.
    Returns:
        (rows_for_json_output, ok_count, empty_count)
    """

    # this is the per-doc TEMPORARY index (discard after doc done)
    memory_texts: List[str] = []          # contextualized chunks so far
    memory_vecs: List[np.ndarray] = []    # normalized embeddings of contextualized chunks
    bm25_index = BM25Index(k1=bm25_k1, b=bm25_b)

    out_rows: List[Dict[str,str]] = []
    ok_count = 0
    empty_count = 0

    for idx, item in enumerate(doc_chunks):
        original_chunk = (item.get("chunk") or "").strip()
        chunk_id       = item.get("id", "")
        doc_name       = item.get("doc", "")

        # 1. local neighbors
        neighbor_text = get_neighbors(doc_chunks, idx, before=2, after=1)

        # 2. retrieve previous contextualized (hybrid)
        if memory_texts and original_chunk:
            mem_matrix = (
                np.stack(memory_vecs, axis=0)
                if memory_vecs else
                np.zeros((0,1), dtype="float32")
            )

            cand_indices = hybrid_retrieve_indices(
                query_text=original_chunk,
                memory_texts=memory_texts,
                memory_vecs_norm=mem_matrix,
                bm25_index=bm25_index,
                embedder=embedder,
                top_k=top_k,
            )
            candidate_passages = [memory_texts[i] for i in cand_indices]
        else:
            candidate_passages = []

        # 3. agentic relevance+compaction
        relevance_prompt = build_relevance_prompt(
            current_chunk=original_chunk,
            candidate_passages=candidate_passages
        )
        try:
            compact_support = ollama_chat(
                server=server,
                model=model,
                system_msg=RELEVANCE_SYSTEM,
                user_msg=relevance_prompt,
                timeout=timeout,
                temperature=temperature,
                num_ctx=num_ctx
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"[warn] relevance-compaction failed for {doc_name}/{chunk_id}: {e}",
                  file=sys.stderr, flush=True)
            compact_support = "(none)"

        # 4. final contextualization rewrite
        contextualize_prompt = build_contextualize_prompt(
            compact_context=compact_support,
            neighbors=neighbor_text,
            current_chunk=original_chunk
        )
        try:
            rewritten = ollama_chat(
                server=server,
                model=model,
                system_msg=CONTEXTUALIZE_SYSTEM,
                user_msg=contextualize_prompt,
                timeout=timeout,
                temperature=temperature,
                num_ctx=num_ctx
            ).strip()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"[warn] rewrite failed for {doc_name}/{chunk_id}: {e}",
                  file=sys.stderr, flush=True)
            rewritten = ""

        merged = (rewritten + "\n\n" + original_chunk).strip() if rewritten else original_chunk

        row = {
            "doc": doc_name,
            "id": chunk_id,
            "chunk": original_chunk,
            "contextualized_chunk": rewritten,
            "merged_chunk": merged,
        }
        out_rows.append(row)

        if rewritten:
            ok_count += 1
        else:
            empty_count += 1

        # 5. update TEMPORARY in-memory retrieval memory for *next* chunks
        if rewritten:
            # embed contextualized version (normalized)
            new_vec = embed_texts(embedder, [rewritten], normalize=True)[0]
            memory_vecs.append(new_vec.astype("float32", copy=False))
            memory_texts.append(rewritten)

            toks = tokenize_for_bm25(rewritten)
            bm25_index.add_doc(toks)
            bm25_index.finalize()

            # 6. persist contextualized passage to global SQLite for long-term RAG
            insert_contextualized_row(
                conn,
                chunk_id       = chunk_id,
                doc            = doc_name,
                original_chunk = flatten_newlines(original_chunk),
                contextualized_chunk = flatten_newlines(rewritten),
                merged_chunk   = flatten_newlines(merged),
                model_name     = source_model_name,
                bm25_tokens    = toks,
                embedding_vec  = new_vec,
            )

        # progress print
        print(f"{progress_prefix} [{idx+1}/{len(doc_chunks)}] {doc_name}/{chunk_id} ok={ok_count} empty={empty_count}",
              flush=True)

    return out_rows, ok_count, empty_count


def main():
    ap = argparse.ArgumentParser(
        description="Agentic contextualization + persistent ingestion"
    )
    ap.add_argument("--chunks", required=True, type=Path,
                    help="Path to *_chunks.json (list of {doc,id,chunk})")
    ap.add_argument("--db", type=Path, default=Path("knowledge.db"),
                    help="Global SQLite knowledge base file")
    ap.add_argument("--model", default="gemma3:1b-it-qat",
                    help="Ollama model tag (e.g. llama3:8b, qwen2.5:7b, etc.)")
    ap.add_argument("--server", default="http://localhost:11434",
                    help="Ollama base URL")
    ap.add_argument("--timeout", type=int, default=120,
                    help="HTTP timeout seconds")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Generation temperature")
    ap.add_argument("--num-ctx", type=int, default=None,
                    help="Context window hint passed to Ollama options.num_ctx")

    # retrieval config
    ap.add_argument("--top-k", type=int, default=3,
                    help="How many prior contextualized passages to retrieve per chunk.")
    ap.add_argument("--bm25-k1", type=float, default=1.5)
    ap.add_argument("--bm25-b",  type=float, default=0.75)

    # dense embed model for retrieval memory
    ap.add_argument("--embed-model", type=str, default="BAAI/bge-small-en-v1.5",
                    help="SentenceTransformer model id for dense retrieval of contextualized chunks")

    # logging / output
    ap.add_argument("--progress-prefix", type=str, default="[ingest]",
                    help="Prefix for progress prints")
    ap.add_argument("--no-json-out", action="store_true",
                    help="Skip writing <base>_contextualized.json (for debugging).")
    args = ap.parse_args()

    # load chunks
    if not args.chunks.exists():
        print(f"[!] chunks not found: {args.chunks}", file=sys.stderr, flush=True)
        sys.exit(1)

    try:
        items = json.loads(args.chunks.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[!] Failed to parse chunks JSON: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    if not isinstance(items, list):
        print("[!] chunks JSON must be a list of objects.", file=sys.stderr, flush=True)
        sys.exit(1)

    base = re.sub(r'(_cleaned|_chunks)$', '', args.chunks.stem)
    out_path = args.chunks.with_name(f"{base}_contextualized.json")

    # group chunks by doc and sort by numeric suffix
    by_doc = group_and_sort_by_doc(items)

    # init embedder once
    embedder = SentenceTransformer(args.embed_model)

    # open / init global DB
    conn = sqlite3.connect(str(args.db))
    ensure_tables(conn)

    all_rows: List[Dict[str,str]] = []
    global_ok = 0
    global_empty = 0
    t0 = time.time()

    # process each doc separately, sequentially
    for doc_name, doc_chunks in by_doc.items():
        rows, ok_count, empty_count = contextualize_document(
            doc_chunks=doc_chunks,
            embedder=embedder,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
            top_k=args.top_k,
            server=args.server,
            model=args.model,
            timeout=args.timeout,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            conn=conn,
            source_model_name=args.model,
            progress_prefix=args.progress_prefix
        )
        all_rows.extend(rows)
        global_ok += ok_count
        global_empty += empty_count

    dt = time.time() - t0
    print(f"Summary: ok={global_ok}, empty={global_empty}, total={len(all_rows)}, time={dt:.1f}s")

    # write debug JSON compatible with extract_claims.py
    if not args.no_json_out:
        # NOTE: we keep flatten_newlines off here on purpose, because your extract_claims.py
        # reads .get("contextualized_chunk") and sends it to the LLM. You said newline
        # flattening is ok for now, but we can leave the raw form in this debug file.
        out_path.write_text(
            json.dumps(all_rows, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"Wrote contextualized debug JSON → {out_path}")

    print("✅ ingest complete")


if __name__ == "__main__":
    main()
