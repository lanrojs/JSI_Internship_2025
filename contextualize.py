"""
contextualize.py — Sequential contextualization pipeline (middle step).

Pipeline:
  1) Take raw chunks (<base>_chunks.json from chunk_and_embed.py).
  2) For each chunk, assemble context from:
       - local neighbors in the same document
       - previous contextualized chunks in the same document (dense + BM25)
       - OPTIONAL external "system" corpus of raw chunks (dense + BM25)
  3) Call a local LLM (Ollama) in either:
       - prefix mode: generate a short context note and prepend to the chunk
       - rewrite mode: rewrite into a self-contained passage
  4) Store contextualized chunks + embeddings in:
       - <base>_contextualized.json
       - <base>_contextualized.sqlite (table: contextualized_chunks)

Input:
  <base>_chunks.json   # produced by chunk_and_embed.py

Outputs:
  <base>_contextualized.json         # list of objects:
    {
      "doc": ...,
      "id": ...,
      "chunk": ...,
      "contextualized_chunk": ...,
      "local_neighbor_ids": [...],
      "doc_retrieved_ids": [...],     # if USE_LLM_RELEVANCE=True → only relevant IDs
      "system_retrieved_ids": [...]   # if USE_LLM_RELEVANCE=True → only relevant IDs
    }

  <base>_contextualized.sqlite (table: contextualized_chunks)
    doc, id, chunk_raw, context_prefix, chunk_ctx, emb_ctx

Modes:
  --mode prefix   (default)
      - LLM writes a short contextual note ("context prefix").
      - contextualized_chunk = context_prefix + " " + chunk

  --mode rewrite
      - LLM REWRITES the passage into a self-contained, compact version,
        integrating necessary context.
      - contextualized_chunk = rewritten_passage
      - context_prefix = rewritten_passage  (no separate note)

System context (use_system / k_sys):
  - External corpus = multiple .sqlite files in a folder (SYSTEM_DB_DIR).
  - Each DB is expected to contain table contextualized_chunks(doc, id, chunk_raw, context_prefix, chunk_ctx, emb_ctx).
  - For each chunk, we:
      * embed the current passage (BGE-small)
      * run dense + BM25 retrieval over all external contextualized_chunks
      * combine scores (dense + BM25) and take top SYSTEM_TOP_K segments
  - Controlled by:
      USE_SYSTEM    (bool)
      SYSTEM_DB_DIR (Path)
      SYSTEM_TOP_K  (int, default 2 = k_sys)
"""

import json
import time
import re
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple
import urllib.request
import numpy as np
from sentence_transformers import SentenceTransformer
from jinja2 import Environment, FileSystemLoader
import argparse
import yaml


# Load configuration YAML
CONFIG_PATH = Path(__file__).resolve().parent / "config" / "contextualize.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# --------------------------------------------------------------------
# Config (from YAML)
# --------------------------------------------------------------------

# LLM setup (Ollama)
OLLAMA_SERVER = CFG["llm"]["server"]
OLLAMA_MODEL = CFG["llm"]["model"]
REQUEST_TIMEOUT = CFG["llm"]["request_timeout"]
TEMPERATURE = CFG["llm"]["temperature"]
MAX_RETRIES = CFG["llm"]["max_retries"]
RETRY_WAIT = CFG["llm"]["retry_wait"]

# Embedding model
EMBED_MODEL_ID = CFG["embeddings"]["model_id"]
NORMALIZE_EMBEDDINGS = CFG["embeddings"]["normalize"]

# Local context window (same-doc neighbors)
LOCAL_BEFORE = CFG["local_context"]["before"]
LOCAL_AFTER = CFG["local_context"]["after"]

# Retrieval config (over *previous* contextualized chunks in the same doc)
TOP_K_RETRIEVE = CFG["doc_retrieval"]["top_k"]
DEDUP_RETRIEVED_AGAINST_LOCAL = CFG["doc_retrieval"]["dedup_against_local"]

# System context (external corpus of contextualized chunks)
USE_SYSTEM = CFG["system_context"]["enabled"]
SYSTEM_DB_DIR = Path(__file__).resolve().parent / CFG["system_context"]["db_dir"]
SYSTEM_TOP_K = CFG["system_context"]["top_k"]

# LLM-based relevance filtering
USE_LLM_RELEVANCE = CFG["relevance_filter"]["enabled"]
RELEVANCE_DEBUG = CFG["relevance_filter"]["debug"]

# Dense/BM25 combination weights (after min–max normalization)
DENSE_WEIGHT = CFG["retrieval_weights"]["dense"]
BM25_WEIGHT = CFG["retrieval_weights"]["bm25"]

TABLE_NAME = CFG["sqlite"]["table_name"]

# Prompt templates (Jinja2)
PROMPTS_DIR = Path(__file__).resolve().parent / CFG["templates"]["dir"]
PREFIX_TEMPLATE_NAME = CFG["templates"]["prefix"]
REWRITE_TEMPLATE_NAME = CFG["templates"]["rewrite"]
RELEVANCE_TEMPLATE_NAME = CFG["templates"]["relevance"]

# Modes
MODE_PREFIX = CFG["modes"]["prefix"]
MODE_REWRITE = CFG["modes"]["rewrite"]

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

    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            return (obj.get("message") or {}).get("content", "").strip()
        except Exception as e:
            wait = RETRY_WAIT * (2 ** attempt)
            print(f"[warn] Ollama call failed (attempt {attempt+1}): {e} → wait {wait:.1f}s")
            time.sleep(wait)

    return ""


# --------------------------------------------------------------------
# LLM relevance helper
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# LLM relevance helper (multi-candidate, yes/no per ID)
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# LLM relevance helper (multi-candidate, yes/no per ID) — DEBUG VERSION
# --------------------------------------------------------------------

def filter_relevant_candidates(
    query_passage: str,
    unified: Dict[str, Tuple[str, str, str]],
    doc: str,
) -> Tuple[set, set]:
    """
    Run a SINGLE LLM call to decide which doc/system candidates are relevant.

    unified: id -> (kind, text, src_type)
    Returns (relevant_doc_ids, relevant_system_ids)
    """

    # Build candidate list for LLM: only doc/system, NOT local neighbors
    candidates = []
    for cid, (kind, txt, src_type) in unified.items():
        if kind in ("doc", "system") and txt.strip():
            candidates.append({
                "id": cid,
                "kind": kind,
                "text": txt,
            })

    if not candidates:
        return set(), set()

    # Build prompt ----------------------------------------------------
    prompt = render_prompt(
        RELEVANCE_TEMPLATE_NAME,
        query_passage=query_passage,
        candidates=candidates,
        doc=doc,
    )

    # DEBUG PRINT – PROMPT
    if RELEVANCE_DEBUG:
        print("\n===== RELEVANCE FILTER PROMPT =====")
        print(prompt)
        print("===== END RELEVANCE FILTER PROMPT =====\n")

    # Call LLM --------------------------------------------------------
    reply = call_ollama(
        OLLAMA_SERVER,
        OLLAMA_MODEL,
        prompt,
        REQUEST_TIMEOUT,
        TEMPERATURE,
    ).strip()

    # DEBUG PRINT – LLM REPLY
    if RELEVANCE_DEBUG:
        print("===== RELEVANCE FILTER RAW LLM REPLY =====")
        print(reply)
        print("===== END RELEVANCE FILTER RAW LLM REPLY =====\n")

    # Parse yes/no lines ----------------------------------------------
    rel_doc_ids: set = set()
    rel_sys_ids: set = set()

    for line in reply.splitlines():
        line = line.strip()
        if not line:
            continue

        # Format: "<id>\t<yes|no>"
        parts = line.split()
        if len(parts) < 2:
            continue

        cid = parts[0].strip()
        decision = parts[-1].strip().lower()

        if decision not in ("yes", "no"):
            continue
        if decision == "no":
            continue

        kind = unified.get(cid, (None, "", ""))[0]
        if kind == "doc":
            rel_doc_ids.add(cid)
        elif kind == "system":
            rel_sys_ids.add(cid)

    return rel_doc_ids, rel_sys_ids


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


def _minmax_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    s_min = min(scores)
    s_max = max(scores)
    if s_max <= s_min:
        # All equal → return zeros
        return [0.0 for _ in scores]
    return [(s - s_min) / (s_max - s_min) for s in scores]


def rank_indices_combined(
    dense_scores: List[float],
    bm25_sc: List[float],
    top_k: int,
) -> List[int]:
    """
    Combine dense+BM25 scores via min–max normalization + weighted sum,
    then return indices of the top_k items.
    """
    n = len(dense_scores)
    if n == 0:
        return []

    if len(bm25_sc) != n:
        # Fallback: dense only
        return sorted(range(n), key=lambda i: dense_scores[i], reverse=True)[:top_k]

    d_norm = _minmax_normalize(dense_scores)
    b_norm = _minmax_normalize(bm25_sc)

    combined = [
        DENSE_WEIGHT * d_norm[i] + BM25_WEIGHT * b_norm[i]
        for i in range(n)
    ]

    return sorted(range(n), key=lambda i: combined[i], reverse=True)[:top_k]


# --------------------------------------------------------------------
# SQLite helpers (contextualized chunks)
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
# System corpus helpers (external contextualized chunks)
# --------------------------------------------------------------------

def from_blob(blob: bytes) -> np.ndarray:
    """Decode a float32 embedding from a SQLite BLOB."""
    return np.frombuffer(blob, dtype=np.float32)


def load_system_corpus(db_dir: Path) -> Dict[str, Any]:
    """
    Load an external corpus from all *contextualized* .sqlite files in `db_dir`.

    Expected schema in each DB (from contextualize.py):
        contextualized_chunks(
            doc            TEXT NOT NULL,
            id             TEXT PRIMARY KEY,
            chunk_raw      TEXT NOT NULL,
            context_prefix TEXT NOT NULL,
            chunk_ctx      TEXT NOT NULL,
            emb_ctx        BLOB NOT NULL
        )

    We use `chunk_ctx` and `emb_ctx` as the system passages and embeddings.
    """
    texts: List[str] = []
    embs: List[np.ndarray] = []
    tokens: List[List[str]] = []
    df: Dict[str, int] = {}
    doc_lengths: List[int] = []
    ids: List[str] = []

    if not db_dir.exists():
        print(f"[system] directory not found: {db_dir} (system context disabled)")
        return {"texts": [], "embs": [], "tokens": [], "df": {}, "doc_lengths": [], "ids": []}

    for db_path in sorted(db_dir.glob("*.sqlite")):
        try:
            conn = sqlite3.connect(db_path)

            # Check for contextualized_chunks table
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='contextualized_chunks'"
            ).fetchall()
            if not rows:
                conn.close()
                continue

            count = 0
            for doc, chunk_id, chunk_raw, context_prefix, chunk_ctx, emb_ctx_blob in conn.execute(
                "SELECT doc, id, chunk_raw, context_prefix, chunk_ctx, emb_ctx FROM contextualized_chunks"
            ):
                if not chunk_ctx:
                    continue

                vec = from_blob(emb_ctx_blob)
                texts.append(chunk_ctx)
                embs.append(vec)

                toks = simple_tokenize(chunk_ctx)
                tokens.append(toks)
                doc_lengths.append(len(toks))

                for t in set(toks):
                    df[t] = df.get(t, 0) + 1

                ids.append(str(chunk_id))
                count += 1

            conn.close()
            print(f"[system] loaded {count} contextualized chunks from {db_path.name}")
        except Exception as e:
            print(f"[system] failed reading {db_path}: {e}")

    print(f"[system] total external contextualized chunks: {len(texts)} from {db_dir}")
    return {
        "texts": texts,
        "embs": embs,
        "tokens": tokens,
        "df": df,
        "doc_lengths": doc_lengths,
        "ids": ids,
    }


def retrieve_system_context(
    passage: str,
    q_emb: np.ndarray,
    system_corpus: Dict[str, Any],
    top_k: int,
) -> Tuple[str, List[str]]:
    """
    Dense + BM25 retrieval over the external system corpus.

    Returns:
      system_context_text: concatenated text block of up to `top_k` passages.
      system_ids:          list of retrieved external chunk IDs.
    """
    texts = system_corpus.get("texts") or []
    embs = system_corpus.get("embs") or []
    tokens = system_corpus.get("tokens") or []
    df = system_corpus.get("df") or {}
    doc_lengths = system_corpus.get("doc_lengths") or []
    ids = system_corpus.get("ids") or []

    if not texts:
        return "(no system context)", []

    dense_scores = [float(np.dot(e, q_emb)) for e in embs]

    query_tokens = simple_tokenize(passage)
    bm25_sc = bm25_scores(query_tokens, tokens, df, doc_lengths)

    top_indices = rank_indices_combined(dense_scores, bm25_sc, top_k)

    if not top_indices:
        return "(no system context)", []

    context_text = "\n\n---\n\n".join(texts[j] for j in top_indices)
    retrieved_ids = [ids[j] for j in top_indices]
    return context_text, retrieved_ids


# --------------------------------------------------------------------
# Core contextualization logic
# --------------------------------------------------------------------

def contextualize_document(
    doc: str,
    chunks: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    mode: str = MODE_PREFIX,
    system_corpus: Dict[str, Any] = None,
):
    if mode not in {MODE_PREFIX, MODE_REWRITE}:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"  Document '{doc}' — {len(chunks)} chunks (mode={mode})")

    # Retrieval memory over *contextualized* history (same doc)
    ctx_texts: List[str] = []
    ctx_embs: List[np.ndarray] = []
    ctx_tokens: List[List[str]] = []
    ctx_ids: List[str] = []
    df_doc: Dict[str, int] = {}
    doc_lengths: List[int] = []

    results: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    n_chunks = len(chunks)

    for i, item in enumerate(chunks):
        passage = item["chunk"]
        chunk_id = item["id"]

        # ============================================================
        # 1) Build LOCAL context (RAW TEXT)
        # ============================================================
        start_local = max(0, i - LOCAL_BEFORE)
        end_local = min(n_chunks, i + 1 + LOCAL_AFTER)

        local_before_ids = [chunks[j]["id"] for j in range(start_local, i)]
        local_after_ids = [chunks[j]["id"] for j in range(i + 1, end_local)]

        local_before_texts = [chunks[j]["chunk"] for j in range(start_local, i)]
        local_after_texts = [chunks[j]["chunk"] for j in range(i + 1, end_local)]

        # ============================================================
        # 2) Embed query chunk once
        # ============================================================
        q_emb = embedder.encode(
            [passage],
            convert_to_numpy=True,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        )[0]

        # ============================================================
        # 3) Document retrieval (contextualized)
        # ============================================================
        doc_retrieved_ids = []
        doc_retrieved_texts = []

        if ctx_texts:
            dense_scores = [float(np.dot(e, q_emb)) for e in ctx_embs]
            query_tokens = simple_tokenize(passage)
            bm25_sc = bm25_scores(query_tokens, ctx_tokens, df_doc, doc_lengths)

            top_indices = rank_indices_combined(dense_scores, bm25_sc, TOP_K_RETRIEVE)

            # Avoid local overlaps if configured
            if DEDUP_RETRIEVED_AGAINST_LOCAL:
                already_local = set(local_before_ids + local_after_ids)
                top_indices = [j for j in top_indices if ctx_ids[j] not in already_local]

            doc_retrieved_ids = [ctx_ids[j] for j in top_indices]
            doc_retrieved_texts = [ctx_texts[j] for j in top_indices]

        # ============================================================
        # 4) System retrieval (contextualized)
        # ============================================================
        system_retrieved_ids = []
        system_retrieved_texts = []

        if system_corpus:
            sys_text, sys_ids = retrieve_system_context(
                passage,
                q_emb,
                system_corpus,
                SYSTEM_TOP_K,
            )

            if sys_ids:
                sys_chunks = sys_text.split("\n\n---\n\n")
                system_retrieved_ids = sys_ids
                system_retrieved_texts = sys_chunks

        # ============================================================
        # 5) Build unified candidate list and deduplicate by ID
        # ============================================================
        # Priority rule C:
        # local_before/raw → local_after/raw → doc/contextualized → system/contextualized

        candidate_list = []

        for cid, txt in zip(local_before_ids, local_before_texts):
            candidate_list.append(("local_before", cid, txt, "raw"))

        for cid, txt in zip(local_after_ids, local_after_texts):
            candidate_list.append(("local_after", cid, txt, "raw"))

        for cid, txt in zip(doc_retrieved_ids, doc_retrieved_texts):
            candidate_list.append(("doc", cid, txt, "ctx"))

        for cid, txt in zip(system_retrieved_ids, system_retrieved_texts):
            candidate_list.append(("system", cid, txt, "ctx"))

        # Deduplicate by ID, preserving FIRST occurrence (priority respected)
        unified: Dict[str, Tuple[str, str, str]] = {}  # id → (kind, text, source_type)
        for kind, cid, txt, src_type in candidate_list:
            if cid not in unified:
                unified[cid] = (kind, txt, src_type)

               # ============================================================
        # 5.5) Optional LLM relevance filtering over doc/system items
        #      (single call, yes/no per ID)
        # ============================================================
        if USE_LLM_RELEVANCE and unified:
            # Ask the LLM once which doc/system candidates are relevant
            rel_doc_ids, rel_system_ids = filter_relevant_candidates(
                query_passage=passage,
                unified=unified,
                doc=doc,
            )

            filtered_unified: Dict[str, Tuple[str, str, str]] = {}
            new_doc_retrieved_ids: List[str] = []
            new_system_retrieved_ids: List[str] = []

            # Preserve ordering inherent in `unified`
            for cid, (kind, txt, src_type) in unified.items():
                if kind == "doc":
                    if cid not in rel_doc_ids:
                        continue
                    new_doc_retrieved_ids.append(cid)
                elif kind == "system":
                    if cid not in rel_system_ids:
                        continue
                    new_system_retrieved_ids.append(cid)

                # Local neighbors are always kept (never filtered by LLM)
                filtered_unified[cid] = (kind, txt, src_type)

            unified = filtered_unified
            doc_retrieved_ids = new_doc_retrieved_ids
            system_retrieved_ids = new_system_retrieved_ids


        # ============================================================
        # 6) Reconstruct final context buckets from unified set
        # ============================================================
        final_local_before: List[str] = []
        final_local_after: List[str] = []
        final_doc_ctx: List[str] = []
        final_system_ctx: List[str] = []

        for cid, (kind, txt, src_type) in unified.items():
            if kind == "local_before":
                final_local_before.append(txt)
            elif kind == "local_after":
                final_local_after.append(txt)
            elif kind == "doc":
                final_doc_ctx.append(txt)
            elif kind == "system":
                final_system_ctx.append(txt)

        local_context_before = "\n\n".join(final_local_before)
        local_context_after = "\n\n".join(final_local_after)
        retrieved_context = "\n\n---\n\n".join(final_doc_ctx)
        system_context = "\n\n---\n\n".join(final_system_ctx)


        # ============================================================
        # 7) Build LLM prompt (prefix or rewrite)
        # ============================================================
        if mode == MODE_PREFIX:
            template_name = PREFIX_TEMPLATE_NAME
        else:
            template_name = REWRITE_TEMPLATE_NAME

        prompt = render_prompt(
            template_name,
            current_passage=passage,
            local_context_before=local_context_before,
            local_context_after=local_context_after,
            retrieved_context=retrieved_context,
            system_context=system_context,
            doc=doc,
        )

        # DEBUG: show the prompt
        print("\n===== PROMPT DEBUG =====")
        print(prompt)
        print("===== END PROMPT DEBUG =====\n")

        reply = call_ollama(
            OLLAMA_SERVER,
            OLLAMA_MODEL,
            prompt,
            REQUEST_TIMEOUT,
            TEMPERATURE,
        ).strip()

        print("\n===== LLM RAW RESPONSE =====")
        print(reply)
        print("===== END LLM RAW RESPONSE =====\n")

        if mode == MODE_PREFIX:
            context_prefix = reply or "(No additional context found.)"
            contextualized = f"{context_prefix} {passage}"
        else:
            # REWRITE mode: LLM returns a self-contained rewritten passage.
            contextualized = reply or passage
            context_prefix = contextualized

        # 8) Embed ONLY the contextualized text
        emb_ctx = embedder.encode(
            [contextualized],
            convert_to_numpy=True,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        )[0]

        # 9) Collect outputs (JSON) including retrieval debugging info
        local_neighbor_ids = local_before_ids + local_after_ids

        result_obj: Dict[str, Any] = {
            "doc": doc,
            "id": chunk_id,
            "chunk": passage,
            "contextualized_chunk": contextualized,
            "local_neighbor_ids": local_neighbor_ids,
            "doc_retrieved_ids": doc_retrieved_ids,
            "system_retrieved_ids": system_retrieved_ids,
        }

        if mode == MODE_PREFIX:
            result_obj["context_prefix"] = context_prefix

        results.append(result_obj)

        rows.append({
            "doc": doc,
            "id": chunk_id,
            "chunk_raw": passage,
            "context_prefix": context_prefix,   # still required for SQLite schema
            "chunk_ctx": contextualized,
            "emb_ctx": emb_ctx,
        })

        # 10) Update retrieval memory for document-level context
        ctx_texts.append(contextualized)
        ctx_embs.append(emb_ctx)
        toks = simple_tokenize(contextualized)
        ctx_tokens.append(toks)
        doc_lengths.append(len(toks))
        ctx_ids.append(chunk_id)
        for t in set(toks):
            df_doc[t] = df_doc.get(t, 0) + 1

        print(f"    [{i+1}/{n_chunks}] contextualized for '{doc}'", flush=True)

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

    # Load external system corpus once (if enabled)
    system_corpus: Dict[str, Any] = {}
    if USE_SYSTEM:
        system_corpus = load_system_corpus(SYSTEM_DB_DIR)
    else:
        system_corpus = {}

    all_results: List[Dict[str, Any]] = []
    t0 = time.time()

    def sort_id(o):  # numeric sort
        m = re.search(r"_(\d+)$", o["id"])
        return int(m.group(1)) if m else 10**9

    for doc, lst in docs.items():
        lst = sorted(lst, key=sort_id)
        res, rows = contextualize_document(
            doc,
            lst,
            embedder,
            mode=mode,
            system_corpus=system_corpus if USE_SYSTEM else None,
        )
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
        default=MODE_REWRITE,
        help="Contextualization mode: 'prefix' (default) or 'rewrite'.",
    )

    args = parser.parse_args()
    contextualize_file(args.input_chunks_json, mode=args.mode)


if __name__ == "__main__":
    main()
