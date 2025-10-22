"""
Agentic Contextual Ingestion — universal version (v3)
For legal, ethical, social, and environmental compliance frameworks.

Processes regulatory or guideline text to extract entities, relationships, and claims for use in multi-jurisdictional LLM compliance advisors.

Outputs in <inputfilename>_output/:
  passages.jsonl, claims.jsonl, entities.jsonl, relationships.jsonl,
  vec_index.npy, vec_meta.jsonl, bm25.json
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Set

import orjson
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from collections import OrderedDict


# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Back-compat: if INPUT_PATH is a valid file, it's used without prompting.
# Otherwise, we prompt from INPUT_DIR (defaults to BASE_DIR).
DEFAULT_INPUT = None  # no hardcoded file by default
DEFAULT_INPUT_DIR = BASE_DIR

CONFIG = {
    "INPUT_PATH": DEFAULT_INPUT,
    "INPUT_DIR": DEFAULT_INPUT_DIR,

    # Kept for backward compat; not used for output naming anymore.
    "OUT_DIR": os.path.join(BASE_DIR, "out"),

    "EMBED_MODEL": "BAAI/bge-small-en-v1.5",

    "LOCAL_K": 1,
    "BM25_K": 2,
    "VEC_K": 2,

    # Updated generation settings
    "MAX_TOKENS": 1024,
    "TEMPERATURE": 0,

    # Removed legacy char caps:
    # "COMPACT_MAX_CHARS": ...
    # "PROMPT_PASSAGE_MAX_CHARS": ...

    # Updated chunking settings
    "SENT_TARGET_WORDS": 400,
    "SENT_OVERLAP": 5,

    "USE_HEURISTIC_CLAIMS": True,
    "USE_RULE_FALLBACK_ER": True,

    "SLEEP_BETWEEN_PASSAGES": 0.0,
}

SMALL_FREE_POOL = [
    "z-ai/glm-4.5-air:free",
    "openai/gpt-oss-20b:free",
    "qwen/qwen2.5-7b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
]

BIG_FALLBACKS = [
    "mistralai/mistral-small:free",
    "deepseek/deepseek-chat:free",
]

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


# ==============================
# Utils
# ==============================
def jdump(obj: Any) -> str:
    return orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode()

def stable_id(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:32]

def _as_str(x):
    return "" if x is None else str(x)

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ==============================
# File picker (console)
# ==============================
def pick_input_txt(base_dir: str) -> str:
    """List .txt files in base_dir and prompt user to pick one (by number or exact name)."""
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"Input directory does not exist: {base_dir}")

    files = sorted(
        f for f in os.listdir(base_dir)
        if f.lower().endswith(".txt") and os.path.isfile(os.path.join(base_dir, f))
    )
    if not files:
        raise RuntimeError(f"No .txt files found in: {base_dir}")

    print("\nAvailable .txt files:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")

    choice = input("\nEnter file number or exact name: ").strip()
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(files):
            return os.path.join(base_dir, files[idx - 1])
        raise RuntimeError(f"Number out of range (1..{len(files)}).")
    else:
        path_try = os.path.join(base_dir, choice)
        if choice in files:
            return path_try
        # also allow a direct relative/absolute path inside base_dir
        if os.path.isfile(path_try):
            return path_try
        raise RuntimeError(f"'{choice}' not found among listed files.")


# ==============================
# Preprocessing & chunking
# ==============================
def preprocess_text(raw: str) -> str:
    t = raw.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"-\n(\w)", r"\1", t)
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    t = re.sub(r"[ \t\u00A0]+", " ", t)
    return t.strip()

def split_sentences(text: str) -> List[str]:
    raw = text.strip()
    if not raw:
        return []
    blocks = re.split(r'\n{2,}', raw)
    sents: List[str] = []
    for b in blocks:
        b = normalize_spaces(b)
        if not b:
            continue
        if len(b) < 80 and not re.search(r'[.!?]', b):
            sents.append(b); continue
        parts = re.split(r'(?<=[.!?])\s+', b)
        for p in parts:
            p = normalize_spaces(p)
            if p: sents.append(p)
    return sents

def sentence_windows(text: str, target_words: int = 140, overlap_sents: int = 2) -> List[str]:
    sents = split_sentences(text)
    if not sents:
        return []
    passages: List[str] = []
    i = 0
    while i < len(sents):
        bag: List[str] = []
        wc = 0; j = i
        while j < len(sents) and wc < target_words:
            bag.append(sents[j]); wc += len(sents[j].split()); j += 1
        passages.append(" ".join(bag).strip())
        if j >= len(sents): break
        step = max(1, len(bag) - overlap_sents)
        i += step
    return passages


# ==============================
# OpenAI (via OpenRouter) + Router
# ==============================
class ORClient:
    def __init__(self, model: str, temperature: float, max_tokens: int):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set.")
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat_json(self, system: str, user: str, timeout: int = 60) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            timeout=timeout,
        )
        return (resp.choices[0].message.content or "{}").strip()

class RoundRobinLLM:
    def __init__(self, small_pool: List[str], big_pool: List[str], temperature: float, max_tokens: int):
        self.small = small_pool[:]
        self.big = big_pool[:]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.i_small = 0

    def _next_small(self) -> str:
        m = self.small[self.i_small % len(self.small)]
        self.i_small += 1
        return m

    def chat_json(self, system: str, user: str, tries: int = 8) -> str:
        for _ in range(tries):
            try:
                model = self._next_small()
                return ORClient(model, self.temperature, self.max_tokens).chat_json(system, user)
            except Exception:
                continue
        for model in self.big:
            try:
                return ORClient(model, self.temperature, self.max_tokens).chat_json(system, user)
            except Exception:
                pass
        return json.dumps({"contextualized_passage":"","entities":[],"relationships":[],"claims":[]})


# ==============================
# Normalizers
# ==============================
def normalize_claims(claims):
    out = []
    if not isinstance(claims, list): return out
    for c in claims:
        text = normalize_spaces(_as_str(c.get("text") if isinstance(c, dict) else c))
        if text: out.append({"text": text})
    return out

def normalize_entities(entities):
    out = []
    if not isinstance(entities, list): return out
    for e in entities:
        if isinstance(e, dict):
            name = normalize_spaces(_as_str(e.get("name"))); typ = normalize_spaces(_as_str(e.get("type"))); cid = normalize_spaces(_as_str(e.get("canonical_id")))
        elif isinstance(e, str):
            name, typ, cid = normalize_spaces(e), "unknown", ""
        else:
            continue
        if name:
            out.append({"name": name, "type": typ, "canonical_id": cid})
    return out

def normalize_relationships(rels):
    out = []
    if not isinstance(rels, list): return out
    for r in rels:
        if isinstance(r, dict):
            src = normalize_spaces(_as_str(r.get("source"))); tgt = normalize_spaces(_as_str(r.get("target")))
            typ = normalize_spaces(_as_str(r.get("type"))); ev = normalize_spaces(_as_str(r.get("evidence_span")))
        else:
            src = tgt = ev = ""; typ = normalize_spaces(_as_str(r))
        out.append({"source": src, "target": tgt, "type": typ, "evidence_span": ev})
    return out


# ==============================
# Retrieval indices (persisted)
# ==============================
class BM25Index:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.docs: List[List[str]] = []
        self.raw: List[str] = []
        self.model: Optional[BM25Okapi] = None
        self._tok = lambda text: re.findall(r"\w+", text.lower())

    def add(self, passage_text: str):
        tokens = self._tok(passage_text)
        if not tokens: return
        self.docs.append(tokens); self.raw.append(passage_text)
        self.model = BM25Okapi(self.docs)

    def search(self, query: str, top_k: int = 2, exclude_idx: Optional[int] = None):
        if not self.docs or self.model is None:
            return []
        q = self._tok(query)
        if not q:
            return []
        scores = self.model.get_scores(q)
        idx_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results: List[Tuple[int, float, str]] = []
        for idx, sc in idx_scores[: top_k + 5]:
            if exclude_idx is not None and idx == exclude_idx:
                continue
            results.append((idx, float(sc), self.raw[idx]))
            if len(results) >= top_k:
                break
        return results

    def save(self):
        with open(self.save_path, "wb") as f:
            f.write(orjson.dumps({"raw": self.raw}))

    def load(self):
        if not os.path.exists(self.save_path): return
        raw = orjson.loads(open(self.save_path, "rb").read()).get("raw", [])
        self.raw = list(map(str, raw))
        self.docs = [re.findall(r"\w+", r.lower()) for r in self.raw]
        if self.docs: self.model = BM25Okapi(self.docs)

class SimpleVectorIndex:
    def __init__(self, dim: int, npy_path: str, meta_path: str):
        self.dim = dim
        self.vecs = np.empty((0, dim), dtype=np.float32)
        self.meta: List[Dict[str, Any]] = []
        self.npy_path = npy_path; self.meta_path = meta_path

    def add(self, vec: np.ndarray, meta: Dict[str, Any]):
        v = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        if v.shape[1] != self.dim: raise ValueError(f"Vector dim {v.shape[1]} != expected {self.dim}")
        self.vecs = np.vstack([self.vecs, v]); self.meta.append(meta)

    def search(self, query_vec: np.ndarray, k: int = 2) -> List[Dict[str, Any]]:
        if self.vecs.shape[0] == 0 or k <= 0: return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dim: return []
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        An = self.vecs / (np.linalg.norm(self.vecs, axis=1, keepdims=True) + 1e-12)
        sims = (An @ qn.T).reshape(-1)
        top_idx = np.argsort(-sims)[:k]
        results = []
        for i in top_idx:
            m = self.meta[i].copy(); m["_score"] = float(sims[i]); results.append(m)
        return results

    def save(self):
        np.save(self.npy_path, self.vecs)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(jdump(m) + "\n")

    def load(self):
        if os.path.exists(self.npy_path): self.vecs = np.load(self.npy_path)
        if os.path.exists(self.meta_path):
            self.meta = [orjson.loads(l) for l in open(self.meta_path, "r", encoding="utf-8")]


# ==============================
# Agentic relevance + compaction
# ==============================
# ------------------------------
# Relevance gate prompt (escaped braces!)
# ------------------------------
REL_SYS = "You label relevance only. Return strict JSON."
REL_PROMPT = """Decide whether the candidate context helps interpret or expand the meaning of the given passage.

Return JSON exactly: {{"label": "likely_relevant" | "maybe" | "irrelevant"}}

# Passage
{passage}

# Candidate
{candidate}
"""


COMP_SYS = "You write concise, lossless bullet summaries for regulatory, ethical, or governance text."
COMP_PROMPT = """Summarize the following context into terse bullet points preserving all key normative or analytical terms.
Return plain text only (no JSON, no markdown).

---
{ctx}
"""


# ----------------------------------------
# Main extraction system + user prompt
# ----------------------------------------
MAIN_SYS = (
    "You are a meticulous legal-research assistant for compliance/AI governance. "
    "Given a passage and compact context, (1) rewrite a compact 'contextualized_passage'; "
    "(2) extract structured entities, relationships, and atomic claims. "
    "Return STRICT JSON only matching the schema."
)

# Removed the (~1800 chars) cap
MAIN_SYS = (
    "You are a meticulous compliance and governance research assistant. "
    "Given a passage and its compact context, produce a contextualized version "
    "and extract key entities, relationships, and atomic claims. "
    "Return STRICT JSON only matching the schema."
)

MAIN_USER_TPL = (
    "Document ID: {doc_id}\n"
    "Passage Index: {idx}\n\n"
    "# Original Passage\n{passage}\n\n"
    "# Compact Context\n{compact_ctx}\n\n"
    "# TASKS\n"
    "1) contextualized_passage: a self-contained, precise restatement with necessary context.\n"
    "2) Extract:\n"
    "   - entities: list of {{name, type, canonical_id?}} (≥2 if possible).\n"
    "   - relationships: list of {{source, target, type, evidence_span?}} "
    "     using types such as: {{obligation, permission, prohibition, scope, exception, "
    "     definition, responsibility, condition, requirement, right, safeguard, risk, metric}}.\n"
    "   - claims: short, atomic, verifiable sentences (15–40 words).\n\n"
    "# STRICT JSON SCHEMA\n"
    "{{\n"
    '  "contextualized_passage": "string",\n'
    '  "entities": [{{"name": "string", "type": "string", "canonical_id": "string?"}}],\n'
    '  "relationships": [{{"source": "string", "target": "string", "type": "string", "evidence_span": "string?"}}],\n'
    '  "claims": ["string"]\n'
    "}}\n"
    "Respond with JSON only."
)



def judge_relevance(router: RoundRobinLLM, passage: str, candidate: str) -> str:
    user = REL_PROMPT.format(passage=passage, candidate=candidate)
    try:
        out = orjson.loads(router.chat_json(REL_SYS, user))
        return out.get("label", "maybe")
    except Exception:
        return "maybe"

def compact_context(router: RoundRobinLLM, items: List[str]) -> str:
    if not items: return "(none)"
    joined = "\n\n".join(items)
    user = COMP_PROMPT.format(ctx=joined)
    txt = router.chat_json(COMP_SYS, user)
    # If a model returns JSON-ish, try to unwrap; else return normalized text
    try:
        parsed = orjson.loads(txt)
        for k in ("text", "summary", "content"):
            if isinstance(parsed.get(k), str):
                return normalize_spaces(parsed[k])
    except Exception:
        pass
    return normalize_spaces(txt)


# ==============================
# Heuristic fallbacks
# ==============================
def looks_like_sentence(s: str) -> bool:
    s = normalize_spaces(s)
    if not s: return False
    if not re.match(r'^[A-Z(“"\']', s): return False
    if not re.search(r'[.!?]$', s): return False
    wc = len(s.split()); return 12 <= wc <= 60

def heuristic_claims_from_text(text: str, max_claims: int = 2) -> List[str]:
    cues = (
    r"\b(must|shall|should|may|may not|prohibit|forbid|require|obliged|"
    r"has to|is responsible|is required|applies to|covers|ensure|guarantee|"
    r"disclose|protect|verify|monitor|evaluate|audit|"
    r"respect|promote|comply|adhere|justify|document|record|limit|"
    r"allow|restrict|balance|fair|transparent|explain|report)\b"
)
    sents = split_sentences(text); out, seen = [], set()
    for s in sents:
        s = normalize_spaces(s)
        if looks_like_sentence(s) and re.search(cues, s, flags=re.I):
            if s not in seen: out.append(s); seen.add(s)
        if len(out) >= max_claims: break
    if len(out) < max_claims:
        for s in sents:
            s = normalize_spaces(s)
            if looks_like_sentence(s) and s not in seen:
                out.append(s); seen.add(s)
            if len(out) >= max_claims: break
    return out

def rule_fallback_entities(contextualized: str) -> List[Dict[str, str]]:
    vocab = [
    # Legal/organizational actors
    ("Member State", "jurisdiction"),
    ("authority", "actor"),
    ("competent authority", "actor"),
    ("organization", "actor"),
    ("institution", "actor"),
    ("public body", "actor"),
    ("controller", "role"),
    ("processor", "role"),
    ("data subject", "role"),
    ("auditor", "role"),
    ("supervisory authority", "actor"),
    ("AI provider", "role"),
    ("deployer", "role"),
    ("user", "role"),
    ("operator", "role"),

    # Regulatory/ethical domains
    ("personal data", "data_type"),
    ("training data", "data_type"),
    ("biometric data", "data_type"),
    ("genetic data", "data_type"),
    ("model output", "data_type"),
    ("risk", "concept"),
    ("bias", "concept"),
    ("transparency", "principle"),
    ("explainability", "principle"),
    ("fairness", "principle"),
    ("accountability", "principle"),
    ("sustainability", "principle"),
    ("environmental impact", "concept"),
    ("governance", "concept"),
    ("ethical guideline", "reference"),
    ("Article", "reference"),
    ("Annex", "reference"),
    ("Section", "reference"),
]
    lower_ctx = contextualized.lower()
    out, seen = [], set()
    for name, typ in vocab:
        if name.lower() in lower_ctx and name not in seen:
            out.append({"name": name, "type": typ, "canonical_id": ""}); seen.add(name)
    return out[:6]

def rule_fallback_relationships(contextualized: str, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if len(entities) < 2: return []
    cues = [
    ("shall be prohibited", "prohibition"),
    ("shall not", "prohibition"),
    ("shall", "obligation"),
    ("must", "obligation"),
    ("should", "recommendation"),
    ("may", "permission"),
    ("is required", "obligation"),
    ("is responsible for", "responsibility"),
    ("is accountable for", "responsibility"),
    ("ensures", "obligation"),
    ("applies to", "scope"),
    ("is subject to", "condition"),
    ("except where", "exception"),
    ("provided that", "condition"),
    ("in accordance with", "reference"),
    ("for the purpose of", "scope"),
    ("based on", "condition"),
]
    for phrase, rtype in cues:
        if phrase in contextualized:
            return [{"source": entities[0]["name"], "target": entities[1]["name"], "type": rtype, "evidence_span": phrase}]
    return []


# ==============================
# Canonicalization
# ==============================
CANON_MAP = {
    # Jurisdictions / institutions
    "union": "kb:eu",
    "european union": "kb:eu",
    "united nations": "kb:un",
    "oecd": "kb:oecd",
    "iso": "kb:iso",
    "iec": "kb:iec",

    # General AI/tech entities
    "ai system": "ai:system",
    "ai model": "ai:model",
    "dataset": "ai:dataset",
    "training data": "ai:training_data",
    "validation data": "ai:validation_data",
    "testing data": "ai:testing_data",
    "algorithm": "ai:algorithm",
    "component": "ai:component",

    # Principles / values
    "transparency": "eth:transparency",
    "explainability": "eth:explainability",
    "fairness": "eth:fairness",
    "accountability": "eth:accountability",
    "privacy": "eth:privacy",
    "sustainability": "eth:sustainability",
    "safety": "eth:safety",
    "security": "eth:security",
    "trustworthiness": "eth:trustworthiness",

    # Legal & procedural
    "risk management": "gov:risk_management",
    "impact assessment": "gov:impact_assessment",
    "data protection": "law:data_protection",
    "data governance": "law:data_governance",
}


ART_RE = re.compile(r"\barticle\s+(\d+[a-z]?)\b", re.I)

def canonicalize_entity(e: Dict[str, str], doc_id: str) -> Dict[str, str]:
    """Assign a stable canonical_id using a mapping, 'Article N' detector, and a deterministic fallback."""
    name = e.get("name","").strip()
    low = name.lower()
    cid = e.get("canonical_id","").strip()

    if not cid:
        cid = CANON_MAP.get(low, "")

    if not cid:
        m = ART_RE.search(low)
        if m:
            cid = f"{doc_id}:art_{m.group(1)}"

    if not cid and name:
        cid = f"{doc_id}:ent_{stable_id(name)[:8]}"

    e["canonical_id"] = cid
    return e

def canonicalize_entities(entities: List[Dict[str, str]], doc_id: str) -> List[Dict[str, str]]:
    return [canonicalize_entity(e, doc_id) for e in entities]

def build_name_to_id(entities: List[Dict[str, str]]) -> Dict[str, str]:
    """Map entity surface name -> canonical_id (last occurrence wins if duplicates)."""
    m = {}
    for e in entities:
        n = e.get("name","")
        if n:
            m[n] = e.get("canonical_id","")
    return m


# ==============================
# Data class
# ==============================
@dataclass
# ==============================
# Data class
# ==============================

@dataclass
class PassageRec:
    doc_id: str
    id: str
    idx: int
    text: str
    contextualized: str
    neighbors_vec: List[str]
    neighbors_bm25: List[str]
    claims: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]

    def to_ordered_dict(self) -> OrderedDict:
        """Return an OrderedDict in the canonical field order for JSON export."""
        return OrderedDict([
            ("doc_id", self.doc_id),
            ("id", self.id),
            ("idx", self.idx),
            ("text", self.text),
            ("contextualized", self.contextualized),
            ("neighbors_vec", self.neighbors_vec),
            ("neighbors_bm25", self.neighbors_bm25),
            ("claims", self.claims),
            ("entities", self.entities),
            ("relationships", self.relationships),
        ])



# ==============================
# Main
# ==============================
def run_pipeline(cfg: Dict[str, Any]):
    # Prefer an explicit INPUT_PATH if provided and valid; otherwise prompt from INPUT_DIR.
    input_path = cfg.get("INPUT_PATH")
    if not (isinstance(input_path, str) and os.path.isfile(input_path)):
        base_dir = cfg.get("INPUT_DIR", BASE_DIR)
        input_path = pick_input_txt(base_dir)

    doc_id = os.path.splitext(os.path.basename(input_path))[0]

    # Output folder = "<inputfilename>_output" next to the input file
    out_dir = os.path.join(os.path.dirname(input_path), f"{doc_id}_contextualized")
    ensure_dir(out_dir)

    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()
    clean = preprocess_text(raw)

    passages = sentence_windows(clean, target_words=cfg["SENT_TARGET_WORDS"], overlap_sents=cfg["SENT_OVERLAP"])

    embedder = SentenceTransformer(cfg["EMBED_MODEL"])
    emb_dim = embedder.get_sentence_embedding_dimension()

    bm25_path = os.path.join(out_dir, "bm25.json")
    vec_npy = os.path.join(out_dir, "vec_index.npy")
    vec_meta = os.path.join(out_dir, "vec_meta.jsonl")

    bm25 = BM25Index(bm25_path); bm25.load()
    vindex = SimpleVectorIndex(dim=emb_dim, npy_path=vec_npy, meta_path=vec_meta); vindex.load()

    router = RoundRobinLLM(SMALL_FREE_POOL, BIG_FALLBACKS, cfg["TEMPERATURE"], cfg["MAX_TOKENS"])

    passages_jsonl = os.path.join(out_dir, "passages.jsonl")
    claims_jsonl = os.path.join(out_dir, "claims.jsonl")
    entities_jsonl = os.path.join(out_dir, "entities.jsonl")
    relationships_jsonl = os.path.join(out_dir, "relationships.jsonl")

    emitted_claims: Set[str] = set()
    if os.path.exists(claims_jsonl):
        for line in open(claims_jsonl, "r", encoding="utf-8"):
            try:
                d = orjson.loads(line); key = stable_id(d.get("doc_id",""), d.get("claim","")); emitted_claims.add(key)
            except Exception: pass

    with open(passages_jsonl, "a", encoding="utf-8") as f_out, \
         open(claims_jsonl, "a", encoding="utf-8") as f_claims, \
         open(entities_jsonl, "a", encoding="utf-8") as f_entities, \
         open(relationships_jsonl, "a", encoding="utf-8") as f_rels:

        for idx, p in enumerate(tqdm(passages, desc="Processing passages")):
            p_for_llm = p
            p_id = stable_id(doc_id, str(idx), p[:120])

            # Local neighbors
            neighbors: List[str] = []
            for off in range(1, cfg["LOCAL_K"] + 1):
                if idx - off >= 0: neighbors.append(passages[idx - off])
                if idx + off < len(passages): neighbors.append(passages[idx + off])
            local_context = neighbors[: 2 * cfg["LOCAL_K"]]

            # Retrieved neighbors (prior only)
            vec_neighbors_text, vec_neighbors_ids = [], []
            if len(vindex.meta) > 0 and cfg["VEC_K"] > 0:
                qvec = embedder.encode([p_for_llm])[0]
                for h in vindex.search(qvec, k=cfg["VEC_K"]):
                    vec_neighbors_text.append(h.get("text","")); vec_neighbors_ids.append(h.get("id",""))

            bm25_neighbors = bm25.search(p_for_llm, top_k=cfg["BM25_K"])
            bm25_neighbors_text = [t for (_i,_s,t) in bm25_neighbors]

            # Relevance gate
            candidates = []
            for src in (local_context, bm25_neighbors_text, vec_neighbors_text):
                for c in src:
                    if judge_relevance(router, p_for_llm, c) in ("likely_relevant","maybe"):
                        candidates.append(c)

            # Compaction
            compact_ctx = compact_context(router, candidates)

            # Main extraction
            user_prompt = MAIN_USER_TPL.format(doc_id=doc_id, idx=idx, passage=p_for_llm, compact_ctx=compact_ctx)
            completion = router.chat_json(MAIN_SYS, user_prompt)

            try:
                payload = orjson.loads(completion)
            except Exception:
                payload = None
                m = re.search(r"\{[\s\S]*\}$", completion.strip())
                if m:
                    try: payload = orjson.loads(m.group(0))
                    except Exception: payload = None
            if not isinstance(payload, dict):
                payload = {"contextualized_passage": p_for_llm, "entities": [], "relationships": [], "claims": []}

            contextualized = normalize_spaces(str(payload.get("contextualized_passage", p_for_llm)))[:4000] or p_for_llm
            entities = normalize_entities(payload.get("entities", []))
            relationships = normalize_relationships(payload.get("relationships", []))
            norm_claims = normalize_claims(payload.get("claims", []))

            # Fallbacks
            if CONFIG.get("USE_RULE_FALLBACK_ER", True):
                if not entities: entities = rule_fallback_entities(contextualized)
                if not relationships: relationships = rule_fallback_relationships(contextualized, entities)
            if CONFIG.get("USE_HEURISTIC_CLAIMS", True) and not norm_claims:
                guessed = heuristic_claims_from_text(contextualized) or heuristic_claims_from_text(p_for_llm)
                norm_claims = [{"text": c} for c in guessed]

            # Canonicalize entities
            entities = canonicalize_entities(entities, doc_id)
            name2id = build_name_to_id(entities)

            # Embed contextualized (for vector index only; not stored in passages.jsonl)
            vec = embedder.encode([contextualized])[0].astype(np.float32)

            rec = PassageRec(
                id=p_id, doc_id=doc_id, idx=idx, text=p, contextualized=contextualized,
                claims=norm_claims, entities=entities, relationships=relationships,
                neighbors_bm25=bm25_neighbors_text, neighbors_vec=vec_neighbors_ids
            )

            # Persist passage (no embedding field)
            f_out.write(jdump(rec.to_ordered_dict()) + "\n")

            # Claims (dedup)
            for c in norm_claims:
                claim_text = normalize_spaces(c["text"])
                if not claim_text: continue
                key = stable_id(doc_id, claim_text)
                if key in emitted_claims: continue
                emitted_claims.add(key)
                f_claims.write(jdump({"doc_id": doc_id, "passage_id": rec.id, "claim": claim_text}) + "\n")

            # Entities
            for e in entities:
                f_entities.write(jdump({
                    "doc_id": doc_id,
                    "passage_id": rec.id,
                    "name": e["name"],
                    "type": e.get("type",""),
                    "canonical_id": e.get("canonical_id",""),
                }) + "\n")

            # Relationships (+ canonical IDs if resolvable)
            for r in relationships:
                f_rels.write(jdump({
                    "doc_id": doc_id,
                    "passage_id": rec.id,
                    "source": r.get("source",""),
                    "target": r.get("target",""),
                    "source_id": name2id.get(r.get("source",""), ""),
                    "target_id": name2id.get(r.get("target",""), ""),
                    "type": r.get("type",""),
                    "evidence": r.get("evidence_span",""),
                }) + "\n")

            # Update indices (persist incrementally)
            vindex.add(vec, {"id": p_id, "text": contextualized}); vindex.save()
            bm25.add(contextualized); bm25.save()

            if CONFIG["SLEEP_BETWEEN_PASSAGES"] > 0:
                time.sleep(CONFIG["SLEEP_BETWEEN_PASSAGES"])

    print("\nDone.")
    print(f"- Dir: {out_dir}")
    print("- Outputs: passages.jsonl, claims.jsonl, entities.jsonl, relationships.jsonl")
    print("- Indices: bm25.json, vec_index.npy, vec_meta.jsonl")
    print("- Entity canonicalization: ON (map + Article detector + stable fallback)")


if __name__ == "__main__":
    run_pipeline(CONFIG)
