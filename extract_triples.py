"""
extract_triples.py — extract (subject, predicate, object) triples from atomic claims.

Expected input:
  - JSONL file produced by extract_claims.py
    Each line is a JSON object with keys:
      {
        "doc": "...",
        "chunk_id": "...",
        "claim_id": "...",
        "claim_text": "...",
        "source_quote": "..."
      }

Behavior:
  • For each claim, send the claim_text to a local Ollama model.
  • The model returns a JSON array of triples:
        - subject
        - predicate
        - object
  • We parse and normalize the model output.
  • Write one JSON line per triple to an output .jsonl file.

Output:
  - <input_basename>_triples.jsonl
"""

import json
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Union

import urllib.request
import urllib.error

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

# Base URL for your local Ollama server
OLLAMA_SERVER = "http://localhost:11434"

# Default Ollama model tag to use for triple extraction
# You can switch between these depending on what you have pulled:
#   - "llama3:8b"
#   - "gemma3:4b-it-qat"
OLLAMA_MODEL = "llama3:8b"

# HTTP timeout for each request (seconds)
REQUEST_TIMEOUT = 120

# LLM sampling temperature (0.0 is best for structured JSON)
TEMPERATURE = 0.0

# Print progress every N claims
PROGRESS_EVERY = 5


# --------------------------------------------------------------------
# Prompt template
# --------------------------------------------------------------------

# IMPORTANT: all literal { and } used for JSON/examples are escaped as {{ and }}
# so that .format(text=...) only fills in {text}.
PROMPT = """You are extracting *semantic triples* (subject, predicate, object) from a single factual claim.

INPUT
------
You are given one atomic factual claim written in plain English.
Each claim expresses one or more closely related relationships.

YOUR TASK
-----------
Represent the meaning of the claim as one or more (subject, predicate, object) triples.

Each triple must be fully grounded in the claim text.
Do NOT invent, generalize, or reorder information.

DEFINITIONS
------------
- subject: the main actor, entity, or concept performing or being described.
- predicate: the action, relation, or attribute connecting subject and object.
- object: the entity, concept, or result affected by the predicate.

OUTPUT FORMAT
---------------
Return a JSON array, where each element has:
    - "subject": concise noun phrase
    - "predicate": concise verb or relational phrase
    - "object": concise noun phrase

STRICT RULES
-------------
- Use ONLY information explicitly present in the claim text.

- SUBJECT and OBJECT:
  • Must be concrete noun phrases taken from the claim (copied exactly or with
    minimal grammatical adjustment).
  • When the claim contains a longer, specific noun phrase (e.g.
    "development of modern foundation models",
    "pre-training data for the Llama 3 models",
    "Llama 3.1 405B"),
    use that full phrase as the subject or object, not just the head noun
    ("development", "data", "Llama 3.1").
  • Avoid vague placeholders like "something", "this", "that", "it".

- PREDICATE:
  • Should be a short verb phrase (e.g. "was pre-trained on", "has", "natively supports").
  • Do NOT use only prepositions or function words as predicates (e.g. "in", "of",
    "with", "on", "using").
  • Do NOT output long clause fragments as the predicate.

- OBJECT:
  • Must NOT be just a function word (e.g. "on", "in", "using", "to", "with").
  • Must contain at least one meaningful content word (noun/adjective/number).

- COORDINATED LISTS:
  • If the claim lists several objects with "and" or commas
    (e.g. "a pre-training stage and a post-training stage",
          "multilinguality, coding, and reasoning"),
    you may either:
      - keep them together in a single object phrase, OR
      - produce multiple triples, one per item,
    but you must NOT drop items from the list.
  • Example (both acceptable):
      subject: "development of modern foundation models"
      predicate: "consists of"
      object: "a pre-training stage and a post-training stage"
    OR:
      subject: "development of modern foundation models"
      predicate: "consists of"
      object: "a pre-training stage"
      subject: "development of modern foundation models"
      predicate: "consists of"
      object: "a post-training stage"

- FIXED EXPRESSIONS:
  • Do NOT split fixed expressions like "was pre-trained on X" into partial triples.
    Represent them as a single triple:
      subject:   "Llama 3"
      predicate: "was pre-trained on"
      object:    "approximately 15 trillion multilingual tokens"

- CARDINALITY:
  • Extract at most 3 triples per claim.
  • If the claim lists similar capabilities or items (e.g. "multilinguality, coding, reasoning, tool usage"),
    you may output up to 4 triples (one per item), but only if they share the same subject and predicate.
  • If there is no valid, well-formed triple, return [].

STYLE
-------
- Use short, lowercase predicates (e.g. "has", "was released in", "natively supports").
- Keep subject, predicate, and object each under about 10 words.
- Maintain technical precision; do not paraphrase more than necessary.

GOOD EXAMPLES
Claim: "Llama 3 was pre-trained on approximately 15 trillion multilingual tokens."
→ [
  {{
    "subject": "Llama 3",
    "predicate": "was pre-trained on",
    "object": "approximately 15 trillion multilingual tokens"
  }}
]

Claim: "The Llama 3 Herd natively supports multilinguality, coding, and reasoning."
→ [
  {{"subject": "Llama 3 Herd", "predicate": "natively supports", "object": "multilinguality"}},
  {{"subject": "Llama 3 Herd", "predicate": "natively supports", "object": "coding"}},
  {{"subject": "Llama 3 Herd", "predicate": "natively supports", "object": "reasoning"}}
]

Claim: "The development of modern foundation models consists of a pre-training stage and a post-training stage."
→ [
  {{
    "subject": "development of modern foundation models",
    "predicate": "consists of",
    "object": "a pre-training stage and a post-training stage"
  }}
]

TEXT
-----
CLAIM_TEXT:
\"\"\"{text}\"\"\"\n
You MUST output ONLY valid JSON, in this exact form:

[
  {{"subject": "...", "predicate": "...", "object": "..."}},
  ...
]

No explanation, no comments, no extra keys. Only the JSON array.
"""


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def call_ollama(
    server: str,
    model: str,
    prompt: str,
    timeout: int = REQUEST_TIMEOUT,
    temperature: float = TEMPERATURE,
) -> str:
    """Call a local Ollama model via /api/chat and return the text reply."""
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

    return (obj.get("message") or {}).get("content", "").strip()


def safe_json_array(s: str) -> List[Dict[str, Any]]:
    """
    Parse a JSON array or object from model output, robustly.

    - If we see a top-level array, try to parse that.
    - If we see just a single object, wrap it in a list.
    - If we see a list of lists/strings, heuristically map to dicts.
    Always return a list (possibly empty).
    """
    s = s.strip()
    start_bracket = s.find("[")
    end_bracket = s.rfind("]")
    start_brace = s.find("{")
    end_brace = s.rfind("}")

    candidate = None

    # Prefer array if present
    if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
        candidate = s[start_bracket:end_bracket + 1]
    elif start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        # single object
        candidate = "[" + s[start_brace:end_brace + 1] + "]"

    if not candidate:
        return []

    try:
        data = json.loads(candidate)
    except Exception:
        return []

    # Normalize shapes
    if isinstance(data, dict):
        return [data]

    if isinstance(data, list):
        # Case 1: already list of dicts
        if all(isinstance(x, dict) for x in data):
            return data

        out: List[Dict[str, Any]] = []
        for x in data:
            if isinstance(x, dict):
                out.append(x)
            elif isinstance(x, list) and len(x) >= 3:
                # ["subj", "pred", "obj ..."]
                out.append(
                    {
                        "subject": str(x[0]),
                        "predicate": str(x[1]),
                        "object": " ".join(str(t) for t in x[2:]),
                    }
                )
            elif isinstance(x, str):
                # treat as object-only fallback
                out.append(
                    {"subject": "", "predicate": "", "object": x}
                )
        return out

    return []


# --------------------------------------------------------------------
# Core extraction function
# --------------------------------------------------------------------

def extract_triples_from_file(input_path: Union[str, Path]) -> Path:
    """Run triple extraction over all claims in a JSONL file."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # e.g. llama3_sample_claims.jsonl → llama3_sample_triples.jsonl
    base = re.sub(r"_claims$", "", input_path.stem)
    out_path = input_path.with_name(f"{base}_triples.jsonl")

    total = sum(1 for _ in input_path.open("r", encoding="utf-8"))
    print(f"Input:  {input_path}", flush=True)
    print(f"Output: {out_path}", flush=True)
    print(f"Server: {OLLAMA_SERVER} | Model: {OLLAMA_MODEL}", flush=True)
    print(f"Total claims: {total}", flush=True)

    written = 0
    t0 = time.time()

    with input_path.open("r", encoding="utf-8") as inp, out_path.open("w", encoding="utf-8") as out:
        for i, line in enumerate(inp, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                claim = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc = (claim.get("doc") or "").strip()
            chunk_id = (claim.get("chunk_id") or "").strip()
            claim_id = (claim.get("claim_id") or "").strip()
            claim_text = (claim.get("claim_text") or "").strip()
            if not claim_text:
                continue

            # Build prompt (note: placeholder is {text} in PROMPT)
            prompt = PROMPT.format(text=claim_text)

            # Call model
            try:
                reply = call_ollama(
                    server=OLLAMA_SERVER,
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    timeout=REQUEST_TIMEOUT,
                    temperature=TEMPERATURE,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                print(f"[warn] {doc}/{claim_id} failed: {e}", flush=True)
                continue

            triples = safe_json_array(reply)

            idx = 0
            for j, tri in enumerate(triples, start=1):
                # Skip anything that's not a dict
                if not isinstance(tri, dict):
                    continue

                subj = (tri.get("subject") or "").strip()
                pred = (tri.get("predicate") or "").strip()
                objt = (tri.get("object") or "").strip()
                if not subj or not pred or not objt:
                    continue

                idx += 1
                rec = {
                    "doc": doc,
                    "chunk_id": chunk_id,
                    "claim_id": claim_id,
                    "triple_id": f"{claim_id}_t{idx}",
                    "subject": subj,
                    "predicate": pred,
                    "object": objt,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

            if i % PROGRESS_EVERY == 0 or i == total:
                print(f"[{i}/{total}] processed, {written} triples written", flush=True)

    dt = time.time() - t0
    print(f"✓ Done. Wrote {written} triples → {out_path} ({dt:.1f}s)", flush=True)
    return out_path


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        prog = Path(sys.argv[0]).name
        print(f"Usage: {prog} INPUT_CLAIMS_JSONL", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    try:
        extract_triples_from_file(input_path)
    except Exception as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
