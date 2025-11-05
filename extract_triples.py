"""
extract_triplets.py — Extract (subject, predicate, object) triples from claims.

Input:
  JSONL produced by extract_claims.py, lines like:
    {
      "doc": "...",
      "chunk_id": "...",
      "claim_id": "...",
      "claim_text": "...",
      "source_quote": "..."
    }

Output:
  <input_basename>_triples.jsonl where each line is:
    {
      "doc": "...",
      "chunk_id": "...",
      "claim_id": "...",
      "triple_id": "<claim_id>@<n>",
      "subject": "...",
      "predicate": "...",
      "object": "..."
    }

Usage from command line:
    python extract_triplets.py INPUT_CLAIMS_JSONL

Usage from notebook:
    from extract_triplets import extract_triplets_from_claims
    out_path = extract_triplets_from_claims("gdpr_sample_claims.jsonl")
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
# Configuration constants — tweak here instead of using CLI flags
# --------------------------------------------------------------------

# Base URL for your local Ollama server
OLLAMA_SERVER = "http://localhost:11434"

# Default Ollama model tag to use for triple extraction
OLLAMA_MODEL = "llama3:8b"     # e.g. "llama3:8b" or "llama3.2"

# HTTP timeout for each request (seconds)
REQUEST_TIMEOUT = 120

# LLM sampling temperature (0.0 = deterministic, good for extraction tasks)
TEMPERATURE = 0.0

# Maximum number of retries on transient HTTP/network errors
MAX_RETRIES = 3

# Initial backoff (in seconds) before retrying; grows exponentially
RETRY_WAIT = 2.0

# Print progress every N claims
PROGRESS_EVERY = 1


# --------------------------------------------------------------------
# Prompt template sent to the model
# --------------------------------------------------------------------

PROMPT = """From the factual claim below, extract subject–predicate–object triples.

Return ONLY a JSON array. Each element MUST be an object with keys:
- "subject": the entity or concept doing/being something
- "predicate": the relation or action, as a concise verb phrase
- "object": the target entity or concept

GROUNDING
- Use ONLY the information in the claim text.
- Do NOT add entities, conditions, or relations that are not explicitly
  present in the claim.
- If you are unsure whether a triple is fully supported by the claim, do not
  include it.

STRUCTURE RULES
- SUBJECT and OBJECT:
  • Must be concrete noun phrases taken from the claim (copied exactly or with
    minimal grammatical adjustment).
  • Avoid vague placeholders like "situations", "reasons", "something".
- PREDICATE:
  • Should be a short verb phrase (e.g. "applies to", "does not apply to",
    "protects", "processes").
  • Do NOT use only prepositions or weak verbs as predicates (e.g. "in",
    "of", "with", "be connected with").
  • Do NOT output long clause fragments as the predicate.

CARDINALITY
- If several triples are possible, extract the 1–2 most central triples that
  best capture the main relation(s) in the claim.
- If there is no valid, well-formed triple, return [].

Claim:
\"\"\"{text}\"\"\""""

# --------------------------------------------------------------------
# Low-level HTTP/model-call helpers
# --------------------------------------------------------------------

def call_ollama(
    server: str,
    model: str,
    prompt: str,
    timeout: int = REQUEST_TIMEOUT,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Call a local Ollama model via the /api/chat endpoint.

    Parameters
    ----------
    server : str
        Base URL of the Ollama server (e.g. "http://localhost:11434").
    model : str
        Model tag, e.g. "llama3.2" or "llama3:8b".
    prompt : str
        The text prompt to send as a single user message.
    timeout : int
        HTTP timeout in seconds.
    temperature : float
        Sampling temperature.

    Returns
    -------
    str
        The content of the model's reply (message.content), stripped.
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


def safe_json_array(s: str) -> List[Any]:
    """
    Best-effort parsing of a JSON array from the model reply string.

    The model *should* return a plain JSON array, but in practice it may
    add explanatory text before/after. This helper:

      1. Finds the substring between the first '[' and last ']'.
      2. Attempts to json.loads() that substring.
      3. Returns the array if successful, otherwise [].

    Parameters
    ----------
    s : str
        Raw string returned by the model.

    Returns
    -------
    list
        Parsed JSON array (possibly empty).
    """
    s = s.strip()
    start = s.find("[")
    end = s.rfind("]")

    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]

    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass

    return []


def sanitize_field(val: Any) -> str:
    """
    Sanitize a text field by:

      • Converting None to empty string.
      • Replacing newlines and carriage returns with spaces.
      • Trimming leading/trailing whitespace.

    This keeps JSONL lines single-line and compact.
    """
    return (str(val or "")).replace("\r", " ").replace("\n", " ").strip()


# --------------------------------------------------------------------
# Core high-level function (for notebooks and scripts)
# --------------------------------------------------------------------

def extract_triplets_from_claims(input_path: Union[str, Path]) -> Path:
    """
    High-level pipeline to extract SPO triples from claims.

    Parameters
    ----------
    input_path : str or Path
        Path to a JSONL file produced by extract_claims.py.
        Each line should be a JSON object with at least:
            "doc", "chunk_id", "claim_id", "claim_text"

    Behavior
    --------
    For each claim:
      1. Build a prompt using PROMPT and the claim_text.
      2. Call the Ollama model and parse its JSON array reply.
      3. For each triple in the array with non-empty
         subject, predicate, and object:
            • Write one JSONL line with fields:
                doc, chunk_id, claim_id, triple_id,
                subject, predicate, object
            • triple_id = "<claim_id>@<n>"

    Output
    ------
    A new JSONL file named:
        <input_basename>_triples.jsonl
    where <input_basename> is the stem of the input file, with
    any trailing "_claims" removed (e.g. "gdpr_claims.jsonl"
    → "gdpr_triples.jsonl").

    Returns
    -------
    Path
        Path to the output JSONL file.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Derive output name by stripping "_claims" from the stem if present.
    base = re.sub(r"_claims$", "", input_path.stem)
    out_path = input_path.with_name(f"{base}_triples.jsonl")

    # Count total lines (claims) up front for progress reporting.
    total = 0
    with input_path.open("r", encoding="utf-8") as f:
        for _ in f:
            total += 1

    print(f"Input:  {input_path}", flush=True)
    print(f"Output: {out_path}", flush=True)
    print(f"Server: {OLLAMA_SERVER} | Model: {OLLAMA_MODEL}", flush=True)
    print(f"Total claims: {total}", flush=True)

    written = 0  # number of triples written
    empty = 0    # number of empty/bad lines or claims with no text
    t0 = time.time()

    # Process the input line by line (streaming JSONL)
    with input_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for i, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                # Empty line; nothing to parse.
                empty += 1
                if i % PROGRESS_EVERY == 0 or i == total:
                    print(
                        f"[{i}/{total}] triples (written={written})",
                        flush=True,
                    )
                continue

            # Parse single JSON object from this line.
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                empty += 1
                if i % PROGRESS_EVERY == 0 or i == total:
                    print(
                        f"[{i}/{total}] triples (written={written})",
                        flush=True,
                    )
                continue

            doc = sanitize_field(obj.get("doc", ""))
            chunk_id = sanitize_field(obj.get("chunk_id", ""))
            claim_id = sanitize_field(obj.get("claim_id", ""))
            claim_text = sanitize_field(obj.get("claim_text", ""))

            if not claim_text:
                # No text → nothing to extract from.
                empty += 1
                if i % PROGRESS_EVERY == 0 or i == total:
                    print(
                        f"[{i}/{total}] triples (written={written})",
                        flush=True,
                    )
                continue

            # Build the prompt for this single claim.
            prompt = PROMPT.format(text=claim_text)

            # --- Call Ollama with retry/backoff logic ---
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
                        f"[warn] {doc}/{claim_id} attempt {attempt}/{MAX_RETRIES} error: {e}. "
                        f"Backing off {wait:.1f}s...",
                        flush=True,
                    )
                    time.sleep(wait)

            # Parse the model reply as a JSON array of triples.
            triples = safe_json_array(reply)

            count_for_claim = 0  # how many triples for this claim
            for idx, t in enumerate(triples, start=1):
                # t is expected to be a dict with subject/predicate/object
                if not isinstance(t, dict):
                    continue

                subj = sanitize_field(t.get("subject", ""))
                pred = sanitize_field(t.get("predicate", ""))
                obj_ = sanitize_field(t.get("object", ""))

                # Skip incomplete triples
                if not (subj and pred and obj_):
                    continue

                rec = {
                    "doc": doc,
                    "chunk_id": chunk_id,
                    "claim_id": claim_id,
                    "triple_id": f"{claim_id}@{idx}",  # unique per claim
                    "subject": subj,
                    "predicate": pred,
                    "object": obj_,
                }

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                count_for_claim += 1

            if i % PROGRESS_EVERY == 0 or i == total:
                print(
                    f"[{i}/{total}] triples (written={written})",
                    flush=True,
                )

    dt = time.time() - t0
    print(
        f"✓ Done. Wrote {written} triples to: {out_path} in {dt:.1f}s",
        flush=True,
    )
    return out_path


# --------------------------------------------------------------------
# Simple CLI wrapper (single positional argument: INPUT_CLAIMS_JSONL)
# --------------------------------------------------------------------

def main() -> None:
    """
    Command-line entry point.

    Usage:
        python extract_triplets.py INPUT_CLAIMS_JSONL

    Where INPUT_CLAIMS_JSONL is a JSONL file produced by extract_claims.py.

    All configuration (server URL, model name, timeouts, etc.) is defined
    by the constants at the top of this file.
    """
    if len(sys.argv) != 2:
        prog = Path(sys.argv[0]).name
        print(f"Usage: {prog} INPUT_CLAIMS_JSONL", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    try:
        extract_triplets_from_claims(input_path)
    except Exception as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
