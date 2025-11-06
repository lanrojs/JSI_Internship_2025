"""
extract_claims.py — extract claims from chunks + contextualized chunks (GraphRAG-style)

Expected input JSON structure (list of objects):
[
  {
    "doc": "...",
    "id": "...",
    "chunk": "...",                 # original text from the source
    "contextualized_chunk": "...",  # a short contextual / paraphrased summary
    "merged_chunk": "..."           # (optional, ignored here)
  },
  ...
]

Behavior:
  • For each item, send BOTH `chunk` and `contextualized_chunk` to a local Ollama model.
  • The model is instructed to base claims primarily on the original chunk, using the
    contextualized text only as an aid for interpretation.
  • Ask the model to return a JSON array of claims, each with:
        - claim_text
        - source_quote
  • Write ONE JSON line per claim to an output .jsonl file.

Output:
  - <input_basename>_claims.jsonl
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

# Base URL of your Ollama server
OLLAMA_SERVER = "http://localhost:11434"

# Name/tag of the LLM model to use
OLLAMA_MODEL = "gemma3:4b-it-qat" # gemma3:4b-it-qat

# HTTP timeout for each request (seconds)
REQUEST_TIMEOUT = 120

# Generation temperature
TEMPERATURE = 0.5

# Print progress every N chunks
PROGRESS_EVERY = 1


# --------------------------------------------------------------------
# Prompt template sent to the LLM
# --------------------------------------------------------------------

PROMPT = """You are extracting *atomic, factual claims* from a legal passage.

You are given TWO related texts:
1) ORIGINAL_CHUNK: the original source passage (authoritative text)
2) CONTEXTUALIZED_CHUNK: a brief contextual/interpretive summary (helper only)

YOUR TASK
- Identify explicit, normative statements in the ORIGINAL_CHUNK and rewrite them
  as *atomic, self-contained factual claims*.

OUTPUT FORMAT
- Return a JSON array. Each element MUST be an object with keys:
    - "claim_text": a single, self-contained factual statement
      (no vague references like "this", "above", "such provision", etc.)
    - "source_quote": the most specific quote from the ORIGINAL_CHUNK
      that supports the claim (a substring of ORIGINAL_CHUNK)

STRICT RULES
- Claims MUST be grounded in the ORIGINAL_CHUNK.
  • Use CONTEXTUALIZED_CHUNK only to clarify wording, not to invent or
    generalize new facts.
- Do NOT:
  • invert meanings (e.g. do not turn an inclusion or example into a
    general exclusion or vice versa),
  • broaden or narrow the scope beyond what is literally stated,
  • add conditions, exceptions, or subjects that are not explicitly present.
- If you are not certain that a statement is *fully* supported by the
  ORIGINAL_CHUNK, SKIP IT (do not output a claim).

CONTENT FOCUS
- Prefer claims that express:
  • conditions of applicability (when the Regulation applies / does not apply),
  • scope (who/what is covered),
  • rights, duties, obligations, prohibitions, permissions, limitations,
  • effects or consequences.
- Skip:
  • purely definitional boilerplate that does not express normative content,
  • high-level recitals or motivation that add no concrete condition or rule.

STYLE
- Prefer one idea per claim_text.
- Paraphrase minimally; preserve legal terms from the ORIGINAL_CHUNK
  wherever possible.
- "source_quote" MUST be a literal substring of ORIGINAL_CHUNK
  (do not paraphrase in source_quote).

Text:
ORIGINAL_CHUNK:
\"\"\"{original_chunk}\"\"\"

CONTEXTUALIZED_CHUNK:
\"\"\"{contextualized_chunk}\"\"\"\n

Output ONLY the JSON array (no prose)."""

# --------------------------------------------------------------------
# Low-level HTTP / model-calling helpers
# --------------------------------------------------------------------

def call_ollama(
    server: str,
    model: str,
    prompt: str,
    timeout: int = REQUEST_TIMEOUT,
    temperature: float = TEMPERATURE,
) -> str:
    
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

    # Ollama returns a structure where the main content is in message.content
    content = (obj.get("message") or {}).get("content", "") or ""
    return content.strip()


def safe_json_array(s: str) -> List[Dict[str, Any]]:
    """
    Best-effort parsing of a JSON array from the model's reply.

    The model is *supposed* to return a plain JSON array, but in practice
    it might add extra text. This function:

      1. Finds the first '[' and last ']' in the string.
      2. Extracts that substring.
      3. Tries to json.loads it.
      4. Returns a list if successful, otherwise [].

    Returns
    -------
    list
        A list of parsed objects (possibly empty if parsing failed).
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


# calling from a notebook
def extract_claims_from_file(input_path: Union[str, Path]) -> Path:
    input_path = Path(input_path)

    # --- Basic existence check ---
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # --- Derive output path: strip trailing '_contextualized' from the stem ---
    base = re.sub(r"_contextualized$", "", input_path.stem)
    out_path = input_path.with_name(f"{base}_claims.jsonl")

    # --- Load input JSON (must be a list) ---
    try:
        items = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}") from e

    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list of objects.")

    total = len(items)
    print(f"Total chunks: {total}", flush=True)

    written = 0       # how many claims we've written so far
    empty_chunks = 0  # how many chunks had no usable text
    t0 = time.time()  # start time

    # --- Open output JSONL for writing ---
    with out_path.open("w", encoding="utf-8") as out:
        for i, obj in enumerate(items, start=1):
            # Document ID (e.g. "gdpr2")
            doc = str(obj.get("doc", ""))

            # Chunk identifier (prefer "id", fallback to "chunk_id")
            cid = str(obj.get("id") or obj.get("chunk_id") or "")

            # Original chunk and contextualized chunk from the input
            original_chunk = (obj.get("chunk") or "").strip()
            contextualized_chunk = (obj.get("contextualized_chunk") or "").strip()

            # If we have absolutely no text, count as empty and continue
            if not original_chunk and not contextualized_chunk:
                empty_chunks += 1
                if i % PROGRESS_EVERY == 0 or i == total:
                    print(
                        f"[{i}/{total}] claims (written={written})",
                        flush=True,
                    )
                continue

            # Fill the prompt template with BOTH texts.
            # The prompt tells the model to ground claims in ORIGINAL_CHUNK.
            prompt = PROMPT.format(
                original_chunk=original_chunk,
                contextualized_chunk=contextualized_chunk,
            )

            # --- Call Ollama model ---
            try:
                reply = call_ollama(
                    server=OLLAMA_SERVER,
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    timeout=REQUEST_TIMEOUT,
                    temperature=TEMPERATURE,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                # Log a warning and skip this chunk
                print(f"[warn] {doc}/{cid} request failed: {e}", flush=True)
                if i % PROGRESS_EVERY == 0 or i == total:
                    print(
                        f"[{i}/{total}] claims (written={written})",
                        flush=True,
                    )
                continue

            # --- Parse the model reply into a JSON array ---
            arr = safe_json_array(reply)

            # --- Write out each valid claim as a JSONL record ---
            idx = 0  # per-chunk claim counter
            for claim in arr:
                idx += 1
                ctext = (claim.get("claim_text") or "").strip()
                if not ctext:
                    # Skip empty or malformed entries
                    continue

                rec = {
                    "doc": doc,
                    "chunk_id": cid,
                    "claim_id": f"{cid}#{idx}",  # unique within this chunk
                    "claim_text": ctext,
                    "source_quote": (claim.get("source_quote") or "").strip(),
                }

                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

            # Periodic progress logging
            if i % PROGRESS_EVERY == 0 or i == total:
                print(
                    f"[{i}/{total}] claims (written={written})",
                    flush=True,
                )

    dt = time.time() - t0
    print(f"✓ Done. Wrote {written} claims to: {out_path} in {dt:.1f}s", flush=True)
    return out_path


# --------------------------------------------------------------------
# Simple CLI wrapper with a single positional argument
# --------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        prog = Path(sys.argv[0]).name
        print(f"Usage: {prog} INPUT_JSON", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    try:
        extract_claims_from_file(input_path)
    except Exception as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
