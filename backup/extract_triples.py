"""
extract_triplets.py — Extract (subject, predicate, object) triples from claims.

Input:
  JSONL produced by extract_claims.py, lines like:
    {"doc": "...", "chunk_id": "...", "claim_id": "...", "claim_text": "...", "source_quote": "..."}

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

Usage:
  python triples_from_claims.py --input gdpr_sample_claims.jsonl
  python triples_from_claims.py --input gdpr_sample_claims.jsonl --model llama3:8b
"""

import argparse
import json
import sys
import time
import re
from pathlib import Path
import urllib.request, urllib.error

PROMPT = """Extract subject–predicate–object triples from the single factual claim below.

Return ONLY a JSON array. Each element MUST be an object with keys:
- "subject": the entity or concept doing/being something
- "predicate": the relation or action, concise verb phrase
- "object": the target entity or concept

Rules:
- Use minimal paraphrasing; prefer terms from the claim.
- Keep each triple atomic and unambiguous.
- If there is no valid triple, return [].

Claim:
\"\"\"{text}\"\"\""""

def call_ollama(server: str, model: str, prompt: str, timeout: int = 120, temperature: float = 0.0):
    url = server.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": float(temperature)}
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    content = (obj.get("message") or {}).get("content", "") or ""
    return content.strip()

def safe_json_array(s: str):
    """Best-effort parse of a JSON array from a model reply."""
    s = s.strip()
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return arr
    except Exception:
        pass
    return []

def sanitize_field(val: str) -> str:
    """Flatten newlines and trim."""
    return (val or "").replace("\r", " ").replace("\n", " ").strip()

def main():
    ap = argparse.ArgumentParser(description="Extract SPO triples from claims JSONL.")
    ap.add_argument("--input", required=True, type=Path, help="Path to *_claims.jsonl")
    ap.add_argument("--output", type=Path, default=None, help="Output JSONL (default: <input>_triples.jsonl)")
    ap.add_argument("--server", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--model", default="llama3:8b", help="Ollama model tag")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds")
    ap.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    ap.add_argument("--max-retries", type=int, default=3, help="Retries on transient errors")
    ap.add_argument("--retry-wait", type=float, default=2.0, help="Initial backoff seconds")
    ap.add_argument("--progress-every", type=int, default=1, help="Print progress every N claims")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"[!] Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Derive output path (strip trailing "_claims" from stem if present)
    base = re.sub(r'_claims$', '', args.input.stem)
    out_path = args.output or args.input.with_name(f"{base}_triples.jsonl")

    total = 0
    with args.input.open("r", encoding="utf-8") as f:
        for _ in f:
            total += 1

    print(f"Input:  {args.input}", flush=True)
    print(f"Output: {out_path}", flush=True)
    print(f"Server: {args.server} | Model: {args.model}", flush=True)
    print(f"Total claims: {total}", flush=True)

    written = 0
    empty = 0
    t0 = time.time()

    # Process
    with args.input.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                empty += 1
                if i % args.progress_every == 0 or i == total:
                    print(f"[{i}/{total}] triples (written={written}, empty={empty})", flush=True)
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                empty += 1
                if i % args.progress_every == 0 or i == total:
                    print(f"[{i}/{total}] triples (written={written}, empty={empty})", flush=True)
                continue

            doc = str(obj.get("doc", ""))
            chunk_id = str(obj.get("chunk_id", ""))
            claim_id = str(obj.get("claim_id", ""))
            claim_text = sanitize_field(obj.get("claim_text", ""))

            if not claim_text:
                empty += 1
                if i % args.progress_every == 0 or i == total:
                    print(f"[{i}/{total}] triples (written={written}, empty={empty})", flush=True)
                continue

            prompt = PROMPT.format(text=claim_text)

            # Call with retries
            attempt = 0
            reply = ""
            while attempt < args.max_retries:
                try:
                    reply = call_ollama(
                        server=args.server,
                        model=args.model,
                        prompt=prompt,
                        timeout=args.timeout,
                        temperature=args.temperature,
                    )
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                    attempt += 1
                    wait = args.retry_wait * (2 ** (attempt - 1))
                    print(f"[warn] {doc}/{claim_id} attempt {attempt}/{args.max_retries} error: {e}. Backing off {wait:.1f}s...", flush=True)
                    time.sleep(wait)

            triples = safe_json_array(reply)

            # Write sanitized triples
            count_for_claim = 0
            for idx, t in enumerate(triples, start=1):
                subj = sanitize_field((t.get("subject") if isinstance(t, dict) else "") if t else "")
                pred = sanitize_field((t.get("predicate") if isinstance(t, dict) else "") if t else "")
                obj_ = sanitize_field((t.get("object") if isinstance(t, dict) else "") if t else "")
                if not (subj and pred and obj_):
                    continue
                rec = {
                    "doc": doc,
                    "chunk_id": chunk_id,
                    "claim_id": claim_id,
                    "triple_id": f"{claim_id}@{idx}",
                    "subject": subj,
                    "predicate": pred,
                    "object": obj_,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                count_for_claim += 1

            if i % args.progress_every == 0 or i == total:
                print(f"[{i}/{total}] triples (written={written}, empty={empty})", flush=True)

    dt = time.time() - t0
    print(f"✓ Done. Wrote {written} triples to: {out_path} in {dt:.1f}s", flush=True)

if __name__ == "__main__":
    main()
