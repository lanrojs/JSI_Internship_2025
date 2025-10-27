#!/usr/bin/env python3
"""
extract_claims.py — extract claims from contextualized chunks (GraphRAG-style)

Reads an input JSON file that contains objects like:
{
  "doc": "...",
  "id": "...",
  "chunk": "...",
  "contextualized_chunk": "...",
  "merged_chunk": "..."
}

Calls a local Ollama model (default: llama3:8b) to extract claims from
`contextualized_chunk` and writes one JSON line per claim to --output.

Progress:
- Prints total chunks
- Prints "[X/Y] claims (written=..., empty_chunks=...)" per-chunk
"""

import argparse
import json
import sys
import time
from pathlib import Path
import urllib.request, urllib.error

PROMPT = """You are extracting *atomic, factual claims* from the provided text. 
Return a JSON array where each element has keys:
- claim_text: a single, self-contained factual statement (no cross-refs like "this", "above", etc.)
- source_quote: the most specific quote supporting the claim (substring)

Rules:
- Do NOT invent facts; use only what's in the text.
- Prefer one idea per claim_text.
- Paraphrase minimally; keep terms from the source where possible.
- Skip non-factual or purely definitional boilerplate unless it asserts a condition, scope, right, duty, prohibition, or permission.

Text:
\"\"\"{text}\"\"\"\n
Output ONLY the JSON array (no prose)."""

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

def main():
    ap = argparse.ArgumentParser(description="Extract claims from contextualized chunks.")
    ap.add_argument("--input", required=True, type=Path, help="Path to *_contextualized.json")
    ap.add_argument("--output", type=Path, default=None, help="Output JSONL (default: <input>_claims.jsonl)")
    ap.add_argument("--server", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--model", default="llama3:8b", help="Ollama model tag")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds")
    ap.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    ap.add_argument("--progress-every", type=int, default=1, help="Print progress every N chunks")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"[!] Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or args.input.with_name(f"{args.input.stem}_claims.jsonl")

    try:
        items = json.loads(args.input.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[!] Failed to parse JSON: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(items, list):
        print("[!] Input JSON must be a list of objects.", file=sys.stderr)
        sys.exit(1)

    total = len(items)
    print(f"Total chunks: {total}", flush=True)

    written = 0
    empty_chunks = 0
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as out:
        for i, obj in enumerate(items, start=1):
            doc = str(obj.get("doc", ""))
            cid = str(obj.get("id") or obj.get("chunk_id") or "")
            text = (obj.get("contextualized_chunk") or "").strip()
            if not text:
                empty_chunks += 1
                if i % args.progress_every == 0 or i == total:
                    print(f"[{i}/{total}] claims (written={written}, empty_chunks={empty_chunks})", flush=True)
                continue

            prompt = PROMPT.format(text=text)
            try:
                reply = call_ollama(
                    server=args.server,
                    model=args.model,
                    prompt=prompt,
                    timeout=args.timeout,
                    temperature=args.temperature,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                print(f"[warn] {doc}/{cid} request failed: {e}", flush=True)
                if i % args.progress_every == 0 or i == total:
                    print(f"[{i}/{total}] claims (written={written}, empty_chunks={empty_chunks})", flush=True)
                continue

            arr = safe_json_array(reply)

            # Filter and write (no confidence)
            idx = 0
            for claim in arr:
                idx += 1
                ctext = (claim.get("claim_text") or "").strip()
                if not ctext:
                    continue
                rec = {
                    "doc": doc,
                    "chunk_id": cid,
                    "claim_id": f"{cid}#{idx}",
                    "claim_text": ctext,
                    "source_quote": (claim.get("source_quote") or "").strip(),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

            if i % args.progress_every == 0 or i == total:
                print(f"[{i}/{total}] claims (written={written}, empty_chunks={empty_chunks})", flush=True)

    dt = time.time() - t0
    print(f"✓ Done. Wrote {written} claims to: {out_path} in {dt:.1f}s", flush=True)

if __name__ == "__main__":
    main()
