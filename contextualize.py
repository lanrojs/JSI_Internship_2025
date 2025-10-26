"""
contextualize.py  —  always start fresh (no resume)

Reads <input>_chunks.json (list of {"doc","id","chunk"}), gathers neighbors
(2 before + 1 after within the same doc), calls a local Llama 3 (Ollama) to
produce a self-contained rewrite, and writes <input>_contextualized.json with:

{
  "doc": "...",
  "id": "...",
  "chunk": "<original>",
  "contextualized_chunk": "<rewritten>",
  "merged_chunk": "<rewritten>\\n\\n<original>"
}

Progress reporting:
- Prints total items at start.
- Prints "[X/Y] contextualized (ok=..., empty=...)" every N items (--progress-every).
- Prints final summary.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request, urllib.error, urllib.parse

PROMPT_TEMPLATE = """You are a helpful assistant that improves text chunks for retrieval systems.

Your task is to rewrite a given chunk so that it is fully self-contained and comprehensible even if read out of its original context.

The rewritten chunk should:
- Preserve all factual information from the original text.
- Add minimal necessary context from neighboring text (if provided) to make references clear (e.g., replace “this law” with “the General Data Protection Regulation”).
- Not add any new facts or opinions.
- Keep the style and tone neutral and informative.

### Neighboring context
{neighbors}

### Original chunk
{chunk}

### Output
Return ONLY the statute text. Preserve every original label and numbering ((1), (2), (a), (b)…), verbatim and in order.
Do not add headings, prefaces, bullets, quotes, or commentary.
Do not introduce any facts, article numbers, or terms not present in the Original chunk or Neighboring context.
If a reference (e.g., “this Regulation”) is ambiguous, replace it with the clearest explicit term present from the provided text.
If something is unknown, leave it as in the source; do not guess.
"""

def build_neighbors(items: List[Dict], idx: int) -> str:
    parts = []
    if idx - 2 >= 0:
        parts.append(items[idx - 2].get("chunk", ""))
    if idx - 1 >= 0:
        parts.append(items[idx - 1].get("chunk", ""))
    if idx + 1 < len(items):
        parts.append(items[idx + 1].get("chunk", ""))
    return "\n\n".join([p for p in parts if p]).strip()

def build_prompt(chunk_text: str, neighbor_text: str) -> str:
    return PROMPT_TEMPLATE.format(
        neighbors=neighbor_text if neighbor_text else "(none provided)",
        chunk=chunk_text
    )

def ollama_chat(server: str, model: str, prompt: str, timeout: int = 120,
                temperature: Optional[float] = 0.0, num_ctx: Optional[int] = None) -> str:
    url = server.rstrip("/") + "/api/chat"
    payload = {
    "model": model,
    "messages": [
        {"role": "system", "content":
         "Rewrite statute chunks to be self-contained for retrieval. "
         "Output ONLY the rewritten statute text. Do not add prefaces, headings, bullets, quotes, or commentary. "
         "Preserve all facts and labels ((1),(2),(a),(b)…). Do not invent facts."},
        {"role": "user", "content": prompt},
    ],
    "stream": False,
    }

    opts = {}
    if temperature is not None:
        opts["temperature"] = float(temperature)
    if num_ctx is not None:
        opts["num_ctx"] = int(num_ctx)
    if opts:
        payload["options"] = opts

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8"))

    msg = obj.get("message") or {}
    content = msg.get("content", "")
    if not content:
        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            content = choices[0].get("message", {}).get("content", "")
    return (content or "").strip()

def main():
    ap = argparse.ArgumentParser(description="Contextualize chunks and write <input>_contextualized.json (fresh run).")
    ap.add_argument("--input", required=True, type=Path, help="Path to *_chunks.json")
    ap.add_argument("--server", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--model", default="llama3:8b", help="Ollama model tag")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds")
    ap.add_argument("--max-retries", type=int, default=3, help="Retries on transient errors")
    ap.add_argument("--retry-wait", type=float, default=2.0, help="Initial backoff seconds")
    ap.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    ap.add_argument("--num-ctx", type=int, default=None, help="Context window hint for Ollama")
    ap.add_argument("--progress-every", type=int, default=1, help="Print a progress line every N items")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"[!] Input not found: {args.input}", file=sys.stderr, flush=True)
        sys.exit(1)

    out_path = args.input.with_name(f"{args.input.stem}_contextualized.json")

    # Load input JSON
    try:
        items = json.loads(args.input.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[!] Failed to parse JSON: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    if not isinstance(items, list):
        print("[!] Input JSON must be a list of objects.", file=sys.stderr, flush=True)
        sys.exit(1)

    # Group by doc for neighbor windows
    by_doc: Dict[str, List[Dict]] = defaultdict(list)
    for it in items:
        d = it.get("doc")
        if d is None:
            d = "_MISC_"
        by_doc[str(d)].append(it)

    # Quick index map per doc
    index_by_doc: Dict[str, Dict[str, int]] = defaultdict(dict)
    for doc, arr in by_doc.items():
        for idx, obj in enumerate(arr):
            index_by_doc[doc][obj.get("id")] = idx

    total = len(items)
    processed = 0
    success = 0
    empty = 0
    output_rows: List[Dict] = []

    print(f"Input: {args.input}", flush=True)
    print(f"Output: {out_path.name}", flush=True)
    print(f"Server: {args.server} | Model: {args.model}", flush=True)
    print(f"Total items: {total}", flush=True)
    t0 = time.time()

    for obj in items:
        doc = str(obj.get("doc", "_MISC_"))
        cid = obj.get("id")
        chunk = (obj.get("chunk") or "").strip()

        row = {
            "doc": obj.get("doc"),
            "id": cid,
            "chunk": chunk,
            "contextualized_chunk": None,
            "merged_chunk": None,
        }

        # Empty chunk → empty outputs
        if not chunk:
            row["contextualized_chunk"] = ""
            row["merged_chunk"] = ""
            output_rows.append(row)
            processed += 1
            empty += 1
            if processed % args.progress_every == 0 or processed == total:
                print(f"[{processed}/{total}] contextualized (ok={success}, empty={empty})", flush=True)
            continue

        # Neighbors within doc
        arr = by_doc[doc]
        idx = index_by_doc[doc].get(cid, None)
        neighbor_text = build_neighbors(arr, idx) if idx is not None else ""

        # Prompt + call with retries
        prompt = build_prompt(chunk, neighbor_text)
        attempt = 0
        rewritten = ""
        while attempt < args.max_retries:
            try:
                rewritten = ollama_chat(
                    server=args.server,
                    model=args.model,
                    prompt=prompt,
                    timeout=args.timeout,
                    temperature=args.temperature,
                    num_ctx=args.num_ctx
                )
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                attempt += 1
                wait = args.retry_wait * (2 ** (attempt - 1))
                print(f"[warn] {doc}/{cid} attempt {attempt}/{args.max_retries} error: {e}. Backing off {wait:.1f}s...", flush=True)
                time.sleep(wait)

        row["contextualized_chunk"] = rewritten or ""
        row["merged_chunk"] = ((rewritten or "").strip() + "\n\n" + chunk) if rewritten else chunk

        output_rows.append(row)
        processed += 1
        if rewritten:
            success += 1
        else:
            empty += 1

        if processed % args.progress_every == 0 or processed == total:
            print(f"[{processed}/{total}] contextualized (ok={success}, empty={empty})", flush=True)

    # Write output (overwrite)
    out_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    dt = time.time() - t0
    print(f"✓ Done. Wrote: {out_path}", flush=True)
    print(f"Summary: {success} ok, {empty} empty, total {total}, time {dt:.1f}s", flush=True)

if __name__ == "__main__":
    main()
