#!/usr/bin/env python3
"""
bge_chunker.py — Clean (via process.py) + chunk .txt files using BGE-small token counts.

Boundary rules:
  - Start: '(' or a LETTER (A–Z/a–z).  ❌ Not a digit.
  - End:   alphanumeric OR ')'
  - Trims punctuation on first/last token but preserves '(' and ')'.
  - Skips redundant chunks (no new words beyond previous chunk).

Output:
  Default name: <input>_chunks.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict

from transformers import AutoTokenizer
from process_text import clean_text  # import from your process.py

# ---------- chunking helpers ----------
ALNUM = re.compile(r"[A-Za-z0-9]", re.UNICODE)
LETTER = re.compile(r"[A-Za-z]", re.UNICODE)
LEAD_NONALNUM_KEEP_LPAREN = re.compile(r"^[^\w(]+", re.UNICODE)   # allow '(' at start
TRAIL_NONALNUM_KEEP_RPAREN = re.compile(r"[^\w)]+$", re.UNICODE)  # allow ')' at end


def is_wordy(tok: str) -> bool:
    return bool(ALNUM.search(tok))


def split_to_words(text: str) -> List[str]:
    return text.split()


def token_len(tokenizer, text_piece: str) -> int:
    return len(tokenizer.encode(text_piece, add_special_tokens=False))


def snap_window_to_wordy(tokens: List[str], i: int, j: int) -> Tuple[int, int]:
    n = len(tokens)
    if i >= j or i >= n:
        return j, j
    while i < j and not is_wordy(tokens[i]):
        i += 1
    if i >= j:
        return j, j
    k = j - 1
    while k >= i and not is_wordy(tokens[k]):
        k -= 1
    if k < i:
        return j, j
    return i, k + 1


def trim_boundary_token_first(tok: str) -> str:
    # Trim leading non-alnum but preserve '('
    return LEAD_NONALNUM_KEEP_LPAREN.sub("", tok)


def trim_boundary_token_last(tok: str) -> str:
    # Trim trailing non-alnum but preserve ')'
    return TRAIL_NONALNUM_KEEP_RPAREN.sub("", tok)


def starts_ok(token_text: str) -> bool:
    """Start must be '(' or a LETTER (A–Z/a–z), not a digit."""
    if not token_text:
        return False
    c = token_text[0]
    return c == "(" or c.isalpha()


def has_boundary_rules(text: str) -> bool:
    """Final guard: start '(' or letter; end alnum or ')'."""
    if not text:
        return False
    start_ok = text[0] == "(" or text[0].isalpha()
    end_ok = text[-1].isalnum() or text[-1] == ")"
    return start_ok and end_ok


def make_chunks_by_bge_tokens(
    text: str,
    tokenizer,
    size: int = 300,
    overlap: int = 50,
) -> List[str]:
    words = split_to_words(text)
    n = len(words)
    if n == 0:
        return []

    per_word_toklen = [token_len(tokenizer, w) for w in words]

    chunks: List[str] = []
    start = 0
    last_max_j = -1  # farthest end index among emitted chunks

    while start < n:
        while start < n and not is_wordy(words[start]):
            start += 1
        if start >= n:
            break

        # grow window up to 'size' model tokens
        total = 0
        end = start
        while end < n:
            tlen = per_word_toklen[end]
            if total + tlen > size:
                break
            total += tlen
            end += 1

        if end == start:
            end = start + 1  # include the very long word alone

        i, j = snap_window_to_wordy(words, start, end)
        if i >= j:
            start = end
            continue

        # Trim only boundaries
        first = trim_boundary_token_first(words[i])
        last = trim_boundary_token_last(words[j - 1])

        # Contract inward if trimming empties boundaries
        while i < j and not first:
            i += 1
            if i < j:
                first = trim_boundary_token_first(words[i])
        while i < j and not last:
            j -= 1
            if i < j:
                last = trim_boundary_token_last(words[j - 1])

        if i >= j:
            start = end
            continue

        # Enforce no-numeric-start rule
        while i < j and not starts_ok(first):
            i += 1
            if i < j:
                first = trim_boundary_token_first(words[i])
        if i >= j:
            start = end
            continue

        # Re-evaluate last (safe)
        last = trim_boundary_token_last(words[j - 1])
        window_tokens = words[i:j].copy()
        window_tokens[0] = first
        window_tokens[-1] = last
        chunk_text = " ".join(window_tokens).strip()

        # Final boundary guard
        if not has_boundary_rules(chunk_text):
            start = j
            continue

        # Skip redundant chunks
        if j <= last_max_j:
            start = max(j, start + 1)
            continue

        chunks.append(chunk_text)
        last_max_j = j

        # Compute next start for overlap
        if overlap <= 0:
            start = j
            continue

        acc = 0
        k = j
        while k > i and acc < overlap:
            k -= 1
            acc += per_word_toklen[k]

        new_start = k if acc >= overlap else i
        if new_start <= start:
            new_start = max(j, start + 1)
        start = new_start

    return chunks


def gather_input_paths(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".txt":
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
    raise ValueError(f"Input must be a .txt file or a directory: {input_path}")


def main():
    ap = argparse.ArgumentParser(description="Clean (via process.py) + chunk .txt files using BGE-small tokenization.")
    ap.add_argument("input", type=Path, help="Path to a .txt file or a directory of .txt files")
    ap.add_argument("-o", "--output", type=Path, help="Output JSON file (default: <input>_chunks.json)")
    ap.add_argument("--size", type=int, default=300, help="Max tokens per chunk (model tokens)")
    ap.add_argument("--overlap", type=int, default=50, help="Overlap in model tokens between consecutive chunks")
    ap.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5", help="HF model id for tokenizer")

    # Cleaning controls (delegated to process.clean_text)
    pc = ap.add_mutually_exclusive_group()
    pc.add_argument("--preclean", dest="preclean", action="store_true", help="Pre-clean text before chunking (default)")
    pc.add_argument("--no-preclean", dest="preclean", action="store_false", help="Skip pre-cleaning")
    ap.set_defaults(preclean=True)

    ap.add_argument("--keep-newlines", action="store_true", help="Cleaner flag: keep original newlines")
    ap.add_argument("--lowercase", action="store_true", help="Cleaner flag: lowercase after cleaning")
    args = ap.parse_args()

    if args.size <= 0:
        raise ValueError("size must be > 0")
    if args.overlap < 0 or args.overlap >= args.size:
        raise ValueError("overlap must be >= 0 and < size")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    paths = gather_input_paths(args.input)

    # Default output filename
    if args.output:
        output_path = args.output
    else:
        if args.input.is_file():
            output_path = args.input.with_name(f"{args.input.stem}_chunks.json")
        else:
            output_path = args.input.with_name(f"{args.input.name}_chunks.json")

    results: List[Dict[str, str]] = []

    for path in paths:
        doc = path.stem
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = clean_text(raw, keep_newlines=args.keep_newlines, lowercase=args.lowercase) if args.preclean else raw

        chunks = make_chunks_by_bge_tokens(text, tokenizer, size=args.size, overlap=args.overlap)
        for idx, chunk in enumerate(chunks, 1):
            results.append({
                "doc": doc,
                "id": f"{doc}_{idx:d}",
                "chunk": chunk
            })

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} chunks to {output_path}")

if __name__ == "__main__":
    main()
