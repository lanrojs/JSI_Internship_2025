"""
chunking.py — Clean + chunk .txt files using BGE-small token counts.
Output file: <input>_chunks.json
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

from transformers import AutoTokenizer

# ---------- text cleaning ----------

def clean_legal_text(text: str) -> str:
    """Cleans and normalizes legal text structure and spacing."""
    text = text.replace('‘', "'").replace('’', "'").replace('“', '"').replace('”', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-').replace('\xa0', ' ')
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove 3+ consecutive blank lines -> at most 2
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    # Merge "(1)" or "(a)" followed by newline/blank line into "(1) "
    text = re.sub(r'\((\d+|[a-z])\)\s*\n+\s*', r'(\1) ', text)

    # Collapse blank lines between consecutive subpoints
    text = re.sub(r'\)\s*\n+\s*\(', ')\n(', text)

    # Merge subpoints split by single newlines
    text = re.sub(r'\(([a-z])\)\s*\n\s*', r'(\1) ', text)

    # Join lines within paragraphs if sentence continues
    text = re.sub(r'(?<![.!?;:])\n(?=[a-z])', ' ', text)

    # Remove leading spaces before list markers
    text = re.sub(r'^[ \t]+(\([a-z0-9]+\))', r'\1', text, flags=re.MULTILINE)

    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    # Trim spaces around newlines
    text = re.sub(r'[ \t]*\n[ \t]*', '\n', text)

    # Remove accidental spaces before punctuation
    text = re.sub(r'\s+([,.;:])', r'\1', text)

    return text.strip()


# ---------- chunking helpers ----------

ALNUM = re.compile(r"[A-Za-z0-9]", re.UNICODE)
LEAD_NONALNUM_KEEP_LPAREN = re.compile(r"^[^\w(]+", re.UNICODE)
TRAIL_NONALNUM_KEEP_RPAREN = re.compile(r"[^\w)]+$", re.UNICODE)


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
    return LEAD_NONALNUM_KEEP_LPAREN.sub("", tok)


def trim_boundary_token_last(tok: str) -> str:
    return TRAIL_NONALNUM_KEEP_RPAREN.sub("", tok)


def starts_ok(token_text: str) -> bool:
    if not token_text:
        return False
    c = token_text[0]
    return c == "(" or c.isalpha()


def has_boundary_rules(text: str) -> bool:
    if not text:
        return False
    return (text[0] == "(" or text[0].isalpha()) and (text[-1].isalnum() or text[-1] == ")")


def make_chunks_by_bge_tokens(
    text: str,
    tokenizer,
    size: int = 300,
    overlap: int = 100,
) -> List[str]:
    words = split_to_words(text)
    n = len(words)
    if n == 0:
        return []

    per_word_toklen = [token_len(tokenizer, w) for w in words]
    chunks: List[str] = []
    start = 0
    last_max_j = -1

    while start < n:
        while start < n and not is_wordy(words[start]):
            start += 1
        if start >= n:
            break

        total = 0
        end = start
        while end < n:
            tlen = per_word_toklen[end]
            if total + tlen > size:
                break
            total += tlen
            end += 1

        if end == start:
            end = start + 1

        i, j = snap_window_to_wordy(words, start, end)
        if i >= j:
            start = end
            continue

        first = trim_boundary_token_first(words[i])
        last = trim_boundary_token_last(words[j - 1])

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

        while i < j and not starts_ok(first):
            i += 1
            if i < j:
                first = trim_boundary_token_first(words[i])
        if i >= j:
            start = end
            continue

        last = trim_boundary_token_last(words[j - 1])
        window_tokens = words[i:j].copy()
        window_tokens[0] = first
        window_tokens[-1] = last
        chunk_text = " ".join(window_tokens).strip()

        if not has_boundary_rules(chunk_text):
            start = j
            continue

        if j <= last_max_j:
            start = max(j, start + 1)
            continue

        chunks.append(chunk_text)
        last_max_j = j

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


# ---------- main ----------

def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python {Path(sys.argv[0]).name} INPUT_FILE.txt", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.is_file():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Prepare paths
    output_path = input_path.with_name(f"{input_path.stem}_chunks.json")

    # Load and clean text
    raw = input_path.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_legal_text(raw)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

    # Chunk
    chunks = make_chunks_by_bge_tokens(cleaned, tokenizer, size=300, overlap=50)

    # Write results
    results: List[Dict[str, str]] = [
        {"doc": input_path.stem, "id": f"{input_path.stem}_{i+1}", "chunk": chunk}
        for i, chunk in enumerate(chunks)
    ]

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} chunks to {output_path}")


if __name__ == "__main__":
    main()
