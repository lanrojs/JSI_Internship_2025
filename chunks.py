"""
chunks.py — Clean (via process_text.clean_text) + chunk .txt files using BGE-small token counts.

Boundary rules for each chunk:
  - Start: '(' or a LETTER (A–Z/a–z).  ❌ Not a digit.
  - End:   alphanumeric OR ')'
  - Trims punctuation on first/last token but preserves '(' and ')'.
  - Skips redundant chunks (no new words beyond previous chunk).

Output:
  Default name: <input>_chunks.json  (if input is a single file)
  or            <directory>_chunks.json  (if input is a directory of .txt files)

This module is designed so that you can:
  • Run it from the command line:  python chunking.py INPUT_PATH
  • Import and call it from a notebook:  from chunking import chunk_path; chunk_path("gdpr.txt")
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Union

from transformers import AutoTokenizer
from process_text import clean_text

# --------------------------------------------------------------------
# Configuration constants (tune here if needed)
# --------------------------------------------------------------------

# HuggingFace model ID whose tokenizer approximates BGE-small token counts
MODEL_ID = "BAAI/bge-small-en-v1.5"

# Maximum number of model tokens per chunk
MAX_TOKENS_PER_CHUNK = 300

# Desired overlap in model tokens between consecutive chunks
TOKEN_OVERLAP = 50

# --------------------------------------------------------------------
# Regex helpers for detecting "wordy" tokens and trimming punctuation
# --------------------------------------------------------------------

# Has at least one alphanumeric character (letter or digit)
ALNUM = re.compile(r"[A-Za-z0-9]", re.UNICODE)

# Trim leading non-alnum but preserve '(' at the start
LEAD_NONALNUM_KEEP_LPAREN = re.compile(r"^[^\w(]+", re.UNICODE)

# Trim trailing non-alnum but preserve ')' at the end
TRAIL_NONALNUM_KEEP_RPAREN = re.compile(r"[^\w)]+$", re.UNICODE)


def is_wordy(tok: str) -> bool:
    """
    Return True if the token contains at least one alphanumeric character.

    We use this to avoid starting/ending chunks on pure punctuation or whitespace-like tokens.
    """
    return bool(ALNUM.search(tok))


def split_to_words(text: str) -> List[str]:
    """
    Split text into a list of whitespace-separated 'words'.

    We treat any whitespace as a separator; punctuation stays attached
    to the surrounding text and is handled later by trimming functions.
    """
    return text.split()


def token_len(tokenizer, text_piece: str) -> int:
    """
    Compute the number of model tokens for a given word or short text piece.

    We disable special tokens so that we measure only the content tokens.
    """
    return len(tokenizer.encode(text_piece, add_special_tokens=False))


def snap_window_to_wordy(tokens: List[str], i: int, j: int) -> Tuple[int, int]:
    """
    Adjust a window [i, j) to ensure it starts and ends on 'wordy' tokens.

    - Move i forward until tokens[i] is wordy.
    - Move j backward until tokens[j-1] is wordy.
    - Return the new (i, j) indices.
    - If there is no wordy token in the window, return (j, j) to signal "empty".
    """
    n = len(tokens)
    if i >= j or i >= n:
        return j, j

    # Move start forward to the first wordy token
    while i < j and not is_wordy(tokens[i]):
        i += 1
    if i >= j:
        return j, j

    # Move end backward to the last wordy token
    k = j - 1
    while k >= i and not is_wordy(tokens[k]):
        k -= 1
    if k < i:
        return j, j

    # Return inclusive start, exclusive end
    return i, k + 1


def trim_boundary_token_first(tok: str) -> str:
    """
    Trim leading punctuation/non-word characters from the first token,
    but keep '(' if present (we want to preserve list markers like '(1)', '(a)').
    """
    return LEAD_NONALNUM_KEEP_LPAREN.sub("", tok)


def trim_boundary_token_last(tok: str) -> str:
    """
    Trim trailing punctuation/non-word characters from the last token,
    but keep ')' if present (to preserve closing parentheses on list markers).
    """
    return TRAIL_NONALNUM_KEEP_RPAREN.sub("", tok)


def starts_ok(token_text: str) -> bool:
    """
    Check that the first visible character of a chunk is valid.

    Rules:
      • Start must be '(' or a LETTER (A–Z/a–z).
      • It must NOT start with a digit.
    """
    if not token_text:
        return False
    c = token_text[0]
    return c == "(" or c.isalpha()


def has_boundary_rules(text: str) -> bool:
    """
    Final guard on a chunk's boundary characters.

    Rules:
      • First character must be '(' or a letter.
      • Last character must be alphanumeric or ')'.
    """
    if not text:
        return False
    start_ok = text[0] == "(" or text[0].isalpha()
    end_ok = text[-1].isalnum() or text[-1] == ")"
    return start_ok and end_ok


def make_chunks(
    text: str,
    tokenizer,
    size: int = MAX_TOKENS_PER_CHUNK,
    overlap: int = TOKEN_OVERLAP,
) -> List[str]:
    """
    Core chunking routine.

    Given:
      • text: already cleaned text (no weird line breaks, normalized spaces, etc.)
      • tokenizer: a HuggingFace tokenizer (e.g. BGE-small)
      • size: maximum model tokens per chunk
      • overlap: how many model tokens should overlap between consecutive chunks

    Returns:
      A list of textual chunks, each a string, satisfying the boundary rules:
        - Start: '(' or a LETTER (no digits).
        - End:   alphanumeric or ')'.
        - Punctuation trimmed at boundaries, but '(' and ')' preserved.
        - Overlapping in terms of token count where possible.
    """
    words = split_to_words(text)
    n = len(words)
    if n == 0:
        return []

    # Precompute token lengths for each word (model token counts)
    per_word_toklen = [token_len(tokenizer, w) for w in words]

    chunks: List[str] = []
    start = 0
    last_max_j = -1  # farthest end index among emitted chunks (to skip redundant chunks)

    while start < n:
        # Skip non-wordy tokens at the beginning
        while start < n and not is_wordy(words[start]):
            start += 1
        if start >= n:
            break

        # Grow window [start, end) until we hit 'size' model tokens
        total = 0
        end = start
        while end < n:
            tlen = per_word_toklen[end]
            if total + tlen > size:
                break
            total += tlen
            end += 1

        # If a single word already exceeds the limit, force it into a chunk alone
        if end == start:
            end = start + 1

        # Snap window to wordy boundaries
        i, j = snap_window_to_wordy(words, start, end)
        if i >= j:
            start = end
            continue

        # --- Trim punctuation only on the first and last tokens in the window ---
        first = trim_boundary_token_first(words[i])
        last = trim_boundary_token_last(words[j - 1])

        # If trimming empties the boundary tokens, contract inward
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

        # Enforce the "no numeric start" rule by moving i forward if needed
        while i < j and not starts_ok(first):
            i += 1
            if i < j:
                first = trim_boundary_token_first(words[i])
        if i >= j:
            start = end
            continue

        # Re-evaluate 'last' safely after possible contractions
        last = trim_boundary_token_last(words[j - 1])

        # Build the final chunk text
        window_tokens = words[i:j].copy()
        window_tokens[0] = first
        window_tokens[-1] = last
        chunk_text = " ".join(window_tokens).strip()

        # Final boundary check: must satisfy start/end rules
        if not has_boundary_rules(chunk_text):
            # If the result is not well-formed at boundaries, skip this window
            start = j
            continue

        # Skip redundant chunks that don't move the 'j' frontier forward
        if j <= last_max_j:
            start = max(j, start + 1)
            continue

        # Accept this chunk
        chunks.append(chunk_text)
        last_max_j = j

        # --- Compute next 'start' to enforce token overlap ---
        if overlap <= 0:
            # No overlap requested; just continue from the end
            start = j
            continue

        acc = 0
        k = j
        # Walk backwards until we accumulate 'overlap' tokens
        while k > i and acc < overlap:
            k -= 1
            acc += per_word_toklen[k]

        # New start is where the backward walk ended, unless we didn't reach 'overlap'
        new_start = k if acc >= overlap else i

        # Ensure forward progress
        if new_start <= start:
            new_start = max(j, start + 1)

        start = new_start

    return chunks


def gather_input_paths(input_path: Path) -> List[Path]:
    """
    Given a path, return a list of .txt files to process.

    - If input_path is a single .txt file → return [input_path].
    - If input_path is a directory      → return all *.txt files (non-recursive).
    - Otherwise, raise an error.
    """
    if input_path.is_file() and input_path.suffix.lower() == ".txt":
        return [input_path]
    if input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() == ".txt"
        )
    raise ValueError(f"Input must be a .txt file or a directory: {input_path}")


def chunk_path(input_path: Union[str, Path]) -> Path:
    """
    High-level function you can call from a notebook or another script.

    Parameters
    ----------
    input_path:
        Path to a .txt file OR to a directory containing multiple .txt files.

    Behavior
    --------
    - Loads the tokenizer for BGE-small.
    - Gathers all .txt files (single file or directory).
    - Reads and cleans each file using process_text.clean_text.
    - Chunks the cleaned text using BGE token counts.
    - Writes all chunks into a single JSON file:
        <input>_chunks.json  (for a single file)
        <directory>_chunks.json  (for a directory)

    Returns
    -------
    Path
        The path to the JSON file that was written.
    """
    input_path = Path(input_path)

    # Load tokenizer once and reuse for all documents
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Collect all .txt files we need to process
    paths = gather_input_paths(input_path)

    # Choose output filename based on the top-level input
    if input_path.is_file():
        base = input_path.stem
        base = re.sub(r"_cleaned$", "", base)  # drop trailing "_cleaned" if present
        output_path = input_path.with_name(f"{base}_chunks.json")
    else:
        base = input_path.name
        base = re.sub(r"_cleaned$", "", base)
        output_path = input_path.with_name(f"{base}_chunks.json")

    results: List[Dict[str, str]] = []

    # Process each document independently, but store all chunks together
    for path in paths:
        doc = path.stem  # document ID prefix
        raw = path.read_text(encoding="utf-8", errors="ignore")

        # Always pre-clean using your process_text.clean_text
        text = clean_text(raw)

        # Perform token-based chunking
        chunks = make_chunks(
            text,
            tokenizer,
            size=MAX_TOKENS_PER_CHUNK,
            overlap=TOKEN_OVERLAP,
        )

        # Collect chunks with doc/id metadata
        for idx, chunk in enumerate(chunks, 1):
            results.append(
                {
                    "doc": doc,
                    "id": f"{doc}_{idx:d}",
                    "chunk": chunk,
                }
            )

    # Write all chunks to JSON (UTF-8, pretty-printed)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(results)} chunks to {output_path}")
    return output_path


def main() -> None:
    """
    Command-line entry point.

    Usage:
        python chunking.py INPUT_PATH

    where INPUT_PATH is:
      • a .txt file, or
      • a directory containing .txt files.

    All configuration (model, chunk size, overlap) is defined at the top of the file.
    """
    if len(sys.argv) != 2:
        prog = Path(sys.argv[0]).name
        print(f"Usage: {prog} INPUT_PATH", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: input path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Run the high-level function
    chunk_path(input_path)


if __name__ == "__main__":
    main()
