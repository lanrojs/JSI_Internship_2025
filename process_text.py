#!/usr/bin/env python3
"""
clean_text.py — Simple text preprocessor for .txt files.

- Removes newlines by default (turns them into spaces)
- Collapses multiple whitespace (spaces/tabs/newlines) into a single space
- Removes spaces before punctuation like , . ; : ! ?
- Trims leading/trailing whitespace
- Optional: keep newlines, lowercase, Unicode normalization

Usage:
  python clean_text.py input.txt [-o output.txt] [--keep-newlines] [--lowercase]
"""

import argparse
import re
import unicodedata
from pathlib import Path

PUNCT = r"\.,;:!\?\)\]\}"

def clean_text(text: str, *, keep_newlines: bool = False, lowercase: bool = False) -> str:
    # Unicode normalization (handles odd spacing chars, compatibility forms, etc.)
    text = unicodedata.normalize("NFKC", text)

    if not keep_newlines:
        # Replace any newline sequence with a single space
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")

    # Collapse all runs of whitespace (space, tab, newline) to a single space
    text = re.sub(r"\s+", " ", text)

    # Remove spaces immediately before common punctuation
    text = re.sub(rf"\s+([{PUNCT}])", r"\1", text)

    # Remove spaces after opening brackets/quotes
    text = re.sub(r"([(\[\{“‘])\s+", r"\1", text)

    # Trim
    text = text.strip()

    if lowercase:
        text = text.lower()

    return text


def main():
    ap = argparse.ArgumentParser(description="Clean a .txt file: remove newlines and unwanted spaces.")
    ap.add_argument("input", type=Path, help="Path to input .txt file")
    ap.add_argument("-o", "--output", type=Path, help="Path to output .txt file (default: <input>_cleaned.txt)")
    ap.add_argument("--keep-newlines", action="store_true", help="Keep original newlines (still trims/fixes spaces)")
    ap.add_argument("--lowercase", action="store_true", help="Lowercase the text after cleaning")
    args = ap.parse_args()

    inp: Path = args.input
    out: Path = args.output or inp.with_name(f"{inp.stem}_cleaned{inp.suffix}")

    text = inp.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_text(text, keep_newlines=args.keep_newlines, lowercase=args.lowercase)
    out.write_text(cleaned, encoding="utf-8")

    print(f"Cleaned text written to: {out}")

if __name__ == "__main__":
    main()
