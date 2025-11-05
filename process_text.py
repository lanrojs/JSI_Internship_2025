import sys
import re
from pathlib import Path

def clean_text(text: str) -> str:
    """
    Clean and normalize a raw text string for chunking.

    This function:
      • Normalizes punctuation and whitespace.
      • Merges line breaks that split sentences.
      • Keeps logical paragraph breaks (double newlines).
      • Collapses redundant spaces.
      • Removes accidental spaces before punctuation.
    """

    # --- Normalize punctuation and whitespace characters ---
    # Replace curly quotes, em/en dashes, and non-breaking spaces
    text = (text.replace('‘', "'")
                .replace('’', "'")
                .replace('“', '"')
                .replace('”', '"')
                .replace('\u2013', '-')   # en dash
                .replace('\u2014', '-')   # em dash
                .replace('\xa0', ' ')     # non-breaking space
                .replace('\r\n', '\n')    # normalize Windows newlines
                .replace('\r', '\n'))     # normalize old Mac newlines

    # --- Collapse 3+ consecutive blank lines into just 2 ---
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    # --- Handle list markers like "(1)" or "(a)" followed by blank lines ---
    # e.g., "(1)\n\nSome text" → "(1) Some text"
    text = re.sub(r'\((\d+|[a-z])\)\s*\n+\s*', r'(\1) ', text)

    # --- Keep consecutive subpoints on separate lines but remove blank gaps ---
    # e.g., "(a)\n\n(b)" → "(a)\n(b)"
    text = re.sub(r'\)\s*\n+\s*\(', ')\n(', text)

    # --- Merge subpoints split by a single newline ---
    # e.g., "(a)\ntext" → "(a) text"
    text = re.sub(r'\(([a-z])\)\s*\n\s*', r'(\1) ', text)

    # --- Join lines within a sentence ---
    # If a line break occurs mid-sentence (no period before it),
    # replace it with a space instead of keeping a newline.
    text = re.sub(r'(?<![.!?;:])\n(?=\S)', ' ', text)

    # --- Remove leading spaces before list markers like "(a)" or "(1)" ---
    text = re.sub(r'^[ \t]+(\([a-z0-9]+\))', r'\1', text, flags=re.MULTILINE)

    # --- Collapse multiple spaces into one ---
    text = re.sub(r' {2,}', ' ', text)

    # --- Trim spaces around newlines ---
    text = re.sub(r'[ \t]*\n[ \t]*', '\n', text)

    # --- Remove accidental spaces before punctuation marks ---
    text = re.sub(r'\s+([,.;:])', r'\1', text)

    # --- Strip leading/trailing whitespace and return result ---
    return text.strip()


def main() -> None:
    """
    Command-line entry point.
    Usage:
        python clean_text.py INPUT_FILE
    Writes a cleaned version of INPUT_FILE to <name>_cleaned.<ext>.
    """
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} INPUT_FILE", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.is_file():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")

    # Read, clean, and write
    raw = input_path.read_text(encoding="utf-8")
    cleaned = clean_text(raw)
    output_path.write_text(cleaned, encoding="utf-8")

    print(f"Wrote cleaned file to: {output_path}")


if __name__ == "__main__":
    main()
