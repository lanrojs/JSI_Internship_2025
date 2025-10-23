#!/usr/bin/env python3
"""
summarize_txt_openrouter.py

Usage:
  python summarize_txt_openrouter.py --input path/to/file.txt [--output summary.txt] [--max_words 250]

Setup:
  export OPENROUTER_API_KEY="your_key_here"
  pip install requests
"""

import os
import sys
import json
import argparse
import textwrap
import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "z-ai/glm-4.5-air:free"  # confirmed model slug on OpenRouter

def summarize_text(text: str, max_words: int = 250, http_referer: str = None, x_title: str = "Txt Summarizer"):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Error: Please set the OPENROUTER_API_KEY environment variable.")

    # Optional headers recommended by OpenRouter (nice-to-have, but not required)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if http_referer:
        headers["HTTP-Referer"] = http_referer  # your app/site URL (optional)
    if x_title:
        headers["X-Title"] = x_title  # short app label (optional)

    system_prompt = (
        "You are a precise technical summarizer. Produce a compact summary that preserves key facts, "
        "numbers, names, definitions, and conclusions. Eliminate fluff and repetition. "
        f"Target length: <= {max_words} words. If the text is structured, use short bullet points; "
        "otherwise, write a tight paragraph. Add a one-line TL;DR at the end."
    )

    # To guard against very large files, we gently trim extreme whitespace
    text = textwrap.dedent(text).strip()

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Compactify/summarize the following text:\n\n"
                    "----- BEGIN TEXT -----\n"
                    f"{text}\n"
                    "----- END TEXT -----"
                )
            }
        ],
        "temperature": 0.2,
    }

    resp = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(payload), timeout=90)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Show useful server message if present
        try:
            details = resp.json()
        except Exception:
            details = resp.text
        sys.exit(f"API error: {e}\nDetails: {details}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected API response format:\n{json.dumps(data, indent=2)}")

def main():
    parser = argparse.ArgumentParser(description="Summarize a .txt file using OpenRouter (GLM-4.5-Air free).")
    parser.add_argument("--input", "-i", required=True, help="Path to the input .txt file")
    parser.add_argument("--output", "-o", help="Optional path to write the summary")
    parser.add_argument("--max_words", type=int, default=250, help="Target summary length in words (default: 250)")
    parser.add_argument("--referer", help="Optional HTTP-Referer header (your app/site URL)")
    parser.add_argument("--title", default="Txt Summarizer", help="Optional X-Title header label")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Error: File not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    summary = summarize_text(
        text=text,
        max_words=args.max_words,
        http_referer=args.referer,
        x_title=args.title
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(summary + "\n")
        print(f"Summary written to: {args.output}")
    else:
        print(summary)

if __name__ == "__main__":
    main()
