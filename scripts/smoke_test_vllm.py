#!/usr/bin/env python
from __future__ import annotations

import argparse

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="local-dev")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt",
        default="Write a Python function that returns the factorial of n.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    print(response.choices[0].message.content or "")


if __name__ == "__main__":
    main()
