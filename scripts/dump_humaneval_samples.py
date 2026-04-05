#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import orjson
from evalplus.data import get_human_eval_plus
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="local-dev")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        required=True,
        help="Repeatable served model name. Example: --model qwen25-1p5b-base --model qwen25-1p5b-sft-run-2",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def fetch_completion(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict:
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = response.choices[0]
    usage = response.usage
    return {
        "text": choice.text,
        "finish_reason": choice.finish_reason,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
    }


def main() -> None:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    problems = list(get_human_eval_plus().items())[: args.max_samples]
    rows: list[dict] = []

    for task_id, problem in problems:
        prompt = problem["prompt"]
        completions = {}
        for model in args.models:
            completions[model] = fetch_completion(
                client,
                model=model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        rows.append(
            {
                "task_id": task_id,
                "entry_point": problem.get("entry_point"),
                "prompt": prompt,
                "canonical_solution": problem.get("canonical_solution"),
                "completions": completions,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(rows, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
