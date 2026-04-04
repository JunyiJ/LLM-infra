#!/usr/bin/env python
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import orjson
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="local-dev")
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark-name", default="serving_smoke")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        default=[],
        help="Repeatable prompt argument. If omitted, built-in coding prompts are used.",
    )
    return parser.parse_args()


def default_prompts() -> list[str]:
    return [
        "Write a Python function that returns the factorial of n.",
        "Write a Python function that merges two sorted lists.",
        "Write a Python function that checks whether a string is a palindrome.",
    ]


def count_tokens(text: str) -> int:
    return len(text.split())


def benchmark(client: OpenAI, *, model: str, prompts: list[str], max_tokens: int, num_runs: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    latencies: list[float] = []
    output_token_counts: list[int] = []
    prompt_token_counts: list[int] = []
    for run_idx in range(num_runs):
        prompt = prompts[run_idx % len(prompts)]
        started_at = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        latency_sec = time.perf_counter() - started_at
        content = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens if response.usage else count_tokens(prompt)
        completion_tokens = response.usage.completion_tokens if response.usage else count_tokens(content)
        total_tokens = response.usage.total_tokens if response.usage else prompt_tokens + completion_tokens
        tokens_per_sec = completion_tokens / latency_sec if latency_sec > 0 else None
        row = {
            "run_index": run_idx,
            "latency_sec": round(latency_sec, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec is not None else None,
            "prompt_preview": prompt[:120],
        }
        rows.append(row)
        latencies.append(latency_sec)
        output_token_counts.append(completion_tokens)
        prompt_token_counts.append(prompt_tokens)

    summary = {
        "model": model,
        "benchmark_name": "serving_smoke",
        "num_runs": num_runs,
        "p50_latency_sec": round(statistics.median(latencies), 4),
        "mean_latency_sec": round(statistics.fmean(latencies), 4),
        "mean_prompt_tokens": round(statistics.fmean(prompt_token_counts), 2),
        "mean_completion_tokens": round(statistics.fmean(output_token_counts), 2),
        "mean_tokens_per_sec": round(
            statistics.fmean(row["tokens_per_sec"] for row in rows if row["tokens_per_sec"] is not None),
            2,
        ),
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else None
    prompts = args.prompts or default_prompts()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    rows, summary = benchmark(
        client,
        model=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens,
        num_runs=args.num_runs,
    )
    payload = {
        "summary": {
            **summary,
            "benchmark_name": args.benchmark_name,
        },
        "rows": rows,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    print(orjson.dumps(payload["summary"], option=orjson.OPT_INDENT_2).decode("utf-8"))


if __name__ == "__main__":
    main()

