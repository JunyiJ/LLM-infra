#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import orjson
from datasets import load_dataset
from huggingface_hub import hf_hub_url
from llm_infra_lab.apps import (
    AppsRecord,
    completion_quality_stats,
    is_completion_acceptable,
    parse_apps_input_output,
    select_passing_solution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="BAAI/TACO")
    parser.add_argument("--output-dir", default="data/taco_verified")
    parser.add_argument("--difficulty", default="easy,medium,hard")
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-val-samples", type=int, default=500)
    parser.add_argument(
        "--task-format",
        choices=["all", "function_only"],
        default="all",
        help="Filter rows by task shape after solution cleaning.",
    )
    return parser.parse_args()


def load_taco_dataset(dataset_name: str):
    if dataset_name != "BAAI/TACO":
        return load_dataset(dataset_name)

    return load_dataset(
        "parquet",
        data_files={
            "train": [
                hf_hub_url(
                    repo_id=dataset_name,
                    repo_type="dataset",
                    filename=f"ALL/train-{index:05d}-of-00009.parquet",
                )
                for index in range(9)
            ],
            "test": hf_hub_url(
                repo_id=dataset_name,
                repo_type="dataset",
                filename="ALL/test-00000-of-00001.parquet",
            ),
        },
    )


def is_function_only_record(record: dict) -> bool:
    starter_code = (record.get("starter_code") or "").strip()
    if not starter_code:
        return False
    first_nonempty_line = next((line for line in starter_code.splitlines() if line.strip()), "")
    return first_nonempty_line.startswith("def ") or first_nonempty_line.startswith("async def ")


def build_record(row: dict, *, skip_counter: Counter[str] | None = None) -> dict | None:
    task_id = str(row.get("problem_id") or row.get("id") or row.get("question") or "")
    question = (row.get("question") or "").strip()
    difficulty = (row.get("difficulty") or "").strip().lower()
    solutions = row.get("solutions")
    starter_code = (row.get("starter_code") or "").strip() or None
    completion = select_passing_solution(solutions, row.get("input_output"))
    if not task_id or not question or not solutions:
        if skip_counter is not None:
            skip_counter["missing_required_fields"] += 1
        return None
    if completion is None:
        if skip_counter is not None:
            skip_counter["no_passing_solution"] += 1
        return None

    target_fn_name = None
    input_output = parse_apps_input_output(row.get("input_output"))
    if isinstance(input_output, dict):
        maybe_name = input_output.get("fn_name")
        target_fn_name = maybe_name if isinstance(maybe_name, str) and maybe_name else None
    acceptable, reason = is_completion_acceptable(completion, target_fn_name=target_fn_name)
    if not acceptable:
        if skip_counter is not None and reason is not None:
            skip_counter[reason] += 1
        return None

    quality_stats = completion_quality_stats(completion)
    record = AppsRecord(
        task_id=task_id,
        difficulty=difficulty,
        question=question,
        starter_code=starter_code,
        completion=completion,
        source="BAAI/TACO",
    )
    return {
        "task_id": record.task_id,
        "difficulty": record.difficulty,
        "question": record.question,
        "starter_code": record.starter_code,
        "completion": record.completion,
        "source": record.source,
        "sample_hash": record.sample_hash,
        "completion_quality": quality_stats,
    }


def write_split(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")


def build_split(
    dataset_split,
    allowed_difficulties: set[str],
    max_samples: int,
    *,
    task_format: str,
    skip_counter: Counter[str] | None = None,
) -> list[dict]:
    output: list[dict] = []
    for row in dataset_split:
        record = build_record(row, skip_counter=skip_counter)
        if record is None:
            continue
        if allowed_difficulties and record["difficulty"] not in allowed_difficulties:
            if skip_counter is not None:
                skip_counter["difficulty_filtered"] += 1
            continue
        if task_format == "function_only" and not is_function_only_record(record):
            if skip_counter is not None:
                skip_counter["task_format_filtered"] += 1
            continue
        output.append(record)
        if len(output) >= max_samples:
            break
    return output


def main() -> None:
    args = parse_args()
    allowed_difficulties = {item.strip().lower() for item in args.difficulty.split(",") if item.strip()}
    dataset = load_taco_dataset(args.dataset_name)
    train_skips: Counter[str] = Counter()
    val_skips: Counter[str] = Counter()

    train_rows = build_split(
        dataset["train"],
        allowed_difficulties,
        args.max_train_samples,
        task_format=args.task_format,
        skip_counter=train_skips,
    )
    val_source = dataset["test"] if "test" in dataset else dataset["train"]
    val_rows = build_split(
        val_source,
        allowed_difficulties,
        args.max_val_samples,
        task_format=args.task_format,
        skip_counter=val_skips,
    )

    output_dir = Path(args.output_dir)
    write_split(train_rows, output_dir / "train.jsonl")
    write_split(val_rows, output_dir / "val.jsonl")

    summary = {
        "dataset_name": args.dataset_name,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "difficulty": sorted(allowed_difficulties),
        "task_format": args.task_format,
        "train_skips": dict(sorted(train_skips.items())),
        "val_skips": dict(sorted(val_skips.items())),
        "next_steps": [
            "Inspect skip counters for task_format_filtered and no_passing_solution",
            "Compare taco_verified against apps_default with the same prompt_style and learning rate",
            "Consider mixing taco_verified with apps_default once single-dataset baselines are stable",
        ],
    }
    with open(output_dir / "summary.json", "wb") as handle:
        handle.write(orjson.dumps(summary, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    main()
