#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time

import orjson
from datasets import load_dataset
from huggingface_hub import hf_hub_url
from llm_infra_lab.apps import (
    AppsRecord,
    completion_quality_stats,
    is_completion_acceptable,
    parse_apps_input_output,
    parse_apps_solutions,
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
    parser.add_argument(
        "--max-candidate-solutions",
        type=int,
        default=16,
        help="Maximum number of candidate solutions to try per problem before verification.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N scanned rows.",
    )
    parser.add_argument(
        "--write-every",
        type=int,
        default=100,
        help="Rewrite partial output files every N kept rows.",
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=None,
        help="Optional wall-clock budget. If reached, write partial outputs and stop early.",
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


def limit_solutions(solutions: str | list[str] | None, max_candidate_solutions: int) -> list[str]:
    parsed = parse_apps_solutions(solutions)
    if max_candidate_solutions <= 0:
        return parsed
    return parsed[:max_candidate_solutions]


def build_record(
    row: dict,
    *,
    max_candidate_solutions: int,
    skip_counter: Counter[str] | None = None,
) -> dict | None:
    task_id = str(row.get("problem_id") or row.get("id") or row.get("question") or "")
    question = (row.get("question") or "").strip()
    difficulty = (row.get("difficulty") or "").strip().lower()
    solutions = limit_solutions(row.get("solutions"), max_candidate_solutions)
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


def write_partial_outputs(
    *,
    output_dir: Path,
    split_name: str,
    rows: list[dict],
    skip_counter: Counter[str],
    scanned_rows: int,
    max_samples: int,
    stopped_early: bool,
) -> None:
    write_split(rows, output_dir / f"{split_name}.jsonl")
    payload = {
        "split": split_name,
        "rows_written": len(rows),
        "rows_scanned": scanned_rows,
        "target_rows": max_samples,
        "stopped_early": stopped_early,
        "skip_counts": dict(sorted(skip_counter.items())),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(output_dir / f"{split_name}_progress.json", "wb") as handle:
        handle.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def build_split(
    dataset_split,
    allowed_difficulties: set[str],
    max_samples: int,
    *,
    split_name: str,
    output_dir: Path,
    task_format: str,
    max_candidate_solutions: int,
    progress_every: int,
    write_every: int,
    started_at: float,
    max_runtime_seconds: float | None,
    skip_counter: Counter[str] | None = None,
) -> tuple[list[dict], int, bool]:
    output: list[dict] = []
    scanned_rows = 0
    stopped_early = False
    for row in dataset_split:
        scanned_rows += 1
        if max_runtime_seconds is not None and time.monotonic() - started_at >= max_runtime_seconds:
            stopped_early = True
            print(
                f"[{split_name}] stopping early after {scanned_rows} scanned rows "
                f"and {len(output)} kept rows due to max-runtime budget"
            )
            break

        record = build_record(
            row,
            max_candidate_solutions=max_candidate_solutions,
            skip_counter=skip_counter,
        )
        if record is None:
            if progress_every and scanned_rows % progress_every == 0:
                print(
                    f"[{split_name}] scanned={scanned_rows} kept={len(output)} "
                    f"elapsed_sec={time.monotonic() - started_at:.1f}"
                )
            continue
        if allowed_difficulties and record["difficulty"] not in allowed_difficulties:
            if skip_counter is not None:
                skip_counter["difficulty_filtered"] += 1
            if progress_every and scanned_rows % progress_every == 0:
                print(
                    f"[{split_name}] scanned={scanned_rows} kept={len(output)} "
                    f"elapsed_sec={time.monotonic() - started_at:.1f}"
                )
            continue
        if task_format == "function_only" and not is_function_only_record(record):
            if skip_counter is not None:
                skip_counter["task_format_filtered"] += 1
            if progress_every and scanned_rows % progress_every == 0:
                print(
                    f"[{split_name}] scanned={scanned_rows} kept={len(output)} "
                    f"elapsed_sec={time.monotonic() - started_at:.1f}"
                )
            continue
        output.append(record)
        if write_every and len(output) % write_every == 0:
            write_partial_outputs(
                output_dir=output_dir,
                split_name=split_name,
                rows=output,
                skip_counter=skip_counter or Counter(),
                scanned_rows=scanned_rows,
                max_samples=max_samples,
                stopped_early=False,
            )
            print(
                f"[{split_name}] wrote partial output with {len(output)} rows "
                f"after scanning {scanned_rows} rows"
            )
        if progress_every and scanned_rows % progress_every == 0:
            print(
                f"[{split_name}] scanned={scanned_rows} kept={len(output)} "
                f"elapsed_sec={time.monotonic() - started_at:.1f}"
            )
        if len(output) >= max_samples:
            break
    write_partial_outputs(
        output_dir=output_dir,
        split_name=split_name,
        rows=output,
        skip_counter=skip_counter or Counter(),
        scanned_rows=scanned_rows,
        max_samples=max_samples,
        stopped_early=stopped_early,
    )
    return output, scanned_rows, stopped_early


def main() -> None:
    args = parse_args()
    allowed_difficulties = {item.strip().lower() for item in args.difficulty.split(",") if item.strip()}
    dataset = load_taco_dataset(args.dataset_name)
    train_skips: Counter[str] = Counter()
    val_skips: Counter[str] = Counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    max_runtime_seconds = None if args.max_runtime_minutes is None else args.max_runtime_minutes * 60.0

    train_rows, train_scanned_rows, train_stopped_early = build_split(
        dataset["train"],
        allowed_difficulties,
        args.max_train_samples,
        split_name="train",
        output_dir=output_dir,
        task_format=args.task_format,
        max_candidate_solutions=args.max_candidate_solutions,
        progress_every=args.progress_every,
        write_every=args.write_every,
        started_at=started_at,
        max_runtime_seconds=max_runtime_seconds,
        skip_counter=train_skips,
    )
    val_source = dataset["test"] if "test" in dataset else dataset["train"]
    val_rows, val_scanned_rows, val_stopped_early = build_split(
        val_source,
        allowed_difficulties,
        args.max_val_samples,
        split_name="val",
        output_dir=output_dir,
        task_format=args.task_format,
        max_candidate_solutions=args.max_candidate_solutions,
        progress_every=args.progress_every,
        write_every=args.write_every,
        started_at=started_at,
        max_runtime_seconds=max_runtime_seconds,
        skip_counter=val_skips,
    )

    write_split(train_rows, output_dir / "train.jsonl")
    write_split(val_rows, output_dir / "val.jsonl")

    summary = {
        "dataset_name": args.dataset_name,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_scanned_rows": train_scanned_rows,
        "val_scanned_rows": val_scanned_rows,
        "train_stopped_early": train_stopped_early,
        "val_stopped_early": val_stopped_early,
        "difficulty": sorted(allowed_difficulties),
        "task_format": args.task_format,
        "max_candidate_solutions": args.max_candidate_solutions,
        "progress_every": args.progress_every,
        "write_every": args.write_every,
        "max_runtime_minutes": args.max_runtime_minutes,
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
