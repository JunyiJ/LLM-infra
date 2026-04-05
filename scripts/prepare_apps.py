#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import orjson
from datasets import load_dataset
from huggingface_hub import hf_hub_url
from llm_infra_lab.apps import AppsRecord, select_passing_solution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="codeparrot/apps")
    parser.add_argument("--output-dir", default="data/apps")
    parser.add_argument("--difficulty", default="interview,competition")
    parser.add_argument("--max-train-samples", type=int, default=3000)
    parser.add_argument("--max-val-samples", type=int, default=300)
    return parser.parse_args()


def load_apps_dataset(dataset_name: str):
    if dataset_name != "codeparrot/apps":
        return load_dataset(dataset_name)

    return load_dataset(
        "json",
        data_files={
            "train": hf_hub_url(repo_id=dataset_name, repo_type="dataset", filename="train.jsonl"),
            "test": hf_hub_url(repo_id=dataset_name, repo_type="dataset", filename="test.jsonl"),
        },
    )


def build_stub_record(row: dict) -> dict | None:
    task_id = str(row.get("problem_id") or row.get("id") or row.get("url") or "")
    question = (row.get("question") or "").strip()
    difficulty = (row.get("difficulty") or "").strip().lower()
    solutions = row.get("solutions")
    starter_code = (row.get("starter_code") or "").strip() or None
    completion = select_passing_solution(solutions, row.get("input_output"))
    if not task_id or not question or not solutions:
        return None
    if completion is None:
        return None

    record = AppsRecord(
        task_id=task_id,
        difficulty=difficulty,
        question=question,
        starter_code=starter_code,
        completion=completion,
    )
    return {
        "task_id": record.task_id,
        "difficulty": record.difficulty,
        "question": record.question,
        "starter_code": record.starter_code,
        "completion": record.completion,
        "source": record.source,
        "sample_hash": record.sample_hash,
    }


def write_split(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")


def build_split(dataset_split, allowed_difficulties: set[str], max_samples: int) -> list[dict]:
    output: list[dict] = []
    for row in dataset_split:
        record = build_stub_record(row)
        if record is None:
            continue
        if allowed_difficulties and record["difficulty"] not in allowed_difficulties:
            continue
        output.append(record)
        if len(output) >= max_samples:
            break
    return output


def main() -> None:
    args = parse_args()
    allowed_difficulties = {item.strip().lower() for item in args.difficulty.split(",") if item.strip()}
    dataset = load_apps_dataset(args.dataset_name)
    train_rows = build_split(dataset["train"], allowed_difficulties, args.max_train_samples)
    val_source = dataset["test"] if "test" in dataset else dataset["train"]
    val_rows = build_split(val_source, allowed_difficulties, args.max_val_samples)

    output_dir = Path(args.output_dir)
    write_split(train_rows, output_dir / "train.jsonl")
    write_split(val_rows, output_dir / "val.jsonl")

    summary = {
        "dataset_name": args.dataset_name,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "difficulty": sorted(allowed_difficulties),
        "next_steps": [
            "Decide whether to keep APPS train/test or create an internal dev split",
            "Inspect rows dropped because no solution passed the provided tests",
            "Tune timeout and output normalization for harder APPS problems",
        ],
    }
    with open(output_dir / "summary.json", "wb") as handle:
        handle.write(orjson.dumps(summary, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    main()
