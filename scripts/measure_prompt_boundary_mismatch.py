#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_infra_lab.apps import row_prompt
from llm_infra_lab.manifest import load_yaml
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--show-examples", type=int, default=5)
    return parser.parse_args()


def resolve_train_path(config_path: str, data_path: str | None) -> Path:
    if data_path is not None:
        return Path(data_path).resolve()

    cfg = load_yaml(config_path)
    config_dir = Path(config_path).resolve().parent
    return (config_dir.parent / cfg["data"]["train_path"]).resolve()


def resolve_model_name(config_path: str, model_name: str | None) -> str:
    if model_name is not None:
        return model_name
    cfg = load_yaml(config_path)
    return cfg["model"]["name_or_path"]


def load_rows(path: Path, max_samples: int | None) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def main() -> None:
    args = parse_args()
    train_path = resolve_train_path(args.config, args.data_path)
    model_name = resolve_model_name(args.config, args.model_name)
    rows = load_rows(train_path, args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    mismatch_count = 0
    mismatch_examples: list[dict] = []
    total = 0

    for row in rows:
        prompt = row_prompt(row)
        completion = row["completion"]
        full_text = prompt + completion

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        total += 1
        if full_ids[: len(prompt_ids)] == prompt_ids:
            continue

        mismatch_count += 1
        if len(mismatch_examples) < args.show_examples:
            mismatch_at = 0
            max_check = min(len(prompt_ids), len(full_ids))
            while mismatch_at < max_check and prompt_ids[mismatch_at] == full_ids[mismatch_at]:
                mismatch_at += 1
            mismatch_examples.append(
                {
                    "task_id": row.get("task_id", ""),
                    "prompt_token_count": len(prompt_ids),
                    "full_prefix_token_count": len(full_ids[: len(prompt_ids)]),
                    "first_mismatch_index": mismatch_at,
                    "prompt_suffix": prompt[-120:].replace("\n", "\\n"),
                    "completion_prefix": completion[:120].replace("\n", "\\n"),
                    "prompt_token_window": prompt_ids[max(0, mismatch_at - 5) : mismatch_at + 5],
                    "full_token_window": full_ids[max(0, mismatch_at - 5) : mismatch_at + 5],
                }
            )

    mismatch_rate = 0.0 if total == 0 else mismatch_count / total

    print(f"train_path: {train_path}")
    print(f"model_name: {model_name}")
    print(f"total_samples: {total}")
    print(f"mismatch_samples: {mismatch_count}")
    print(f"mismatch_rate: {mismatch_rate:.6f}")

    if mismatch_examples:
        print("\nexamples:")
        for index, example in enumerate(mismatch_examples, start=1):
            print(f"[{index}] task_id={example['task_id']}")
            print(f"  first_mismatch_index={example['first_mismatch_index']}")
            print(f"  prompt_token_count={example['prompt_token_count']}")
            print(f"  full_prefix_token_count={example['full_prefix_token_count']}")
            print(f"  prompt_suffix={example['prompt_suffix']}")
            print(f"  completion_prefix={example['completion_prefix']}")
            print(f"  prompt_token_window={example['prompt_token_window']}")
            print(f"  full_token_window={example['full_token_window']}")


if __name__ == "__main__":
    main()
