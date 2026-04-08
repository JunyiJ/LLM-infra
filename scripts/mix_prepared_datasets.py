#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import orjson

from llm_infra_lab.manifest import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/apps_taco_mix")
    parser.add_argument(
        "--mix",
        action="append",
        required=True,
        help="Repeatable source spec in the form name=dir:train_count:val_count",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_mix_spec(spec: str) -> dict:
    try:
        name_part, rest = spec.split("=", 1)
        dir_part, train_count_part, val_count_part = rest.rsplit(":", 2)
    except ValueError as exc:
        raise ValueError(f"Invalid --mix spec {spec!r}. Expected name=dir:train_count:val_count") from exc

    name = name_part.strip()
    source_dir = Path(dir_part).expanduser()
    train_count = int(train_count_part)
    val_count = int(val_count_part)
    if not name:
        raise ValueError(f"Invalid --mix spec {spec!r}: missing source name")
    if train_count < 0 or val_count < 0:
        raise ValueError(f"Invalid --mix spec {spec!r}: counts must be non-negative")
    return {
        "name": name,
        "source_dir": source_dir,
        "train_count": train_count,
        "val_count": val_count,
    }


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")


def sample_rows(rows: list[dict], count: int, *, rng: random.Random) -> list[dict]:
    if count > len(rows):
        raise ValueError(f"Requested {count} rows from a source with only {len(rows)} rows")
    if count == len(rows):
        sampled = list(rows)
        rng.shuffle(sampled)
        return sampled
    return rng.sample(rows, count)


def main() -> None:
    args = parse_args()
    specs = [parse_mix_spec(spec) for spec in args.mix]
    rng = random.Random(args.seed)

    mixed_train_rows: list[dict] = []
    mixed_val_rows: list[dict] = []
    source_summaries: list[dict] = []

    for spec in specs:
        source_dir = spec["source_dir"].resolve()
        train_path = source_dir / "train.jsonl"
        val_path = source_dir / "val.jsonl"
        train_rows = load_jsonl(train_path)
        val_rows: list[dict] = []
        if spec["val_count"] > 0:
            if not val_path.exists():
                raise FileNotFoundError(
                    f"Requested {spec['val_count']} validation rows from {source_dir}, "
                    f"but {val_path} does not exist"
                )
            val_rows = load_jsonl(val_path)

        sampled_train = sample_rows(train_rows, spec["train_count"], rng=rng)
        sampled_val = sample_rows(val_rows, spec["val_count"], rng=rng) if spec["val_count"] > 0 else []

        mixed_train_rows.extend(sampled_train)
        mixed_val_rows.extend(sampled_val)
        source_summaries.append(
            {
                "name": spec["name"],
                "source_dir": str(source_dir),
                "train_path": str(train_path),
                "val_path": str(val_path),
                "train_path_sha256": sha256_file(train_path),
                "val_path_sha256": sha256_file(val_path) if val_path.exists() else None,
                "available_train_rows": len(train_rows),
                "available_val_rows": len(val_rows),
                "sampled_train_rows": len(sampled_train),
                "sampled_val_rows": len(sampled_val),
            }
        )

    rng.shuffle(mixed_train_rows)
    rng.shuffle(mixed_val_rows)

    output_dir = Path(args.output_dir)
    write_jsonl(mixed_train_rows, output_dir / "train.jsonl")
    write_jsonl(mixed_val_rows, output_dir / "val.jsonl")

    summary = {
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "train_rows": len(mixed_train_rows),
        "val_rows": len(mixed_val_rows),
        "sources": source_summaries,
    }
    with open(output_dir / "summary.json", "wb") as handle:
        handle.write(orjson.dumps(summary, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))

    print(f"wrote {output_dir / 'train.jsonl'}")
    print(f"wrote {output_dir / 'val.jsonl'}")
    print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
