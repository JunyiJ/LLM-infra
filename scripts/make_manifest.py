#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from llm_infra_lab.manifest import resolve_data_config, load_yaml, sha256_file, utc_now, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = load_yaml(config_path)
    data_cfg = resolve_data_config(cfg)
    output_dir = Path(cfg["train"]["output_root"]) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = (config_dir.parent / data_cfg["train_path"]).resolve()
    val_path = (config_dir.parent / data_cfg["val_path"]).resolve()
    manifest = {
        "created_at": utc_now(),
        "run_name": args.run_name,
        "project": cfg["project"]["name"],
        "model": cfg["model"]["name_or_path"],
        "data": {
            "variant": data_cfg.get("variant"),
            "train_path": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val_path": str(val_path),
            "val_sha256": sha256_file(val_path),
        },
        "train": cfg["train"],
    }
    write_json(output_dir / "manifest.json", manifest)
    print(output_dir / "manifest.json")


if __name__ == "__main__":
    main()
