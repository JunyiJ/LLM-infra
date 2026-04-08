from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import orjson
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_data_config(cfg: dict) -> dict:
    data_cfg = dict(cfg["data"])
    variant_name = data_cfg.get("variant")
    variants = data_cfg.get("variants")
    if not variant_name or not isinstance(variants, dict):
        return data_cfg

    variant_cfg = variants.get(variant_name)
    if not isinstance(variant_cfg, dict):
        available = ", ".join(sorted(str(key) for key in variants))
        raise KeyError(f"Unknown data.variant={variant_name!r}. Available variants: {available}")

    merged = {**data_cfg, **variant_cfg}
    merged["variant"] = variant_name
    merged["selected_variant"] = variant_name
    return merged


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def write_yaml(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
