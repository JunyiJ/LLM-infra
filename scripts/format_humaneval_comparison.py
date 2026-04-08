#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Path to base model HumanEval sample dump")
    parser.add_argument("--candidate", required=True, help="Path to candidate model HumanEval sample dump")
    parser.add_argument("--output", required=True, help="Markdown output path")
    parser.add_argument("--max-completion-chars", type=int, default=1200)
    return parser.parse_args()


def load_rows(path: Path) -> tuple[str, list[dict]]:
    rows = json.load(open(path, "r", encoding="utf-8"))
    if not rows:
        raise ValueError(f"{path} is empty")
    model_name = next(iter(rows[0]["completions"]))
    return model_name, rows


def analyze_completion(text: str, entry_point: str) -> dict:
    defs = re.findall(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text, flags=re.M)
    extra_defs = [name for name in defs if name != entry_point]
    return {
        "has_main_block": "__name__" in text or "def main" in text,
        "defines_entry_point_again": entry_point in defs,
        "extra_defs": extra_defs,
        "has_test_helper": any(name == "main" or name.startswith("test_") or name.startswith("check_") for name in defs),
        "has_variant_defs": any(re.search(r"_\d+$", name) for name in defs),
        "line_count": text.count("\n") + 1 if text else 0,
        "char_count": len(text),
    }


def issue_list(analysis: dict) -> list[str]:
    issues: list[str] = []
    if analysis["has_main_block"]:
        issues.append("main-block")
    if analysis["defines_entry_point_again"]:
        issues.append("redefines-entry-point")
    if analysis["has_test_helper"]:
        issues.append("test-helper")
    if analysis["has_variant_defs"]:
        issues.append("variant-defs")
    if analysis["extra_defs"]:
        issues.append("extra-defs")
    return issues


def clipped_code_block(text: str, max_chars: int) -> str:
    clipped = text if len(text) <= max_chars else text[:max_chars] + "\n# ... truncated ..."
    return f"```python\n{clipped.rstrip()}\n```"


def main() -> None:
    args = parse_args()
    base_model, base_rows = load_rows(Path(args.base))
    candidate_model, candidate_rows = load_rows(Path(args.candidate))

    if [row["task_id"] for row in base_rows] != [row["task_id"] for row in candidate_rows]:
        raise ValueError("Base and candidate dumps do not contain the same task ordering")

    lines: list[str] = []
    lines.append(f"# HumanEval Comparison: {base_model} vs {candidate_model}")
    lines.append("")
    lines.append(f"- Base dump: `{args.base}`")
    lines.append(f"- Candidate dump: `{args.candidate}`")
    lines.append("")

    summary = {
        "base_main_blocks": 0,
        "candidate_main_blocks": 0,
        "base_extra_defs": 0,
        "candidate_extra_defs": 0,
        "base_variant_defs": 0,
        "candidate_variant_defs": 0,
        "base_test_helpers": 0,
        "candidate_test_helpers": 0,
    }

    for base_row, candidate_row in zip(base_rows, candidate_rows):
        task_id = base_row["task_id"]
        entry_point = base_row.get("entry_point") or ""
        prompt = base_row["prompt"]
        base_completion = base_row["completions"][base_model]["text"]
        candidate_completion = candidate_row["completions"][candidate_model]["text"]

        base_analysis = analyze_completion(base_completion, entry_point)
        candidate_analysis = analyze_completion(candidate_completion, entry_point)

        summary["base_main_blocks"] += int(base_analysis["has_main_block"])
        summary["candidate_main_blocks"] += int(candidate_analysis["has_main_block"])
        summary["base_extra_defs"] += int(bool(base_analysis["extra_defs"]))
        summary["candidate_extra_defs"] += int(bool(candidate_analysis["extra_defs"]))
        summary["base_variant_defs"] += int(base_analysis["has_variant_defs"])
        summary["candidate_variant_defs"] += int(candidate_analysis["has_variant_defs"])
        summary["base_test_helpers"] += int(base_analysis["has_test_helper"])
        summary["candidate_test_helpers"] += int(candidate_analysis["has_test_helper"])

        lines.append(f"## {task_id} `{entry_point}`")
        lines.append("")
        lines.append("**Prompt**")
        lines.append("")
        lines.append(clipped_code_block(prompt, args.max_completion_chars))
        lines.append("")
        lines.append(f"**Base Issues**: `{', '.join(issue_list(base_analysis)) or 'none'}`")
        lines.append("")
        lines.append(clipped_code_block(base_completion, args.max_completion_chars))
        lines.append("")
        lines.append(f"**Candidate Issues**: `{', '.join(issue_list(candidate_analysis)) or 'none'}`")
        lines.append("")
        lines.append(clipped_code_block(candidate_completion, args.max_completion_chars))
        lines.append("")

    lines.insert(
        4,
        "\n".join(
            [
                "## Summary",
                "",
                "| Metric | Base | Candidate |",
                "| --- | --- | --- |",
                f"| main/test harness blocks | {summary['base_main_blocks']} | {summary['candidate_main_blocks']} |",
                f"| completions with extra top-level defs | {summary['base_extra_defs']} | {summary['candidate_extra_defs']} |",
                f"| completions with variant defs like `_2` | {summary['base_variant_defs']} | {summary['candidate_variant_defs']} |",
                f"| completions with test/helper functions | {summary['base_test_helpers']} | {summary['candidate_test_helpers']} |",
                "",
            ]
        ),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
