from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_infra_lab.prompting import PROMPT_STYLE_CHATML, render_apps_prompt


@dataclass(slots=True)
class AppsRecord:
    task_id: str
    difficulty: str
    question: str
    starter_code: str | None
    completion: str
    source: str = "codeparrot/apps"

    @property
    def prompt(self) -> str:
        return render_apps_prompt(self.question, self.starter_code)

    @property
    def text(self) -> str:
        return f"{self.prompt}{self.completion}"

    @property
    def sample_hash(self) -> str:
        payload = (
            f"{self.task_id}\n{self.question}\n{self.starter_code or ''}\n{self.completion}"
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def row_prompt(row: dict, *, prompt_style: str = PROMPT_STYLE_CHATML) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt
    question = (row.get("question") or "").strip()
    starter_code = (row.get("starter_code") or "").strip() or None
    return render_apps_prompt(question, starter_code, prompt_style=prompt_style)


def row_text(row: dict, *, prompt_style: str = PROMPT_STYLE_CHATML) -> str:
    text = row.get("text")
    if isinstance(text, str) and text:
        return text
    return f"{row_prompt(row, prompt_style=prompt_style)}{row['completion']}"


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def top_level_function_names(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]


def has_main_block(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
            continue
        if not isinstance(test.ops[0], ast.Eq):
            continue
        left = test.left
        right = test.comparators[0]
        if isinstance(left, ast.Name) and left.id == "__name__" and isinstance(right, ast.Constant) and right.value == "__main__":
            return True
    return False


def strip_main_block(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    lines = code.splitlines()
    removal_ranges: list[tuple[int, int]] = []
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
            continue
        if not isinstance(test.ops[0], ast.Eq):
            continue
        left = test.left
        right = test.comparators[0]
        if isinstance(left, ast.Name) and left.id == "__name__" and isinstance(right, ast.Constant) and right.value == "__main__":
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                removal_ranges.append((node.lineno - 1, node.end_lineno))
    if not removal_ranges:
        return code

    kept: list[str] = []
    cursor = 0
    for start, end in sorted(removal_ranges):
        kept.extend(lines[cursor:start])
        cursor = end
    kept.extend(lines[cursor:])
    cleaned = "\n".join(kept).rstrip()
    return f"{cleaned}\n" if cleaned else ""


def completion_quality_stats(code: str) -> dict[str, Any]:
    function_names = top_level_function_names(code)
    stripped_lines = [line for line in code.splitlines() if line.strip()]
    variant_function_names = [name for name in function_names if re.search(r"_\d+$", name)]
    test_like_function_names = [
        name for name in function_names if name == "main" or name.startswith("test_") or name.startswith("check_")
    ]
    return {
        "top_level_function_names": function_names,
        "top_level_function_count": len(function_names),
        "variant_function_names": variant_function_names,
        "test_like_function_names": test_like_function_names,
        "has_main_block": has_main_block(code),
        "has_doctest": "doctest" in code,
        "has_code_fence": "```" in code,
        "line_count": len(stripped_lines),
        "char_count": len(code),
    }


def completion_quality_score(
    code: str,
    *,
    target_fn_name: str | None,
) -> float:
    stats = completion_quality_stats(code)
    score = 0.0
    score += min(stats["line_count"], 400) / 20.0
    score += min(stats["char_count"], 8000) / 1000.0
    if stats["has_main_block"]:
        score += 8.0
    if stats["has_doctest"]:
        score += 6.0
    if stats["has_code_fence"]:
        score += 10.0
    score += len(stats["test_like_function_names"]) * 12.0
    score += len(stats["variant_function_names"]) * 10.0
    if target_fn_name is not None:
        fn_names = stats["top_level_function_names"]
        if target_fn_name not in fn_names:
            score += 100.0
        extra_names = [name for name in fn_names if name != target_fn_name]
        score += len(extra_names) * 12.0
    else:
        # Script-style tasks can legitimately avoid function definitions, but a long list of
        # unrelated helper functions is still a bad fit for HumanEval-like behavior.
        score += max(stats["top_level_function_count"] - 2, 0) * 8.0
    return score


def is_completion_acceptable(
    code: str,
    *,
    target_fn_name: str | None,
) -> tuple[bool, str | None]:
    stats = completion_quality_stats(code)
    if stats["has_code_fence"]:
        return False, "code_fence"
    if stats["test_like_function_names"]:
        return False, "test_or_main_function"
    if stats["variant_function_names"]:
        return False, "variant_function_names"
    if target_fn_name is not None:
        fn_names = stats["top_level_function_names"]
        if target_fn_name not in fn_names:
            return False, "missing_target_fn"
        extra_names = [name for name in fn_names if name != target_fn_name]
        if extra_names:
            return False, "extra_top_level_functions"
    if stats["line_count"] > 200:
        return False, "too_long"
    return True, None


_CALL_BASED_RUNNER = """
import importlib.util
import json
import sys


def to_jsonable(value):
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


spec = importlib.util.spec_from_file_location("candidate_solution", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

payload = json.loads(sys.argv[2])
fn = getattr(module, payload["fn_name"])
args = payload["args"]
if not isinstance(args, list):
    args = [args]
result = fn(*args)
json.dump(to_jsonable(result), sys.stdout)
"""

_EXECUTION_FAILED = object()


def parse_apps_solutions(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [item.strip() for item in raw if isinstance(item, str) and item.strip()]
    text = raw.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(payload, list):
        return [item.strip() for item in payload if isinstance(item, str) and item.strip()]
    return [text]


def parse_apps_input_output(raw: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    text = raw.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    except ValueError as exc:
        # Some APPS rows contain extremely large integer literals in JSON.
        # This dataset is trusted input, so temporarily disable the digit cap
        # and retry instead of failing the whole preparation job.
        if "Exceeds the limit" not in str(exc) or not hasattr(sys, "set_int_max_str_digits"):
            return None
        previous_limit = sys.get_int_max_str_digits()
        try:
            sys.set_int_max_str_digits(0)
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
        finally:
            sys.set_int_max_str_digits(previous_limit)
    return payload if isinstance(payload, dict) else None


def select_passing_solution(
    solutions_raw: str | list[str] | None,
    input_output_raw: str | dict[str, Any] | None,
    timeout_sec: float = 2.0,
) -> str | None:
    solutions = parse_apps_solutions(solutions_raw)
    if not solutions:
        return None

    spec = parse_apps_input_output(input_output_raw)
    if spec is None:
        valid_solutions = [solution for solution in solutions if is_valid_python(solution)]
        if not valid_solutions:
            return None
        cleaned = [strip_main_block(solution) for solution in valid_solutions]
        ranked = sorted(cleaned, key=lambda solution: completion_quality_score(solution, target_fn_name=None))
        for solution in ranked:
            ok, _ = is_completion_acceptable(solution, target_fn_name=None)
            if ok:
                return solution
        return ranked[0] if ranked else None
    target_fn_name = spec.get("fn_name") if isinstance(spec.get("fn_name"), str) and spec.get("fn_name") else None
    inputs = spec.get("inputs")
    outputs = spec.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs):
        return None

    passing_solutions: list[str] = []
    for solution in solutions:
        if not is_valid_python(solution):
            continue
        passed = True
        for case_input, expected_output in zip(inputs, outputs):
            actual_output = run_python_solution(
                solution,
                case_input,
                fn_name=target_fn_name,
                timeout_sec=timeout_sec,
            )
            if actual_output is _EXECUTION_FAILED or not outputs_match(actual_output, expected_output):
                passed = False
                break
        if passed:
            passing_solutions.append(strip_main_block(solution))
    if not passing_solutions:
        return None

    ranked = sorted(
        passing_solutions,
        key=lambda solution: completion_quality_score(solution, target_fn_name=target_fn_name),
    )
    acceptable_ranked = [
        solution for solution in ranked if is_completion_acceptable(solution, target_fn_name=target_fn_name)[0]
    ]
    return acceptable_ranked[0] if acceptable_ranked else ranked[0]


def run_python_solution(
    solution: str,
    case_input: Any,
    *,
    fn_name: str | None = None,
    timeout_sec: float = 2.0,
) -> Any:
    if fn_name:
        return _run_call_based_solution(solution, fn_name, case_input, timeout_sec)
    return _run_stdin_solution(solution, case_input, timeout_sec)


def outputs_match(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    return _normalize_output(actual) == _normalize_output(expected)


def _run_stdin_solution(solution: str, case_input: Any, timeout_sec: float) -> Any:
    stdin_text = _coerce_stdin(case_input)
    with tempfile.TemporaryDirectory(prefix="apps_solution_") as temp_dir:
        solution_path = Path(temp_dir) / "candidate.py"
        solution_path.write_text(solution, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(solution_path)],
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return _EXECUTION_FAILED
    if completed.returncode != 0:
        return _EXECUTION_FAILED
    return completed.stdout


def _run_call_based_solution(solution: str, fn_name: str, case_input: Any, timeout_sec: float) -> Any:
    payload = json.dumps({"fn_name": fn_name, "args": case_input})
    with tempfile.TemporaryDirectory(prefix="apps_solution_") as temp_dir:
        solution_path = Path(temp_dir) / "candidate.py"
        runner_path = Path(temp_dir) / "runner.py"
        solution_path.write_text(solution, encoding="utf-8")
        runner_path.write_text(_CALL_BASED_RUNNER, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(runner_path), str(solution_path), payload],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return _EXECUTION_FAILED
    if completed.returncode != 0:
        return _EXECUTION_FAILED
    stdout = completed.stdout.strip()
    if not stdout:
        return None
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return stdout


def _coerce_stdin(case_input: Any) -> str:
    if isinstance(case_input, list):
        text = "\n".join("" if item is None else str(item) for item in case_input)
    else:
        text = "" if case_input is None else str(case_input)
    return text if text.endswith("\n") else f"{text}\n"


def _normalize_output(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(_normalize_output(item) for item in value).strip()
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").strip()
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()
