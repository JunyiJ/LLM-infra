from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AppsRecord:
    task_id: str
    difficulty: str
    prompt: str
    completion: str
    source: str = "codeparrot/apps"

    @property
    def text(self) -> str:
        return f"{self.prompt}{self.completion}"

    @property
    def sample_hash(self) -> str:
        payload = f"{self.task_id}\n{self.prompt}\n{self.completion}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


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
        for solution in solutions:
            if is_valid_python(solution):
                return solution
        return None

    inputs = spec.get("inputs")
    outputs = spec.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs):
        return None

    fn_name = spec.get("fn_name")
    for solution in solutions:
        if not is_valid_python(solution):
            continue
        passed = True
        for case_input, expected_output in zip(inputs, outputs):
            actual_output = run_python_solution(
                solution,
                case_input,
                fn_name=fn_name if isinstance(fn_name, str) and fn_name else None,
                timeout_sec=timeout_sec,
            )
            if actual_output is _EXECUTION_FAILED or not outputs_match(actual_output, expected_output):
                passed = False
                break
        if passed:
            return solution
    return None


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
