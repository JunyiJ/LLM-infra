SYSTEM_PROMPT = (
    "You are a careful Python programmer. "
    "Return only valid Python code that solves the task."
)

USER_INSTRUCTION = (
    "Solve the following programming task in Python. "
    "Return only the Python solution with no explanation."
)

PROMPT_STYLE_CHATML = "chatml"
PROMPT_STYLE_PLAIN = "plain"
PROMPT_STYLE_CODE_PREFIX = "code_prefix"
PROMPT_STYLE_SIGNATURE_DOCSTRING = "signature_docstring"
PROMPT_STYLE_AUTO_BENCHMARK_LIKE = "auto_benchmark_like"
SUPPORTED_PROMPT_STYLES = {
    PROMPT_STYLE_CHATML,
    PROMPT_STYLE_PLAIN,
    PROMPT_STYLE_CODE_PREFIX,
    PROMPT_STYLE_SIGNATURE_DOCSTRING,
    PROMPT_STYLE_AUTO_BENCHMARK_LIKE,
}


def _render_user_message(question: str, starter_code: str | None = None) -> str:
    user_parts = [
        USER_INSTRUCTION,
        "",
        "Problem:",
        question.strip(),
    ]
    if starter_code:
        user_parts.extend(
            [
                "",
                "Starter code:",
                starter_code.rstrip(),
            ]
        )
    return "\n".join(user_parts)


def _render_code_prefix(question: str, starter_code: str | None = None) -> str:
    comment_lines = ["# " if not line.strip() else f"# {line}" for line in question.strip().splitlines()]
    parts = comment_lines if comment_lines else ["#"]
    if starter_code:
        parts.extend(["", starter_code.rstrip()])
    else:
        parts.append("")
    return "\n".join(parts) + "\n"


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" "))]


def _has_top_level_function_signature(starter_code: str | None) -> bool:
    if not starter_code:
        return False
    for line in starter_code.splitlines():
        if not line.strip():
            continue
        return line.startswith("def ") or line.startswith("async def ")
    return False


def _render_signature_docstring(question: str, starter_code: str | None = None) -> str:
    if not starter_code:
        return _render_code_prefix(question, starter_code)

    starter_lines = starter_code.rstrip().splitlines()
    signature_index = None
    signature_indent = ""
    for index, line in enumerate(starter_lines):
        stripped = line.lstrip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            signature_index = index
            signature_indent = _indent_of(line)
            break

    if signature_index is None:
        return _render_code_prefix(question, starter_code)

    body_indent = f"{signature_indent}    "
    docstring_lines = [f'{body_indent}"""']
    for line in question.strip().splitlines():
        docstring_lines.append(body_indent if not line.strip() else f"{body_indent}{line}")
    docstring_lines.append(f'{body_indent}"""')

    parts = starter_lines[: signature_index + 1] + docstring_lines
    return "\n".join(parts) + "\n"


def render_apps_prompt(
    question: str,
    starter_code: str | None = None,
    *,
    prompt_style: str = PROMPT_STYLE_CHATML,
) -> str:
    if prompt_style not in SUPPORTED_PROMPT_STYLES:
        supported = ", ".join(sorted(SUPPORTED_PROMPT_STYLES))
        raise ValueError(f"Unsupported prompt_style={prompt_style!r}. Expected one of: {supported}")

    if prompt_style == PROMPT_STYLE_AUTO_BENCHMARK_LIKE:
        if _has_top_level_function_signature(starter_code):
            return _render_signature_docstring(question, starter_code)
        return _render_code_prefix(question, starter_code)

    user_message = _render_user_message(question, starter_code)
    if prompt_style == PROMPT_STYLE_SIGNATURE_DOCSTRING:
        return _render_signature_docstring(question, starter_code)
    if prompt_style == PROMPT_STYLE_CODE_PREFIX:
        return _render_code_prefix(question, starter_code)
    if prompt_style == PROMPT_STYLE_PLAIN:
        return f"{SYSTEM_PROMPT}\n\n{user_message}\n\n"

    # Qwen2.5-Instruct follows a ChatML-style conversation format.
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
