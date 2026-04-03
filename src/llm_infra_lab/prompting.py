SYSTEM_PROMPT = (
    "You are a careful Python programmer. "
    "Return only valid Python code that solves the task."
)


def render_apps_prompt(question: str, starter_code: str | None = None) -> str:
    parts = [
        "<|system|>",
        SYSTEM_PROMPT,
        "</s>",
        "<|user|>",
        "Solve the following programming task in Python.",
        "",
        "Problem:",
        question.strip(),
    ]
    if starter_code:
        parts.extend(
            [
                "",
                "Starter code:",
                starter_code.rstrip(),
            ]
        )
    parts.extend(
        [
            "",
            "Return only the Python solution.",
            "</s>",
            "<|assistant|>",
        ]
    )
    return "\n".join(parts)

