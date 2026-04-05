SYSTEM_PROMPT = (
    "You are a careful Python programmer. "
    "Return only valid Python code that solves the task."
)

USER_INSTRUCTION = (
    "Solve the following programming task in Python. "
    "Return only the Python solution with no explanation."
)


def render_apps_prompt(question: str, starter_code: str | None = None) -> str:
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

    user_message = "\n".join(user_parts)

    # Qwen2.5-Instruct follows a ChatML-style conversation format.
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
