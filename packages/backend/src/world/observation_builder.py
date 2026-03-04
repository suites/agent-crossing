from typing import Literal


def format_self_said(language: Literal["ko", "en"], reply: str) -> str:
    if language == "ko":
        return f"나는 이렇게 말했다: {reply}"
    return f"I said: {reply}"


def format_other_said(
    language: Literal["ko", "en"], speaker_name: str, reply: str
) -> str:
    if language == "ko":
        return f"{speaker_name}가 이렇게 말했다: {reply}"
    return f"{speaker_name} said: {reply}"
