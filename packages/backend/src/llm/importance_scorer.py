import json
import re
from dataclasses import dataclass
from typing import Protocol, cast

from .clients.ollama import (
    JsonObject,
    OllamaGenerateOptions,
)
from .prompt_builders import build_importance_scoring_prompt


def clamp_importance(value: int) -> int:
    return max(1, min(10, value))


def _parse_json_object(text: str) -> JsonObject | None:
    try:
        parsed = cast(object, json.loads(text))
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(parsed, dict):
            return cast(JsonObject, parsed)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None

    try:
        nested = cast(object, json.loads(match.group(0)))
    except json.JSONDecodeError:
        return None

    return cast(JsonObject, nested) if isinstance(nested, dict) else None


def parse_importance_value(text: str, fallback_importance: int) -> int:
    payload = _parse_json_object(text)
    if payload is None:
        return fallback_importance

    raw_importance = payload.get("importance")

    if isinstance(raw_importance, bool):
        return fallback_importance

    if isinstance(raw_importance, int):
        return clamp_importance(raw_importance)

    if isinstance(raw_importance, float):
        return clamp_importance(round(raw_importance))

    if isinstance(raw_importance, str):
        stripped = raw_importance.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return clamp_importance(int(stripped))

    return fallback_importance


@dataclass(frozen=True)
class ImportanceScoringContext:
    observation: str
    agent_name: str
    identity_stable_set: list[str]
    current_plan: str | None = None


class ImportanceScorer(Protocol):
    def score(self, context: ImportanceScoringContext) -> int: ...


class ImportanceGenerateClient(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        options: OllamaGenerateOptions | None = None,
        system: str | None = None,
        format_json: bool = False,
    ) -> str: ...


class LlmImportanceScorer:
    def __init__(
        self,
        client: ImportanceGenerateClient,
        fallback_importance: int = 3,
        options: OllamaGenerateOptions | None = None,
    ) -> None:
        self.client: ImportanceGenerateClient = client
        self.fallback_importance: int = clamp_importance(fallback_importance)
        self.options: OllamaGenerateOptions = options or OllamaGenerateOptions()

    def score(self, context: ImportanceScoringContext) -> int:
        prompt = self._build_prompt(context)

        try:
            response = self.client.generate(
                prompt=prompt,
                options=self.options,
                format_json=True,
            )
        except (RuntimeError, TimeoutError, ValueError):
            return self.fallback_importance

        return parse_importance_value(response, self.fallback_importance)

    @staticmethod
    def _build_prompt(context: ImportanceScoringContext) -> str:
        return build_importance_scoring_prompt(
            agent_name=context.agent_name,
            identity_stable_set=context.identity_stable_set,
            current_plan=context.current_plan,
            observation=context.observation,
        )


OllamaImportanceScorer = LlmImportanceScorer
