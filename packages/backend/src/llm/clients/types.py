from dataclasses import dataclass
from typing import TypeAlias

JsonObject: TypeAlias = dict[str, object]


@dataclass(frozen=True)
class LlmGenerateOptions:
    temperature: float = 0.0
    top_p: float = 0.9
    num_predict: int = 80
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
