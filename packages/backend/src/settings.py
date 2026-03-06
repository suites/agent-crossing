"""Backend-wide configuration constants."""

import os
from typing import Final, Literal, cast

EMBEDDING_DIMENSION: Final[int] = 1024

_raw_llm_provider = os.getenv("LLM_PROVIDER", "ollama")
if _raw_llm_provider not in {"ollama", "google_ai_studio"}:
    _raw_llm_provider = "ollama"
LLM_PROVIDER: Final[Literal["ollama", "google_ai_studio"]] = cast(
    Literal["ollama", "google_ai_studio"],
    _raw_llm_provider,
)
LLM_BASE_URL: Final[str] = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_MODEL: Final[str] = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
EMBEDDING_MODEL: Final[str] = os.getenv("EMBEDDING_MODEL", "bge-m3")
LLM_TIMEOUT_SECONDS: Final[float] = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
GOOGLE_AI_STUDIO_API_KEY: Final[str] = os.getenv("GOOGLE_AI_STUDIO_API_KEY", "")
