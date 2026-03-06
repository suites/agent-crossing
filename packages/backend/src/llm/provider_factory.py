from typing import Literal, Protocol

from .google_ai_studio_client import GoogleAiStudioClient
from .ollama_client import OllamaClient, OllamaGenerateOptions

ProviderName = Literal["ollama", "google_ai_studio"]


class ProviderClient(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        system: str | None = None,
        options: OllamaGenerateOptions | None = None,
        format_json: bool = False,
    ) -> str: ...

    def embed(
        self,
        *,
        model: str | None = None,
        input: str,
        truncate: bool = True,
        keep_alive: str = "30m",
        expected_dimension: int | None = None,
    ) -> list[float]: ...


def build_provider_client(
    *,
    provider: str,
    timeout_seconds: float,
    generation_model: str,
    embedding_model: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ProviderClient:
    if provider == "ollama":
        return OllamaClient(
            base_url=base_url or "http://localhost:11434",
            timeout_seconds=timeout_seconds,
            default_generate_model=generation_model,
            default_embedding_model=embedding_model,
        )

    if provider == "google_ai_studio":
        return GoogleAiStudioClient(
            api_key=api_key or "",
            timeout_seconds=timeout_seconds,
            default_generate_model=generation_model,
            default_embedding_model=embedding_model,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
