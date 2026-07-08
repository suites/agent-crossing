import pytest
from typing import cast

from llm.clients.google_ai_studio import GoogleAiStudioClient
from llm.clients.litellm_client import LiteLlmClient
from llm.clients.ollama import OllamaClient
from llm.clients.provider_factory import ProviderName, build_provider_client


def test_build_provider_client_returns_ollama_client() -> None:
    client = build_provider_client(
        provider="ollama",
        timeout_seconds=4.0,
        generation_model="qwen2.5:7b-instruct",
        embedding_model="bge-m3",
        base_url="http://localhost:11434",
    )

    assert isinstance(client, OllamaClient)


def test_build_provider_client_returns_google_client() -> None:
    client = build_provider_client(
        provider="google_ai_studio",
        timeout_seconds=4.0,
        generation_model="gemini-1.5-flash",
        embedding_model="text-embedding-004",
        api_key="test-key",
    )

    assert isinstance(client, GoogleAiStudioClient)


def test_build_provider_client_returns_litellm_client() -> None:
    client = build_provider_client(
        provider="litellm",
        timeout_seconds=4.0,
        generation_model="ollama_chat/qwen2.5:7b-instruct",
        embedding_model="ollama/bge-m3",
        base_url="https://model.fredly.dev",
        api_key="test-key",
    )

    assert isinstance(client, LiteLlmClient)
    assert client.base_url == "https://model.fredly.dev"
    assert client.api_key == "test-key"


def test_build_provider_client_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError):
        _ = build_provider_client(
            provider=cast(ProviderName, cast(object, "unknown")),
            timeout_seconds=4.0,
            generation_model="model",
            embedding_model="embedding",
        )
