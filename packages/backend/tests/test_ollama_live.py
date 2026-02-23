"""Live Ollama smoke tests (opt-in).

Run with:
  RUN_LIVE_OLLAMA=1 uv run pytest -m live_ollama packages/backend/tests/test_ollama_live.py -v
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import pytest

from llm import OllamaClient, OllamaGenerateOptions
from llm.ollama_client import JsonObject


@dataclass(frozen=True)
class OllamaContext:
    base_url: str
    version: str
    model: str


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
RUN_LIVE_OLLAMA = os.getenv("RUN_LIVE_OLLAMA", "").strip().lower() in {"1", "true", "yes", "on"}

TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TEST_TIMEOUT_SECONDS", "30"))
RETRY_COUNT = int(os.getenv("OLLAMA_TEST_RETRY_COUNT", "3"))
RETRY_BACKOFF_SECONDS = float(os.getenv("OLLAMA_TEST_RETRY_BACKOFF_SECONDS", "0.5"))
INFERENCE_BUDGET_SECONDS = float(os.getenv("OLLAMA_TEST_INFERENCE_BUDGET_SECONDS", "20"))
INFERENCE_MAX_TOKENS = int(os.getenv("OLLAMA_TEST_INFERENCE_MAX_TOKENS", "64"))


pytestmark = [
    pytest.mark.live_ollama,
    pytest.mark.skipif(
        not RUN_LIVE_OLLAMA,
        reason="Set RUN_LIVE_OLLAMA=1 to run live Ollama smoke tests",
    ),
]


def _request_json(
    *,
    url: str,
    method: str,
    payload: JsonObject | None,
    timeout_seconds: float,
    attempts: int,
) -> JsonObject:
    payload_bytes = None
    if payload is not None:
        payload_bytes = json.dumps(payload).encode("utf-8")

    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url,
                data=payload_bytes,
                method=method,
                headers={"Content-Type": "application/json"} if payload is not None else {},
            )
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read()
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise AssertionError("Ollama response must be a JSON object")
            return parsed
        except urllib.error.HTTPError as exc:
            body = None
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except OSError:
                body = None
            message = f"HTTP {exc.code} from {url}"
            if body:
                message = f"{message}: {body}"
            error: Exception = RuntimeError(message)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, AssertionError) as exc:
            error = exc

        if attempt >= attempts:
            raise RuntimeError(f"Ollama request failed after {attempts} attempts") from error

        time.sleep(RETRY_BACKOFF_SECONDS * attempt)


@pytest.fixture(scope="session")
def ollama_context() -> OllamaContext:
    version_payload = _request_json(
        url=f"{OLLAMA_BASE_URL}/api/version",
        method="GET",
        payload=None,
        timeout_seconds=TIMEOUT_SECONDS,
        attempts=RETRY_COUNT,
    )
    version = version_payload.get("version")
    if not isinstance(version, str) or not version:
        raise RuntimeError("Ollama /api/version response missing version")

    tags_payload = _request_json(
        url=f"{OLLAMA_BASE_URL}/api/tags",
        method="GET",
        payload=None,
        timeout_seconds=TIMEOUT_SECONDS,
        attempts=RETRY_COUNT,
    )
    raw_models = tags_payload.get("models")
    if not isinstance(raw_models, list):
        raise RuntimeError("Ollama /api/tags response missing models list")

    model_names = [
        model.get("name") for model in raw_models if isinstance(model, dict) and isinstance(model.get("name"), str)
    ]

    if not model_names:
        pytest.skip("Ollama has no installed models. Run ollama pull first.")

    if OLLAMA_MODEL not in model_names:
        pytest.skip(f"Configured OLLAMA_MODEL '{OLLAMA_MODEL}' is not installed")

    return OllamaContext(base_url=OLLAMA_BASE_URL, version=version, model=OLLAMA_MODEL)


def test_ollama_smoke_health(ollama_context: OllamaContext) -> None:
    assert ollama_context.version
    assert ollama_context.model


def test_ollama_generate_smoke_and_deterministic_shape(ollama_context: OllamaContext) -> None:
    options = OllamaGenerateOptions(temperature=0.0, top_p=1.0, num_predict=INFERENCE_MAX_TOKENS)

    start = time.perf_counter()
    client = OllamaClient(
        base_url=ollama_context.base_url,
        timeout_seconds=TIMEOUT_SECONDS,
    )
    response_text = client.generate(
        model=ollama_context.model,
        prompt='Return strict JSON only: {"status":"ok","topic":"ollama-smoke"}.',
        options=options,
        format_json=True,
    )
    elapsed = time.perf_counter() - start

    assert response_text, "Ollama generate returned empty response"
    parsed = json.loads(response_text)
    assert parsed.get("status") == "ok"
    assert parsed.get("topic") == "ollama-smoke"
    assert elapsed < INFERENCE_BUDGET_SECONDS


def test_ollama_generate_payload_contract(ollama_context: OllamaContext) -> None:
    generate_payload = {
        "model": ollama_context.model,
        "prompt": "Hello. Return one short sentence in Korean.",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": INFERENCE_MAX_TOKENS,
        },
    }

    response = _request_json(
        url=f"{ollama_context.base_url}/api/generate",
        method="POST",
        payload=generate_payload,
        timeout_seconds=TIMEOUT_SECONDS,
        attempts=RETRY_COUNT,
    )

    assert response.get("done") is True
    assert response.get("model") == ollama_context.model
    response_text = response.get("response")
    assert isinstance(response_text, str) and response_text.strip(), "Generate response text must be non-empty"
