from __future__ import annotations

from llm.ollama_client import JsonObject, OllamaClient, OllamaGenerateOptions


def test_generate_includes_penalty_options_when_configured() -> None:
    captured_payloads: list[JsonObject] = []

    def request_fn(
        _url: str, payload: JsonObject, _timeout_seconds: float
    ) -> JsonObject:
        captured_payloads.append(payload)
        return {"response": "ok"}

    client = OllamaClient(request_fn=request_fn)
    options = OllamaGenerateOptions(
        temperature=0.3,
        top_p=0.95,
        num_predict=42,
        repeat_penalty=1.2,
        presence_penalty=0.4,
        frequency_penalty=0.1,
    )

    result = client.generate(prompt="hello", options=options)

    assert result == "ok"
    assert captured_payloads
    options_payload = captured_payloads[0]["options"]
    assert isinstance(options_payload, dict)
    assert options_payload["temperature"] == 0.3
    assert options_payload["top_p"] == 0.95
    assert options_payload["num_predict"] == 42
    assert options_payload["repeat_penalty"] == 1.2
    assert options_payload["presence_penalty"] == 0.4
    assert options_payload["frequency_penalty"] == 0.1


def test_generate_omits_penalty_options_by_default() -> None:
    captured_payloads: list[JsonObject] = []

    def request_fn(
        _url: str, payload: JsonObject, _timeout_seconds: float
    ) -> JsonObject:
        captured_payloads.append(payload)
        return {"response": "ok"}

    client = OllamaClient(request_fn=request_fn)

    result = client.generate(prompt="hello")

    assert result == "ok"
    assert captured_payloads
    options_payload = captured_payloads[0]["options"]
    assert isinstance(options_payload, dict)
    assert "repeat_penalty" not in options_payload
    assert "presence_penalty" not in options_payload
    assert "frequency_penalty" not in options_payload
