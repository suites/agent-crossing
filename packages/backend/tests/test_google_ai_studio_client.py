from llm.clients.google_ai_studio import GoogleAiStudioClient
from llm.clients.ollama import LlmGenerateOptions


def test_generate_uses_google_generate_content_shape() -> None:
    captured: list[tuple[str, dict[str, object], float]] = []

    def request_fn(
        url: str, payload: dict[str, object], timeout_seconds: float
    ) -> dict[str, object]:
        captured.append((url, payload, timeout_seconds))
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": '{"status":"ok"}'}],
                    }
                }
            ]
        }

    client = GoogleAiStudioClient(
        api_key="test-key",
        timeout_seconds=7.0,
        request_fn=request_fn,
    )

    response = client.generate(
        prompt="Return JSON",
        system="You are concise.",
        options=LlmGenerateOptions(temperature=0.3, top_p=0.8, num_predict=60),
        format_json=True,
    )

    assert response == '{"status":"ok"}'
    assert captured
    url, payload, timeout_seconds = captured[0]
    assert "generateContent" in url
    assert "key=test-key" in url
    assert timeout_seconds == 7.0
    assert payload["contents"] == [{"parts": [{"text": "Return JSON"}]}]
    assert payload["systemInstruction"] == {"parts": [{"text": "You are concise."}]}
    generation_config = payload["generationConfig"]
    assert isinstance(generation_config, dict)
    assert generation_config["responseMimeType"] == "application/json"


def test_embed_reads_embedding_values() -> None:
    captured_payloads: list[dict[str, object]] = []

    def request_fn(
        _url: str, payload: dict[str, object], _timeout_seconds: float
    ) -> dict[str, object]:
        captured_payloads.append(payload)
        return {
            "embedding": {
                "values": [0.1, 0.2, 0.3],
            }
        }

    client = GoogleAiStudioClient(api_key="test-key", request_fn=request_fn)

    embedding = client.embed(input="hello", expected_dimension=3)

    assert embedding == [0.1, 0.2, 0.3]
    assert captured_payloads
    assert captured_payloads[0]["outputDimensionality"] == 3
