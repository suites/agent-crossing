import json

from llm import (
    ImportanceScoringContext,
    OllamaClient,
    OllamaImportanceScorer,
    clamp_importance,
    parse_importance_value,
)
from llm.ollama_client import JsonObject


def test_clamp_importance_bounds() -> None:
    assert clamp_importance(-3) == 1
    assert clamp_importance(5) == 5
    assert clamp_importance(42) == 10


def test_parse_importance_value_json_success() -> None:
    text = json.dumps({"importance": 8, "reason": "important"})
    assert parse_importance_value(text, fallback_importance=3) == 8


def test_parse_importance_value_string_and_clamp() -> None:
    text = json.dumps({"importance": "11", "reason": "critical"})
    assert parse_importance_value(text, fallback_importance=3) == 10


def test_parse_importance_value_malformed_fallback() -> None:
    assert parse_importance_value("not-json", fallback_importance=3) == 3


def test_ollama_importance_scorer_success() -> None:
    def request_fn(url: str, payload: JsonObject, timeout_seconds: float) -> JsonObject:
        assert url.endswith("/api/generate")
        assert payload["model"] == "qwen2.5:7b-instruct"
        assert payload["format"] == "json"
        assert timeout_seconds == 5.0
        return {"response": '{"importance": 7, "reason": "goal relevant"}'}

    client = OllamaClient(timeout_seconds=5.0, request_fn=request_fn)
    scorer = OllamaImportanceScorer(client=client)

    score = scorer.score(
        ImportanceScoringContext(
            observation="지호가 오늘 중요한 회의 약속을 잡았다.",
            agent_name="Jiho Park",
            identity_stable_set=["Jiho is reliable and values promises."],
            current_plan="오후 3시 미팅 참석",
        )
    )

    assert score == 7


def test_ollama_importance_scorer_fallback_on_client_error() -> None:
    def request_fn(
        _url: str, _payload: JsonObject, _timeout_seconds: float
    ) -> JsonObject:
        raise TimeoutError("timed out")

    client = OllamaClient(request_fn=request_fn)
    scorer = OllamaImportanceScorer(client=client, fallback_importance=3)

    score = scorer.score(
        ImportanceScoringContext(
            observation="일상 산책",
            agent_name="Jiho Park",
            identity_stable_set=[],
        )
    )
    assert score == 3
