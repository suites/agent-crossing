import datetime
import json

import numpy as np
from agents.agent import AgentIdentity, AgentProfile, ExtendedPersona, FixedPersona
from agents.memory.memory_object import MemoryObject, NodeType
from llm.embedding_encoder import EmbeddingEncodingContext
from llm.llm_service import LlmService
from llm.reaction_types import ReactionDecisionInput
from llm.prompt_builders import build_reaction_decision_prompt


class StubOllamaClient:
    def __init__(self, responses: list[str]):
        self.responses: list[str] = list(responses)
        self.calls: int = 0

    def generate(self, **_kwargs: object) -> str:
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[index]


class StubEmbeddingEncoder:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors: dict[str, list[float]] = vectors

    def encode(self, context: EmbeddingEncodingContext) -> np.ndarray:
        if context.text not in self.vectors:
            return np.asarray([0.0, 1.0], dtype=np.float32)
        return np.asarray(self.vectors[context.text], dtype=np.float32)


def _reaction_json(
    *,
    should_react: bool,
    reaction: str,
    reason: str,
    thought: str = "",
    critique: str = "",
    use_utterance_field: bool = False,
) -> str:
    payload: dict[str, object] = {
        "should_react": should_react,
        "reason": reason,
    }
    if use_utterance_field:
        payload["utterance"] = reaction
    else:
        payload["reaction"] = reaction

    if thought:
        payload["thought"] = thought
    if critique:
        payload["critique"] = critique

    return json.dumps(payload)


def _input(
    dialogue_history: list[tuple[str, str]],
    *,
    retrieved_memories: list[MemoryObject] | None = None,
) -> ReactionDecisionInput:
    return ReactionDecisionInput(
        agent_identity=AgentIdentity(
            id="jiho",
            name="Jiho Park",
            age=29,
            traits=["kind"],
        ),
        current_time=datetime.datetime(2026, 2, 27, 14, 0, 0),
        observation_content="Jiho encountered Sujin near the cafe.",
        dialogue_history=dialogue_history,
        profile=AgentProfile(
            fixed=FixedPersona(identity_stable_set=["Jiho helps neighbors."]),
            extended=ExtendedPersona(
                lifestyle_and_routine=["Morning library routine."],
                current_plan_context=["Finish workbook."],
            ),
        ),
        retrieved_memories=retrieved_memories or [],
        language="ko",
    )


def test_decide_reaction_retries_when_first_output_is_repetitive() -> None:
    client = StubOllamaClient(
        responses=[
            _reaction_json(
                should_react=True,
                reaction="안녕하세요 오늘 연습 문제를 마무리하려고 해요",
                reason="first",
            ),
            _reaction_json(
                should_react=True,
                reaction="수진 씨, 디카프 테스트 반응은 어땠나요?",
                reason="retry",
            ),
        ]
    )
    service = LlmService(client)

    decision = service.decide_reaction(
        _input(
            dialogue_history=[
                ("안녕하세요 오늘 연습 문제를 마무리하려고 해요", "none"),
            ]
        )
    )

    assert decision.reaction == "수진 씨, 디카프 테스트 반응은 어땠나요?"
    assert client.calls == 2


def test_decide_reaction_uses_first_output_when_not_repetitive() -> None:
    client = StubOllamaClient(
        responses=[
            _reaction_json(
                should_react=True,
                reaction="수진 씨, 이번 주 테스트한 블렌드는 어땠어요?",
                reason="ok",
            )
        ]
    )
    service = LlmService(client)

    decision = service.decide_reaction(
        _input(
            dialogue_history=[
                ("안녕하세요", "none"),
            ]
        )
    )

    assert decision.reaction == "수진 씨, 이번 주 테스트한 블렌드는 어땠어요?"
    assert client.calls == 1


def test_decide_reaction_semantic_retry_when_embedding_similarity_is_high() -> None:
    client = StubOllamaClient(
        responses=[
            _reaction_json(
                should_react=True,
                reaction="같은 말 반복",
                reason="first",
            ),
            _reaction_json(
                should_react=True,
                reaction="다른 반응으로 바꿔볼게요",
                reason="retry",
            ),
        ]
    )
    embedding = StubEmbeddingEncoder(
        vectors={
            "같은 말 반복": [1.0, 0.0],
            "다른 반응으로 바꿔볼게요": [0.0, 1.0],
            "같은 말": [1.0, 0.0],
        }
    )
    service = LlmService(client, embedding_encoder=embedding)

    decision = service.decide_reaction(
        _input(dialogue_history=[("partner", "같은 말")])
    )

    assert client.calls == 2
    assert decision.reaction == "다른 반응으로 바꿔볼게요"
    assert decision.trace.semantic_retry_count == 1


def test_decide_reaction_parses_thought_critique_and_utterance() -> None:
    client = StubOllamaClient(
        responses=[
            _reaction_json(
                should_react=True,
                reaction="수진 씨, 테스트한 디카프 반응은 어땠어요?",
                reason="short",
                thought="상대 근황 확인과 카페 맥락 연결",
                critique="반복 인사를 피하고 구체 질문으로 시작",
                use_utterance_field=True,
            )
        ]
    )
    service = LlmService(client)

    decision = service.decide_reaction(_input(dialogue_history=[]))

    assert decision.should_react is True
    assert decision.reaction == "수진 씨, 테스트한 디카프 반응은 어땠어요?"
    assert decision.thought == "상대 근황 확인과 카페 맥락 연결"
    assert decision.critique == "반복 인사를 피하고 구체 질문으로 시작"
    assert decision.trace.parse_success is True


def test_decide_reaction_records_parse_failure_trace() -> None:
    client = StubOllamaClient(responses=["not-json"])
    service = LlmService(client)

    decision = service.decide_reaction(_input(dialogue_history=[]))

    assert decision.should_react is False
    assert decision.trace.parse_success is False
    assert decision.trace.fallback_reason == "parse_failure"


def test_decide_reaction_retries_once_for_partner_utterance_when_silent() -> None:
    client = StubOllamaClient(
        responses=[
            _reaction_json(should_react=False, reaction="", reason="first_decline"),
            _reaction_json(
                should_react=True,
                reaction="좋아요, 방금 이야기해준 블렌드가 궁금해요.",
                reason="respond_after_nudge",
            ),
        ]
    )
    service = LlmService(client)

    decision = service.decide_reaction(
        _input(dialogue_history=[("수진 씨, 오늘 테스트 어땠어요?", "none")])
    )

    assert client.calls == 2
    assert decision.should_react is True
    assert decision.trace.partner_retry_count == 1


def test_decide_reaction_repairs_truncated_json_once() -> None:
    client = StubOllamaClient(
        responses=['{"should_react": true, "utterance": "좋아요", "reason": "ok"']
    )
    service = LlmService(client)

    decision = service.decide_reaction(_input(dialogue_history=[]))

    assert decision.should_react is True
    assert decision.reaction == "좋아요"
    assert decision.trace.parse_success is True
    assert decision.trace.parse_error == "repaired_once"


def test_reaction_prompt_includes_concise_utterance_constraint() -> None:
    request = _input(dialogue_history=[])
    prompt = build_reaction_decision_prompt(
        agent_identity=request.agent_identity,
        current_time=request.current_time,
        observation_content=request.observation_content,
        dialogue_history=request.dialogue_history,
        profile=request.profile,
        retrieved_memories=request.retrieved_memories,
    )

    assert "Keep utterance concise and short" in prompt
    assert "80 Korean characters" in prompt


def test_reaction_prompt_includes_few_shot_and_reflection_anchor() -> None:
    reflection_memory = MemoryObject(
        id=1,
        node_type=NodeType.REFLECTION,
        citations=None,
        content="상대의 요청이 원칙과 충돌하면 정중히 거절한다.",
        created_at=datetime.datetime(2026, 2, 27, 13, 0, 0),
        last_accessed_at=datetime.datetime(2026, 2, 27, 13, 0, 0),
        importance=8,
        embedding=np.asarray([0.1, 0.2], dtype=np.float32),
    )

    request = _input(
        dialogue_history=[],
        retrieved_memories=[reflection_memory],
    )
    prompt = build_reaction_decision_prompt(
        agent_identity=request.agent_identity,
        current_time=request.current_time,
        observation_content=request.observation_content,
        dialogue_history=request.dialogue_history,
        profile=request.profile,
        retrieved_memories=request.retrieved_memories,
    )

    assert "[Identity Anchor - highest priority]" in prompt
    assert "Few-shot calibration examples" in prompt
    assert "polite refusal" in prompt
