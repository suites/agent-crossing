import datetime
import json
from typing import cast

import numpy as np
from agents.agent import AgentIdentity, AgentProfile, ExtendedPersona, FixedPersona
from agents.memory.memory_object import MemoryObject, NodeType
from agents.planning.models import DayPlanItem, HourlyPlanItem
from llm.embedding_encoder import EmbeddingEncodingContext
from llm.guardrails.similarity import EmbeddingEncoder
from llm.governance import ReactionDecisionInput
from llm.llm_gateway import LlmGateway
from llm.prompt_builders import (
    build_day_plan_prompt,
    build_reaction_intent_prompt,
    build_reaction_utterance_prompt,
    build_hourly_plan_prompt,
    build_minute_plan_prompt,
    build_salient_questions_prompt,
)


class StubOllamaClient:
    def __init__(self, responses: list[str]):
        self.responses: list[str] = list(responses)
        self.calls: int = 0
        self.call_kwargs: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> str:
        self.call_kwargs.append(kwargs)
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


def _intent_json(
    *,
    should_react: bool,
    reason: str,
    thought: str = "",
    critique: str = "",
) -> str:
    payload: dict[str, object] = {
        "should_react": should_react,
        "reason": reason,
    }
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
            _intent_json(should_react=True, reason="react"),
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
    service = LlmGateway(client)

    decision = service.decide_reaction(
        _input(
            dialogue_history=[
                ("안녕하세요 오늘 연습 문제를 마무리하려고 해요", "none"),
            ]
        )
    )

    assert decision.reaction == "수진 씨, 디카프 테스트 반응은 어땠나요?"
    assert client.calls == 3


def test_decide_reaction_uses_first_output_when_not_repetitive() -> None:
    client = StubOllamaClient(
        responses=[
            _intent_json(should_react=True, reason="react"),
            _reaction_json(
                should_react=True,
                reaction="수진 씨, 이번 주 테스트한 블렌드는 어땠어요?",
                reason="ok",
            ),
        ]
    )
    service = LlmGateway(client)

    decision = service.decide_reaction(
        _input(
            dialogue_history=[
                ("안녕하세요", "none"),
            ]
        )
    )

    assert decision.reaction == "수진 씨, 이번 주 테스트한 블렌드는 어땠어요?"
    assert client.calls == 2


def test_decide_reaction_semantic_retry_when_embedding_similarity_is_high() -> None:
    client = StubOllamaClient(
        responses=[
            _intent_json(should_react=True, reason="react"),
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
    service = LlmGateway(client, embedding_encoder=cast(EmbeddingEncoder, embedding))

    decision = service.decide_reaction(
        _input(dialogue_history=[("partner", "같은 말")])
    )

    assert client.calls == 3
    assert decision.reaction == "다른 반응으로 바꿔볼게요"
    assert decision.trace.semantic_retry_count == 1


def test_decide_reaction_parses_thought_critique_and_utterance() -> None:
    client = StubOllamaClient(
        responses=[
            _intent_json(
                should_react=True,
                reason="short",
                thought="상대 근황 확인과 카페 맥락 연결",
                critique="반복 인사를 피하고 구체 질문으로 시작",
            ),
            _reaction_json(
                should_react=True,
                reaction="수진 씨, 테스트한 디카프 반응은 어땠어요?",
                reason="utterance_stage",
                use_utterance_field=True,
            ),
        ]
    )
    service = LlmGateway(client)

    decision = service.decide_reaction(_input(dialogue_history=[]))

    assert decision.should_react is True
    assert decision.reaction == "수진 씨, 테스트한 디카프 반응은 어땠어요?"
    assert decision.thought == "상대 근황 확인과 카페 맥락 연결"
    assert decision.critique == "반복 인사를 피하고 구체 질문으로 시작"
    assert decision.trace.parse_success is True


def test_decide_reaction_records_parse_failure_trace() -> None:
    client = StubOllamaClient(responses=["not-json"])
    service = LlmGateway(client)

    decision = service.decide_reaction(_input(dialogue_history=[]))

    assert decision.should_react is False
    assert decision.trace.parse_success is False
    assert decision.trace.fallback_reason == "parse_failure"


def test_decide_reaction_retries_once_for_partner_utterance_when_silent() -> None:
    client = StubOllamaClient(
        responses=[
            _intent_json(should_react=True, reason="react"),
            _reaction_json(should_react=True, reaction="", reason="first_silent"),
            _reaction_json(
                should_react=True,
                reaction="좋아요, 방금 이야기해준 블렌드가 궁금해요.",
                reason="respond_after_nudge",
            ),
        ]
    )
    service = LlmGateway(client)

    decision = service.decide_reaction(
        _input(dialogue_history=[("수진 씨, 오늘 테스트 어땠어요?", "none")])
    )

    assert client.calls == 3
    assert decision.should_react is True
    assert decision.trace.partner_retry_count == 1


def test_decide_reaction_repairs_truncated_json_once() -> None:
    client = StubOllamaClient(
        responses=[
            _intent_json(should_react=True, reason="react"),
            '{"should_react": true, "utterance": "좋아요", "reason": "ok"',
        ]
    )
    service = LlmGateway(client)

    decision = service.decide_reaction(_input(dialogue_history=[]))

    assert decision.should_react is True
    assert decision.reaction == "좋아요"
    assert decision.trace.parse_success is True
    assert decision.trace.parse_error == "repaired_once"


def test_reaction_prompt_includes_concise_utterance_constraint() -> None:
    request = _input(dialogue_history=[])
    prompt = build_reaction_utterance_prompt(
        agent_identity=request.agent_identity,
        current_time=request.current_time,
        observation_content=request.observation_content,
        dialogue_history=request.dialogue_history,
        profile=request.profile,
        retrieved_memories=request.retrieved_memories,
        intent_reason="react",
        intent_thought="",
        intent_critique="",
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
    prompt = build_reaction_utterance_prompt(
        agent_identity=request.agent_identity,
        current_time=request.current_time,
        observation_content=request.observation_content,
        dialogue_history=request.dialogue_history,
        profile=request.profile,
        retrieved_memories=request.retrieved_memories,
        intent_reason="react",
        intent_thought="",
        intent_critique="",
    )

    assert "[Identity Anchor - highest priority]" in prompt
    assert "Few-shot calibration examples" in prompt
    assert "polite refusal" in prompt


def test_reaction_intent_prompt_requests_intent_shape() -> None:
    request = _input(dialogue_history=[])
    prompt = build_reaction_intent_prompt(
        agent_identity=request.agent_identity,
        current_time=request.current_time,
        observation_content=request.observation_content,
        dialogue_history=request.dialogue_history,
        profile=request.profile,
        retrieved_memories=request.retrieved_memories,
    )

    assert "Should [Jiho Park] react to the observation right now?" in prompt
    assert '"should_react": <boolean>' in prompt


def test_day_plan_prompt_contains_persona_and_json_shape() -> None:
    prompt = build_day_plan_prompt(
        agent_name="Eddy Lin",
        age=19,
        innate_traits=["friendly", "outgoing", "hospitable"],
        persona_background=(
            "Eddy Lin is a student at Oak Hill College studying music theory and composition."
        ),
        yesterday_date_text="Tuesday February 12",
        yesterday_summary=(
            "woke up and completed the morning routine at 7:00 am, [...] got ready to sleep around 10 pm."
        ),
        today_date_text="Wednesday February 13",
    )

    assert "Name: Eddy Lin (age: 19)" in prompt
    assert "Innate traits: friendly, outgoing, hospitable" in prompt
    assert "Today is Wednesday February 13" in prompt
    assert "Draft Eddy Lin's structured day plan." in prompt
    assert "Framing reference (for style, not output format):" in prompt
    assert "Return strict JSON only with this exact shape and no extra text:" in prompt
    assert '"items": [' in prompt
    assert '"start_time": "<ISO-8601 datetime>"' in prompt


def test_hourly_plan_prompt_contains_context_and_json_shape() -> None:
    prompt = build_hourly_plan_prompt(
        agent_name="Eddy Lin",
        today_date_text="Wednesday February 13",
        day_plan_items=[
            DayPlanItem(
                start_time=datetime.datetime(2026, 2, 13, 8, 0, 0),
                duration_minutes=120,
                location="Town > Home > Kitchen",
                action_content="Review composition notes and plan",
            ),
            DayPlanItem(
                start_time=datetime.datetime(2026, 2, 13, 10, 0, 0),
                duration_minutes=90,
                location="Town > College > Lab",
                action_content="Attend class",
            ),
        ],
    )

    assert "Convert the day plan into a chronological hourly plan." in prompt
    assert "Return strict JSON only with this exact shape and no extra text:" in prompt
    assert "'items': [" not in prompt
    assert '"items": [' in prompt


def test_minute_plan_prompt_contains_context_and_json_shape() -> None:
    prompt = build_minute_plan_prompt(
        agent_name="Eddy Lin",
        current_time=datetime.datetime(2026, 2, 13, 12, 0, 0),
        hourly_plan_items=[
            HourlyPlanItem(
                start_time=datetime.datetime(2026, 2, 13, 12, 0, 0),
                duration_minutes=60,
                location="Town > Home > Study",
                action_content="Draft project outline",
            ),
        ],
    )

    assert "Generate an executable minute plan for the current phase." in prompt
    assert "duration_minutes" in prompt
    assert "Return strict JSON only with this exact shape and no extra text:" in prompt
    assert '"duration_minutes": <int 5-15>' in prompt


def test_salient_prompt_uses_strict_json_contract_line() -> None:
    memory = MemoryObject(
        id=1,
        node_type=NodeType.OBSERVATION,
        citations=None,
        content="Met classmate at the cafe.",
        created_at=datetime.datetime(2026, 2, 27, 10, 0, 0),
        last_accessed_at=datetime.datetime(2026, 2, 27, 10, 0, 0),
        importance=4,
        embedding=np.asarray([0.1, 0.2], dtype=np.float32),
    )
    prompt = build_salient_questions_prompt(agent_name="Eddy Lin", memories=[memory])

    assert "Return strict JSON only with this exact shape and no extra text:" in prompt
    assert '"questions": [' in prompt


def test_generate_salient_questions_requests_json_format() -> None:
    client = StubOllamaClient(
        responses=[
            json.dumps(
                {
                    "questions": [
                        "What should Eddy focus on first today?",
                        "Who can help Eddy improve the composition?",
                        "Which task gives the highest progress today?",
                    ]
                }
            )
        ]
    )
    service = LlmGateway(client)
    memory = MemoryObject(
        id=2,
        node_type=NodeType.OBSERVATION,
        citations=None,
        content="Eddy wants to spend more hours on composition work.",
        created_at=datetime.datetime(2026, 2, 27, 10, 0, 0),
        last_accessed_at=datetime.datetime(2026, 2, 27, 10, 0, 0),
        importance=5,
        embedding=np.asarray([0.2, 0.3], dtype=np.float32),
    )

    questions = service.generate_salient_high_level_questions(
        agent_name="Eddy Lin",
        memories=[memory],
    )

    assert len(questions) == 3
    assert client.call_kwargs[0].get("format_json") is True


def test_generate_hour_plan_parses_json_items() -> None:
    service = LlmGateway(
        StubOllamaClient(
            responses=[
                json.dumps(
                    {
                        "items": [
                            {
                                "start_time": "2026-02-13T08:00:00",
                                "duration_minutes": 90,
                                "location": "Town > Home > Kitchen",
                                "action_content": "Review composition notes over breakfast.",
                            },
                            {
                                "start_time": "2026-02-13T09:45:00",
                                "duration_minutes": 60,
                                "location": "Town > College > Theory Room",
                                "action_content": "Attend morning music theory class.",
                            },
                        ]
                    }
                )
            ]
        )
    )

    items = service.generate_hour_plan(
        agent_name="Eddy Lin",
        today_date_text="Wednesday February 13",
        day_plan_items=[
            DayPlanItem(
                start_time=datetime.datetime(2026, 2, 13, 8, 0, 0),
                duration_minutes=120,
                location="Town > Home > Kitchen",
                action_content="Review composition notes and plan",
            ),
            DayPlanItem(
                start_time=datetime.datetime(2026, 2, 13, 10, 0, 0),
                duration_minutes=90,
                location="Town > College > Lab",
                action_content="Attend class",
            ),
        ],
    )

    assert len(items) == 2
    assert items[0].duration_minutes == 90
    assert items[0].start_time == datetime.datetime(2026, 2, 13, 8, 0, 0)


def test_generate_hour_plan_retries_once_on_truncated_json() -> None:
    client = StubOllamaClient(
        responses=[
            '{"items": [{"start_time": "2026-02-13T08:00:00", "duration_minutes": 90',
            json.dumps(
                {
                    "items": [
                        {
                            "start_time": "2026-02-13T08:00:00",
                            "duration_minutes": 90,
                            "location": "Town > Home > Kitchen",
                            "action_content": "Review composition notes over breakfast.",
                        },
                        {
                            "start_time": "2026-02-13T09:45:00",
                            "duration_minutes": 60,
                            "location": "Town > College > Theory Room",
                            "action_content": "Attend morning music theory class.",
                        },
                    ]
                }
            ),
        ]
    )
    service = LlmGateway(client)

    items = service.generate_hour_plan(
        agent_name="Eddy Lin",
        today_date_text="Wednesday February 13",
        day_plan_items=[
            DayPlanItem(
                start_time=datetime.datetime(2026, 2, 13, 8, 0, 0),
                duration_minutes=120,
                location="Town > Home > Kitchen",
                action_content="Review composition notes and plan",
            ),
            DayPlanItem(
                start_time=datetime.datetime(2026, 2, 13, 10, 0, 0),
                duration_minutes=90,
                location="Town > College > Lab",
                action_content="Attend class",
            ),
        ],
    )

    assert len(items) == 2
    assert client.calls == 2


def test_generate_hour_plan_returns_empty_after_retry_exhaustion() -> None:
    client = StubOllamaClient(
        responses=[
            "not-json",
            json.dumps(
                {
                    "items": [
                        {
                            "start_time": "bad-time",
                            "duration_minutes": 30,
                            "location": "Town > Home > Kitchen",
                            "action_content": "Wake up",
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "items": [
                        {
                            "start_time": "also-bad",
                            "duration_minutes": 30,
                            "location": "Town > Home > Kitchen",
                            "action_content": "Breakfast",
                        }
                    ]
                }
            ),
        ]
    )
    service = LlmGateway(client)

    items = service.generate_hour_plan(
        agent_name="Eddy Lin",
        today_date_text="Wednesday February 13",
        day_plan_items=[
            DayPlanItem(
                start_time=datetime.datetime(2026, 2, 13, 8, 0, 0),
                duration_minutes=120,
                location="Town > Home > Kitchen",
                action_content="Review composition notes and plan",
            ),
            DayPlanItem(
                start_time=datetime.datetime(2026, 2, 13, 10, 0, 0),
                duration_minutes=90,
                location="Town > College > Lab",
                action_content="Attend class",
            ),
        ],
    )

    assert items == []
    assert client.calls == 3


def test_generate_minute_plan_parses_json_items() -> None:
    service = LlmGateway(
        StubOllamaClient(
            responses=[
                json.dumps(
                    {
                        "items": [
                            {
                                "start_time": "2026-02-13T12:00:00",
                                "duration_minutes": 10,
                                "location": "Town > Home > Study",
                                "action_content": "Review motif variations.",
                            },
                            {
                                "start_time": "2026-02-13T12:10:00",
                                "duration_minutes": 5,
                                "location": "Town > Home > Study",
                                "action_content": "Write transition phrase.",
                            },
                        ]
                    }
                )
            ]
        )
    )

    items = service.generate_minute_plan(
        agent_name="Eddy Lin",
        current_time=datetime.datetime(2026, 2, 13, 12, 0, 0),
        hourly_plan_items=[
            HourlyPlanItem(
                start_time=datetime.datetime(2026, 2, 13, 12, 0, 0),
                duration_minutes=60,
                location="Town > Home > Study",
                action_content="Draft project outline",
            )
        ],
    )

    assert len(items) == 2
    assert items[1].duration_minutes == 5
    assert items[0].action_content == "Review motif variations."


def test_generate_minute_plan_retries_once_on_schema_validation_error() -> None:
    client = StubOllamaClient(
        responses=[
            json.dumps({"items": "invalid-format"}),
            json.dumps(
                {
                    "items": [
                        {
                            "start_time": "2026-02-13T12:00:00",
                            "duration_minutes": 12,
                            "location": "Town > Home > Study",
                            "action_content": "Review motif variations.",
                        }
                    ]
                }
            ),
        ]
    )
    service = LlmGateway(client)

    items = service.generate_minute_plan(
        agent_name="Eddy Lin",
        current_time=datetime.datetime(2026, 2, 13, 12, 0, 0),
        hourly_plan_items=[
            HourlyPlanItem(
                start_time=datetime.datetime(2026, 2, 13, 12, 0, 0),
                duration_minutes=60,
                location="Town > Home > Study",
                action_content="Draft project outline",
            )
        ],
    )

    assert len(items) == 1
    assert items[0].duration_minutes == 12
    assert client.calls == 2


def test_generate_day_plan_parses_json_items() -> None:
    client = StubOllamaClient(
        responses=[
            json.dumps(
                {
                    "items": [
                        {
                            "start_time": "2026-02-13T08:00:00",
                            "duration_minutes": 90,
                            "location": "Town > Home > Kitchen",
                            "action_content": "Review composition notes over breakfast.",
                        },
                        {
                            "start_time": "2026-02-13T09:45:00",
                            "duration_minutes": 60,
                            "location": "Town > College > Theory Room",
                            "action_content": "Attend morning music theory class.",
                        },
                        {
                            "start_time": "2026-02-13T11:00:00",
                            "duration_minutes": 75,
                            "location": "Town > College > Studio",
                            "action_content": "Draft harmonic progression for project.",
                        },
                        {
                            "start_time": "2026-02-13T13:00:00",
                            "duration_minutes": 60,
                            "location": "Town > Cafe > Patio",
                            "action_content": "Meet classmate for feedback session.",
                        },
                        {
                            "start_time": "2026-02-13T15:00:00",
                            "duration_minutes": 80,
                            "location": "Town > Home > Desk",
                            "action_content": "Revise composition and annotate changes.",
                        },
                    ]
                }
            )
        ]
    )
    service = LlmGateway(client)

    items = service.generate_day_plan(
        agent_name="Eddy Lin",
        age=19,
        innate_traits=["friendly", "outgoing", "hospitable"],
        persona_background="Music theory student focusing on composition.",
        yesterday_date_text="Tuesday February 12",
        yesterday_summary="woke up at 7:00 am and got ready to sleep around 10 pm.",
        today_date_text="Wednesday February 13",
    )

    assert len(items) == 5
    assert items[0].action_content == "Review composition notes over breakfast."
    assert items[0].location == "Town > Home > Kitchen"
    assert items[0].start_time == datetime.datetime(2026, 2, 13, 8, 0, 0)


def test_generate_day_plan_returns_empty_on_parse_failure() -> None:
    client = StubOllamaClient(responses=["not-json"])
    service = LlmGateway(client)

    items = service.generate_day_plan(
        agent_name="Eddy Lin",
        age=19,
        innate_traits=["friendly", "outgoing", "hospitable"],
        persona_background="Music theory student focusing on composition.",
        yesterday_date_text="Tuesday February 12",
        yesterday_summary="woke up at 7:00 am and got ready to sleep around 10 pm.",
        today_date_text="Wednesday February 13",
    )

    assert items == []


def test_generate_day_plan_returns_empty_if_too_few_items() -> None:
    client = StubOllamaClient(
        responses=[
            json.dumps(
                {
                    "items": [
                        {
                            "start_time": "2026-02-13T08:00:00",
                            "duration_minutes": 30,
                            "location": "Town > Home > Room",
                            "action_content": "Wake up and brush teeth.",
                        },
                        {
                            "start_time": "2026-02-13T08:40:00",
                            "duration_minutes": 30,
                            "location": "Town > Home > Kitchen",
                            "action_content": "Have breakfast.",
                        },
                        {
                            "start_time": "2026-02-13T09:30:00",
                            "duration_minutes": 50,
                            "location": "Town > Street > Bus Stop",
                            "action_content": "Head to class.",
                        },
                    ]
                }
            )
        ]
    )
    service = LlmGateway(client)

    items = service.generate_day_plan(
        agent_name="Eddy Lin",
        age=19,
        innate_traits=["friendly", "outgoing", "hospitable"],
        persona_background="Music theory student focusing on composition.",
        yesterday_date_text="Tuesday February 12",
        yesterday_summary="woke up at 7:00 am and got ready to sleep around 10 pm.",
        today_date_text="Wednesday February 13",
    )

    assert items == []


def test_generate_day_plan_dedupes_and_truncates_to_max() -> None:
    client = StubOllamaClient(
        responses=[
            json.dumps(
                {
                    "items": [
                        {
                            "start_time": "2026-02-13T08:00:00",
                            "duration_minutes": 90,
                            "location": "Town > Home > Kitchen",
                            "action_content": "Review composition notes over breakfast.",
                        },
                        {
                            "start_time": "2026-02-13T08:00:00",
                            "duration_minutes": 90,
                            "location": "Town > Home > Kitchen",
                            "action_content": "Review composition notes over breakfast. ",
                        },
                        {
                            "start_time": "2026-02-13T09:45:00",
                            "duration_minutes": 60,
                            "location": "Town > College > Theory Room",
                            "action_content": "Attend morning music theory class.",
                        },
                        {
                            "start_time": "2026-02-13T11:00:00",
                            "duration_minutes": 75,
                            "location": "Town > College > Studio",
                            "action_content": "Draft harmonic progression for project.",
                        },
                        {
                            "start_time": "2026-02-13T13:00:00",
                            "duration_minutes": 60,
                            "location": "Town > Cafe > Patio",
                            "action_content": "Meet classmate for feedback session.",
                        },
                        {
                            "start_time": "2026-02-13T15:00:00",
                            "duration_minutes": 80,
                            "location": "Town > Home > Desk",
                            "action_content": "Revise composition and annotate changes.",
                        },
                        {
                            "start_time": "2026-02-13T17:00:00",
                            "duration_minutes": 60,
                            "location": "Town > Cafe > Table",
                            "action_content": "Take lunch with classmate.",
                        },
                        {
                            "start_time": "2026-02-13T19:00:00",
                            "duration_minutes": 30,
                            "location": "Town > Home > Practice Room",
                            "action_content": "Practice instrument for 30 minutes.",
                        },
                        {
                            "start_time": "2026-02-13T20:00:00",
                            "duration_minutes": 40,
                            "location": "Town > Home > Desk",
                            "action_content": "Log today's notes in planner.",
                        },
                    ]
                }
            )
        ]
    )
    service = LlmGateway(client)

    items = service.generate_day_plan(
        agent_name="Eddy Lin",
        age=19,
        innate_traits=["friendly", "outgoing", "hospitable"],
        persona_background="Music theory student focusing on composition.",
        yesterday_date_text="Tuesday February 12",
        yesterday_summary="woke up at 7:00 am and got ready to sleep around 10 pm.",
        today_date_text="Wednesday February 13",
    )

    assert len(items) == 8
    assert items[0].action_content == "Review composition notes over breakfast."
    assert items[1].action_content == "Attend morning music theory class."
    assert items[-1].action_content == "Log today's notes in planner."


def test_generate_day_plan_retries_once_on_truncated_json() -> None:
    client = StubOllamaClient(
        responses=[
            '{"items": [{"start_time": "2026-02-13T08:00:00", "duration_minutes": 20',
            json.dumps(
                {
                    "items": [
                        {
                            "start_time": "2026-02-13T08:00:00",
                            "duration_minutes": 20,
                            "location": "Town > Home > Room",
                            "action_content": "Wake up",
                        },
                        {
                            "start_time": "2026-02-13T08:30:00",
                            "duration_minutes": 30,
                            "location": "Town > Home > Kitchen",
                            "action_content": "Eat breakfast",
                        },
                        {
                            "start_time": "2026-02-13T09:30:00",
                            "duration_minutes": 60,
                            "location": "Town > College > Theory Room",
                            "action_content": "Class",
                        },
                        {
                            "start_time": "2026-02-13T11:00:00",
                            "duration_minutes": 40,
                            "location": "Town > Home > Practice Room",
                            "action_content": "Practice",
                        },
                        {
                            "start_time": "2026-02-13T12:00:00",
                            "duration_minutes": 40,
                            "location": "Town > Home > Desk",
                            "action_content": "Review",
                        },
                        {
                            "start_time": "2026-02-13T13:00:00",
                            "duration_minutes": 30,
                            "location": "Town > Park > Bench",
                            "action_content": "Reflect",
                        },
                    ]
                }
            ),
        ]
    )
    service = LlmGateway(client)

    items = service.generate_day_plan(
        agent_name="Eddy Lin",
        age=19,
        innate_traits=["friendly", "outgoing", "hospitable"],
        persona_background="Music theory student focusing on composition.",
        yesterday_date_text="Tuesday February 12",
        yesterday_summary="woke up at 7:00 am and got ready to sleep around 10 pm.",
        today_date_text="Wednesday February 13",
    )

    assert len(items) == 6
    assert items[0].action_content == "Wake up"
    assert client.calls == 2


def test_generate_day_plan_retries_once_on_schema_validation_error() -> None:
    client = StubOllamaClient(
        responses=[
            json.dumps({"items": "invalid-format"}),
            json.dumps(
                {
                    "items": [
                        {
                            "start_time": "2026-02-13T08:00:00",
                            "duration_minutes": 90,
                            "location": "Town > Home > Kitchen",
                            "action_content": "Review composition notes over breakfast.",
                        },
                        {
                            "start_time": "2026-02-13T09:45:00",
                            "duration_minutes": 60,
                            "location": "Town > College > Theory Room",
                            "action_content": "Attend morning music theory class.",
                        },
                        {
                            "start_time": "2026-02-13T11:00:00",
                            "duration_minutes": 75,
                            "location": "Town > College > Studio",
                            "action_content": "Draft harmonic progression for project.",
                        },
                        {
                            "start_time": "2026-02-13T13:00:00",
                            "duration_minutes": 60,
                            "location": "Town > Cafe > Patio",
                            "action_content": "Meet classmate for feedback session.",
                        },
                        {
                            "start_time": "2026-02-13T15:00:00",
                            "duration_minutes": 80,
                            "location": "Town > Home > Desk",
                            "action_content": "Revise composition and annotate changes.",
                        },
                    ]
                }
            ),
        ]
    )
    service = LlmGateway(client)

    items = service.generate_day_plan(
        agent_name="Eddy Lin",
        age=19,
        innate_traits=["friendly", "outgoing", "hospitable"],
        persona_background="Music theory student focusing on composition.",
        yesterday_date_text="Tuesday February 12",
        yesterday_summary="woke up at 7:00 am and got ready to sleep around 10 pm.",
        today_date_text="Wednesday February 13",
    )

    assert len(items) == 5
    assert client.calls == 2


def test_generate_day_plan_returns_empty_after_retry_exhaustion() -> None:
    client = StubOllamaClient(
        responses=[
            "not-json",
            json.dumps(
                {
                    "items": [
                        {
                            "start_time": "2026-02-13T08:00:00",
                            "duration_minutes": 30,
                            "location": "Town > Home > Room",
                            "action_content": "Too few",
                        },
                        {
                            "start_time": "2026-02-13T08:40:00",
                            "duration_minutes": 20,
                            "location": "Town > Home > Kitchen",
                            "action_content": "items",
                        },
                    ]
                }
            ),
            json.dumps(
                {
                    "items": [
                        {
                            "start_time": "bad-time",
                            "duration_minutes": 20,
                            "location": "Town > Home > Room",
                            "action_content": "Wake",
                        },
                        {
                            "start_time": "bad-time",
                            "duration_minutes": 20,
                            "location": "Town > Home > Room",
                            "action_content": "Eat",
                        },
                        {
                            "start_time": "bad-time",
                            "duration_minutes": 20,
                            "location": "Town > Home > Room",
                            "action_content": "Class",
                        },
                        {
                            "start_time": "bad-time",
                            "duration_minutes": 20,
                            "location": "Town > Home > Room",
                            "action_content": "Practice",
                        },
                        {
                            "start_time": "bad-time",
                            "duration_minutes": 20,
                            "location": "Town > Home > Room",
                            "action_content": "Review",
                        },
                    ]
                }
            ),
        ]
    )
    service = LlmGateway(client)

    items = service.generate_day_plan(
        agent_name="Eddy Lin",
        age=19,
        innate_traits=["friendly", "outgoing", "hospitable"],
        persona_background="Music theory student focusing on composition.",
        yesterday_date_text="Tuesday February 12",
        yesterday_summary="woke up at 7:00 am and got ready to sleep around 10 pm.",
        today_date_text="Wednesday February 13",
    )

    assert items == []
    assert client.calls == 3
