import datetime
from dataclasses import dataclass
from typing import cast

import pytest
from agents.sim_agent import SimAgent
from world.session import (
    WorldConversationSession,
    build_turn_observed_events,
    build_turn_world_context,
)


@dataclass(frozen=True)
class DummyAgent:
    name: str


@dataclass
class DummyBrain:
    queued: list[str]

    def queue_observation(
        self,
        *,
        content: str,
        now: datetime.datetime,
        profile: object,
    ) -> None:
        _ = now
        _ = profile
        self.queued.append(content)


@dataclass
class DummyInteractiveAgent:
    name: str
    profile: object
    brain: DummyBrain


def test_dialogue_context_for_returns_full_history_when_window_is_none() -> None:
    agents = cast(list[SimAgent], [DummyAgent(name="Jiho"), DummyAgent(name="Sujin")])
    session = WorldConversationSession(agents=agents, dialogue_turn_window=None)

    session.dialogue_history_by_agent["Jiho"] = [
        ("안녕하세요", "안녕하세요"),
        ("오늘 어땠어요?", "도서관에 있었어요."),
    ]

    context = session.dialogue_context_for(speaker=agents[0])

    assert context == [
        ("안녕하세요", "안녕하세요"),
        ("오늘 어땠어요?", "도서관에 있었어요."),
    ]


def test_dialogue_context_for_respects_window_when_configured() -> None:
    agents = cast(list[SimAgent], [DummyAgent(name="Jiho"), DummyAgent(name="Sujin")])
    session = WorldConversationSession(agents=agents, dialogue_turn_window=1)

    session.dialogue_history_by_agent["Jiho"] = [
        ("안녕하세요", "안녕하세요"),
        ("오늘 어땠어요?", "도서관에 있었어요."),
    ]

    context = session.dialogue_context_for(speaker=agents[0])

    assert context == [("오늘 어땠어요?", "도서관에 있었어요.")]


def test_dialogue_turn_window_must_be_positive_if_provided() -> None:
    agents = cast(list[SimAgent], [DummyAgent(name="Jiho"), DummyAgent(name="Sujin")])

    with pytest.raises(ValueError):
        _ = WorldConversationSession(agents=agents, dialogue_turn_window=0)


def test_build_turn_world_context_rotates_locations() -> None:
    context = build_turn_world_context(
        speaker_name="Jiho",
        partner_name="Sujin",
        turn=5,
    )

    assert context["location"] == "town square near Sujin"
    assert context["focus"] == "Jiho is facing Sujin"


def test_build_turn_observed_events_uses_partner_utterance_when_available() -> None:
    events = build_turn_observed_events(
        language="en",
        speaker_name="Jiho",
        partner_name="Sujin",
        incoming_partner_utterance="How was the decaf test?",
    )

    assert events == ["Heard Sujin's latest utterance: How was the decaf test?"]


def test_broadcast_reply_enqueues_observations_and_incoming_queue() -> None:
    speaker = DummyInteractiveAgent(
        name="Jiho",
        profile=object(),
        brain=DummyBrain(queued=[]),
    )
    observer = DummyInteractiveAgent(
        name="Sujin",
        profile=object(),
        brain=DummyBrain(queued=[]),
    )
    agents = cast(list[SimAgent], [speaker, observer])
    session = WorldConversationSession(agents=agents, dialogue_turn_window=None)

    session.broadcast_reply(
        speaker=cast(SimAgent, cast(object, speaker)),
        reply="안녕하세요",
        now=datetime.datetime(2026, 3, 3, 12, 0, 0),
        language="ko",
    )

    assert speaker.brain.queued == ["나는 이렇게 말했다: 안녕하세요"]
    assert observer.brain.queued == ["Jiho가 이렇게 말했다: 안녕하세요"]
    assert session.incoming_utterances_by_agent["Sujin"] == ["안녕하세요"]
