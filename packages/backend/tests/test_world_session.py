from dataclasses import dataclass
from typing import cast

import pytest
from agents.sim_agent import SimAgent
from world.session import WorldConversationSession


@dataclass(frozen=True)
class DummyAgent:
    name: str


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
