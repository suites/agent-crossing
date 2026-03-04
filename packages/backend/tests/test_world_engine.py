import datetime
from dataclasses import dataclass
from typing import Literal, cast

from agents.agent_brain import ActionLoopInput, ActionLoopResult
from agents.sim_agent import SimAgent
from llm.ollama_client import OllamaGenerateOptions
from world.engine import SimulationEngine, SimulationEngineConfig
from world.session import WorldConversationSession


@dataclass
class DummyBrain:
    next_result: ActionLoopResult
    queued: list[str]

    def action_loop(self, input: ActionLoopInput) -> ActionLoopResult:
        _ = input
        return self.next_result

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
class DummyAgent:
    name: str
    profile: object
    brain: DummyBrain


def _engine_config(
    *,
    suppress_repeated_replies: bool = False,
    fallback_on_empty_reply: bool = False,
) -> SimulationEngineConfig:
    return SimulationEngineConfig(
        language=cast(Literal["ko", "en"], "ko"),
        turn_time_step_seconds=45,
        suppress_repeated_replies=suppress_repeated_replies,
        repetition_window=4,
        fallback_on_empty_reply=fallback_on_empty_reply,
        reaction_generation_options=OllamaGenerateOptions(num_predict=128),
    )


def test_step_commits_reply_and_broadcasts_to_partner() -> None:
    speaker = DummyAgent(
        name="Jiho",
        profile=object(),
        brain=DummyBrain(
            next_result=ActionLoopResult(
                current_time=datetime.datetime(2026, 3, 3, 12, 0, 0),
                talk="안녕하세요",
                utterance="안녕하세요",
                thought="반갑게 인사",
                action_summary="react_to_partner",
                reaction_trace={"parse_success": True},
            ),
            queued=[],
        ),
    )
    partner = DummyAgent(
        name="Sujin",
        profile=object(),
        brain=DummyBrain(
            next_result=ActionLoopResult(
                current_time=datetime.datetime(2026, 3, 3, 12, 0, 0),
                talk=None,
            ),
            queued=[],
        ),
    )
    session = WorldConversationSession(
        agents=cast(list[SimAgent], [speaker, partner]),
        dialogue_turn_window=None,
    )
    engine = SimulationEngine(session=session, config=_engine_config())

    result = engine.step(
        turn=1,
        current_time=datetime.datetime(2026, 3, 3, 12, 0, 0),
        speaker=cast(SimAgent, cast(object, speaker)),
        speaking_partner=cast(SimAgent, cast(object, partner)),
    )

    assert result.reply == "안녕하세요"
    assert result.silent_reason == ""
    assert session.history == [("Jiho", "안녕하세요")]
    assert speaker.brain.queued == ["나는 이렇게 말했다: 안녕하세요"]
    assert partner.brain.queued == ["Jiho가 이렇게 말했다: 안녕하세요"]


def test_step_suppresses_repeated_reply_when_policy_enabled() -> None:
    speaker = DummyAgent(
        name="Jiho",
        profile=object(),
        brain=DummyBrain(
            next_result=ActionLoopResult(
                current_time=datetime.datetime(2026, 3, 3, 12, 0, 0),
                talk="안녕하세요",
                utterance="안녕하세요",
                silent_reason="",
                reaction_trace={"parse_success": True},
            ),
            queued=[],
        ),
    )
    partner = DummyAgent(
        name="Sujin",
        profile=object(),
        brain=DummyBrain(
            next_result=ActionLoopResult(
                current_time=datetime.datetime(2026, 3, 3, 12, 0, 0),
                talk=None,
            ),
            queued=[],
        ),
    )
    session = WorldConversationSession(
        agents=cast(list[SimAgent], [speaker, partner]),
        dialogue_turn_window=None,
    )
    session.history.append(("Sujin", "안녕하세요"))
    engine = SimulationEngine(
        session=session,
        config=_engine_config(suppress_repeated_replies=True),
    )

    result = engine.step(
        turn=2,
        current_time=datetime.datetime(2026, 3, 3, 12, 0, 0),
        speaker=cast(SimAgent, cast(object, speaker)),
        speaking_partner=cast(SimAgent, cast(object, partner)),
    )

    assert result.reply == ""
    assert "repeat_echo_suppressed" in result.silent_reason
    assert session.history == [("Sujin", "안녕하세요")]
