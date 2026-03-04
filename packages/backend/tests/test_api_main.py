import datetime
from dataclasses import dataclass

import pytest
from fastapi import HTTPException
from metrics.conversation_metrics import ConversationMetrics
from world.engine import SimulationStepResult

from api.main import _require_runtime, app, get_world_state, post_world_step


@dataclass(frozen=True)
class DummyState:
    turn: int
    current_time: datetime.datetime
    parse_failures: int
    silent_turns: int
    history_size: int


@dataclass
class DummyAgent:
    name: str


@dataclass
class DummyRuntime:
    turn: int
    agents: list[DummyAgent]
    _state: DummyState
    _step: SimulationStepResult
    _metrics: ConversationMetrics

    def state(self) -> DummyState:
        return self._state

    def step(self) -> SimulationStepResult:
        self.turn += 1
        return self._step

    def metrics(self) -> ConversationMetrics:
        return self._metrics


def test_require_runtime_raises_when_unavailable() -> None:
    app.state.world_runtime = None

    with pytest.raises(HTTPException):
        _ = _require_runtime()


@pytest.mark.anyio
async def test_get_world_state_returns_runtime_snapshot() -> None:
    runtime = DummyRuntime(
        turn=3,
        agents=[DummyAgent(name="Jiho"), DummyAgent(name="Sujin")],
        _state=DummyState(
            turn=3,
            current_time=datetime.datetime(2026, 3, 4, 10, 30, 0),
            parse_failures=1,
            silent_turns=1,
            history_size=2,
        ),
        _step=SimulationStepResult(
            now=datetime.datetime(2026, 3, 4, 10, 31, 0),
            speaker_name="Jiho",
            thought="hi",
            action_summary="react",
            trace={"parse_success": True},
            reply="안녕하세요",
            silent_reason="",
            parse_failure=False,
        ),
        _metrics=ConversationMetrics(
            parse_failure_rate=0.1,
            silent_rate=0.1,
            semantic_repeat_rate=0.2,
            topic_progress_rate=0.8,
        ),
    )
    app.state.world_runtime = runtime

    response = await get_world_state()

    assert response.turn == 3
    assert response.history_size == 2
    assert response.agent_names == ["Jiho", "Sujin"]


@pytest.mark.anyio
async def test_post_world_step_returns_metrics_and_trace() -> None:
    runtime = DummyRuntime(
        turn=0,
        agents=[DummyAgent(name="Jiho"), DummyAgent(name="Sujin")],
        _state=DummyState(
            turn=0,
            current_time=datetime.datetime(2026, 3, 4, 10, 30, 0),
            parse_failures=0,
            silent_turns=0,
            history_size=0,
        ),
        _step=SimulationStepResult(
            now=datetime.datetime(2026, 3, 4, 10, 31, 0),
            speaker_name="Jiho",
            thought="greet",
            action_summary="react_to_partner",
            trace={"parse_success": True},
            reply="안녕하세요",
            silent_reason="",
            parse_failure=False,
        ),
        _metrics=ConversationMetrics(
            parse_failure_rate=0.0,
            silent_rate=0.0,
            semantic_repeat_rate=0.0,
            topic_progress_rate=1.0,
        ),
    )
    app.state.world_runtime = runtime

    response = await post_world_step()

    assert response.turn == 1
    assert response.speaker_name == "Jiho"
    assert response.reply == "안녕하세요"
    assert response.parse_failure_rate == 0.0
