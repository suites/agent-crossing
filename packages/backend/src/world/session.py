import datetime
from typing import Literal

from agents.sim_agent import SimAgent


def build_turn_world_context(
    *, speaker_name: str, partner_name: str, turn: int
) -> dict[str, str]:
    locations = [
        "town square",
        "cafe entrance",
        "library walkway",
        "park bench",
    ]
    location = locations[(turn - 1) % len(locations)]
    return {
        "location": f"{location} near {partner_name}",
        "focus": f"{speaker_name} is facing {partner_name}",
    }


def build_turn_observed_events(
    *,
    language: Literal["ko", "en"],
    speaker_name: str,
    partner_name: str,
    incoming_partner_utterance: str | None,
) -> list[str]:
    if incoming_partner_utterance and incoming_partner_utterance.strip():
        if language == "ko":
            return [f"{partner_name}의 직전 발화를 들음: {incoming_partner_utterance}"]
        return [
            f"Heard {partner_name}'s latest utterance: {incoming_partner_utterance}"
        ]

    if language == "ko":
        return [f"{speaker_name}가 {partner_name}를 근처에서 마주쳤다."]
    return [f"{speaker_name} encountered {partner_name} nearby."]


def _format_self_said(language: Literal["ko", "en"], reply: str) -> str:
    if language == "ko":
        return f"나는 이렇게 말했다: {reply}"
    return f"I said: {reply}"


def _format_other_said(
    language: Literal["ko", "en"], speaker_name: str, reply: str
) -> str:
    if language == "ko":
        return f"{speaker_name}가 이렇게 말했다: {reply}"
    return f"{speaker_name} said: {reply}"


class WorldConversationSession:
    def __init__(
        self,
        *,
        agents: list[SimAgent],
        dialogue_turn_window: int | None,
    ):
        if len(agents) < 2:
            raise ValueError("At least two agents are required")
        if dialogue_turn_window is not None and dialogue_turn_window < 1:
            raise ValueError("dialogue_turn_window must be at least 1")

        self.agents = agents
        self.dialogue_turn_window = dialogue_turn_window
        self.turn_index = 0
        self.history: list[tuple[str, str]] = []
        self.dialogue_history_by_agent: dict[str, list[tuple[str, str]]] = {
            agent.name: [] for agent in agents
        }
        self.incoming_utterances_by_agent: dict[str, list[str]] = {
            agent.name: [] for agent in agents
        }

    def next_speaker(self) -> SimAgent:
        speaker = self.agents[self.turn_index % len(self.agents)]
        self.turn_index += 1
        return speaker

    def consume_incoming_partner_utterance(
        self,
        *,
        speaker: SimAgent,
    ) -> str | None:
        incoming_queue = self.incoming_utterances_by_agent[speaker.name]
        if not incoming_queue:
            return None

        incoming_partner_utterance = incoming_queue.pop(0)
        self.dialogue_history_by_agent[speaker.name].append(
            (incoming_partner_utterance, "")
        )
        return incoming_partner_utterance

    def dialogue_context_for(
        self,
        *,
        speaker: SimAgent,
    ) -> list[tuple[str, str]]:
        history = self.dialogue_history_by_agent[speaker.name]
        if self.dialogue_turn_window is None:
            return history
        return history[-self.dialogue_turn_window :]

    def commit_speaker_reply(
        self,
        *,
        speaker: SimAgent,
        incoming_partner_utterance: str | None,
        reply: str,
    ) -> None:
        if incoming_partner_utterance is not None:
            self.dialogue_history_by_agent[speaker.name][-1] = (
                incoming_partner_utterance,
                reply,
            )
        else:
            self.dialogue_history_by_agent[speaker.name].append(("", reply))

        self.history.append((speaker.name, reply))

    def broadcast_reply(
        self,
        *,
        speaker: SimAgent,
        reply: str,
        now: datetime.datetime,
        language: Literal["ko", "en"],
    ) -> None:
        for observer in self.agents:
            if observer is speaker:
                observer.brain.queue_observation(
                    content=_format_self_said(language, reply),
                    now=now,
                    profile=observer.profile,
                )
                continue

            observer.brain.queue_observation(
                content=_format_other_said(language, speaker.name, reply),
                now=now,
                profile=observer.profile,
            )
            self.incoming_utterances_by_agent[observer.name].append(reply)
