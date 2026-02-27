import datetime
from typing import Literal

from agents.sim_agent import SimAgent


class WorldConversationSession:
    def __init__(
        self,
        *,
        agents: list[SimAgent],
        dialogue_turn_window: int,
        language: Literal["ko", "en"],
    ):
        if len(agents) < 2:
            raise ValueError("At least two agents are required")
        if dialogue_turn_window < 1:
            raise ValueError("dialogue_turn_window must be at least 1")

        self.agents = agents
        self.dialogue_turn_window = dialogue_turn_window
        self.language = language
        self.turn_index = 0
        self.history: list[tuple[str, str]] = []
        self.dialogue_history_by_agent: dict[str, list[tuple[str, str]]] = {
            agent.name: [] for agent in agents
        }
        self.incoming_utterances_by_agent: dict[str, list[str]] = {
            agent.name: [] for agent in agents
        }

    def seed_conversation_start_intent(
        self,
        *,
        initiator: SimAgent,
        target: SimAgent,
        now: datetime.datetime,
    ) -> None:
        self._ingest_line(
            initiator,
            self._format_conversation_start_intent(target_name=target.name),
            now,
        )

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
        return self.dialogue_history_by_agent[speaker.name][
            -self.dialogue_turn_window :
        ]

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
    ) -> None:
        for observer in self.agents:
            if observer is speaker:
                self._ingest_line(observer, self._format_self_said(reply), now)
                continue

            self._ingest_line(
                observer, self._format_other_said(speaker.name, reply), now
            )
            self.incoming_utterances_by_agent[observer.name].append(reply)

    def _ingest_line(
        self,
        observer: SimAgent,
        content: str,
        now: datetime.datetime,
    ) -> None:
        observer.brain.queue_observation(
            content=content,
            now=now,
            profile=observer.profile,
        )

    def _format_conversation_start_intent(self, *, target_name: str) -> str:
        if self.language == "ko":
            return f"{target_name}한테 말을 걸려는 행동을 하기로 결정했다."
        return f"I decided to initiate a conversation with {target_name}."

    def _format_self_said(self, reply: str) -> str:
        if self.language == "ko":
            return f"나는 이렇게 말했다: {reply}"
        return f"I said: {reply}"

    def _format_other_said(self, speaker_name: str, reply: str) -> str:
        if self.language == "ko":
            return f"{speaker_name}가 이렇게 말했다: {reply}"
        return f"{speaker_name} said: {reply}"
