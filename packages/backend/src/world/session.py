from agents.sim_agent import SimAgent


class WorldConversationSession:
    def __init__(
        self,
        *,
        agents: list[SimAgent],
        dialogue_turn_window: int,
    ):
        if len(agents) < 2:
            raise ValueError("At least two agents are required")
        if dialogue_turn_window < 1:
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
