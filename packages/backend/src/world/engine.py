import datetime
from dataclasses import dataclass
from typing import Literal

from agents.agent_brain import ActionLoopInput, ActionLoopResult
from agents.sim_agent import SimAgent
from dialogue.reply_policy import apply_reply_policy, recent_replies_for_echo_check
from llm.ollama_client import OllamaGenerateOptions

from .session import (
    WorldConversationSession,
    build_turn_observed_events,
    build_turn_world_context,
)


@dataclass(frozen=True)
class SimulationStepResult:
    now: datetime.datetime
    speaker_name: str
    thought: str
    action_summary: str
    trace: dict[str, object]
    reply: str
    silent_reason: str
    parse_failure: bool


@dataclass(frozen=True)
class SimulationEngineConfig:
    language: Literal["ko", "en"]
    turn_time_step_seconds: int
    suppress_repeated_replies: bool
    repetition_window: int
    fallback_on_empty_reply: bool
    reaction_generation_options: OllamaGenerateOptions


class SimulationEngine:
    def __init__(
        self,
        *,
        session: WorldConversationSession,
        config: SimulationEngineConfig,
    ):
        self.session: WorldConversationSession = session
        self.config: SimulationEngineConfig = config

    def step(
        self,
        *,
        turn: int,
        current_time: datetime.datetime,
        speaker: SimAgent,
        speaking_partner: SimAgent,
    ) -> SimulationStepResult:
        incoming_partner_utterance = self.session.consume_incoming_partner_utterance(
            speaker=speaker
        )
        now = current_time + datetime.timedelta(
            seconds=self.config.turn_time_step_seconds
        )
        action_result = self._run_action_loop(
            turn=turn,
            now=now,
            speaker=speaker,
            speaking_partner=speaking_partner,
            incoming_partner_utterance=incoming_partner_utterance,
        )
        raw_reply = (action_result.utterance or action_result.talk or "").strip()
        recent_replies = recent_replies_for_echo_check(
            session_history=self.session.history,
            window=self.config.repetition_window,
        )
        policy_result = apply_reply_policy(
            raw_reply=raw_reply,
            recent_replies=recent_replies,
            language=self.config.language,
            suppress_repeated_replies=self.config.suppress_repeated_replies,
            fallback_on_empty_reply=self.config.fallback_on_empty_reply,
        )

        trace = dict(action_result.reaction_trace or {})
        parse_failure = not bool(trace.get("parse_success", True))
        if policy_result.suppress_reason:
            trace["suppress_reason"] = policy_result.suppress_reason
        if policy_result.fallback_reason:
            trace["fallback_reason"] = policy_result.fallback_reason

        if not policy_result.reply:
            silent_reason = ",".join(
                reason
                for reason in [
                    action_result.silent_reason,
                    policy_result.suppress_reason,
                ]
                if reason
            )
            if not silent_reason:
                silent_reason = "unknown"
            return SimulationStepResult(
                now=now,
                speaker_name=speaker.name,
                thought=action_result.thought,
                action_summary=action_result.action_summary,
                trace=trace,
                reply="",
                silent_reason=silent_reason,
                parse_failure=parse_failure,
            )

        self.session.commit_speaker_reply(
            speaker=speaker,
            incoming_partner_utterance=incoming_partner_utterance,
            reply=policy_result.reply,
        )
        self.session.broadcast_reply(
            speaker=speaker,
            reply=policy_result.reply,
            now=now,
            language=self.config.language,
        )
        return SimulationStepResult(
            now=now,
            speaker_name=speaker.name,
            thought=action_result.thought,
            action_summary=action_result.action_summary,
            trace=trace,
            reply=policy_result.reply,
            silent_reason="",
            parse_failure=parse_failure,
        )

    def _run_action_loop(
        self,
        *,
        turn: int,
        now: datetime.datetime,
        speaker: SimAgent,
        speaking_partner: SimAgent,
        incoming_partner_utterance: str | None,
    ) -> ActionLoopResult:
        observed_events = build_turn_observed_events(
            language=self.config.language,
            speaker_name=speaker.name,
            partner_name=speaking_partner.name,
            incoming_partner_utterance=incoming_partner_utterance,
        )
        return speaker.brain.action_loop(
            ActionLoopInput(
                current_time=now,
                dialogue_history=self.session.dialogue_context_for(speaker=speaker),
                profile=speaker.profile,
                language=self.config.language,
                world_context=build_turn_world_context(
                    speaker_name=speaker.name,
                    partner_name=speaking_partner.name,
                    turn=turn,
                ),
                observed_entities=[speaking_partner.name],
                observed_events=observed_events,
                reaction_generation_options=self.config.reaction_generation_options,
            )
        )
