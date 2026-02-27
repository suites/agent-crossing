from run_agent_loop_simulation import (
    _build_turn_observed_events,
    _is_repetitive_reply,
    _recent_replies_for_echo_check,
)


def test_recent_replies_for_echo_check_collects_cross_speaker_history() -> None:
    history = [
        ("Jiho Park", "안녕하세요!"),
        ("Sujin Lee", "안녕하세요!"),
        ("Jiho Park", "책 정리 마무리 중이에요."),
    ]

    replies = _recent_replies_for_echo_check(session_history=history, window=2)

    assert replies == ["안녕하세요!", "책 정리 마무리 중이에요."]


def test_is_repetitive_reply_blocks_cross_speaker_echo() -> None:
    history = [
        ("Jiho Park", "안녕하세요!"),
        ("Sujin Lee", "안녕하세요!"),
    ]
    recent_replies = _recent_replies_for_echo_check(session_history=history, window=4)

    assert _is_repetitive_reply("안녕하세요.", recent_replies) is True


def test_recent_replies_for_echo_check_returns_empty_for_non_positive_window() -> None:
    history = [("Jiho Park", "안녕하세요!")]

    replies = _recent_replies_for_echo_check(session_history=history, window=0)

    assert replies == []


def test_build_turn_observed_events_uses_pure_encounter_on_opening_turn() -> None:
    observed_events = _build_turn_observed_events(
        language="ko",
        speaker_name="Jiho Park",
        partner_name="Sujin Lee",
        incoming_partner_utterance=None,
    )

    assert observed_events == ["Jiho Park가 Sujin Lee를 근처에서 마주쳤다."]


def test_build_turn_observed_events_prefers_latest_utterance_when_present() -> None:
    observed_events = _build_turn_observed_events(
        language="ko",
        speaker_name="Jiho Park",
        partner_name="Sujin Lee",
        incoming_partner_utterance="새로운 디카프 블렌드 테스트 중이에요.",
    )

    assert observed_events == [
        "Sujin Lee의 직전 발화를 들음: 새로운 디카프 블렌드 테스트 중이에요."
    ]
