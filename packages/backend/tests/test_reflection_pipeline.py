import datetime

from agents.reflection_service import ReflectionConfig, ReflectionService


def test_should_reflect_checks_threshold_minimal_contract() -> None:
    service = ReflectionService(config=ReflectionConfig(threshold=10))

    service.record_observation_importance(4)
    assert service.should_reflect() is False

    service.record_observation_importance(6)
    assert service.should_reflect() is True


def test_run_resets_accumulated_importance_and_returns_empty_template() -> None:
    service = ReflectionService(config=ReflectionConfig(threshold=1))
    service.record_observation_importance(10)

    result = service.run(now=datetime.datetime(2026, 2, 23, 19, 0, 0))

    assert result == []
    assert service.accumulated_importance == 0
