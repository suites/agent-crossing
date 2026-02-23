import datetime

from agents.reflection_pipeline import ReflectionConfig, ReflectionPipeline


def test_should_reflect_checks_threshold_minimal_contract() -> None:
    pipeline = ReflectionPipeline(config=ReflectionConfig(threshold=10))

    pipeline.record_observation_importance(4)
    assert pipeline.should_reflect() is False

    pipeline.record_observation_importance(6)
    assert pipeline.should_reflect() is True


def test_run_resets_accumulated_importance_and_returns_empty_template() -> None:
    pipeline = ReflectionPipeline(config=ReflectionConfig(threshold=1))
    pipeline.record_observation_importance(10)

    result = pipeline.run(now=datetime.datetime(2026, 2, 23, 19, 0, 0))

    assert result == []
    assert pipeline.accumulated_importance == 0
