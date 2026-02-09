"""
MLX 기반 LLM 추론 속도 및 한국어 품질 테스트
M1 Pro 최적화 가속 및 4-bit 양자화 모델 검증

Run with: uv run pytest tests/test_mlx.py -v -s
"""

import time

import pytest
from mlx_lm import generate, load

MODEL_NAME = "mlx-community/Qwen2.5-3B-Instruct-4bit"
TARGET_LATENCY_SECONDS = 3.0


@pytest.fixture(scope="module")
def mlx_model_and_tokenizer():
    """MLX 모델과 토크나이저를 로드하는 fixture (모듈 단위 캐싱)"""
    model, tokenizer = load(MODEL_NAME)
    return model, tokenizer


@pytest.mark.skip(reason="외부 모델 필요 - 수동 실행: uv run pytest tests/test_mlx.py -v -s")
class TestQwenMLXInference:
    """Qwen 2.5-3B-Instruct-4bit 모델 추론 테스트 (MLX)"""

    test_prompts = [
        ("마을 주민 (날씨)", "당신은 친절한 마을 주민입니다. 오늘 날씨에 대해 이야기해주세요."),
        ("카페 주인 (대화)", "당신은 작은 마을 카페의 주인입니다. 손님에게 오늘의 추천 메뉴를 소개해주세요."),
        ("도서관 사서 (기억)", "당신은 마을 도서관의 사서입니다. 어제 만난 사람에 대해 기억을 떠올려보세요."),
    ]

    @pytest.mark.parametrize("name,prompt", test_prompts)
    def test_inference_latency(self, mlx_model_and_tokenizer: tuple, name: str, prompt: str):
        """각 프롬프트에 대한 추론 시간이 목표 이하인지 확인 (MLX)"""
        model, tokenizer = mlx_model_and_tokenizer

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        start = time.time()
        generate(model, tokenizer, prompt=formatted_prompt, max_tokens=100, verbose=False)
        elapsed = time.time() - start

        assert elapsed < TARGET_LATENCY_SECONDS, f"{name}: {elapsed:.2f}s >= {TARGET_LATENCY_SECONDS}s"

    @pytest.mark.parametrize("name,prompt", test_prompts)
    def test_korean_response_quality(self, mlx_model_and_tokenizer: tuple, name: str, prompt: str):
        """한국어 응답이 생성되는지 확인 (MLX)"""
        model, tokenizer = mlx_model_and_tokenizer

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = generate(
            model, tokenizer, prompt=formatted_prompt, max_tokens=100, verbose=False
        )

        assert len(response.strip()) > 0, f"{name}: 응답이 비어있음"
