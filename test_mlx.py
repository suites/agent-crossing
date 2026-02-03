"""
MLX 기반 LLM 추론 속도 및 한국어 품질 테스트
M1 Pro 최적화 가속 및 4-bit 양자화 모델 검증
"""

import time
from mlx_lm import load, generate


def test_qwen_mlx_inference():
    """Qwen 2.5-3B-Instruct-4bit 모델 추론 테스트 (MLX)"""

    print("=" * 60)
    print("Qwen 2.5-3B-Instruct-4bit 모델 로드 중 (MLX)...")
    print("=" * 60)

    model_name = "mlx-community/Qwen2.5-3B-Instruct-4bit"

    load_start = time.time()
    model, tokenizer = load(model_name)
    load_time = time.time() - load_start

    print(f"✓ 모델 로드 완료: {load_time:.2f}초 (Device: MLX/Metal)\n")

    test_prompts = [
        {
            "name": "마을 주민 (날씨)",
            "prompt": "당신은 친절한 마을 주민입니다. 오늘 날씨에 대해 이야기해주세요.",
        },
        {
            "name": "카페 주인 (대화)",
            "prompt": "당신은 작은 마을 카페의 주인입니다. 손님에게 오늘의 추천 메뉴를 소개해주세요.",
        },
        {
            "name": "도서관 사서 (기억)",
            "prompt": "당신은 마을 도서관의 사서입니다. 어제 만난 사람에 대해 기억을 떠올려보세요.",
        },
    ]

    results = []

    print("=" * 60)
    print("추론 테스트 시작 (MLX)")
    print("=" * 60)

    for i, test in enumerate(test_prompts, 1):
        print(f"\n[테스트 {i}] {test['name']}")
        print(f"프롬프트: {test['prompt']}")
        print("-" * 60)

        messages = [{"role": "user", "content": test["prompt"]}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        start = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=100,
            verbose=False,
        )
        elapsed = time.time() - start

        print(f"응답 시간: {elapsed:.2f}초")
        print(f"응답:\n{response.strip()}")

        results.append(
            {"name": test["name"], "latency": elapsed, "response": response.strip()}
        )

    print("\n" + "=" * 60)
    print("테스트 결과 요약 (MLX)")
    print("=" * 60)

    avg_latency = sum(r["latency"] for r in results) / len(results)

    print(f"\n평균 응답 시간: {avg_latency:.2f}초")
    print(f"목표: <3초")

    if avg_latency < 3.0:
        print(f"✓ 목표 달성! ({avg_latency:.2f}초 < 3초)")
        decision = "수용 가능 (MLX 최적화 성공)"
    else:
        print(f"✗ 목표 미달성 ({avg_latency:.2f}초 >= 3초)")
        decision = "추가 최적화 또는 모델 변경 필요"

    print(f"\n최종 결정: {decision}")

    print("\n개별 테스트 결과:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['name']}: {result['latency']:.2f}초")

    return {"avg_latency": avg_latency, "decision": decision, "results": results}


if __name__ == "__main__":
    try:
        result = test_qwen_mlx_inference()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback

        traceback.print_exc()
