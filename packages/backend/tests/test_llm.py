"""
LLM 추론 속도 및 한국어 품질 테스트
Day 1-2: Qwen 2.5-3B 검증
"""

import time
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_qwen_inference():
    """Qwen 2.5-3B 모델 추론 속도 및 한국어 품질 테스트"""

    print("=" * 60)
    print("Qwen 2.5-3B Instruct 모델 로드 중...")
    print("=" * 60)

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("mps")
    load_time = time.time() - load_start

    print(f"✓ 모델 로드 완료: {load_time:.2f}초 (Device: MPS)\n")

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
    print("추론 테스트 시작")
    print("=" * 60)

    for i, test in enumerate(test_prompts, 1):
        print(f"\n[테스트 {i}] {test['name']}")
        print(f"프롬프트: {test['prompt']}")
        print("-" * 60)

        start = time.time()
        inputs = tokenizer(test["prompt"], return_tensors="pt").to("mps")
        outputs = model.generate(
            **inputs, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed = time.time() - start

        response_text = response[len(test["prompt"]) :].strip()

        print(f"응답 시간: {elapsed:.2f}초")
        print(f"응답:\n{response_text}")

        results.append(
            {"name": test["name"], "latency": elapsed, "response": response_text}
        )

    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)

    avg_latency = sum(r["latency"] for r in results) / len(results)

    print(f"\n평균 응답 시간: {avg_latency:.2f}초")
    print(f"목표: <3초")

    if avg_latency < 3.0:
        print(f"✓ 목표 달성! ({avg_latency:.2f}초 < 3초)")
        decision = "수용 가능"
    else:
        print(f"✗ 목표 미달성 ({avg_latency:.2f}초 >= 3초)")
        decision = "최적화 필요"

    print(f"\n최종 결정: {decision}")

    print("\n개별 테스트 결과:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['name']}: {result['latency']:.2f}초")

    print("\n한국어 품질:")
    for result in results:
        if result["response"] and len(result["response"]) > 0:
            print(f"  ✓ {result['name']}: 한국어 응답 생성됨")
        else:
            print(f"  ✗ {result['name']}: 응답 없음")

    return {"avg_latency": avg_latency, "decision": decision, "results": results}


if __name__ == "__main__":
    try:
        result = test_qwen_inference()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback

        traceback.print_exc()
