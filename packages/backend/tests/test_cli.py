from agents.agent_brain import AgentBrain
from dotenv import load_dotenv

load_dotenv()


def run_cli_poc():
    print("Agent Crossing: CLI POC 시작")
    print("-" * 30)

    # 에이전트 초기화
    jiho = AgentBrain(
        agent_id="jiho",
        name="지호",
        persona="당신은 조용한 책벌레입니다. 마을 도서관에서 일하며, 항상 차분하고 예의 바릅니다.",
    )

    sujin = AgentBrain(
        agent_id="sujin",
        name="수진",
        persona="당신은 활기찬 마을 카페 주인입니다. 사람들과 이야기하는 것을 좋아하고, 항상 긍정적입니다.",
    )

    print("에이전트 로드 완료.\n")

    # 1. 인지 (Perceive)
    jiho.perceive("오늘 아침 도서관 창밖으로 수진이 카페 문을 여는 것을 보았다.")
    sujin.perceive(
        "지호가 도서관으로 걸어가는 것을 보았다. 손에는 두꺼운 책이 들려 있었다."
    )

    # 2. 대화 (Converse)
    print("대화 시뮬레이션:")
    msg1 = "안녕하세요 수진님! 오늘도 일찍 시작하시네요."
    print(f"지호: {msg1}")

    resp1 = sujin.converse("지호", msg1)
    print(f"수진: {resp1}")

    msg2 = "네, 도서관도 곧 열 시간이네요. 오늘 그 책 다 읽으셨어요?"
    print(f"수진: {msg2}")

    resp2 = jiho.converse("수진", msg2)
    print(f"지호: {resp2}")


if __name__ == "__main__":
    try:
        run_cli_poc()
    except Exception as e:
        print(f"오류 발생: {e}")
