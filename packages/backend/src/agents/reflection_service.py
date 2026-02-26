from agents.reflection import Reflection
from llm.llm_service import LlmService

from .memory.memory_object import MemoryObject
from .memory.memory_service import MemoryService


class ReflectionService:
    """아주 최소한의 reflection 시작점(스캐폴딩 전용)."""

    def __init__(
        self,
        reflection: Reflection,
        memory_service: MemoryService,
        llm_service: LlmService,
    ):
        self.reflection = reflection
        self.memory_service = memory_service
        self.llm_service = llm_service

    def record_observation_importance(self, importance: int) -> None:
        """
        중요도 카운터를 기록한다.
        - 입력/출력: importance(int) -> None
        """
        self.reflection.record_observation_importance(importance=importance)

    def should_reflect(self) -> bool:
        """
        reflect 실행 여부를 판단한다.
        - 입력/출력: None -> bool
        """
        return self.reflection.should_reflect()

    def reflect(self) -> None:
        """
        reflect를 실행하여 reflection 결과를 반환한다.
        - 입력/출력: now(datetime) -> list[MemoryObject] (현재는 빈 리스트)
        """

        agent_name = "temp"  # TODO: agent 이름을 받아오는 로직 필요
        # 2. 최근 100건의 메모리를 가져온다.
        memories: list[MemoryObject] = self.memory_service.get_recent_memories(
            limit=100
        )

        # 3. LLM에 reflection 프롬프트를 던지고, 결과를 받아온다.
        questions = self.llm_service.generate_salient_high_level_questions(
            agent_name=agent_name, memories=memories
        )

        # 4. 질문별로 전체 메모리 스트림에 Query해서 관련성이 높은 기억들을 뽑아온다.
        for question in questions:
            relation_questions: list[MemoryObject] = (
                self.memory_service.get_retrieval_memories(query=question)
            )

            # 5. 가져온 기억들을 가지고 다시 LLM에 질문해서 insight 5가지를 얻는다.
            insights = self.llm_service.generate_insights_with_citation_key(
                agent_name=agent_name, memories=relation_questions
            )

            # 6. 얻은 통찰들과 인용 키들을 외래키로 해서 메모리스트림에 성찰로 반환한다.
            for insight in insights:
                self.memory_service.create_reflection(insight)

        # 7. 카운터를 0으로 초기화한다.
        self.reflection.clear_importance()
