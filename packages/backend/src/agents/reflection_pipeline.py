import datetime
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from memory.memory_object import MemoryObject, NodeType
from memory.memory_stream import MemoryStream


@dataclass(frozen=True)
class ReflectionConfig:
    threshold: int = 150
    recent_memory_window: int = 100
    question_count: int = 3
    insight_count: int = 5
    retrieval_top_k_per_question: int = 5


class SalientQuestionGenerator(Protocol):
    def generate(self, memories: list[MemoryObject], count: int) -> list[str]: ...


class QuestionMemoryRetriever(Protocol):
    def retrieve(
        self,
        *,
        question: str,
        memories: list[MemoryObject],
        top_k: int,
    ) -> list[MemoryObject]: ...


class InsightGenerator(Protocol):
    def generate(
        self,
        *,
        questions: list[str],
        retrieved_by_question: dict[str, list[MemoryObject]],
        count: int,
    ) -> list[str]: ...


class CitationLinker(Protocol):
    def link(
        self,
        *,
        insights: list[str],
        retrieved_by_question: dict[str, list[MemoryObject]],
    ) -> list[list[int]]: ...


class ReflectionRolloverPolicy(Protocol):
    def rollover(self, *, accumulated_importance_before_run: int) -> int: ...


class PlaceholderQuestionGenerator:
    def generate(self, memories: list[MemoryObject], count: int) -> list[str]:
        return []


class PlaceholderRetriever:
    def retrieve(
        self,
        *,
        question: str,
        memories: list[MemoryObject],
        top_k: int,
    ) -> list[MemoryObject]:
        return []


class PlaceholderInsightGenerator:
    def generate(
        self,
        *,
        questions: list[str],
        retrieved_by_question: dict[str, list[MemoryObject]],
        count: int,
    ) -> list[str]:
        return []


class PlaceholderCitationLinker:
    def link(
        self,
        *,
        insights: list[str],
        retrieved_by_question: dict[str, list[MemoryObject]],
    ) -> list[list[int]]:
        return [[] for _ in insights]


class ResetToZeroRolloverPolicy:
    def rollover(self, *, accumulated_importance_before_run: int) -> int:
        return 0


class ReflectionPipelineService:
    def __init__(
        self,
        *,
        memory_stream: MemoryStream,
        config: ReflectionConfig | None = None,
        question_generator: SalientQuestionGenerator | None = None,
        retriever: QuestionMemoryRetriever | None = None,
        insight_generator: InsightGenerator | None = None,
        citation_linker: CitationLinker | None = None,
        rollover_policy: ReflectionRolloverPolicy | None = None,
    ):
        self.memory_stream = memory_stream
        self.config = config or ReflectionConfig()
        self.question_generator = question_generator or PlaceholderQuestionGenerator()
        self.retriever = retriever or PlaceholderRetriever()
        self.insight_generator = insight_generator or PlaceholderInsightGenerator()
        self.citation_linker = citation_linker or PlaceholderCitationLinker()
        self.rollover_policy = rollover_policy or ResetToZeroRolloverPolicy()

        self.accumulated_importance: int = 0

    def record_observation_importance(self, importance: int) -> None:
        """
        메모 노트:
        - 의도: 관찰 이벤트의 중요도를 누적해 reflection 트리거 판단 입력값으로 쌓는다.
        - 입력: importance(정수, 일반적으로 1~10)
        - 출력: 없음(내부 상태 accumulated_importance 갱신)
        - TODO: 에이전트별/세션별 분리 카운터 저장소로 확장
        - 엣지케이스: 음수 값 입력 시에도 현재는 그대로 누적하므로, 향후 clamp 정책 검토 필요
        """
        self.accumulated_importance += importance

    def check_reflection_trigger(self) -> bool:
        """
        메모 노트:
        - 의도: 현재 누적 중요도가 reflection 임계치를 넘었는지 확인한다.
        - 입력: 없음(내부 상태와 config.threshold 사용)
        - 출력: bool (True면 reflection 실행 후보)
        - TODO: 최근 시간창 기반 누적(예: last N hours) 규칙으로 교체 가능
        - 엣지케이스: threshold가 0 이하인 설정이면 항상 True가 될 수 있음
        """
        return self.accumulated_importance >= self.config.threshold

    def generate_salient_questions(self, memories: list[MemoryObject]) -> list[str]:
        """
        메모 노트:
        - 의도: 최근 기억에서 고수준 성찰 질문 생성 훅을 호출한다.
        - 입력: 최근 메모리 리스트
        - 출력: 질문 문자열 리스트
        - TODO: LLM 프롬프트/페르소나/현재 계획을 context로 포함
        - 엣지케이스: 메모리가 비어도 빈 리스트를 허용하고 파이프라인을 중단하지 않음
        """
        return self.question_generator.generate(memories, self.config.question_count)

    def retrieve_memories_per_question(
        self,
        *,
        questions: list[str],
        memories: list[MemoryObject],
    ) -> dict[str, list[MemoryObject]]:
        """
        메모 노트:
        - 의도: 질문별로 관련 메모리 검색 훅을 호출한다.
        - 입력: questions, 검색 대상 memories
        - 출력: {question: [MemoryObject, ...]} 매핑
        - TODO: MemoryStream.retrieve와 임베딩 기반 검색 어댑터 연결
        - 엣지케이스: 질문이 중복되면 마지막 키가 덮어쓸 수 있어 향후 식별자 키 분리 고려
        """
        result: dict[str, list[MemoryObject]] = {}
        for question in questions:
            result[question] = self.retriever.retrieve(
                question=question,
                memories=memories,
                top_k=self.config.retrieval_top_k_per_question,
            )
        return result

    def generate_insights(
        self,
        *,
        questions: list[str],
        retrieved_by_question: dict[str, list[MemoryObject]],
    ) -> list[str]:
        """
        메모 노트:
        - 의도: 질문/근거기억 묶음을 바탕으로 인사이트 생성 훅을 호출한다.
        - 입력: questions, retrieved_by_question
        - 출력: 인사이트 문자열 리스트
        - TODO: insight 품질 검증(중복 제거, 추상화 수준 체크) 추가
        - 엣지케이스: 질문은 있으나 검색 결과가 비어도 빈/저신뢰 인사이트 허용
        """
        return self.insight_generator.generate(
            questions=questions,
            retrieved_by_question=retrieved_by_question,
            count=self.config.insight_count,
        )

    def link_citations(
        self,
        *,
        insights: list[str],
        retrieved_by_question: dict[str, list[MemoryObject]],
    ) -> list[list[int]]:
        """
        메모 노트:
        - 의도: 인사이트별 근거 memory id 연결 훅을 호출한다.
        - 입력: insights, retrieved_by_question
        - 출력: insight 순서와 정렬된 citation id 리스트 목록
        - TODO: 질문-인사이트 매핑 정보를 분리 보관하여 traceability 강화
        - 엣지케이스: 길이 불일치(인사이트 수 != citation 목록 수) 시 저장 단계에서 보정 필요
        """
        return self.citation_linker.link(
            insights=insights,
            retrieved_by_question=retrieved_by_question,
        )

    def apply_post_reflection_rollover_policy(self) -> int:
        """
        메모 노트:
        - 의도: reflection 실행 이후 누적 중요도 카운터의 리셋/이월 정책 훅을 적용한다.
        - 입력: 없음(내부 accumulated_importance 사용)
        - 출력: 갱신된 accumulated_importance 값
        - TODO: 완전 리셋 외에 일부 이월(예: threshold 초과분 유지) 정책 도입
        - 엣지케이스: 정책이 음수 반환 시 방어 처리 필요(현재는 그대로 반영)
        """
        self.accumulated_importance = self.rollover_policy.rollover(
            accumulated_importance_before_run=self.accumulated_importance
        )
        return self.accumulated_importance

    def run_reflection(self, *, now: datetime.datetime) -> list[MemoryObject]:
        """
        메모 노트:
        - 의도: reflection 파이프라인 스켈레톤을 순서대로 실행하고 REFLECTION 메모리를 저장한다.
        - 입력: now(저장 타임스탬프)
        - 출력: 생성된 REFLECTION MemoryObject 리스트
        - TODO: 실제 임베딩 생성, 질문-인사이트 정합성 검증, 예외 로깅/복구 추가
        - 엣지케이스: 훅이 모두 빈 결과를 반환해도 실행은 정상 종료되고 빈 리스트 반환
        """
        recent_memories = self.memory_stream.memories[-self.config.recent_memory_window :]
        questions = self.generate_salient_questions(recent_memories)
        retrieved_by_question = self.retrieve_memories_per_question(
            questions=questions,
            memories=recent_memories,
        )
        insights = self.generate_insights(
            questions=questions,
            retrieved_by_question=retrieved_by_question,
        )
        citations_by_insight = self.link_citations(
            insights=insights,
            retrieved_by_question=retrieved_by_question,
        )

        created: list[MemoryObject] = []
        for idx, insight in enumerate(insights):
            citations = citations_by_insight[idx] if idx < len(citations_by_insight) else []
            self.memory_stream.add_memory(
                node_type=NodeType.REFLECTION,
                citations=citations,
                content=insight,
                now=now,
                importance=5,
                embedding=self._build_placeholder_embedding(recent_memories),
            )
            created.append(self.memory_stream.memories[-1])

        self.apply_post_reflection_rollover_policy()
        return created

    def _build_placeholder_embedding(self, memories: list[MemoryObject]) -> np.ndarray:
        if memories:
            return np.zeros_like(memories[0].embedding)
        return np.zeros(1)
