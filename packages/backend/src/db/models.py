import datetime

from pgvector.sqlalchemy import Vector
from settings import EMBEDDING_DIMENSION
from sqlalchemy import DateTime, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class VectorMemory(Base):
    """에이전트가 생성한 기억을 벡터와 함께 저장하는 테이블 모델."""

    __tablename__ = "vector_memories"

    """각 메모리 레코드를 고유하게 식별하는 기본 키."""
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    """메모리의 원문 텍스트(검색 및 표시 기준 설명)."""
    description: Mapped[str] = mapped_column(String, nullable=False)

    """메모리 중요도 점수(회상 우선순위 계산에 사용)."""
    importance: Mapped[int] = mapped_column(Integer, nullable=False)

    """메모리 임베딩 벡터(pgvector, 차원은 EMBEDDING_DIMENSION)."""
    embedding: Mapped[Vector] = mapped_column(
        Vector(EMBEDDING_DIMENSION), nullable=False
    )

    """메모리가 생성된 시각(서버 기준 현재 시각으로 자동 기록)."""
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
