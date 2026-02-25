import datetime

from pgvector.sqlalchemy import Vector
from settings import EMBEDDING_DIMENSION
from sqlalchemy import DateTime, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class VectorMemory(Base):
    __tablename__ = "vector_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    description: Mapped[str] = mapped_column(String, nullable=False)
    importance: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[Vector] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
