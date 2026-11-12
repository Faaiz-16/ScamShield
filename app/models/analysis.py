"""Stored analyses and user feedback.

Only anonymised data is stored: message text with personal details replaced,
link domains (never full URLs), and keyed hashes of client identifiers.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base, utcnow

MYSQL_TABLE_ARGS: dict[str, Any] = {
    "mysql_engine": "InnoDB",
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_unicode_ci",
}


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (MYSQL_TABLE_ARGS,)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Exposed to clients instead of the sequential id, so analyses cannot be enumerated.
    public_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    channel: Mapped[str] = mapped_column(String(20))
    client_hash: Mapped[str] = mapped_column(String(64), index=True)

    anonymized_text: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(10))
    response_language: Mapped[str] = mapped_column(String(5))

    verdict: Mapped[str] = mapped_column(String(20), index=True)
    risk_score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    scam_type: Mapped[str | None] = mapped_column(String(40), index=True, nullable=True)

    rule_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    link_domains: Mapped[list[str]] = mapped_column(JSON, default=list)
    model_name: Mapped[str | None] = mapped_column(String(40), nullable=True)
    scam_probability: Mapped[float | None] = mapped_column(Float, nullable=True)

    feedback: Mapped[list["Feedback"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan", passive_deletes=True
    )


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (UniqueConstraint("analysis_id", "client_hash"), MYSQL_TABLE_ARGS)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    client_hash: Mapped[str] = mapped_column(String(64))
    is_correct: Mapped[bool] = mapped_column(Boolean)
    # The label the user believes is right, when they say the verdict was wrong.
    correct_label: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    analysis: Mapped[Analysis] = relationship(back_populates="feedback")
