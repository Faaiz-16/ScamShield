"""Persistence of analyses and feedback, and aggregate statistics."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.core.database import utcnow
from app.models import Analysis, Feedback
from app.services.analysis import AnalysisResult


class AnalysisNotFoundError(LookupError):
    pass


def save_analysis(
    session: Session,
    result: AnalysisResult,
    *,
    channel: str,
    client_hash: str,
    response_language: str,
) -> str:
    verdict = result.verdict
    domains = sorted(
        {r.registered_domain or r.host for r in result.url_reports if r.registered_domain or r.host}
    )
    record = Analysis(
        public_id=str(uuid.uuid4()),
        channel=channel,
        client_hash=client_hash,
        anonymized_text=result.anonymized_text,
        language=result.detected_language,
        response_language=response_language,
        verdict=verdict.level,
        risk_score=verdict.risk_score,
        confidence=verdict.confidence,
        scam_type=verdict.scam_type,
        rule_ids=[hit.rule_id for hit in result.rule_hits],
        link_domains=domains,
        model_name=result.model_name,
        scam_probability=result.prediction.scam_probability if result.prediction else None,
    )
    session.add(record)
    session.commit()
    return record.public_id


def record_feedback(
    session: Session,
    *,
    public_id: str,
    client_hash: str,
    is_correct: bool,
    correct_label: str | None,
) -> bool:
    """Store feedback. Returns True if it replaced earlier feedback from the same client."""
    analysis = session.scalar(select(Analysis).where(Analysis.public_id == public_id))
    if analysis is None:
        raise AnalysisNotFoundError(public_id)

    existing = session.scalar(
        select(Feedback).where(
            Feedback.analysis_id == analysis.id, Feedback.client_hash == client_hash
        )
    )
    label = None if is_correct else correct_label
    if existing:
        existing.is_correct = is_correct
        existing.correct_label = label
    else:
        session.add(
            Feedback(
                analysis_id=analysis.id,
                client_hash=client_hash,
                is_correct=is_correct,
                correct_label=label,
            )
        )
    session.commit()
    return existing is not None


@dataclass
class Stats:
    days: int
    total: int
    verdicts: dict[str, int]
    scam_types: list[tuple[str, int]]
    feedback_correct: int
    feedback_wrong: int
    daily: list[dict[str, Any]]


def get_stats(session: Session, days: int = 30, top_types: int = 10) -> Stats:
    since = utcnow() - timedelta(days=days)
    in_window = Analysis.created_at >= since

    verdict_rows = session.execute(
        select(Analysis.verdict, func.count()).where(in_window).group_by(Analysis.verdict)
    ).all()
    verdicts = {"safe": 0, "suspicious": 0, "likely_scam": 0}
    verdicts.update({verdict: count for verdict, count in verdict_rows})

    type_count = func.count().label("n")
    type_rows = session.execute(
        select(Analysis.scam_type, type_count)
        .where(in_window, Analysis.scam_type.is_not(None), Analysis.verdict != "safe")
        .group_by(Analysis.scam_type)
        .order_by(type_count.desc(), Analysis.scam_type)
        .limit(top_types)
    ).all()

    feedback_rows = session.execute(
        select(Feedback.is_correct, func.count())
        .join(Analysis)
        .where(in_window)
        .group_by(Feedback.is_correct)
    ).all()
    feedback = {bool(is_correct): count for is_correct, count in feedback_rows}

    day = func.date(Analysis.created_at)
    daily_rows = session.execute(
        select(
            day.label("day"),
            func.count(),
            func.sum(case((Analysis.verdict == "likely_scam", 1), else_=0)),
        )
        .where(in_window)
        .group_by(day)
        .order_by(day)
    ).all()

    return Stats(
        days=days,
        total=sum(verdicts.values()),
        verdicts=verdicts,
        scam_types=[(scam_type, count) for scam_type, count in type_rows],
        feedback_correct=feedback.get(True, 0),
        feedback_wrong=feedback.get(False, 0),
        daily=[
            {"date": _as_date_string(d), "total": int(total), "likely_scam": int(scams or 0)}
            for d, total, scams in daily_rows
        ],
    )


def _as_date_string(value: object) -> str:
    # MySQL returns a date object, SQLite a string.
    return value.isoformat() if isinstance(value, date) else str(value)
