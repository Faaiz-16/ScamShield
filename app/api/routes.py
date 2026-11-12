import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.deps import (
    Client,
    get_analysis_service,
    get_client,
    get_session,
    get_settings,
    rate_limit,
)
from app.core.config import Settings
from app.schemas.analysis import (
    ActionOut,
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackSummary,
    HealthResponse,
    Language,
    LinkFindingOut,
    LinkOut,
    LocalizedTexts,
    ModelOut,
    ReasonOut,
    ScamTypeCount,
    StatsResponse,
)
from app.services import records
from app.services.analysis import AnalysisResult, AnalysisService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", responses={429: {"model": ErrorResponse}})

ServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]
SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
ClientDep = Annotated[Client, Depends(get_client)]


def localized_texts(
    result: AnalysisResult, language: Language, service: AnalysisService
) -> LocalizedTexts:
    verdict = result.verdict
    return LocalizedTexts(
        verdict_label=verdict.label(language),
        scam_type_label=service.verdict_engine.category_label(verdict.scam_type, language),
        reasons=[r.text(language) for r in verdict.reasons],
        actions=[a.text(language) for a in verdict.actions],
        link_findings=[[f.reason(language) for f in r.findings] for r in result.url_reports],
    )


def build_response(
    result: AnalysisResult,
    analysis_id: str | None,
    language: Language,
    service: AnalysisService,
) -> AnalyzeResponse:
    verdict = result.verdict
    return AnalyzeResponse(
        analysis_id=analysis_id,
        verdict=verdict.level,
        verdict_label=verdict.label(language),
        risk_score=verdict.risk_score,
        confidence=verdict.confidence,
        scam_type=verdict.scam_type,
        scam_type_label=service.verdict_engine.category_label(verdict.scam_type, language),
        reasons=[
            ReasonOut(code=r.code, source=r.source, severity=r.severity, text=r.text(language))
            for r in verdict.reasons
        ],
        actions=[ActionOut(code=a.code, text=a.text(language)) for a in verdict.actions],
        links=[
            LinkOut(
                url=report.url,
                final_url=report.final_url,
                domain=report.registered_domain or report.host,
                is_shortened=report.is_shortened,
                is_official=report.is_official,
                official_brand=report.official_brand,
                domain_age_days=report.domain_age_days,
                findings=[
                    LinkFindingOut(code=f.code, severity=f.severity, text=f.reason(language))
                    for f in report.findings
                ],
            )
            for report in result.url_reports
        ],
        language=language,
        detected_language=result.detected_language,
        model=ModelOut(
            name=result.model_name,
            scam_probability=result.prediction.scam_probability if result.prediction else None,
        ),
        translations={lang: localized_texts(result, lang, service) for lang in ("en", "hi")},
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    dependencies=[Depends(rate_limit("analyze"))],
    responses={422: {"model": ErrorResponse}},
)
async def analyze(
    payload: AnalyzeRequest,
    service: ServiceDep,
    session: SessionDep,
    settings: SettingsDep,
    client: ClientDep,
) -> AnalyzeResponse:
    result = await service.analyze(payload.text)
    language: Language = payload.language or (
        "hi" if result.detected_language == "hi" else settings.default_language
    )

    try:
        analysis_id = await run_in_threadpool(
            records.save_analysis,
            session,
            result,
            channel=client.channel,
            client_hash=client.hash,
            response_language=language,
        )
    except SQLAlchemyError:
        # The user still gets a verdict; they just cannot leave feedback on it.
        logger.exception("could not store analysis")
        session.rollback()
        analysis_id = None

    return build_response(result, analysis_id, language, service)


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    dependencies=[Depends(rate_limit("feedback"))],
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def feedback(payload: FeedbackRequest, session: SessionDep, client: ClientDep) -> FeedbackResponse:
    try:
        replaced = records.record_feedback(
            session,
            public_id=str(payload.analysis_id),
            client_hash=client.hash,
            is_correct=payload.is_correct,
            correct_label=payload.correct_label,
        )
    except records.AnalysisNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Analysis not found.") from None
    except IntegrityError:
        # Two simultaneous submissions from the same client; the first one won.
        session.rollback()
        replaced = True
    return FeedbackResponse(status="updated" if replaced else "recorded")


@router.get("/stats", response_model=StatsResponse, dependencies=[Depends(rate_limit("stats"))])
def stats(
    session: SessionDep,
    service: ServiceDep,
    days: Annotated[int, Query(ge=1, le=365)] = 30,
    language: Language = "en",
) -> StatsResponse:
    data = records.get_stats(session, days=days)
    rated = data.feedback_correct + data.feedback_wrong
    return StatsResponse(
        days=data.days,
        total=data.total,
        verdicts=data.verdicts,
        scam_types=[
            ScamTypeCount(
                scam_type=scam_type,
                label=service.verdict_engine.category_label(scam_type, language) or scam_type,
                count=count,
            )
            for scam_type, count in data.scam_types
        ],
        feedback=FeedbackSummary(
            correct=data.feedback_correct,
            wrong=data.feedback_wrong,
            accuracy=round(data.feedback_correct / rated, 4) if rated else None,
        ),
        daily=data.daily,
    )


@router.get("/health", response_model=HealthResponse)
def health(session: SessionDep, service: ServiceDep) -> HealthResponse:
    try:
        session.execute(text("SELECT 1"))
        database = "ok"
    except SQLAlchemyError:
        database = "unavailable"
    model = service.classifier.name if service.classifier else None
    return HealthResponse(status="ok", database=database, model=model)
