import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_MESSAGE_CHARS = 5000

Language = Literal["en", "hi"]
VerdictLevel = Literal["safe", "suspicious", "likely_scam"]
MessageLabel = Literal["scam", "legit", "spam_not_scam"]

# Control characters other than tab, newline and carriage return.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    language: Language | None = Field(
        default=None,
        description="Language for reasons and actions. Defaults to Hindi for Devanagari "
        "messages and to the server default otherwise.",
    )

    @field_validator("text")
    @classmethod
    def _clean_text(cls, value: str) -> str:
        value = _CONTROL_CHARS.sub("", value).strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class ReasonOut(BaseModel):
    code: str
    source: Literal["rule", "link", "model", "info"]
    severity: float
    text: str


class ActionOut(BaseModel):
    code: str
    text: str


class LinkFindingOut(BaseModel):
    code: str
    severity: float
    text: str


class LinkOut(BaseModel):
    url: str
    final_url: str
    domain: str
    is_shortened: bool
    is_official: bool
    official_brand: str | None
    domain_age_days: int | None
    findings: list[LinkFindingOut]


class ModelOut(BaseModel):
    name: str | None
    scam_probability: float | None


class LocalizedTexts(BaseModel):
    """All human-readable text of a result in one language, in the same order as the
    structured fields. Lets clients switch language without re-analysing."""

    verdict_label: str
    scam_type_label: str | None
    reasons: list[str]
    actions: list[str]
    link_findings: list[list[str]]


class AnalyzeResponse(BaseModel):
    analysis_id: str | None = Field(
        description="Pass this to /api/feedback. Null if the analysis could not be stored."
    )
    verdict: VerdictLevel
    verdict_label: str
    risk_score: float
    confidence: float
    scam_type: str | None
    scam_type_label: str | None
    reasons: list[ReasonOut]
    actions: list[ActionOut]
    links: list[LinkOut]
    language: Language
    detected_language: Literal["en", "hi", "hinglish"]
    model: ModelOut
    translations: dict[Language, LocalizedTexts]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: UUID
    is_correct: bool
    correct_label: MessageLabel | None = Field(
        default=None,
        description="Optional: what the message really was, when the verdict was wrong.",
    )


class FeedbackResponse(BaseModel):
    status: Literal["recorded", "updated"]


class ScamTypeCount(BaseModel):
    scam_type: str
    label: str
    count: int


class FeedbackSummary(BaseModel):
    correct: int
    wrong: int
    accuracy: float | None


class DailyCount(BaseModel):
    date: str
    total: int
    likely_scam: int


class StatsResponse(BaseModel):
    days: int
    total: int
    verdicts: dict[VerdictLevel, int]
    scam_types: list[ScamTypeCount]
    feedback: FeedbackSummary
    daily: list[DailyCount]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: Literal["ok", "unavailable"]
    model: str | None


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list[dict[str, str]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
