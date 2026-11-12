from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.detection.anonymize import anonymize
from app.detection.classifier import Prediction, TextClassifier
from app.detection.language import MessageLanguage, detect_language
from app.detection.rules import RuleEngine, RuleHit
from app.detection.url_analyzer import UrlAnalyzer, UrlReport
from app.detection.verdict import Verdict, VerdictEngine

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    verdict: Verdict
    rule_hits: list[RuleHit]
    url_reports: list[UrlReport]
    prediction: Prediction | None
    detected_language: MessageLanguage
    anonymized_text: str
    model_name: str | None


class AnalysisService:
    """Runs every detector on a message and turns the signals into a verdict."""

    def __init__(
        self,
        rule_engine: RuleEngine,
        url_analyzer: UrlAnalyzer,
        verdict_engine: VerdictEngine,
        classifier: TextClassifier | None = None,
    ) -> None:
        self.rule_engine = rule_engine
        self.url_analyzer = url_analyzer
        self.verdict_engine = verdict_engine
        self.classifier = classifier

    async def analyze(self, text: str) -> AnalysisResult:
        rule_hits = self.rule_engine.evaluate(text)
        url_reports, prediction = await asyncio.gather(
            self._analyze_links(text), self._predict(text)
        )
        verdict = self.verdict_engine.decide(rule_hits, url_reports, prediction)
        return AnalysisResult(
            verdict=verdict,
            rule_hits=rule_hits,
            url_reports=url_reports,
            prediction=prediction,
            detected_language=detect_language(text),
            anonymized_text=anonymize(text),
            model_name=self.classifier.name if self.classifier else None,
        )

    async def _analyze_links(self, text: str) -> list[UrlReport]:
        try:
            return await self.url_analyzer.analyze_text(text)
        except Exception:
            # Link checks are best effort: a failure must not block the verdict.
            logger.exception("link analysis failed")
            return []

    async def _predict(self, text: str) -> Prediction | None:
        if self.classifier is None:
            return None
        try:
            predictions = await asyncio.to_thread(self.classifier.predict, [text])
            return predictions[0]
        except Exception:
            logger.exception("classifier prediction failed")
            return None
