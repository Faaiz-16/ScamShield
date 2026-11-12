"""Combine rule hits, link findings and the ML probability into one verdict.

Scoring
-------
Each signal source produces a score in [0, 1]:

* rules: noisy-OR of the severities of all rule hits, 1 - prod(1 - s_i), so
  several weak signs add up but no single source exceeds 1;
* links: noisy-OR of the severities of all link findings;
* model: the classifier's probability that the message is a scam.

Each source score is scaled by its weight and the sources are combined with
another noisy-OR. A source that is absent (no links in the message, no trained
model) contributes nothing, rather than dragging an average down. With the
default model weight of 0.6, the model alone can raise a message to
"suspicious" but never to "likely scam".

When every link in the message points to an official domain and no critical
sign was found, the score is multiplied by ``official_link_factor`` to
reduce false positives on genuine bank and government messages.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from app.core.config import Settings
from app.detection.classifier import Prediction
from app.detection.rules import Language, LocalizedText, RuleHit
from app.detection.url_analyzer import UrlFinding, UrlReport

VerdictLevel = Literal["safe", "suspicious", "likely_scam"]
ReasonSource = Literal["rule", "link", "model", "info"]

VERDICT_LABELS: dict[VerdictLevel, LocalizedText] = {
    "safe": LocalizedText(en="Safe", hi="सुरक्षित"),
    "suspicious": LocalizedText(en="Suspicious", hi="संदिग्ध"),
    "likely_scam": LocalizedText(en="Likely Scam", hi="संभावित धोखाधड़ी"),
}

_PHISHING_LINK_CODES = {"LOOKALIKE_DOMAIN", "BRAND_IMPERSONATION", "IDN_HOMOGRAPH"}
# For an official destination, a shortened link is expected, not a warning sign.
_IGNORED_ON_OFFICIAL_LINKS = {"SHORTENED_URL"}


@dataclass(frozen=True)
class VerdictConfig:
    suspicious_threshold: float = 0.35
    scam_threshold: float = 0.65
    rule_weight: float = 0.9
    url_weight: float = 0.85
    ml_weight: float = 0.6
    official_link_factor: float = 0.5
    critical_severity: float = 0.85
    ml_reason_threshold: float = 0.7
    max_reasons: int = 3
    max_actions: int = 4

    @classmethod
    def from_settings(cls, settings: Settings) -> VerdictConfig:
        return cls(
            suspicious_threshold=settings.suspicious_threshold,
            scam_threshold=settings.scam_threshold,
            rule_weight=settings.verdict_rule_weight,
            url_weight=settings.verdict_url_weight,
            ml_weight=settings.verdict_ml_weight,
            official_link_factor=settings.verdict_official_link_factor,
        )


@dataclass(frozen=True)
class Reason:
    code: str
    source: ReasonSource
    severity: float
    text_en: str
    text_hi: str

    def text(self, language: Language = "en") -> str:
        return self.text_hi if language == "hi" else self.text_en


@dataclass(frozen=True)
class Action:
    code: str
    text_en: str
    text_hi: str

    def text(self, language: Language = "en") -> str:
        return self.text_hi if language == "hi" else self.text_en


@dataclass(frozen=True)
class Verdict:
    level: VerdictLevel
    risk_score: float
    confidence: float
    scam_type: str | None
    reasons: list[Reason]
    actions: list[Action]
    signal_scores: dict[str, float | None] = field(default_factory=dict)

    def label(self, language: Language = "en") -> str:
        return VERDICT_LABELS[self.level].get(language)


# --------------------------------------------------------------------------
# Fixed explanation texts
# --------------------------------------------------------------------------

_INFO_REASONS: dict[str, tuple[str, str]] = {
    "NO_SCAM_PATTERNS": (
        "No common scam patterns were found, such as requests for an OTP, PIN or urgent payment.",
        "ओटीपी, पिन या तुरंत भुगतान की मांग जैसे धोखाधड़ी के आम संकेत नहीं मिले।",
    ),
    "MINOR_SIGNALS_ONLY": (
        "Only minor warning signs were found, such as urgent wording or a request to open a link.",
        "केवल मामूली चेतावनी संकेत मिले, जैसे जल्दबाजी वाले शब्द या लिंक खोलने का अनुरोध।",
    ),
    "OFFICIAL_LINK": (
        "The link goes to an official website ({brand}).",
        "लिंक आधिकारिक वेबसाइट ({brand}) पर जाता है।",
    ),
    "MODEL_LOOKS_GENUINE": (
        "The wording is similar to genuine messages.",
        "संदेश के शब्द असली संदेशों से मिलते-जुलते हैं।",
    ),
    "MODEL_LOOKS_LIKE_SCAM": (
        "The wording closely matches known scam messages.",
        "संदेश के शब्द ज्ञात धोखाधड़ी वाले संदेशों से काफी मिलते हैं।",
    ),
    "UNCLEAR": (
        "Some parts of this message resemble scams, but there is no clear sign either way.",
        "इस संदेश के कुछ हिस्से धोखाधड़ी जैसे लगते हैं, लेकिन कोई स्पष्ट संकेत नहीं है।",
    ),
}

_ACTIONS: dict[str, tuple[str, str]] = {
    "DO_NOT_CLICK": (
        "Do not click any link in this message.",
        "इस संदेश के किसी भी लिंक पर क्लिक न करें।",
    ),
    "NEVER_SHARE_CREDENTIALS": (
        "Never share your OTP, PIN, CVV or password with anyone, even if they say they are from your bank.",
        "अपना ओटीपी, पिन, सीवीवी या पासवर्ड किसी को न बताएं, चाहे वह खुद को बैंक का कर्मचारी बताए।",
    ),
    "UPI_PIN_ONLY_TO_PAY": (
        "A UPI PIN is only needed to send money, never to receive it. Decline unknown collect requests.",
        "यूपीआई पिन केवल पैसे भेजने के लिए होता है, पाने के लिए कभी नहीं। अनजान कलेक्ट रिक्वेस्ट अस्वीकार करें।",
    ),
    "DO_NOT_INSTALL": (
        "Do not install apps from links. If you installed one because of this message, uninstall it and contact your bank.",
        "लिंक से कोई ऐप इंस्टॉल न करें। अगर इस संदेश की वजह से कोई ऐप इंस्टॉल किया है, तो उसे हटाएं और अपने बैंक से संपर्क करें।",
    ),
    "DO_NOT_CALL_BACK": (
        "Do not call the phone number given in the message.",
        "संदेश में दिए गए फोन नंबर पर कॉल न करें।",
    ),
    "REPORT_CYBERCRIME": (
        "Report it at cybercrime.gov.in or call the 1930 helpline. If you have lost money, call 1930 immediately.",
        "cybercrime.gov.in पर शिकायत करें या 1930 हेल्पलाइन पर कॉल करें। अगर पैसे कट गए हैं तो तुरंत 1930 पर कॉल करें।",
    ),
    "CALL_1930_IF_AFFECTED": (
        "If you have already shared details or lost money, call the 1930 helpline immediately.",
        "अगर आपने जानकारी साझा कर दी है या पैसे कट गए हैं, तो तुरंत 1930 हेल्पलाइन पर कॉल करें।",
    ),
    "CONTACT_BANK_OFFICIALLY": (
        "Contact your bank only through the number on its official website or the back of your card.",
        "अपने बैंक से केवल उसकी आधिकारिक वेबसाइट या कार्ड के पीछे लिखे नंबर पर ही संपर्क करें।",
    ),
    "VERIFY_WITH_SENDER": (
        "If you are unsure, check with the sender directly through their official website or app.",
        "अगर आप निश्चित नहीं हैं, तो भेजने वाले से उनकी आधिकारिक वेबसाइट या ऐप के जरिए सीधे पुष्टि करें।",
    ),
    "STAY_ALERT": (
        "This message looks safe, but never share your OTP or PIN with anyone.",
        "यह संदेश सुरक्षित लगता है, फिर भी अपना ओटीपी या पिन किसी को न बताएं।",
    ),
}

_BANKING_CATEGORIES = {
    "credential_theft",
    "kyc_scam",
    "account_threat",
    "upi_fraud",
    "reward_scam",
    "loan_scam",
    "phishing_link",
}
_CREDENTIAL_CATEGORIES = _BANKING_CATEGORIES | {"customer_care_scam", "remote_access_scam"}
_PHONE_CATEGORIES = {"customer_care_scam", "utility_scam", "remote_access_scam", "telecom_scam"}


def _info(code: str, severity: float = 0.0, **params: object) -> Reason:
    en, hi = _INFO_REASONS[code]
    source: ReasonSource = "model" if code.startswith("MODEL_") else "info"
    return Reason(code, source, severity, en.format(**params), hi.format(**params))


def _action(code: str) -> Action:
    en, hi = _ACTIONS[code]
    return Action(code, en, hi)


def noisy_or(values: Iterable[float]) -> float:
    return 1.0 - math.prod(1.0 - min(max(v, 0.0), 1.0) for v in values)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


class VerdictEngine:
    def __init__(
        self,
        config: VerdictConfig | None = None,
        categories: Mapping[str, LocalizedText] | None = None,
    ) -> None:
        self.config = config or VerdictConfig()
        self.categories = dict(categories or {})

    def category_label(self, category: str | None, language: Language = "en") -> str | None:
        if category is None:
            return None
        label = self.categories.get(category)
        return label.get(language) if label else category

    def decide(
        self,
        rule_hits: Sequence[RuleHit],
        url_reports: Sequence[UrlReport] = (),
        prediction: Prediction | None = None,
    ) -> Verdict:
        cfg = self.config
        url_findings = self._scored_url_findings(url_reports)

        rule_score = noisy_or(hit.severity for hit in rule_hits)
        url_score = noisy_or(f.severity for f in url_findings) if url_reports else None
        ml_score = prediction.scam_probability if prediction else None

        weighted = [cfg.rule_weight * rule_score]
        if url_score is not None:
            weighted.append(cfg.url_weight * url_score)
        if ml_score is not None:
            weighted.append(cfg.ml_weight * ml_score)
        risk = noisy_or(weighted)

        max_signal = max(
            [h.severity for h in rule_hits] + [f.severity for f in url_findings], default=0.0
        )
        all_links_official = bool(url_reports) and all(r.is_official for r in url_reports)
        if all_links_official and max_signal < cfg.critical_severity:
            risk *= cfg.official_link_factor

        level = self._level(risk)
        return Verdict(
            level=level,
            risk_score=round(risk, 4),
            confidence=round(self._confidence(risk, level), 4),
            scam_type=self._scam_type(level, rule_hits, url_findings),
            reasons=self._reasons(level, rule_hits, url_findings, url_reports, ml_score),
            actions=self._actions(level, rule_hits, url_findings, url_reports),
            signal_scores={"rules": rule_score, "links": url_score, "model": ml_score},
        )

    # -- scoring helpers ----------------------------------------------------

    @staticmethod
    def _scored_url_findings(reports: Sequence[UrlReport]) -> list[UrlFinding]:
        findings: dict[str, UrlFinding] = {}
        for report in reports:
            for finding in report.findings:
                if report.is_official and finding.code in _IGNORED_ON_OFFICIAL_LINKS:
                    continue
                current = findings.get(finding.code)
                if current is None or finding.severity > current.severity:
                    findings[finding.code] = finding
        return sorted(findings.values(), key=lambda f: (-f.severity, f.code))

    def _level(self, risk: float) -> VerdictLevel:
        if risk >= self.config.scam_threshold:
            return "likely_scam"
        if risk >= self.config.suspicious_threshold:
            return "suspicious"
        return "safe"

    def _confidence(self, risk: float, level: VerdictLevel) -> float:
        """How far the score sits inside its band: 0.5 at a boundary, 1.0 at the far end."""
        low, high = self.config.suspicious_threshold, self.config.scam_threshold
        if level == "likely_scam":
            depth = (risk - high) / (1.0 - high)
        elif level == "safe":
            depth = (low - risk) / low
        else:
            middle, half_width = (low + high) / 2, (high - low) / 2
            depth = 1.0 - abs(risk - middle) / half_width
        return 0.5 + 0.5 * min(max(depth, 0.0), 1.0)

    @staticmethod
    def _scam_type(
        level: VerdictLevel, rule_hits: Sequence[RuleHit], url_findings: Sequence[UrlFinding]
    ) -> str | None:
        if level == "safe":
            return None
        for hit in rule_hits:  # already sorted most severe first
            if hit.category != "generic":
                return hit.category
        if any(f.code in _PHISHING_LINK_CODES for f in url_findings):
            return "phishing_link"
        return None

    # -- explanations -------------------------------------------------------

    def _reasons(
        self,
        level: VerdictLevel,
        rule_hits: Sequence[RuleHit],
        url_findings: Sequence[UrlFinding],
        url_reports: Sequence[UrlReport],
        ml_score: float | None,
    ) -> list[Reason]:
        cfg = self.config
        if level == "safe":
            reasons = [
                _info("MINOR_SIGNALS_ONLY" if rule_hits or url_findings else "NO_SCAM_PATTERNS")
            ]
            brands = sorted(
                {r.official_brand for r in url_reports if r.is_official and r.official_brand}
            )
            if brands:
                reasons.append(_info("OFFICIAL_LINK", brand=", ".join(brands)))
            if ml_score is not None and ml_score < 0.3:
                reasons.append(_info("MODEL_LOOKS_GENUINE"))
            return reasons[: cfg.max_reasons]

        candidates = [
            Reason(h.rule_id, "rule", h.severity, h.reason_en, h.reason_hi) for h in rule_hits
        ]
        candidates += [
            Reason(f.code, "link", f.severity, f.reason_en, f.reason_hi) for f in url_findings
        ]
        model_threshold = cfg.ml_reason_threshold if candidates else 0.5
        if ml_score is not None and ml_score >= model_threshold:
            # Ranked below explicit rule and link evidence of similar strength.
            candidates.append(_info("MODEL_LOOKS_LIKE_SCAM", severity=round(ml_score * 0.8, 4)))

        candidates.sort(key=lambda r: -r.severity)
        return candidates[: cfg.max_reasons] or [_info("UNCLEAR")]

    def _actions(
        self,
        level: VerdictLevel,
        rule_hits: Sequence[RuleHit],
        url_findings: Sequence[UrlFinding],
        url_reports: Sequence[UrlReport],
    ) -> list[Action]:
        if level == "safe":
            return [_action("STAY_ALERT"), _action("VERIFY_WITH_SENDER")]

        categories = {h.category for h in rule_hits}
        if any(f.code in _PHISHING_LINK_CODES for f in url_findings):
            categories.add("phishing_link")
        codes = {h.rule_id for h in rule_hits} | {f.code for f in url_findings}

        wanted: list[str] = []
        if url_reports:
            wanted.append("DO_NOT_CLICK")
        if level == "likely_scam" or categories & _CREDENTIAL_CATEGORIES:
            wanted.append("NEVER_SHARE_CREDENTIALS")
        if "upi_fraud" in categories:
            wanted.append("UPI_PIN_ONLY_TO_PAY")
        if categories & {"remote_access_scam", "malware"} or "APK_LINK" in codes:
            wanted.append("DO_NOT_INSTALL")
        if categories & _PHONE_CATEGORIES:
            wanted.append("DO_NOT_CALL_BACK")
        if categories & _BANKING_CATEGORIES:
            wanted.append("CONTACT_BANK_OFFICIALLY")
        if level == "suspicious":
            wanted.append("VERIFY_WITH_SENDER")

        # The reporting step always stays in the list, even when it is trimmed.
        final = "REPORT_CYBERCRIME" if level == "likely_scam" else "CALL_1930_IF_AFFECTED"
        trimmed = wanted[: self.config.max_actions - 1] + [final]
        return [_action(code) for code in trimmed]
