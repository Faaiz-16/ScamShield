import pytest

from app.core.config import Settings
from app.detection.classifier import Prediction
from app.detection.rules import RuleHit, get_rule_engine
from app.detection.url_analyzer import UrlReport, make_finding
from app.detection.verdict import VerdictConfig, VerdictEngine, noisy_or


def hit(rule_id: str = "R", severity: float = 0.5, category: str = "kyc_scam") -> RuleHit:
    return RuleHit(rule_id, category, severity, f"{rule_id} en", f"{rule_id} hi", "match")


def link(*codes: str, official: bool = False, brand: str | None = None) -> UrlReport:
    params = {"domain": "x.com", "brand": "Bank", "official": "bank.com", "tld": "xyz", "days": 3}
    return UrlReport(
        url="http://x.com",
        final_url="http://x.com",
        host="x.com",
        registered_domain="x.com",
        is_official=official,
        official_brand=brand,
        findings=[make_finding(code, **params) for code in codes],
    )


def model(p_scam: float) -> Prediction:
    rest = (1 - p_scam) / 2
    return Prediction(
        "scam" if p_scam > 0.5 else "legit",
        {"scam": p_scam, "legit": rest, "spam_not_scam": rest},
        "test",
    )


@pytest.fixture(scope="module")
def engine() -> VerdictEngine:
    return VerdictEngine(categories=get_rule_engine().categories)


def test_noisy_or() -> None:
    assert noisy_or([]) == 0
    assert noisy_or([0.5, 0.5]) == pytest.approx(0.75)
    assert noisy_or([1.5, -1]) == 1.0  # values are clamped


# --- Levels ------------------------------------------------------------------


def test_no_signals_is_safe(engine: VerdictEngine) -> None:
    verdict = engine.decide([], [], None)
    assert verdict.level == "safe"
    assert verdict.risk_score == 0
    assert verdict.confidence == 1.0
    assert verdict.scam_type is None
    assert [r.code for r in verdict.reasons] == ["NO_SCAM_PATTERNS"]


def test_single_critical_rule_is_likely_scam(engine: VerdictEngine) -> None:
    verdict = engine.decide([hit("OTP_SHARE_REQUEST", 0.95, "credential_theft")])
    assert verdict.level == "likely_scam"
    assert verdict.scam_type == "credential_theft"


def test_weak_signals_accumulate(engine: VerdictEngine) -> None:
    one = engine.decide([hit("A", 0.3, "generic")])
    three = engine.decide([hit("A", 0.3, "generic"), hit("B", 0.3, "generic"), hit("C", 0.3)])
    assert one.level == "safe"
    assert three.risk_score > one.risk_score
    assert three.level == "suspicious"


def test_model_alone_cannot_reach_likely_scam(engine: VerdictEngine) -> None:
    verdict = engine.decide([], [], model(1.0))
    assert verdict.level == "suspicious"
    assert verdict.reasons[0].code == "MODEL_LOOKS_LIKE_SCAM"


def test_lookalike_link_alone_is_likely_scam(engine: VerdictEngine) -> None:
    verdict = engine.decide([], [link("LOOKALIKE_DOMAIN")])
    assert verdict.level == "likely_scam"
    assert verdict.scam_type == "phishing_link"


def test_missing_sources_do_not_dilute_the_score(engine: VerdictEngine) -> None:
    rules_only = engine.decide([hit("X", 0.8)])
    with_clean_link = engine.decide([hit("X", 0.8)], [link()])
    assert with_clean_link.risk_score == pytest.approx(rules_only.risk_score)


def test_official_links_reduce_score(engine: VerdictEngine) -> None:
    hits = [hit("TRAFFIC_CHALLAN", 0.5, "challan_scam")]
    unofficial = engine.decide(hits, [link()], model(0.4))
    official = engine.decide(
        hits, [link(official=True, brand="verified .gov.in domain")], model(0.4)
    )
    assert unofficial.level == "suspicious"
    assert official.level == "safe"
    assert official.risk_score == pytest.approx(unofficial.risk_score * 0.5, abs=1e-3)
    assert "OFFICIAL_LINK" in [r.code for r in official.reasons]


def test_critical_signal_is_not_discounted_by_official_link(engine: VerdictEngine) -> None:
    hits = [hit("OTP_SHARE_REQUEST", 0.95, "credential_theft")]
    verdict = engine.decide(hits, [link(official=True, brand="SBI")])
    assert verdict.level == "likely_scam"


def test_shortened_official_link_is_not_a_warning(engine: VerdictEngine) -> None:
    verdict = engine.decide([], [link("SHORTENED_URL", official=True, brand="Amazon")])
    assert verdict.signal_scores["links"] == 0
    assert verdict.level == "safe"


def test_thresholds_are_configurable() -> None:
    strict = VerdictEngine(VerdictConfig(suspicious_threshold=0.1, scam_threshold=0.3))
    assert strict.decide([hit("A", 0.4)]).level == "likely_scam"


def test_config_reads_settings() -> None:
    settings = Settings(_env_file=None, scam_threshold=0.8, verdict_ml_weight=0.3)
    config = VerdictConfig.from_settings(settings)
    assert config.scam_threshold == 0.8
    assert config.ml_weight == 0.3


@pytest.mark.parametrize("risk_hits", [[], [hit("A", 0.3)], [hit("A", 0.5)], [hit("A", 0.99)]])
def test_confidence_is_between_half_and_one(engine: VerdictEngine, risk_hits) -> None:
    assert 0.5 <= engine.decide(risk_hits).confidence <= 1.0


# --- Reasons -----------------------------------------------------------------


def test_reasons_are_capped_and_most_severe_first(engine: VerdictEngine) -> None:
    hits = [hit("LOW", 0.3, "generic"), hit("HIGH", 0.9), hit("MID", 0.6), hit("MID2", 0.5)]
    verdict = engine.decide(hits, [link("SUSPICIOUS_TLD")])
    assert [r.code for r in verdict.reasons] == ["HIGH", "MID", "MID2"]


def test_reasons_mix_rules_and_links(engine: VerdictEngine) -> None:
    verdict = engine.decide([hit("KYC_UPDATE", 0.65)], [link("BRAND_IMPERSONATION")])
    assert [(r.code, r.source) for r in verdict.reasons] == [
        ("BRAND_IMPERSONATION", "link"),
        ("KYC_UPDATE", "rule"),
    ]


def test_scam_type_skips_generic_rules(engine: VerdictEngine) -> None:
    hits = [hit("URGENT", 0.9, "generic"), hit("KYC", 0.6, "kyc_scam")]
    assert engine.decide(hits).scam_type == "kyc_scam"


def test_minor_signals_reason_for_safe_with_weak_hits(engine: VerdictEngine) -> None:
    verdict = engine.decide([hit("URGENCY_PRESSURE", 0.25, "generic")])
    assert verdict.level == "safe"
    assert verdict.reasons[0].code == "MINOR_SIGNALS_ONLY"


def test_unclear_reason_when_nothing_explains_the_score() -> None:
    engine = VerdictEngine(VerdictConfig(suspicious_threshold=0.05, ml_weight=0.2))
    verdict = engine.decide([], [], model(0.4))
    assert verdict.level == "suspicious"
    assert [r.code for r in verdict.reasons] == ["UNCLEAR"]


def test_hindi_text_is_available(engine: VerdictEngine) -> None:
    verdict = engine.decide([hit("OTP_SHARE_REQUEST", 0.95, "credential_theft")])
    assert verdict.label("hi") == "संभावित धोखाधड़ी"
    assert verdict.reasons[0].text("hi") == "OTP_SHARE_REQUEST hi"
    assert any("1930" in a.text("hi") for a in verdict.actions)
    assert engine.category_label("credential_theft", "hi") != engine.category_label(
        "credential_theft", "en"
    )


# --- Actions -----------------------------------------------------------------


def action_codes(verdict) -> list[str]:
    return [a.code for a in verdict.actions]


def test_scam_actions_always_end_with_reporting(engine: VerdictEngine) -> None:
    hits = [hit("A", 0.95, "upi_fraud"), hit("B", 0.9, "remote_access_scam")]
    verdict = engine.decide(hits, [link("APK_LINK")])
    codes = action_codes(verdict)
    assert len(codes) <= 4
    assert codes[-1] == "REPORT_CYBERCRIME"
    assert codes[0] == "DO_NOT_CLICK"
    assert "cybercrime.gov.in" in verdict.actions[-1].text_en


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("upi_fraud", "UPI_PIN_ONLY_TO_PAY"),
        ("remote_access_scam", "DO_NOT_INSTALL"),
        ("customer_care_scam", "DO_NOT_CALL_BACK"),
        ("kyc_scam", "CONTACT_BANK_OFFICIALLY"),
    ],
)
def test_actions_depend_on_scam_type(engine: VerdictEngine, category: str, expected: str) -> None:
    assert expected in action_codes(engine.decide([hit("X", 0.9, category)]))


def test_suspicious_actions(engine: VerdictEngine) -> None:
    verdict = engine.decide([hit("LINK", 0.3, "generic"), hit("URG", 0.25, "generic")], [link()])
    assert verdict.level == "suspicious"
    assert action_codes(verdict) == ["DO_NOT_CLICK", "VERIFY_WITH_SENDER", "CALL_1930_IF_AFFECTED"]


def test_safe_actions(engine: VerdictEngine) -> None:
    assert action_codes(engine.decide([])) == ["STAY_ALERT", "VERIFY_WITH_SENDER"]


# --- With the real rule engine -----------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "482913 is your OTP for login. Do not share it with anyone. SBI never asks for OTP.",
            "safe",
        ),
        ("Bhai kal cricket khelne chalega? 6 baje ground pe milte hai", "safe"),
        ("Please share the OTP you just received to complete verification.", "likely_scam"),
        ("Paise receive karne ke liye UPI PIN daalein", "likely_scam"),
        ("आपकी बिजली आज रात काट दी जाएगी। तुरंत बिजली अधिकारी से संपर्क करें 9812345670", "likely_scam"),
        ("Final reminder: click the link to view your statement", "suspicious"),
    ],
)
def test_end_to_end_with_rules(engine: VerdictEngine, text: str, expected: str) -> None:
    assert engine.decide(get_rule_engine().evaluate(text)).level == expected
