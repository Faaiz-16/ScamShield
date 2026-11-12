import re
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from app.core.config import Settings
from app.detection.classifier import Prediction
from app.detection.rules import get_rule_engine
from app.detection.url_analyzer import UrlAnalyzer
from app.detection.verdict import VerdictEngine
from app.main import create_app
from app.models import Analysis, Feedback
from app.services import records
from app.services.analysis import AnalysisService

SCAM_TEXT = (
    "Dear Rahul, your SBI account will be blocked today. Update KYC at "
    "http://sbi-kyc.xyz/login?m=9876543210 or call customer care 9876543210"
)
LEGIT_TEXT = "482913 is your OTP for login. Do not share it with anyone. SBI never asks for OTP."
BOT_KEY = "bot-secret"


class FakeClassifier:
    name = "fake"

    def __init__(self, scam_probability: float = 0.2) -> None:
        self.scam_probability = scam_probability

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        p = self.scam_probability
        probs = {"scam": p, "legit": 1 - p, "spam_not_scam": 0.0}
        return [Prediction("scam" if p > 0.5 else "legit", probs, self.name) for _ in texts]


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": f"sqlite:///{tmp_path / 'api.db'}",
        "database_auto_create": True,
        "url_network_checks": False,
        "rate_limit_per_minute": 1000,
        "hash_salt": "test-salt",
        "bot_api_key": BOT_KEY,
        "cors_origins": "http://allowed.test",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_service(settings: Settings, classifier: FakeClassifier | None = None) -> AnalysisService:
    rules = get_rule_engine()
    return AnalysisService(
        rule_engine=rules,
        url_analyzer=UrlAnalyzer(settings),
        verdict_engine=VerdictEngine(categories=rules.categories),
        classifier=classifier or FakeClassifier(),
    )


@pytest.fixture
def build(tmp_path: Path) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def factory(service_factory=make_service, raise_errors: bool = True, **overrides) -> TestClient:
        settings = make_settings(tmp_path, **overrides)
        app = create_app(settings, analysis_service=service_factory(settings))
        client = TestClient(app, raise_server_exceptions=raise_errors)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(build) -> TestClient:
    return build()


def stored(client: TestClient, model):
    with client.app.state.session_factory() as session:
        return session.scalars(select(model)).all()


def analyze(client: TestClient, text: str = SCAM_TEXT, **extra) -> dict:
    response = client.post("/api/analyze", json={"text": text, **extra})
    assert response.status_code == 200, response.text
    return response.json()


# --- /api/analyze ------------------------------------------------------------


def test_analyze_returns_full_verdict(client: TestClient) -> None:
    body = analyze(client)
    assert body["verdict"] == "likely_scam"
    assert body["verdict_label"] == "Likely Scam"
    assert 0 <= body["risk_score"] <= 1 and 0.5 <= body["confidence"] <= 1
    assert body["scam_type"] and body["scam_type_label"]
    assert 1 <= len(body["reasons"]) <= 3
    severities = [r["severity"] for r in body["reasons"]]
    assert severities == sorted(severities, reverse=True)
    assert body["actions"][-1]["code"] == "REPORT_CYBERCRIME"
    assert body["links"][0]["domain"] == "sbi-kyc.xyz"
    assert body["model"] == {"name": "fake", "scam_probability": 0.2}
    assert body["language"] == "en" and body["detected_language"] == "en"
    assert re.fullmatch(r"[0-9a-f-]{36}", body["analysis_id"])


def test_legit_message_is_safe(client: TestClient) -> None:
    body = analyze(client, LEGIT_TEXT)
    assert body["verdict"] == "safe"
    assert body["scam_type"] is None


def test_hindi_message_gets_hindi_response(client: TestClient) -> None:
    body = analyze(client, "आपके मोबाइल पर आया ओटीपी तुरंत बताएं")
    assert body["language"] == "hi"
    assert body["verdict_label"] == "संभावित धोखाधड़ी"
    assert re.search(r"[ऀ-ॿ]", body["reasons"][0]["text"])


def test_language_can_be_requested(client: TestClient) -> None:
    assert analyze(client, "आपके मोबाइल पर आया ओटीपी तुरंत बताएं", language="en")["language"] == "en"
    hindi = analyze(client, SCAM_TEXT, language="hi")
    assert re.search(r"[ऀ-ॿ]", hindi["actions"][0]["text"])


def test_only_anonymised_data_is_stored(client: TestClient) -> None:
    analyze(client)
    record = stored(client, Analysis)[0]
    assert "9876543210" not in record.anonymized_text
    assert "Rahul" not in record.anonymized_text
    assert "<PHONE>" in record.anonymized_text
    assert "http://sbi-kyc.xyz/<URL_PATH>" in record.anonymized_text
    assert record.link_domains == ["sbi-kyc.xyz"]
    assert re.fullmatch(r"[0-9a-f]{64}", record.client_hash)
    assert record.channel == "web"
    assert record.verdict == "likely_scam"
    assert "ACCOUNT_BLOCK_THREAT" in record.rule_ids
    assert record.scam_probability == pytest.approx(0.2)


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": "   \n "},
        {"text": "x" * 5001},
        {"text": "hello", "language": "ta"},
        {"text": "hello", "unexpected": 1},
        {},
    ],
)
def test_invalid_input_is_rejected(client: TestClient, payload: dict) -> None:
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["details"]


def test_validation_errors_do_not_echo_the_message(client: TestClient) -> None:
    secret = "my phone is 9876543210 " * 300
    response = client.post("/api/analyze", json={"text": secret})
    assert response.status_code == 422
    assert "9876543210" not in response.text


def test_control_characters_are_removed(client: TestClient) -> None:
    analyze(client, "hello\x00 there\x07")
    assert stored(client, Analysis)[0].anonymized_text == "hello there"


def test_storage_failure_still_returns_verdict(client: TestClient, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise OperationalError("INSERT", {}, Exception("database down"))

    monkeypatch.setattr(records, "save_analysis", fail)
    body = analyze(client)
    assert body["analysis_id"] is None
    assert body["verdict"] == "likely_scam"


def test_link_analysis_failure_does_not_break_analysis(build) -> None:
    def broken_service(settings: Settings) -> AnalysisService:
        service = make_service(settings)

        async def explode(text: str):
            raise RuntimeError("network stack on fire")

        service.url_analyzer.analyze_text = explode
        return service

    body = analyze(build(service_factory=broken_service))
    assert body["links"] == []
    assert body["verdict"] == "likely_scam"


def test_runs_without_a_trained_model(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, baseline_model_path=tmp_path / "missing.joblib")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json()["model"] is None
        body = analyze(client)
        assert body["model"] == {"name": None, "scam_probability": None}
        assert body["verdict"] == "likely_scam"


# --- /api/feedback -----------------------------------------------------------


def test_feedback_is_recorded_then_updated(client: TestClient) -> None:
    analysis_id = analyze(client)["analysis_id"]
    first = client.post("/api/feedback", json={"analysis_id": analysis_id, "is_correct": True})
    assert first.json() == {"status": "recorded"}

    second = client.post(
        "/api/feedback",
        json={"analysis_id": analysis_id, "is_correct": False, "correct_label": "legit"},
    )
    assert second.json() == {"status": "updated"}

    rows = stored(client, Feedback)
    assert len(rows) == 1
    assert (rows[0].is_correct, rows[0].correct_label) == (False, "legit")


def test_correct_label_is_ignored_when_verdict_was_right(client: TestClient) -> None:
    analysis_id = analyze(client)["analysis_id"]
    client.post(
        "/api/feedback",
        json={"analysis_id": analysis_id, "is_correct": True, "correct_label": "legit"},
    )
    assert stored(client, Feedback)[0].correct_label is None


def test_feedback_for_unknown_analysis_is_404(client: TestClient) -> None:
    response = client.post(
        "/api/feedback",
        json={"analysis_id": "00000000-0000-4000-8000-000000000000", "is_correct": True},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    "payload",
    [
        {"analysis_id": "not-a-uuid", "is_correct": True},
        {"analysis_id": "00000000-0000-4000-8000-000000000000"},
        {
            "analysis_id": "00000000-0000-4000-8000-000000000000",
            "is_correct": False,
            "correct_label": "maybe",
        },
    ],
)
def test_invalid_feedback_is_rejected(client: TestClient, payload: dict) -> None:
    assert client.post("/api/feedback", json=payload).status_code == 422


# --- /api/stats --------------------------------------------------------------


def test_stats_aggregate_verdicts_types_and_feedback(client: TestClient) -> None:
    scam_ids = [analyze(client)["analysis_id"] for _ in range(2)]
    analyze(client, "Paise receive karne ke liye UPI PIN daalein")
    analyze(client, LEGIT_TEXT)
    client.post("/api/feedback", json={"analysis_id": scam_ids[0], "is_correct": True})
    client.post("/api/feedback", json={"analysis_id": scam_ids[1], "is_correct": False})

    body = client.get("/api/stats").json()
    assert body["total"] == 4
    assert body["verdicts"] == {"safe": 1, "suspicious": 0, "likely_scam": 3}
    assert [(t["scam_type"], t["count"]) for t in body["scam_types"]] == [
        ("account_threat", 2),
        ("upi_fraud", 1),
    ]
    assert body["scam_types"][0]["label"] == "Account blocking threat"
    assert body["feedback"] == {"correct": 1, "wrong": 1, "accuracy": 0.5}
    assert sum(day["total"] for day in body["daily"]) == 4
    assert sum(day["likely_scam"] for day in body["daily"]) == 3


def test_stats_labels_in_hindi(client: TestClient) -> None:
    analyze(client)
    body = client.get("/api/stats", params={"language": "hi"}).json()
    assert body["scam_types"][0]["label"] == "खाता बंद करने की धमकी"


def test_stats_on_empty_database(client: TestClient) -> None:
    body = client.get("/api/stats").json()
    assert body["total"] == 0
    assert body["feedback"]["accuracy"] is None
    assert body["daily"] == []


@pytest.mark.parametrize("days", [0, 366, "abc"])
def test_stats_validates_days(client: TestClient, days) -> None:
    assert client.get("/api/stats", params={"days": days}).status_code == 422


# --- Rate limiting and client identity ---------------------------------------


def test_rate_limit_returns_429(build) -> None:
    client = build(rate_limit_per_minute=2)
    for _ in range(2):
        analyze(client, "hello")
    response = client.post("/api/analyze", json={"text": "hello"})
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1
    assert response.json()["error"]["code"] == "rate_limited"


def test_bot_users_have_separate_limits(build) -> None:
    client = build(rate_limit_per_minute=1)
    for user in ("user-1", "user-2"):
        response = client.post(
            "/api/analyze",
            json={"text": "hello"},
            headers={"X-ScamShield-Key": BOT_KEY, "X-Client-Ref": user},
        )
        assert response.status_code == 200
    assert {r.channel for r in stored(client, Analysis)} == {"telegram"}


def test_client_ref_without_bot_key_is_ignored(build) -> None:
    client = build(rate_limit_per_minute=1)
    statuses = [
        client.post(
            "/api/analyze",
            json={"text": "hello"},
            headers={"X-ScamShield-Key": "wrong", "X-Client-Ref": user},
        ).status_code
        for user in ("user-1", "user-2")
    ]
    assert statuses == [200, 429]
    assert stored(client, Analysis)[0].channel == "web"


def test_forwarded_header_only_trusted_when_enabled(build) -> None:
    untrusted = build(rate_limit_per_minute=1)
    codes = [
        untrusted.post(
            "/api/analyze", json={"text": "hi"}, headers={"X-Forwarded-For": ip}
        ).status_code
        for ip in ("1.1.1.1", "2.2.2.2")
    ]
    assert codes == [200, 429]

    trusted = build(rate_limit_per_minute=1, trust_proxy_headers=True)
    codes = [
        trusted.post(
            "/api/analyze", json={"text": "hi"}, headers={"X-Forwarded-For": ip}
        ).status_code
        for ip in ("3.3.3.3", "4.4.4.4")
    ]
    assert codes == [200, 200]


# --- Cross-cutting -----------------------------------------------------------


def test_cors_allows_only_configured_origins(client: TestClient) -> None:
    allowed = client.options(
        "/api/analyze",
        headers={"Origin": "http://allowed.test", "Access-Control-Request-Method": "POST"},
    )
    assert allowed.headers["access-control-allow-origin"] == "http://allowed.test"

    denied = client.options(
        "/api/analyze",
        headers={"Origin": "http://evil.test", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in denied.headers


def test_security_headers(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["cache-control"] == "no-store"


def test_health(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok", "database": "ok", "model": "fake"}


def test_unknown_route_uses_error_format(client: TestClient) -> None:
    response = client.get("/api/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert client.get("/api/analyze").json()["error"]["code"] == "method_not_allowed"


def test_unexpected_errors_return_500_without_details(build) -> None:
    def failing_service(settings: Settings) -> AnalysisService:
        service = make_service(settings)

        async def explode(text: str):
            raise RuntimeError("secret internal detail")

        service.analyze = explode
        return service

    client = build(service_factory=failing_service, raise_errors=False)
    response = client.post("/api/analyze", json={"text": "hello"})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "secret internal detail" not in response.text


def test_docs_hidden_in_production(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, environment="production", hash_salt="x" * 32)
    app = create_app(settings, analysis_service=make_service(settings))
    with TestClient(app) as client:
        assert client.get("/api/docs").status_code == 404
