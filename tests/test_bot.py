import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import select

from app.core.config import Settings
from app.main import create_app
from app.models import Analysis
from bot.api_client import ApiError, ScamShieldClient
from bot.formatting import parse_callback, render_result, result_keyboard
from bot.handlers import (
    MAX_STORED_RESULTS,
    RESULTS_KEY,
    analyze_message,
    handle_button,
    help_command,
    language_command,
    remember_result,
    start,
)
from bot.main import build_application
from bot.texts import TEXTS

SALT = SecretStr("bot-test-salt")
ANALYSIS_ID = "3c9089fe-a629-47a8-a3d1-c9f412d16be5"


def sample_result(**overrides: Any) -> dict[str, Any]:
    result = {
        "analysis_id": ANALYSIS_ID,
        "verdict": "likely_scam",
        "risk_score": 0.97,
        "links": [
            {"domain": "sbi-kyc.xyz", "is_official": False},
            {"domain": "onlinesbi.sbi", "is_official": True},
        ],
        "translations": {
            "en": {
                "verdict_label": "Likely Scam",
                "scam_type_label": "Fake KYC update",
                "reasons": ["It asks you to update KYC.", "The link imitates SBI."],
                "actions": ["Do not click any link.", "Call 1930."],
                "link_findings": [["The link imitates SBI."], []],
            },
            "hi": {
                "verdict_label": "संभावित धोखाधड़ी",
                "scam_type_label": "फर्जी केवाईसी अपडेट",
                "reasons": ["केवाईसी अपडेट करने को कहा गया है।", "लिंक एसबीआई की नकल है।"],
                "actions": ["लिंक पर क्लिक न करें।", "1930 पर कॉल करें।"],
                "link_findings": [["लिंक एसबीआई की नकल है।"], []],
            },
        },
    }
    result.update(overrides)
    return result


# --- API client --------------------------------------------------------------


def make_client(handler: Callable[[httpx.Request], httpx.Response], **kwargs) -> ScamShieldClient:
    http = httpx.AsyncClient(base_url="http://api.test", transport=httpx.MockTransport(handler))
    return ScamShieldClient(
        "http://api.test", bot_key=SecretStr("k"), salt=SALT, http_client=http, **kwargs
    )


async def test_client_sends_hashed_user_reference_and_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"verdict": "safe"})

    client = make_client(handler)
    assert await client.analyze("hello", user_id=123456789, language="hi") == {"verdict": "safe"}

    request = seen[0]
    assert request.url.path == "/api/analyze"
    assert json.loads(request.content) == {"text": "hello", "language": "hi"}
    assert request.headers["X-ScamShield-Key"] == "k"
    ref = request.headers["X-Client-Ref"]
    assert len(ref) == 64 and "123456789" not in ref


async def test_client_sends_feedback() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "updated"})

    status = await make_client(handler).feedback(ANALYSIS_ID, 1, False, "legit")
    assert status == "updated"
    assert seen == [{"analysis_id": ANALYSIS_ID, "is_correct": False, "correct_label": "legit"}]


@pytest.mark.parametrize(
    ("response", "kind", "retry_after"),
    [
        (httpx.Response(429, headers={"Retry-After": "12"}), "rate_limited", 12),
        (httpx.Response(422, json={}), "invalid", None),
        (httpx.Response(404, json={}), "not_found", None),
        (httpx.Response(500, json={}), "unavailable", None),
        (httpx.Response(200, content=b"not json"), "unavailable", None),
    ],
)
async def test_client_maps_errors(response, kind, retry_after) -> None:
    client = make_client(lambda request: response)
    with pytest.raises(ApiError) as caught:
        await client.analyze("hello", 1, "en")
    assert caught.value.kind == kind
    assert caught.value.retry_after == retry_after


async def test_client_handles_connection_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ApiError, match="unavailable"):
        await make_client(handler).analyze("hello", 1, "en")


async def test_client_against_real_api(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path / 'bot.db'}",
        database_auto_create=True,
        url_network_checks=False,
        classifier="none",
        hash_salt="api-salt",
        bot_api_key="shared-secret",
        rate_limit_per_minute=1,
    )
    with TestClient(create_app(settings)) as test_client:
        http = httpx.AsyncClient(
            base_url="http://api.test", transport=httpx.ASGITransport(app=test_client.app)
        )
        client = ScamShieldClient(
            "http://api.test", bot_key=SecretStr("shared-secret"), salt=SALT, http_client=http
        )
        first = await client.analyze("Please share the OTP you received", user_id=1, language="en")
        second = await client.analyze("Please share the OTP you received", user_id=2, language="hi")
        assert first["verdict"] == second["verdict"] == "likely_scam"
        assert set(first["translations"]) == {"en", "hi"}

        with pytest.raises(ApiError, match="rate_limited"):
            await client.analyze("hello again", user_id=1, language="en")

        assert await client.feedback(first["analysis_id"], 1, True) == "recorded"
        await http.aclose()

        with test_client.app.state.session_factory() as session:
            channels = session.scalars(select(Analysis.channel)).all()
        assert channels == ["telegram", "telegram"]


# --- Formatting --------------------------------------------------------------


def test_render_result_in_english() -> None:
    message = render_result(sample_result(), "en")
    assert "<b>Likely Scam</b>" in message
    assert "Risk: 97%" in message
    assert "• It asks you to update KYC." in message
    assert "1. Do not click any link." in message
    assert "<code>sbi-kyc.xyz</code> (risky): The link imitates SBI." in message
    assert "<code>onlinesbi.sbi</code>: official site" in message
    assert message.rstrip().endswith("Was this result correct?")


def test_render_result_in_hindi() -> None:
    message = render_result(sample_result(), "hi", feedback="done")
    assert "<b>संभावित धोखाधड़ी</b>" in message
    assert "जोखिम: 97%" in message
    assert message.rstrip().endswith(TEXTS["hi"]["feedback_thanks"])


def test_render_escapes_html() -> None:
    result = sample_result()
    result["translations"]["en"]["reasons"] = ["<script>alert(1)</script> & more"]
    result["links"] = [{"domain": "<b>evil</b>.xyz", "is_official": False}]
    result["translations"]["en"]["link_findings"] = [[]]
    message = render_result(result, "en")
    assert "<script>" not in message and "&lt;script&gt;" in message
    assert "&lt;b&gt;evil&lt;/b&gt;.xyz" in message


def test_render_without_analysis_id_has_no_feedback_prompt() -> None:
    message = render_result(sample_result(analysis_id=None), "en")
    assert "Was this result correct?" not in message


def buttons(markup) -> list[tuple[str, str]]:
    return [(b.text, b.callback_data) for row in markup.inline_keyboard for b in row]


def test_keyboards_for_each_feedback_stage() -> None:
    ask = buttons(result_keyboard(sample_result(), "en", "ask"))
    assert ask == [
        ("Correct", f"fb:y:{ANALYSIS_ID}"),
        ("Wrong", f"fb:n:{ANALYSIS_ID}"),
        ("हिंदी में देखें", "lang:hi"),
    ]
    choose = buttons(result_keyboard(sample_result(), "hi", "choose_label"))
    assert [data.split(":")[1] for _, data in choose[:4]] == ["s", "l", "p", "x"]
    assert choose[-1] == ("View in English", "lang:en")
    assert buttons(result_keyboard(sample_result(), "en", "done")) == [("हिंदी में देखें", "lang:hi")]
    assert all(len(data.encode()) <= 64 for _, data in ask + choose)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ("lang:hi", ("lang", "hi", None)),
        ("setlang:en", ("setlang", "en", None)),
        (f"fb:y:{ANALYSIS_ID}", ("feedback", "y", ANALYSIS_ID)),
        (f"fl:p:{ANALYSIS_ID}", ("label", "p", ANALYSIS_ID)),
        ("lang:fr", None),
        ("fb:maybe:x", None),
        ("fl:z:x", None),
        ("fb:y:", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_callback(data, expected) -> None:
    parsed = parse_callback(data)
    assert (parsed and (parsed.action, parsed.value, parsed.analysis_id)) == expected or (
        parsed is None and expected is None
    )


def test_both_languages_define_the_same_texts() -> None:
    assert set(TEXTS["en"]) == set(TEXTS["hi"])


# --- Handlers ----------------------------------------------------------------


class FakeApi:
    def __init__(self, result: dict | None = None, error: ApiError | None = None) -> None:
        self.analyze = AsyncMock(side_effect=error, return_value=result or sample_result())
        self.feedback = AsyncMock(return_value="recorded")


def make_context(api: FakeApi | None = None, **user_data) -> SimpleNamespace:
    return SimpleNamespace(
        user_data=dict(user_data),
        bot_data={"api": api or FakeApi()},
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
    )


def message_update(text: str | None = None, caption: str | None = None, language_code="en"):
    message = SimpleNamespace(
        text=text,
        caption=caption,
        chat_id=7,
        reply_text=AsyncMock(),
        reply_html=AsyncMock(return_value=SimpleNamespace(message_id=100)),
    )
    user = SimpleNamespace(id=42, language_code=language_code)
    return SimpleNamespace(effective_message=message, effective_user=user, callback_query=None)


def button_update(data: str, message_id: int = 100):
    query = SimpleNamespace(
        data=data,
        message=SimpleNamespace(message_id=message_id),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )
    user = SimpleNamespace(id=42, language_code="en")
    return SimpleNamespace(callback_query=query, effective_user=user, effective_message=None)


async def test_start_and_help_use_user_language() -> None:
    update = message_update(language_code="hi-IN")
    await start(update, make_context())
    assert "स्वागत" in update.effective_message.reply_html.await_args.args[0]

    update = message_update()
    await help_command(update, make_context(language="en"))
    assert "1930" in update.effective_message.reply_html.await_args.args[0]


async def test_language_command_shows_choices() -> None:
    update = message_update()
    await language_command(update, make_context())
    markup = update.effective_message.reply_text.await_args.kwargs["reply_markup"]
    assert [d for _, d in buttons(markup)] == ["setlang:en", "setlang:hi"]


async def test_message_is_analysed_and_result_remembered() -> None:
    api = FakeApi()
    context = make_context(api)
    update = message_update("Your KYC has expired, update now")
    await analyze_message(update, context)

    api.analyze.assert_awaited_once_with("Your KYC has expired, update now", 42, "en")
    context.bot.send_chat_action.assert_awaited_once()
    reply = update.effective_message.reply_html.await_args
    assert "<b>Likely Scam</b>" in reply.args[0]
    assert reply.kwargs["link_preview_options"].is_disabled
    assert context.user_data[RESULTS_KEY][100]["feedback"] == "ask"


async def test_forwarded_media_caption_is_used() -> None:
    api = FakeApi()
    await analyze_message(message_update(caption="Scan QR to receive Rs 5000"), make_context(api))
    assert api.analyze.await_args.args[0] == "Scan QR to receive Rs 5000"


@pytest.mark.parametrize(
    ("update", "expected_key"),
    [
        (message_update(None), "not_text"),
        (message_update("   "), "not_text"),
    ],
)
async def test_messages_without_text_are_rejected(update, expected_key) -> None:
    api = FakeApi()
    await analyze_message(update, make_context(api))
    api.analyze.assert_not_awaited()
    assert update.effective_message.reply_text.await_args.args[0] == TEXTS["en"][expected_key]


async def test_too_long_message_is_rejected() -> None:
    api = FakeApi()
    update = message_update("x" * 5001)
    await analyze_message(update, make_context(api))
    api.analyze.assert_not_awaited()
    assert "5000" in update.effective_message.reply_text.await_args.args[0]


@pytest.mark.parametrize(
    ("error", "fragment"),
    [
        (ApiError("rate_limited", 9), "9 seconds"),
        (ApiError("unavailable"), "not reachable"),
        (ApiError("invalid"), "could not read"),
    ],
)
async def test_api_errors_become_friendly_replies(error, fragment) -> None:
    update = message_update("hello")
    await analyze_message(update, make_context(FakeApi(error=error)))
    assert fragment in update.effective_message.reply_text.await_args.args[0]


async def test_language_button_redraws_result() -> None:
    context = make_context()
    remember_result(context, 100, sample_result(), "en")
    update = button_update("lang:hi")
    await handle_button(update, context)

    edited = update.callback_query.edit_message_text.await_args
    assert "संभावित धोखाधड़ी" in edited.args[0]
    assert context.user_data["language"] == "hi"
    assert context.user_data[RESULTS_KEY][100]["language"] == "hi"


async def test_correct_button_sends_feedback() -> None:
    api = FakeApi()
    context = make_context(api)
    remember_result(context, 100, sample_result(), "en")
    update = button_update(f"fb:y:{ANALYSIS_ID}")
    await handle_button(update, context)

    api.feedback.assert_awaited_once_with(ANALYSIS_ID, 42, True, None)
    assert context.user_data[RESULTS_KEY][100]["feedback"] == "done"
    assert "Thank you" in update.callback_query.edit_message_text.await_args.args[0]


async def test_wrong_button_asks_for_label_then_sends_it() -> None:
    api = FakeApi()
    context = make_context(api)
    remember_result(context, 100, sample_result(), "en")

    await handle_button(button_update(f"fb:n:{ANALYSIS_ID}"), context)
    api.feedback.assert_not_awaited()
    assert context.user_data[RESULTS_KEY][100]["feedback"] == "choose_label"

    await handle_button(button_update(f"fl:l:{ANALYSIS_ID}"), context)
    api.feedback.assert_awaited_once_with(ANALYSIS_ID, 42, False, "legit")


async def test_unsure_label_sends_no_label() -> None:
    api = FakeApi()
    context = make_context(api)
    remember_result(context, 100, sample_result(), "en")
    await handle_button(button_update(f"fl:x:{ANALYSIS_ID}"), context)
    api.feedback.assert_awaited_once_with(ANALYSIS_ID, 42, False, None)


async def test_feedback_failure_keeps_buttons() -> None:
    api = FakeApi()
    api.feedback.side_effect = ApiError("unavailable")
    context = make_context(api)
    remember_result(context, 100, sample_result(), "en")
    update = button_update(f"fb:y:{ANALYSIS_ID}")
    await handle_button(update, context)

    assert update.callback_query.answer.await_args.kwargs["show_alert"] is True
    update.callback_query.edit_message_text.assert_not_awaited()
    assert context.user_data[RESULTS_KEY][100]["feedback"] == "ask"


async def test_expired_result_shows_alert() -> None:
    update = button_update("lang:hi", message_id=999)
    await handle_button(update, make_context())
    assert update.callback_query.answer.await_args.kwargs["show_alert"] is True
    update.callback_query.edit_message_text.assert_not_awaited()


async def test_mismatched_analysis_id_is_ignored() -> None:
    api = FakeApi()
    context = make_context(api)
    remember_result(context, 100, sample_result(), "en")
    update = button_update("fb:y:00000000-0000-4000-8000-000000000000")
    await handle_button(update, context)
    api.feedback.assert_not_awaited()
    update.callback_query.edit_message_text.assert_not_awaited()


async def test_set_language_button() -> None:
    context = make_context()
    update = button_update("setlang:hi")
    await handle_button(update, context)
    assert context.user_data["language"] == "hi"
    assert update.callback_query.edit_message_text.await_args.args[0] == TEXTS["hi"]["language_set"]


def test_only_recent_results_are_kept() -> None:
    context = make_context()
    for message_id in range(MAX_STORED_RESULTS + 5):
        remember_result(context, message_id, sample_result(), "en")
    stored = context.user_data[RESULTS_KEY]
    assert len(stored) == MAX_STORED_RESULTS
    assert min(stored) == 5


# --- Application wiring ------------------------------------------------------


def test_build_application_requires_token() -> None:
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        build_application(Settings(_env_file=None, telegram_bot_token=None))


def test_build_application_registers_handlers() -> None:
    settings = Settings(_env_file=None, telegram_bot_token="123456:TEST-TOKEN")
    application = build_application(settings)
    assert isinstance(application.bot_data["api"], ScamShieldClient)
    assert len(application.handlers[0]) == 5
    assert application.error_handlers
