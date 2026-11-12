"""Turn API results into Telegram HTML messages and inline keyboards."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Literal

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from bot.texts import text

FeedbackStage = Literal["ask", "choose_label", "done"]

# Callback data must stay under Telegram's 64-byte limit, so labels are coded.
LABEL_CODES = {"s": "scam", "l": "legit", "p": "spam_not_scam", "x": None}

VERDICT_MARKERS = {"safe": "[OK]", "suspicious": "[!]", "likely_scam": "[!!!]"}


@dataclass(frozen=True)
class Callback:
    action: Literal["lang", "feedback", "label", "setlang"]
    value: str
    analysis_id: str | None = None


def parse_callback(data: str | None) -> Callback | None:
    """Parse callback data built by the keyboards below; None if it is not ours."""
    if not data:
        return None
    parts = data.split(":", 2)
    match parts:
        case ["lang", ("en" | "hi") as language]:
            return Callback("lang", language)
        case ["setlang", ("en" | "hi") as language]:
            return Callback("setlang", language)
        case ["fb", ("y" | "n") as answer, analysis_id] if analysis_id:
            return Callback("feedback", answer, analysis_id)
        case ["fl", code, analysis_id] if code in LABEL_CODES and analysis_id:
            return Callback("label", code, analysis_id)
    return None


def other_language(language: str) -> str:
    return "en" if language == "hi" else "hi"


def render_result(result: dict[str, Any], language: str, feedback: FeedbackStage = "ask") -> str:
    texts = result["translations"].get(language) or result["translations"]["en"]
    marker = VERDICT_MARKERS.get(result["verdict"], "")
    risk = round(result["risk_score"] * 100)

    lines = [
        f"{marker} <b>{escape(texts['verdict_label'])}</b>  ({text(language, 'risk')}: {risk}%)"
    ]
    if texts.get("scam_type_label"):
        lines.append(f"<i>{escape(texts['scam_type_label'])}</i>")

    lines += ["", f"<b>{text(language, 'why')}</b>"]
    lines += [f"• {escape(reason)}" for reason in texts["reasons"]]

    lines += ["", f"<b>{text(language, 'what_to_do')}</b>"]
    lines += [f"{i}. {escape(action)}" for i, action in enumerate(texts["actions"], start=1)]

    if result.get("links"):
        lines += ["", f"<b>{text(language, 'links')}</b>"]
        for link, findings in zip(result["links"], texts["link_findings"], strict=False):
            # <code> keeps Telegram from turning the scam domain into a tappable link.
            domain = f"<code>{escape(link['domain'])}</code>"
            if link.get("is_official"):
                lines.append(f"• {domain}: {text(language, 'official')}")
            elif findings:
                lines.append(f"• {domain} ({text(language, 'risky')}): {escape(findings[0])}")
            else:
                lines.append(f"• {domain}: {text(language, 'no_link_issues')}")

    lines += ["", f"<i>{text(language, 'disclaimer')}</i>"]
    if result.get("analysis_id"):
        if feedback == "done":
            lines.append(text(language, "feedback_thanks"))
        else:
            lines.append(text(language, "feedback_question"))
    return "\n".join(lines)


def result_keyboard(
    result: dict[str, Any], language: str, feedback: FeedbackStage = "ask"
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    analysis_id = result.get("analysis_id")

    if analysis_id and feedback == "ask":
        rows.append(
            [
                InlineKeyboardButton(
                    text(language, "button_correct"), callback_data=f"fb:y:{analysis_id}"
                ),
                InlineKeyboardButton(
                    text(language, "button_wrong"), callback_data=f"fb:n:{analysis_id}"
                ),
            ]
        )
    elif analysis_id and feedback == "choose_label":
        rows += [
            [
                InlineKeyboardButton(
                    text(language, "button_scam"), callback_data=f"fl:s:{analysis_id}"
                ),
                InlineKeyboardButton(
                    text(language, "button_legit"), callback_data=f"fl:l:{analysis_id}"
                ),
            ],
            [
                InlineKeyboardButton(
                    text(language, "button_spam"), callback_data=f"fl:p:{analysis_id}"
                ),
                InlineKeyboardButton(
                    text(language, "button_unsure"), callback_data=f"fl:x:{analysis_id}"
                ),
            ],
        ]

    rows.append(
        [
            InlineKeyboardButton(
                text(language, "button_other_language"),
                callback_data=f"lang:{other_language(language)}",
            )
        ]
    )
    return InlineKeyboardMarkup(rows)


def language_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("English", callback_data="setlang:en"),
                InlineKeyboardButton("हिंदी", callback_data="setlang:hi"),
            ]
        ]
    )
