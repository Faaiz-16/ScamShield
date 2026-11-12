from __future__ import annotations

import logging
from typing import Any

from telegram import LinkPreviewOptions, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from bot.api_client import ApiError, ScamShieldClient
from bot.formatting import (
    LABEL_CODES,
    language_keyboard,
    parse_callback,
    render_result,
    result_keyboard,
)
from bot.texts import text

logger = logging.getLogger(__name__)

MAX_CHARS = 5000
LANGUAGE_KEY = "language"
RESULTS_KEY = "results"
# Results are kept in memory so the language and feedback buttons can redraw
# them. Only the most recent ones per user are kept.
MAX_STORED_RESULTS = 20
NO_PREVIEW = LinkPreviewOptions(is_disabled=True)


def user_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    stored = context.user_data.get(LANGUAGE_KEY)
    if stored in ("en", "hi"):
        return stored
    code = update.effective_user.language_code if update.effective_user else None
    return "hi" if (code or "").lower().startswith("hi") else "en"


def api_client(context: ContextTypes.DEFAULT_TYPE) -> ScamShieldClient:
    return context.bot_data["api"]


def remember_result(
    context: ContextTypes.DEFAULT_TYPE, message_id: int, result: dict[str, Any], language: str
) -> None:
    results: dict[int, dict[str, Any]] = context.user_data.setdefault(RESULTS_KEY, {})
    results[message_id] = {"result": result, "language": language, "feedback": "ask"}
    while len(results) > MAX_STORED_RESULTS:
        results.pop(next(iter(results)))


def error_message(language: str, error: ApiError) -> str:
    if error.kind == "rate_limited":
        return text(language, "rate_limited", seconds=error.retry_after or 30)
    if error.kind == "invalid":
        return text(language, "invalid")
    return text(language, "unavailable")


# --- Commands ----------------------------------------------------------------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_html(text(user_language(update, context), "start"))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_html(
        text(user_language(update, context), "help"), link_preview_options=NO_PREVIEW
    )


async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        text(user_language(update, context), "choose_language"), reply_markup=language_keyboard()
    )


# --- Messages ----------------------------------------------------------------


async def analyze_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    language = user_language(update, context)
    content = (message.text or message.caption or "").strip()

    if not content:
        await message.reply_text(text(language, "not_text"))
        return
    if len(content) > MAX_CHARS:
        await message.reply_text(text(language, "too_long", max=MAX_CHARS))
        return

    await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)
    try:
        result = await api_client(context).analyze(content, update.effective_user.id, language)
    except ApiError as exc:
        await message.reply_text(error_message(language, exc))
        return

    sent = await message.reply_html(
        render_result(result, language),
        reply_markup=result_keyboard(result, language),
        link_preview_options=NO_PREVIEW,
        do_quote=True,
    )
    remember_result(context, sent.message_id, result, language)


# --- Buttons -----------------------------------------------------------------


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    callback = parse_callback(query.data)
    if callback is None:
        await query.answer()
        return

    if callback.action == "setlang":
        context.user_data[LANGUAGE_KEY] = callback.value
        await query.answer()
        await query.edit_message_text(text(callback.value, "language_set"))
        return

    entry = context.user_data.get(RESULTS_KEY, {}).get(query.message.message_id)
    if entry is None:
        await query.answer(text(user_language(update, context), "expired"), show_alert=True)
        return

    result = entry["result"]
    if callback.action == "lang":
        entry["language"] = callback.value
        context.user_data[LANGUAGE_KEY] = callback.value
    elif callback.analysis_id != result.get("analysis_id"):
        await query.answer()
        return
    elif callback.action == "feedback" and callback.value == "n":
        entry["feedback"] = "choose_label"
    else:
        is_correct = callback.action == "feedback"
        label = None if is_correct else LABEL_CODES[callback.value]
        try:
            await api_client(context).feedback(
                result["analysis_id"], update.effective_user.id, is_correct, label
            )
        except ApiError:
            await query.answer(text(entry["language"], "feedback_failed"), show_alert=True)
            return
        entry["feedback"] = "done"

    await query.answer()
    await query.edit_message_text(
        render_result(result, entry["language"], entry["feedback"]),
        parse_mode=ParseMode.HTML,
        reply_markup=result_keyboard(result, entry["language"], entry["feedback"]),
        link_preview_options=NO_PREVIEW,
    )


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Message text is deliberately not logged.
    logger.error("error while handling an update", exc_info=context.error)


def register_handlers(application: Application) -> None:
    private = filters.ChatType.PRIVATE
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("language", language_command))
    application.add_handler(
        MessageHandler(private & ~filters.COMMAND & ~filters.StatusUpdate.ALL, analyze_message)
    )
    application.add_handler(CallbackQueryHandler(handle_button))
    application.add_error_handler(on_error)
