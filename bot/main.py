"""Run the ScamShield Telegram bot.

Usage:
    python -m bot.main

Needs TELEGRAM_BOT_TOKEN in .env and the API running at API_BASE_URL.
"""

import logging
import sys

from telegram import BotCommand, Update
from telegram.ext import Application

from app.core.config import Settings, get_settings
from bot.api_client import ScamShieldClient
from bot.handlers import register_handlers

logger = logging.getLogger("scamshield.bot")

COMMANDS = {
    "en": [
        BotCommand("start", "Introduction"),
        BotCommand("help", "How to use ScamShield"),
        BotCommand("language", "Switch between English and Hindi"),
    ],
    "hi": [
        BotCommand("start", "परिचय"),
        BotCommand("help", "ScamShield का उपयोग कैसे करें"),
        BotCommand("language", "अंग्रेज़ी और हिंदी के बीच बदलें"),
    ],
}


def build_application(settings: Settings) -> Application:
    if settings.telegram_bot_token is None:
        raise ValueError("TELEGRAM_BOT_TOKEN is not set")

    client = ScamShieldClient(
        settings.api_base_url, bot_key=settings.bot_api_key, salt=settings.hash_salt
    )

    async def post_init(application: Application) -> None:
        await application.bot.set_my_commands(COMMANDS["en"])
        await application.bot.set_my_commands(COMMANDS["hi"], language_code="hi")

    async def post_shutdown(application: Application) -> None:
        await client.aclose()

    application = (
        Application.builder()
        .token(settings.telegram_bot_token.get_secret_value())
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    application.bot_data["api"] = client
    register_handlers(application)
    return application


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # httpx logs every request URL at INFO level, and Telegram URLs contain the bot token.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()
    try:
        application = build_application(settings)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if settings.bot_api_key is None:
        logger.warning(
            "BOT_API_KEY is not set: all Telegram users will share one rate limit on the API"
        )
    logger.info("starting bot, API at %s", settings.api_base_url)
    application.run_polling(allowed_updates=[Update.MESSAGE, Update.CALLBACK_QUERY])
    return 0


if __name__ == "__main__":
    sys.exit(main())
