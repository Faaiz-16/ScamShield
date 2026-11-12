import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.database import session_scope
from app.core.rate_limit import RateLimiter
from app.core.security import hash_identifier, secrets_match
from app.services.analysis import AnalysisService

BOT_KEY_HEADER = "X-ScamShield-Key"
CLIENT_REF_HEADER = "X-Client-Ref"


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_session(request: Request) -> Iterator[Session]:
    yield from session_scope(request.app.state.session_factory)


def get_analysis_service(request: Request) -> AnalysisService:
    return request.app.state.analysis


@dataclass(frozen=True)
class Client:
    hash: str
    channel: Literal["web", "telegram"]


def get_client(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    bot_key: Annotated[str | None, Header(alias=BOT_KEY_HEADER)] = None,
    client_ref: Annotated[str | None, Header(alias=CLIENT_REF_HEADER, max_length=128)] = None,
) -> Client:
    """Identify the caller by a salted hash, never by a raw IP or user ID.

    Requests from the Telegram bot carry a shared secret plus an opaque
    per-user reference, so each Telegram user is rate limited separately even
    though all bot traffic comes from one IP. Anyone else is identified by IP,
    and a client reference sent without the secret is ignored.
    """
    if client_ref and secrets_match(bot_key, settings.bot_api_key):
        return Client(hash_identifier(f"tg:{client_ref}", settings.hash_salt), "telegram")

    ip = request.client.host if request.client else "unknown"
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        ip = forwarded.split(",")[0].strip() or ip
    return Client(hash_identifier(f"ip:{ip}", settings.hash_salt), "web")


def rate_limit(scope: str) -> Callable[..., None]:
    def dependency(request: Request, client: Annotated[Client, Depends(get_client)]) -> None:
        limiter: RateLimiter = request.app.state.rate_limiter
        retry_after = limiter.check(f"{scope}:{client.hash}")
        if retry_after:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please wait a moment and try again.",
                headers={"Retry-After": str(math.ceil(retry_after))},
            )

    return dependency
