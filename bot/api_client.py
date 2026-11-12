"""Async client for the ScamShield HTTP API, used by the Telegram bot."""

from __future__ import annotations

from typing import Any, Literal

import httpx
from pydantic import SecretStr

from app.core.security import hash_identifier

ErrorKind = Literal["rate_limited", "invalid", "not_found", "unavailable"]


class ApiError(Exception):
    def __init__(self, kind: ErrorKind, retry_after: int | None = None) -> None:
        super().__init__(kind)
        self.kind = kind
        self.retry_after = retry_after


class ScamShieldClient:
    def __init__(
        self,
        base_url: str,
        *,
        bot_key: SecretStr | None,
        salt: SecretStr,
        timeout: float = 20.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._bot_key = bot_key
        self._salt = salt
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(self, user_id: int) -> dict[str, str]:
        # The raw Telegram user ID never leaves the bot: the API only sees a
        # keyed hash, which it hashes again before storing.
        headers = {"X-Client-Ref": hash_identifier(f"tg-user:{user_id}", self._salt)}
        if self._bot_key:
            headers["X-ScamShield-Key"] = self._bot_key.get_secret_value()
        return headers

    async def _post(self, path: str, payload: dict[str, Any], user_id: int) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=payload, headers=self._headers(user_id))
        except httpx.HTTPError as exc:
            raise ApiError("unavailable") from exc

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "")
            raise ApiError("rate_limited", int(retry_after) if retry_after.isdigit() else None)
        if response.status_code == 422:
            raise ApiError("invalid")
        if response.status_code == 404:
            raise ApiError("not_found")
        if response.status_code >= 400:
            raise ApiError("unavailable")
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError("unavailable") from exc

    async def analyze(self, text: str, user_id: int, language: str) -> dict[str, Any]:
        return await self._post("/api/analyze", {"text": text, "language": language}, user_id)

    async def feedback(
        self, analysis_id: str, user_id: int, is_correct: bool, correct_label: str | None = None
    ) -> str:
        payload: dict[str, Any] = {"analysis_id": analysis_id, "is_correct": is_correct}
        if correct_label:
            payload["correct_label"] = correct_label
        body = await self._post("/api/feedback", payload, user_id)
        return str(body.get("status", "recorded"))
