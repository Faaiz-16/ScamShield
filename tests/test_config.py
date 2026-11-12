import pytest
from pydantic import ValidationError

from app.core.config import Settings


def make_settings(**overrides: object) -> Settings:
    # _env_file=None keeps a developer's local .env out of the tests.
    return Settings(_env_file=None, **overrides)


def test_defaults_are_valid() -> None:
    settings = make_settings()
    assert settings.suspicious_threshold < settings.scam_threshold
    assert settings.rules_path.exists()


def test_cors_origins_are_split_and_trimmed() -> None:
    settings = make_settings(cors_origins=" http://a.test , http://b.test ,")
    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]


def test_blank_api_keys_become_none() -> None:
    settings = make_settings(virustotal_api_key="  ", telegram_bot_token="")
    assert settings.virustotal_api_key is None
    assert settings.telegram_bot_token is None


def test_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="lower than"):
        make_settings(suspicious_threshold=0.7, scam_threshold=0.6)


def test_production_requires_real_salt() -> None:
    with pytest.raises(ValidationError, match="HASH_SALT"):
        make_settings(environment="production")
    assert make_settings(environment="production", hash_salt="a" * 32)
