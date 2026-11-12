from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

_INSECURE_SALTS = {"", "change-me"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "ScamShield"
    environment: Literal["development", "test", "production"] = "development"
    debug: bool = False
    default_language: Literal["en", "hi"] = "en"

    database_url: str = (
        "mysql+pymysql://scamshield:change-me@localhost:3306/scamshield?charset=utf8mb4"
    )
    database_auto_create: bool = False
    hash_salt: SecretStr = SecretStr("change-me")

    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"
    rate_limit_per_minute: int = 30
    trust_proxy_headers: bool = False
    bot_api_key: SecretStr | None = None

    suspicious_threshold: float = 0.35
    scam_threshold: float = 0.65
    verdict_rule_weight: float = 0.9
    verdict_url_weight: float = 0.85
    verdict_ml_weight: float = 0.6
    verdict_official_link_factor: float = 0.5

    rules_path: Path = BASE_DIR / "app" / "detection" / "rules.yaml"
    # Which trained model the API uses: "baseline", "muril" or "none".
    classifier: str = "baseline"
    # Device for transformer inference: cpu, cuda, mps or auto.
    model_device: str = "cpu"
    baseline_model_path: Path = BASE_DIR / "ml" / "models" / "baseline.joblib"
    muril_model_dir: Path = BASE_DIR / "ml" / "models" / "muril"

    url_network_checks: bool = True
    url_request_timeout: float = 5.0
    url_max_redirects: int = 5
    url_max_response_bytes: int = 64 * 1024
    url_max_per_message: int = 5
    young_domain_days: int = 30
    whois_cache_ttl_seconds: int = 24 * 3600
    reputation_cache_ttl_seconds: int = 6 * 3600

    google_safe_browsing_api_key: SecretStr | None = None
    virustotal_api_key: SecretStr | None = None

    telegram_bot_token: SecretStr | None = None
    api_base_url: str = "http://localhost:8000"

    @field_validator(
        "google_safe_browsing_api_key",
        "virustotal_api_key",
        "telegram_bot_token",
        "bot_api_key",
        mode="before",
    )
    @classmethod
    def _blank_secret_to_none(cls, value: object) -> object:
        # A blank line in .env should mean "not configured", not an empty key.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("suspicious_threshold", "scam_threshold")
    @classmethod
    def _threshold_in_range(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("thresholds must be between 0 and 1 (exclusive)")
        return value

    @field_validator(
        "verdict_rule_weight",
        "verdict_url_weight",
        "verdict_ml_weight",
        "verdict_official_link_factor",
    )
    @classmethod
    def _weight_in_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("verdict weights must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.suspicious_threshold >= self.scam_threshold:
            raise ValueError("SUSPICIOUS_THRESHOLD must be lower than SCAM_THRESHOLD")
        if self.environment == "production" and (
            self.hash_salt.get_secret_value() in _INSECURE_SALTS
        ):
            raise ValueError("HASH_SALT must be set to a random secret in production")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
