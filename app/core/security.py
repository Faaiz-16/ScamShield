import hashlib
import hmac

from pydantic import SecretStr


def hash_identifier(value: str, salt: SecretStr) -> str:
    """Keyed hash of a client identifier (IP address, Telegram user ID).

    HMAC with a secret salt means the stored value cannot be reversed by
    hashing every possible phone number or IP address.
    """
    return hmac.new(salt.get_secret_value().encode(), value.encode(), hashlib.sha256).hexdigest()


def secrets_match(provided: str | None, expected: SecretStr | None) -> bool:
    if not provided or expected is None:
        return False
    return hmac.compare_digest(provided.encode(), expected.get_secret_value().encode())
