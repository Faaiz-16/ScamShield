import pytest
from pydantic import SecretStr

from app.core.rate_limit import RateLimiter
from app.core.security import hash_identifier, secrets_match


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_allows_burst_then_blocks() -> None:
    limiter = RateLimiter(3, clock=FakeClock())
    assert [limiter.check("a") for _ in range(3)] == [0, 0, 0]
    assert limiter.check("a") == pytest.approx(20.0)  # one token every 20 s at 3/min


def test_tokens_refill_over_time() -> None:
    clock = FakeClock()
    limiter = RateLimiter(60, clock=clock)
    for _ in range(60):
        limiter.check("a")
    assert limiter.check("a") > 0
    clock.now += 1.0
    assert limiter.check("a") == 0


def test_clients_are_independent() -> None:
    limiter = RateLimiter(1, clock=FakeClock())
    assert limiter.check("a") == 0
    assert limiter.check("a") > 0
    assert limiter.check("b") == 0


def test_idle_clients_are_evicted_beyond_capacity() -> None:
    limiter = RateLimiter(1, max_clients=2, clock=FakeClock())
    limiter.check("a")
    limiter.check("b")
    limiter.check("c")  # evicts "a", which then starts with a full bucket again
    assert limiter.check("a") == 0


def test_rejects_non_positive_rate() -> None:
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_hash_identifier_is_keyed_and_stable() -> None:
    salt_a, salt_b = SecretStr("salt-a"), SecretStr("salt-b")
    assert hash_identifier("1.2.3.4", salt_a) == hash_identifier("1.2.3.4", salt_a)
    assert hash_identifier("1.2.3.4", salt_a) != hash_identifier("1.2.3.4", salt_b)
    assert "1.2.3.4" not in hash_identifier("1.2.3.4", salt_a)
    assert len(hash_identifier("x", salt_a)) == 64


def test_secrets_match() -> None:
    assert secrets_match("key", SecretStr("key"))
    assert not secrets_match("other", SecretStr("key"))
    assert not secrets_match(None, SecretStr("key"))
    assert not secrets_match("key", None)
