from app.core.cache import TTLCache


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_entries_expire_after_ttl() -> None:
    clock = FakeClock()
    cache: TTLCache[str] = TTLCache(default_ttl=10, clock=clock)
    cache.set("a", "value")
    clock.now = 9.9
    assert cache.get("a") == "value"
    clock.now = 10.0
    assert cache.get("a") is None
    assert "a" not in cache


def test_per_entry_ttl_overrides_default() -> None:
    clock = FakeClock()
    cache: TTLCache[str] = TTLCache(default_ttl=100, clock=clock)
    cache.set("short", "x", ttl=1)
    clock.now = 2
    assert "short" not in cache


def test_none_values_are_cached() -> None:
    cache: TTLCache[str | None] = TTLCache(default_ttl=10)
    cache.set("missing", None)
    assert "missing" in cache
    assert cache.get("missing", "default") is None


def test_least_recently_used_entry_is_evicted() -> None:
    cache: TTLCache[int] = TTLCache(default_ttl=10, max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")
    cache.set("c", 3)
    assert "a" in cache and "c" in cache
    assert "b" not in cache
