import time
from collections import OrderedDict
from collections.abc import Callable
from threading import Lock


class RateLimiter:
    """In-memory token bucket per client key.

    Each client may make ``per_minute`` requests per minute, with bursts up to
    the same number. State lives in this process only; when running several
    workers, each enforces its own limit (use a shared store such as Redis if
    that matters).
    """

    def __init__(
        self,
        per_minute: int,
        *,
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self._capacity = float(per_minute)
        self._refill_per_second = per_minute / 60.0
        self._max_clients = max_clients
        self._clock = clock
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._lock = Lock()

    def check(self, key: str) -> float:
        """Consume one token. Returns 0 if allowed, else seconds until retry."""
        now = self._clock()
        with self._lock:
            tokens, updated = self._buckets.get(key, (self._capacity, now))
            tokens = min(self._capacity, tokens + (now - updated) * self._refill_per_second)

            if tokens >= 1.0:
                tokens -= 1.0
                retry_after = 0.0
            else:
                retry_after = (1.0 - tokens) / self._refill_per_second

            self._buckets[key] = (tokens, now)
            self._buckets.move_to_end(key)
            while len(self._buckets) > self._max_clients:
                self._buckets.popitem(last=False)
        return retry_after
