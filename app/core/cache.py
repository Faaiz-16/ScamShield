import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from threading import Lock
from typing import Generic, TypeVar

V = TypeVar("V")

_MISSING = object()


class TTLCache(Generic[V]):
    """Small in-memory cache with per-entry expiry and LRU eviction.

    Values may legitimately be ``None`` (for example "WHOIS had no creation
    date"), so ``get`` takes an explicit default and ``__contains__`` is
    provided for presence checks.
    """

    def __init__(
        self,
        default_ttl: float,
        max_size: int = 2048,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._clock = clock
        self._data: OrderedDict[Hashable, tuple[float, V]] = OrderedDict()
        self._lock = Lock()

    def _lookup(self, key: Hashable) -> object:
        entry = self._data.get(key)
        if entry is None:
            return _MISSING
        expires_at, value = entry
        if expires_at <= self._clock():
            del self._data[key]
            return _MISSING
        self._data.move_to_end(key)
        return value

    def get(self, key: Hashable, default: V | None = None) -> V | None:
        with self._lock:
            value = self._lookup(key)
        return default if value is _MISSING else value  # type: ignore[return-value]

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return self._lookup(key) is not _MISSING

    def set(self, key: Hashable, value: V, ttl: float | None = None) -> None:
        expires_at = self._clock() + (self._default_ttl if ttl is None else ttl)
        with self._lock:
            self._data[key] = (expires_at, value)
            self._data.move_to_end(key)
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
