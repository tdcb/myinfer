"""Single-owner prompt prefix cache.

Entries are moved out of the cache while a request uses them (no deep copies),
so reusing a conversation's KV costs no extra memory.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache


def cache_nbytes(cache: List[Any]) -> int:
    total = 0
    for c in cache:
        try:
            total += c.nbytes
        except (NotImplementedError, AttributeError, TypeError):
            pass
    return total


def cache_offset(cache: List[Any]) -> Optional[int]:
    """Number of tokens held by the attention caches (None if no positional cache)."""
    for c in cache:
        off = getattr(c, "offset", None)
        if isinstance(off, int):
            return off
        sub = getattr(c, "caches", None)
        if sub:
            off = cache_offset(list(sub))
            if off is not None:
                return off
    return None


def common_prefix_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class _Entry:
    key: str
    tokens: List[int]
    cache: List[Any]
    nbytes: int
    last_used: float = field(default_factory=time.monotonic)


class PrefixCache:
    def __init__(self, max_entries: int = 8):
        self.max_entries = max_entries
        self._entries: List[_Entry] = []
        self._lock = threading.Lock()
        self.total_hit_tokens = 0
        self.total_prompt_tokens = 0

    @property
    def nbytes(self) -> int:
        return sum(e.nbytes for e in self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def take(self, key: str, tokens: List[int]) -> Tuple[Optional[List[Any]], int]:
        """Remove and return the best cache for `tokens` and the number of reusable tokens.

        At least one prompt token is always left to be processed by the model.
        """
        with self._lock:
            self.total_prompt_tokens += len(tokens)
            best, best_len = None, 0
            for e in self._entries:
                if e.key != key:
                    continue
                n = common_prefix_len(e.tokens, tokens)
                n = min(n, len(tokens) - 1)
                if n < len(e.tokens) and not can_trim_prompt_cache(e.cache):
                    continue
                if n > best_len:
                    best, best_len = e, n
            if best is None:
                return None, 0
            self._entries.remove(best)
        excess = len(best.tokens) - best_len
        if excess > 0:
            trim_prompt_cache(best.cache, excess)
        self.total_hit_tokens += best_len
        return best.cache, best_len

    def put(self, key: str, tokens: List[int], cache: List[Any]) -> None:
        offsets = {getattr(c, "offset", None) for c in cache}
        offsets.discard(None)
        if len(offsets) > 1:
            return  # caches disagree on length (e.g. after speculative decoding); not reusable
        off = cache_offset(cache)
        if off is not None:
            tokens = tokens[:off]
        if not tokens:
            return
        entry = _Entry(key, list(tokens), cache, cache_nbytes(cache))
        with self._lock:
            self._entries = [
                e
                for e in self._entries
                if not (e.key == key and common_prefix_len(e.tokens, tokens) == len(e.tokens))
            ]
            self._entries.append(entry)
            self._entries.sort(key=lambda e: e.last_used)
            while len(self._entries) > self.max_entries:
                self._entries.pop(0)

    def evict_to(self, max_bytes: int) -> int:
        """Drop least-recently-used entries until the cache uses at most `max_bytes`."""
        freed = 0
        with self._lock:
            self._entries.sort(key=lambda e: e.last_used)
            while self._entries and self.nbytes > max(0, max_bytes):
                freed += self._entries.pop(0).nbytes
        return freed
