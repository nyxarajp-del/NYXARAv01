"""Error taxonomy + self-heal + learn-once.

Errors are categorised as *transient* (retry / back off) or *permanent* (escalate, do
not retry). A tiny in-process ledger records the first occurrence of each distinct
failure so the system "learns once" — repeated identical permanent errors short-circuit
instead of re-attempting the same doomed path.
"""
from __future__ import annotations

import enum
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable


class Category(str, enum.Enum):
    TRANSIENT = "transient"     # network blip, rate limit, timeout — safe to retry
    PERMANENT = "permanent"     # bug, contract violation — do not retry blindly
    INVARIANT = "invariant"     # a non-negotiable was breached — halt / escalate
    RESOURCE = "resource"       # budget / limit exhausted — back off, not a bug


class NyxaraError(Exception):
    """Base for all kernel-aware errors."""

    category: Category = Category.PERMANENT

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class TransientError(NyxaraError):
    category = Category.TRANSIENT


class PermanentError(NyxaraError):
    category = Category.PERMANENT


class InvariantBreach(NyxaraError):
    """A formal non-negotiable was violated. The kernel fails closed on these."""

    category = Category.INVARIANT


class ResourceExhausted(NyxaraError):
    category = Category.RESOURCE


def categorize(exc: BaseException) -> Category:
    """Best-effort classification of an arbitrary exception."""
    if isinstance(exc, NyxaraError):
        return exc.category
    transient = (TimeoutError, ConnectionError, ConnectionResetError)
    if isinstance(exc, transient):
        return Category.TRANSIENT
    if isinstance(exc, (MemoryError, OSError)):
        return Category.RESOURCE
    return Category.PERMANENT


@dataclass
class _Record:
    signature: str
    category: Category
    count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    note: str = ""


class ErrorLedger:
    """Learn-once memory of failures keyed by a stable signature."""

    def __init__(self) -> None:
        self._records: dict[str, _Record] = {}

    @staticmethod
    def signature(exc: BaseException) -> str:
        tb = traceback.extract_tb(exc.__traceback__)
        site = f"{tb[-1].filename}:{tb[-1].lineno}" if tb else "?"
        return f"{type(exc).__name__}@{site}"

    def record(self, exc: BaseException, note: str = "") -> _Record:
        sig = self.signature(exc)
        rec = self._records.get(sig)
        if rec is None:
            rec = _Record(signature=sig, category=categorize(exc), note=note)
            self._records[sig] = rec
        rec.count += 1
        rec.last_seen = time.time()
        return rec

    def seen_before(self, exc: BaseException) -> bool:
        return self.signature(exc) in self._records

    def should_retry(self, exc: BaseException) -> bool:
        """Retry transient errors; retry a permanent one at most once."""
        cat = categorize(exc)
        if cat is Category.TRANSIENT:
            return True
        if cat in (Category.INVARIANT,):
            return False
        return not self.seen_before(exc)


def with_self_heal(
    fn: Callable[[], object],
    *,
    ledger: ErrorLedger,
    retries: int = 3,
    base_delay: float = 0.5,
) -> object:
    """Run ``fn`` with category-aware retry + exponential backoff.

    Invariant breaches are never retried — they re-raise immediately so the kernel can
    fail closed.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except InvariantBreach:
            raise
        except BaseException as exc:  # noqa: BLE001 — deliberate broad catch at boundary
            rec = ledger.record(exc)
            attempt += 1
            if attempt > retries or not ledger.should_retry(exc):
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)))


__all__ = [
    "Category",
    "NyxaraError",
    "TransientError",
    "PermanentError",
    "InvariantBreach",
    "ResourceExhausted",
    "categorize",
    "ErrorLedger",
    "with_self_heal",
]
