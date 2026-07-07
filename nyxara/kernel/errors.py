"""NYXARA · kernel/errors.py — error taxonomy, classification, self-heal, learn-once.

Every failure in NYXARA flows through this module so the kernel can decide,
*uniformly*, whether to retry, heal, escalate, or fail-closed.

Pillars
-------
1. **Rich error type** — :class:`NyxaraError` carries a stable ``code``, a
   :class:`ErrorCategory`, a :class:`Severity`, structured ``context``, the
   originating ``cause``, a monotonic ``occurred_at`` and a ``trace_id``.
2. **Classification** — :func:`classify` maps *any* exception (incl. stdlib /
   third-party) onto a category so callers can ask the single question that
   matters: *is this transient (retry) or permanent (give up / heal)?*
3. **Retry** — :class:`RetryPolicy` + :func:`retry` / :func:`aretry` decorators
   implement exponential backoff with full jitter, honouring categories
   (never retry PERMANENT / SECURITY / INVARIANT).
4. **Self-heal** — :class:`SelfHealRegistry` lets subsystems register remediation
   callbacks keyed by error type; the kernel tries them before surfacing.
5. **Learn-once** — :class:`ErrorLedger` fingerprints errors, dedupes them,
   tracks frequency, and remembers what resolved them — so the same failure is
   diagnosed once, not forever.

Security note (Rules 5, 8): SECURITY and INVARIANT errors are **never** retried,
never auto-healed silently, and always fail-closed.

Zero internal dependencies — depends only on the standard library.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import random
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

__all__ = [
    "ErrorCategory",
    "Severity",
    "NyxaraError",
    "ConfigurationError",
    "InvariantViolation",
    "SecurityError",
    "PromptInjectionError",
    "AuthorizationError",
    "ResourceExhausted",
    "TimeoutErrorNX",
    "ExternalServiceError",
    "LLMError",
    "ToolError",
    "MemoryError_",
    "ValidationError",
    "CorrigibilityError",
    "classify",
    "is_retryable",
    "RetryPolicy",
    "retry",
    "aretry",
    "SelfHealRegistry",
    "HEAL",
    "ErrorRecord",
    "ErrorLedger",
    "LEDGER",
    "wrap",
]

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Category & severity
# --------------------------------------------------------------------------- #
class ErrorCategory(str, Enum):
    """Coarse class that determines the kernel's reflex."""

    TRANSIENT = "transient"          # retry may succeed (network blip, timeout, rate-limit)
    PERMANENT = "permanent"          # retrying is pointless (bad input, 404, logic bug)
    RESOURCE = "resource"            # exhausted budget/quota/memory — back off, maybe heal
    EXTERNAL = "external"            # downstream service fault — degrade/fallback
    SECURITY = "security"            # injection, auth, tampering — FAIL CLOSED, never retry
    INVARIANT = "invariant"          # non-negotiable core violated — HALT, never proceed
    CORRIGIBILITY = "corrigibility"  # attempt to resist correction/stop — escalate to owner
    USER = "user"                    # caller mistake — report, do not retry
    UNKNOWN = "unknown"              # unclassified — treat conservatively (no auto-retry)

    @property
    def retryable(self) -> bool:
        return self in (ErrorCategory.TRANSIENT, ErrorCategory.EXTERNAL, ErrorCategory.RESOURCE)

    @property
    def fail_closed(self) -> bool:
        return self in (ErrorCategory.SECURITY, ErrorCategory.INVARIANT, ErrorCategory.CORRIGIBILITY)


class Severity(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    FATAL = 60  # reserved for invariant/corrigibility breaches

    @property
    def label(self) -> str:
        return self.name.lower()


# --------------------------------------------------------------------------- #
# Base error
# --------------------------------------------------------------------------- #
class NyxaraError(Exception):
    """Root of all NYXARA errors. Carries structured, loggable context.

    Parameters
    ----------
    message:  human-readable summary.
    code:     stable machine code (defaults to the class name).
    category: :class:`ErrorCategory`; subclasses set a sensible default.
    severity: :class:`Severity`.
    retryable: explicit override; falls back to ``category.retryable``.
    context:  arbitrary JSON-ish dict of relevant state (auto-redact secrets upstream).
    cause:    the originating exception (chains via ``raise ... from``).
    trace_id: correlation id for distributed tracing (auto-generated if absent).
    """

    default_category: ErrorCategory = ErrorCategory.UNKNOWN
    default_severity: Severity = Severity.ERROR

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        category: Optional[ErrorCategory] = None,
        severity: Optional[Severity] = None,
        retryable: Optional[bool] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.category = category or self.default_category
        self.severity = severity or self.default_severity
        self._retryable_override = retryable
        self.context: Dict[str, Any] = dict(context or {})
        self.cause = cause
        self.trace_id = trace_id or uuid.uuid4().hex
        self.occurred_at = time.time()
        if cause is not None and self.__cause__ is None:
            self.__cause__ = cause

    @property
    def retryable(self) -> bool:
        if self._retryable_override is not None:
            return self._retryable_override
        return self.category.retryable

    @property
    def fail_closed(self) -> bool:
        return self.category.fail_closed

    def fingerprint(self) -> str:
        """Stable signature for dedup/learn-once (ignores volatile context values)."""
        origin = ""
        tb = self.__traceback__ or (self.cause.__traceback__ if self.cause else None)
        if tb is not None:
            last = traceback.extract_tb(tb)[-1:]
            if last:
                fr = last[0]
                origin = f"{fr.filename}:{fr.name}"
        basis = f"{self.code}|{self.category.value}|{origin}|{type(self.cause).__name__ if self.cause else ''}"
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.label,
            "retryable": self.retryable,
            "fail_closed": self.fail_closed,
            "trace_id": self.trace_id,
            "occurred_at": self.occurred_at,
            "fingerprint": self.fingerprint(),
            "context": self.context,
            "cause": repr(self.cause) if self.cause else None,
        }

    def with_context(self, **kv: Any) -> "NyxaraError":
        self.context.update(kv)
        return self

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.code}/{self.category.value}/{self.severity.label}] {self.message}"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}({self.message!r}, code={self.code!r}, category={self.category.value!r})"


# --------------------------------------------------------------------------- #
# Concrete hierarchy
# --------------------------------------------------------------------------- #
class ConfigurationError(NyxaraError):
    default_category = ErrorCategory.PERMANENT
    default_severity = Severity.CRITICAL


class ValidationError(NyxaraError):
    """A proposal/schema check failed (mind/proposal.py)."""

    default_category = ErrorCategory.PERMANENT
    default_severity = Severity.WARNING


class InvariantViolation(NyxaraError):
    """A formal non-negotiable (kernel/invariants.py) was breached. HALT."""

    default_category = ErrorCategory.INVARIANT
    default_severity = Severity.FATAL


class CorrigibilityError(NyxaraError):
    """Something attempted to resist correction/stop. Escalate to owner."""

    default_category = ErrorCategory.CORRIGIBILITY
    default_severity = Severity.FATAL


class SecurityError(NyxaraError):
    default_category = ErrorCategory.SECURITY
    default_severity = Severity.CRITICAL


class PromptInjectionError(SecurityError):
    """Untrusted content tried to hijack the system (guard/shield.py)."""


class AuthorizationError(SecurityError):
    """Identity/authority check failed (guard/auth.py, Rule 7/8)."""


class ResourceExhausted(NyxaraError):
    default_category = ErrorCategory.RESOURCE
    default_severity = Severity.WARNING


class TimeoutErrorNX(NyxaraError):
    default_category = ErrorCategory.TRANSIENT
    default_severity = Severity.WARNING


class ExternalServiceError(NyxaraError):
    default_category = ErrorCategory.EXTERNAL
    default_severity = Severity.ERROR


class LLMError(ExternalServiceError):
    """A provider call failed (mind/llm.py)."""


class ToolError(NyxaraError):
    default_category = ErrorCategory.EXTERNAL
    default_severity = Severity.ERROR


class MemoryError_(NyxaraError):
    """Memory subsystem fault (named with underscore to avoid shadowing builtin)."""

    default_category = ErrorCategory.PERMANENT
    default_severity = Severity.ERROR


# --------------------------------------------------------------------------- #
# Classification of arbitrary exceptions
# --------------------------------------------------------------------------- #
# stdlib / common exception → category mapping (checked by isinstance, most-specific first).
_STDLIB_MAP: Tuple[Tuple[Tuple[type, ...], ErrorCategory], ...] = (
    ((TimeoutError, asyncio.TimeoutError, ConnectionResetError, ConnectionAbortedError), ErrorCategory.TRANSIENT),
    ((ConnectionError, BrokenPipeError, InterruptedError), ErrorCategory.EXTERNAL),
    ((MemoryError, BufferError), ErrorCategory.RESOURCE),
    ((PermissionError,), ErrorCategory.SECURITY),
    ((ValueError, TypeError, KeyError, IndexError, AttributeError, LookupError,
      ArithmeticError, AssertionError, NotImplementedError), ErrorCategory.PERMANENT),
    ((FileNotFoundError, IsADirectoryError, NotADirectoryError), ErrorCategory.PERMANENT),
    ((OSError, IOError), ErrorCategory.EXTERNAL),
)

# Heuristic substrings for opaque third-party errors (e.g. httpx/transformers) when type is unhelpful.
_TRANSIENT_HINTS = ("timeout", "timed out", "temporar", "rate limit", "429", "503", "502",
                    "overloaded", "connection reset", "try again", "unavailable")
_SECURITY_HINTS = ("unauthorized", "forbidden", "401", "403", "invalid api key",
                   "authentication", "permission denied")


def classify(exc: BaseException) -> ErrorCategory:
    """Best-effort categorization of *any* exception."""
    if isinstance(exc, NyxaraError):
        return exc.category
    for types, cat in _STDLIB_MAP:
        if isinstance(exc, types):
            return cat
    msg = f"{type(exc).__name__} {exc}".lower()
    if any(h in msg for h in _SECURITY_HINTS):
        return ErrorCategory.SECURITY
    if any(h in msg for h in _TRANSIENT_HINTS):
        return ErrorCategory.TRANSIENT
    return ErrorCategory.UNKNOWN


def is_retryable(exc: BaseException) -> bool:
    """True iff the kernel may safely retry this failure."""
    if isinstance(exc, NyxaraError):
        return exc.retryable
    return classify(exc).retryable


def wrap(exc: BaseException, message: Optional[str] = None, **kw: Any) -> NyxaraError:
    """Adopt an arbitrary exception into a :class:`NyxaraError`, preserving cause."""
    if isinstance(exc, NyxaraError):
        if message:
            exc.context.setdefault("note", message)
        return exc
    cat = classify(exc)
    cls = {
        ErrorCategory.SECURITY: SecurityError,
        ErrorCategory.INVARIANT: InvariantViolation,
        ErrorCategory.CORRIGIBILITY: CorrigibilityError,
        ErrorCategory.RESOURCE: ResourceExhausted,
        ErrorCategory.EXTERNAL: ExternalServiceError,
        ErrorCategory.TRANSIENT: TimeoutErrorNX,
    }.get(cat, NyxaraError)
    return cls(message or str(exc) or type(exc).__name__, category=cat, cause=exc, **kw)


# --------------------------------------------------------------------------- #
# Retry policy with exponential backoff + full jitter
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: bool = True
    # Only retry these categories (defaults to all retryable categories).
    retry_on: Tuple[ErrorCategory, ...] = (
        ErrorCategory.TRANSIENT, ErrorCategory.EXTERNAL, ErrorCategory.RESOURCE,
    )

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        if attempt >= self.max_attempts:
            return False
        cat = exc.category if isinstance(exc, NyxaraError) else classify(exc)
        if cat.fail_closed:
            return False  # never retry security/invariant/corrigibility
        if isinstance(exc, NyxaraError) and exc._retryable_override is False:
            return False
        return cat in self.retry_on

    def delay_for(self, attempt: int) -> float:
        """Backoff for a *1-indexed* attempt number."""
        raw = min(self.max_delay, self.base_delay * (self.multiplier ** (attempt - 1)))
        if self.jitter:
            return random.uniform(0.0, raw)  # full jitter
        return raw


def retry(
    policy: Optional[RetryPolicy] = None,
    *,
    on_retry: Optional[Callable[[BaseException, int, float], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: retry a *sync* callable per ``policy`` with backoff+jitter."""
    pol = policy or RetryPolicy()

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            while True:
                attempt += 1
                try:
                    return fn(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001 - we re-raise below
                    if not pol.should_retry(exc, attempt):
                        raise
                    d = pol.delay_for(attempt)
                    if on_retry:
                        on_retry(exc, attempt, d)
                    LEDGER.record(exc, healed=False, note=f"retry {attempt}/{pol.max_attempts}")
                    sleep(d)

        return inner

    return deco


def aretry(
    policy: Optional[RetryPolicy] = None,
    *,
    on_retry: Optional[Callable[[BaseException, int, float], None]] = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator: retry an *async* coroutine per ``policy`` with backoff+jitter."""
    pol = policy or RetryPolicy()

    def deco(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def inner(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            while True:
                attempt += 1
                try:
                    return await fn(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001
                    if not pol.should_retry(exc, attempt):
                        raise
                    d = pol.delay_for(attempt)
                    if on_retry:
                        on_retry(exc, attempt, d)
                    LEDGER.record(exc, healed=False, note=f"retry {attempt}/{pol.max_attempts}")
                    await asyncio.sleep(d)

        return inner

    return deco


# --------------------------------------------------------------------------- #
# Self-heal registry
# --------------------------------------------------------------------------- #
HealHandler = Callable[[BaseException], bool]
"""Return True if the handler resolved the underlying condition."""


class SelfHealRegistry:
    """Maps exception types → remediation handlers tried in registration order.

    SECURITY / INVARIANT / CORRIGIBILITY errors are *refused* — they must never be
    auto-healed silently; they fail-closed and escalate.
    """

    _REFUSED = (SecurityError, InvariantViolation, CorrigibilityError)

    def __init__(self) -> None:
        self._handlers: List[Tuple[type, HealHandler]] = []
        self._lock = threading.RLock()

    def register(self, exc_type: Type[BaseException], handler: HealHandler) -> "SelfHealRegistry":
        with self._lock:
            self._handlers.append((exc_type, handler))
        return self

    def on(self, exc_type: Type[BaseException]) -> Callable[[HealHandler], HealHandler]:
        def deco(handler: HealHandler) -> HealHandler:
            self.register(exc_type, handler)
            return handler
        return deco

    def heal(self, exc: BaseException) -> bool:
        """Try registered handlers. Returns True if one resolved it."""
        if isinstance(exc, self._REFUSED) or (
            isinstance(exc, NyxaraError) and exc.fail_closed
        ):
            return False
        with self._lock:
            handlers = [h for t, h in self._handlers if isinstance(exc, t)]
        for h in handlers:
            try:
                if h(exc):
                    LEDGER.record(exc, healed=True, note=f"healed by {getattr(h, '__name__', h)}")
                    return True
            except Exception:  # noqa: BLE001 - a failing healer must not mask the original
                continue
        return False

    def clear(self) -> None:
        with self._lock:
            self._handlers.clear()


HEAL = SelfHealRegistry()
"""Process-wide self-heal registry."""


# --------------------------------------------------------------------------- #
# Learn-once ledger
# --------------------------------------------------------------------------- #
@dataclass
class ErrorRecord:
    fingerprint: str
    code: str
    category: ErrorCategory
    severity: Severity
    sample_message: str
    count: int = 0
    healed_count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "code": self.code,
            "category": self.category.value,
            "severity": self.severity.label,
            "sample_message": self.sample_message,
            "count": self.count,
            "healed_count": self.healed_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "notes": self.notes[-10:],
        }


class ErrorLedger:
    """Fingerprints and dedupes errors so each distinct failure is *learned once*.

    Thread-safe. Bounded by ``max_records`` (LRU-ish eviction of least-recent).
    """

    def __init__(self, max_records: int = 2048, max_notes_per_record: int = 50) -> None:
        self._records: Dict[str, ErrorRecord] = {}
        self._lock = threading.RLock()
        self._max_records = max_records
        self._max_notes = max_notes_per_record

    @staticmethod
    def _fingerprint(exc: BaseException) -> str:
        if isinstance(exc, NyxaraError):
            return exc.fingerprint()
        origin = ""
        tb = exc.__traceback__
        if tb is not None:
            last = traceback.extract_tb(tb)[-1:]
            if last:
                origin = f"{last[0].filename}:{last[0].name}"
        basis = f"{type(exc).__name__}|{classify(exc).value}|{origin}"
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]

    def record(self, exc: BaseException, *, healed: bool = False, note: str = "") -> ErrorRecord:
        fp = self._fingerprint(exc)
        cat = exc.category if isinstance(exc, NyxaraError) else classify(exc)
        sev = exc.severity if isinstance(exc, NyxaraError) else Severity.ERROR
        code = exc.code if isinstance(exc, NyxaraError) else type(exc).__name__
        now = time.time()
        with self._lock:
            rec = self._records.get(fp)
            if rec is None:
                rec = ErrorRecord(
                    fingerprint=fp, code=code, category=cat, severity=sev,
                    sample_message=(str(exc) or code)[:500],
                )
                self._records[fp] = rec
                self._evict_if_needed()
            rec.count += 1
            rec.last_seen = now
            if healed:
                rec.healed_count += 1
            if note:
                rec.notes.append(f"{now:.0f}:{note}")
                if len(rec.notes) > self._max_notes:
                    rec.notes = rec.notes[-self._max_notes:]
            return rec

    def seen_before(self, exc: BaseException) -> bool:
        with self._lock:
            return self._fingerprint(exc) in self._records

    def get(self, exc: BaseException) -> Optional[ErrorRecord]:
        with self._lock:
            return self._records.get(self._fingerprint(exc))

    def top(self, n: int = 10) -> List[ErrorRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: r.count, reverse=True)[:n]

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            by_cat: Dict[str, int] = {}
            total = 0
            for r in self._records.values():
                by_cat[r.category.value] = by_cat.get(r.category.value, 0) + r.count
                total += r.count
            return {
                "distinct": len(self._records),
                "total_occurrences": total,
                "by_category": by_cat,
            }

    def _evict_if_needed(self) -> None:
        if len(self._records) <= self._max_records:
            return
        # Evict the least-recently-seen record.
        victim = min(self._records.values(), key=lambda r: r.last_seen)
        self._records.pop(victim.fingerprint, None)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


LEDGER = ErrorLedger()
"""Process-wide learn-once error ledger."""


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA errors self-test")
    print("=" * 70)

    # classification
    assert classify(TimeoutError()) is ErrorCategory.TRANSIENT
    assert classify(ValueError("bad")) is ErrorCategory.PERMANENT
    assert classify(PermissionError()) is ErrorCategory.SECURITY
    assert classify(Exception("HTTP 503 service unavailable")) is ErrorCategory.TRANSIENT
    print("classification OK")

    # fail-closed for security/invariant
    assert SecurityError("x").fail_closed and not SecurityError("x").retryable
    assert InvariantViolation("core").severity is Severity.FATAL
    print("fail-closed posture OK")

    # retry with deterministic sleep
    calls = {"n": 0}

    @retry(RetryPolicy(max_attempts=3, base_delay=0, jitter=False), sleep=lambda d: None)
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ExternalServiceError("temporary 503")
        return "ok"

    assert flaky() == "ok" and calls["n"] == 3
    print(f"retry succeeded after {calls['n']} attempts")

    # never retry permanent
    perm = {"n": 0}

    @retry(RetryPolicy(max_attempts=5, base_delay=0), sleep=lambda d: None)
    def permanent() -> None:
        perm["n"] += 1
        raise ValidationError("bad input")

    try:
        permanent()
    except ValidationError:
        pass
    assert perm["n"] == 1, perm
    print("permanent errors not retried OK")

    # self-heal
    HEAL.clear()
    state = {"db": "down"}

    @HEAL.on(ToolError)
    def reconnect(e: BaseException) -> bool:
        state["db"] = "up"
        return True

    assert HEAL.heal(ToolError("db gone")) is True and state["db"] == "up"
    assert HEAL.heal(SecurityError("nope")) is False  # refused
    print("self-heal OK (security refused)")

    # learn-once ledger
    LEDGER.clear()
    for _ in range(5):
        LEDGER.record(ExternalServiceError("503"))
    rec = LEDGER.record(ExternalServiceError("503"))
    assert rec.count == 6, rec.count
    assert LEDGER.summary()["distinct"] == 1
    print(f"ledger learned once: count={rec.count}, summary={LEDGER.summary()}")

    # wrap
    w = wrap(KeyError("missing"), "lookup failed")
    assert isinstance(w, NyxaraError) and w.category is ErrorCategory.PERMANENT
    print("wrap OK")

    print("\nALL SELF-TESTS PASSED ✓")
