"""Tests for nyxara.kernel.errors."""

from __future__ import annotations

import asyncio

import pytest

from nyxara.kernel.errors import (
    HEAL,
    LEDGER,
    AuthorizationError,
    CorrigibilityError,
    ErrorCategory,
    ExternalServiceError,
    InvariantViolation,
    LLMError,
    NyxaraError,
    PromptInjectionError,
    ResourceExhausted,
    RetryPolicy,
    SecurityError,
    Severity,
    SelfHealRegistry,
    ToolError,
    ValidationError,
    aretry,
    classify,
    is_retryable,
    retry,
    wrap,
)


# -------------------- classification -------------------- #
@pytest.mark.parametrize(
    "exc,expected",
    [
        (TimeoutError(), ErrorCategory.TRANSIENT),
        (asyncio.TimeoutError(), ErrorCategory.TRANSIENT),
        (ConnectionError(), ErrorCategory.EXTERNAL),
        (MemoryError(), ErrorCategory.RESOURCE),
        (PermissionError(), ErrorCategory.SECURITY),
        (ValueError("x"), ErrorCategory.PERMANENT),
        (KeyError("k"), ErrorCategory.PERMANENT),
        (OSError("io"), ErrorCategory.EXTERNAL),
    ],
)
def test_classify_stdlib(exc, expected):
    assert classify(exc) is expected


def test_classify_heuristics():
    assert classify(Exception("Request timed out")) is ErrorCategory.TRANSIENT
    assert classify(Exception("HTTP 429 rate limit exceeded")) is ErrorCategory.TRANSIENT
    assert classify(Exception("401 Unauthorized")) is ErrorCategory.SECURITY
    assert classify(Exception("totally novel weirdness")) is ErrorCategory.UNKNOWN


def test_category_properties():
    assert ErrorCategory.TRANSIENT.retryable
    assert not ErrorCategory.PERMANENT.retryable
    assert ErrorCategory.SECURITY.fail_closed
    assert ErrorCategory.INVARIANT.fail_closed
    assert ErrorCategory.CORRIGIBILITY.fail_closed
    assert not ErrorCategory.TRANSIENT.fail_closed


# -------------------- error object -------------------- #
def test_base_error_fields_and_dict():
    e = NyxaraError("boom", code="X1", context={"k": 1})
    d = e.to_dict()
    assert d["code"] == "X1"
    assert d["category"] == ErrorCategory.UNKNOWN.value
    assert d["context"] == {"k": 1}
    assert len(e.trace_id) == 32
    assert d["fingerprint"]


def test_retryable_override():
    e = ExternalServiceError("x", retryable=False)
    assert e.retryable is False
    e2 = ValidationError("x", retryable=True)
    assert e2.retryable is True


def test_security_and_invariant_posture():
    assert SecurityError("x").fail_closed and not SecurityError("x").retryable
    assert isinstance(PromptInjectionError("x"), SecurityError)
    assert isinstance(AuthorizationError("x"), SecurityError)
    assert InvariantViolation("x").severity is Severity.FATAL
    assert CorrigibilityError("x").category is ErrorCategory.CORRIGIBILITY


def test_is_retryable():
    assert is_retryable(TimeoutError())
    assert not is_retryable(ValueError())
    assert not is_retryable(SecurityError("x"))
    assert is_retryable(ResourceExhausted("quota"))


def test_with_context_chaining():
    e = ToolError("t").with_context(tool="x").with_context(arg=1)
    assert e.context == {"tool": "x", "arg": 1}


def test_wrap_preserves_cause_and_category():
    src = KeyError("missing")
    w = wrap(src, "lookup failed")
    assert isinstance(w, NyxaraError)
    assert w.category is ErrorCategory.PERMANENT
    assert w.cause is src
    # wrapping an existing NyxaraError returns it unchanged
    nx = LLMError("provider down")
    assert wrap(nx) is nx


# -------------------- retry -------------------- #
def test_retry_succeeds_after_transient():
    calls = {"n": 0}

    @retry(RetryPolicy(max_attempts=4, base_delay=0, jitter=False), sleep=lambda d: None)
    def f():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ExternalServiceError("503")
        return "ok"

    assert f() == "ok"
    assert calls["n"] == 3


def test_retry_gives_up_and_reraises():
    calls = {"n": 0}

    @retry(RetryPolicy(max_attempts=3, base_delay=0), sleep=lambda d: None)
    def f():
        calls["n"] += 1
        raise ExternalServiceError("always")

    with pytest.raises(ExternalServiceError):
        f()
    assert calls["n"] == 3


def test_retry_does_not_retry_permanent():
    calls = {"n": 0}

    @retry(RetryPolicy(max_attempts=5, base_delay=0), sleep=lambda d: None)
    def f():
        calls["n"] += 1
        raise ValidationError("bad")

    with pytest.raises(ValidationError):
        f()
    assert calls["n"] == 1


def test_retry_never_retries_security():
    calls = {"n": 0}

    @retry(RetryPolicy(max_attempts=5, base_delay=0), sleep=lambda d: None)
    def f():
        calls["n"] += 1
        raise SecurityError("nope")

    with pytest.raises(SecurityError):
        f()
    assert calls["n"] == 1


def test_retry_policy_backoff_growth():
    pol = RetryPolicy(base_delay=1.0, multiplier=2.0, jitter=False, max_delay=100)
    assert pol.delay_for(1) == 1.0
    assert pol.delay_for(2) == 2.0
    assert pol.delay_for(3) == 4.0
    # capped
    capped = RetryPolicy(base_delay=1.0, multiplier=10.0, jitter=False, max_delay=5.0)
    assert capped.delay_for(5) == 5.0


def test_retry_full_jitter_within_bounds():
    pol = RetryPolicy(base_delay=2.0, multiplier=2.0, jitter=True, max_delay=100)
    for attempt in range(1, 5):
        raw = min(100, 2.0 * (2.0 ** (attempt - 1)))
        for _ in range(20):
            d = pol.delay_for(attempt)
            assert 0.0 <= d <= raw


def test_aretry_async():
    calls = {"n": 0}

    @aretry(RetryPolicy(max_attempts=4, base_delay=0))
    async def f():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ExternalServiceError("503")
        return 42

    assert asyncio.run(f()) == 42
    assert calls["n"] == 2


# -------------------- self-heal -------------------- #
def test_self_heal_runs_handler():
    reg = SelfHealRegistry()
    state = {"ok": False}

    @reg.on(ToolError)
    def fix(e):
        state["ok"] = True
        return True

    assert reg.heal(ToolError("x")) is True
    assert state["ok"] is True


def test_self_heal_refuses_security_and_invariant():
    reg = SelfHealRegistry()

    @reg.on(SecurityError)
    def should_not_run(e):  # pragma: no cover - must never run
        return True

    assert reg.heal(SecurityError("x")) is False
    assert reg.heal(InvariantViolation("x")) is False


def test_self_heal_tries_multiple_until_success():
    reg = SelfHealRegistry()
    log = []

    @reg.on(ToolError)
    def a(e):
        log.append("a")
        return False

    @reg.on(ToolError)
    def b(e):
        log.append("b")
        return True

    assert reg.heal(ToolError("x")) is True
    assert log == ["a", "b"]


def test_self_heal_handler_exception_is_swallowed():
    reg = SelfHealRegistry()

    @reg.on(ToolError)
    def boom(e):
        raise RuntimeError("healer broke")

    assert reg.heal(ToolError("x")) is False


def test_global_heal_registry_exists():
    assert isinstance(HEAL, SelfHealRegistry)


# -------------------- ledger -------------------- #
def test_ledger_learns_once(monkeypatch):
    LEDGER.clear()
    for _ in range(7):
        LEDGER.record(ExternalServiceError("503"))
    assert LEDGER.summary()["distinct"] == 1
    assert LEDGER.summary()["total_occurrences"] == 7
    LEDGER.clear()


def test_ledger_distinguishes_categories():
    LEDGER.clear()
    LEDGER.record(ExternalServiceError("a"))
    LEDGER.record(ValidationError("b"))
    s = LEDGER.summary()
    assert s["distinct"] == 2
    assert set(s["by_category"]) == {ErrorCategory.EXTERNAL.value, ErrorCategory.PERMANENT.value}
    LEDGER.clear()


def test_ledger_seen_before_and_get():
    LEDGER.clear()
    e = ToolError("x")
    assert not LEDGER.seen_before(e)
    LEDGER.record(e)
    assert LEDGER.seen_before(ToolError("x"))  # same fingerprint
    rec = LEDGER.get(e)
    assert rec is not None and rec.count == 1
    LEDGER.clear()


def test_ledger_top_ordering():
    LEDGER.clear()
    for _ in range(3):
        LEDGER.record(ToolError("rare"))
    for _ in range(10):
        LEDGER.record(ExternalServiceError("common"))
    top = LEDGER.top(2)
    assert top[0].count == 10
    assert top[1].count == 3
    LEDGER.clear()


def test_ledger_healed_count():
    LEDGER.clear()
    LEDGER.record(ToolError("x"), healed=True, note="fixed")
    rec = LEDGER.get(ToolError("x"))
    assert rec.healed_count == 1
    assert rec.notes
    LEDGER.clear()
