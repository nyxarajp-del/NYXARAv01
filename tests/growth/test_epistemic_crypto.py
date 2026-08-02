"""Tests for nyxara.growth.epistemic_crypto — Quantum-Resistant Epistemic Cryptography.

Every learned fact is HMAC-signed and chained; tampering breaks the chain; a context-safe policy
decides sharing/execution only when the signature verifies. These tests assert signing, tamper
detection, and the share/execute immune system.
"""

from __future__ import annotations

from nyxara.growth.epistemic_crypto import (
    Context,
    EpistemicLedger,
    Sensitivity,
    ShareDecision,
)


def _ledger():
    led = EpistemicLedger()
    led.record("the master is jp", source="identity", sensitivity=Sensitivity.PUBLIC)
    led.record("owner secret", source="owner", sensitivity=Sensitivity.SECRET)
    led.record("a+b = b+a", source="meta_epistemology", sensitivity=Sensitivity.INTERNAL)
    led.record("def f(): return 1", source="tool_forge", sensitivity=Sensitivity.EXECUTABLE)
    return led


def test_signs_and_chain_verifies():
    led = _ledger()
    ok, broken = led.verify_chain()
    assert ok and broken is None
    assert all(led.verify(e) for e in led.chain)


def test_tamper_breaks_chain_at_index():
    led = _ledger()
    led.chain[1].content = "owner secret = leaked"
    ok, broken = led.verify_chain()
    assert not ok and broken == 1
    assert not led.verify(led.chain[1])


def test_public_shareable_everywhere():
    led = _ledger()
    d = led.may_share(led.chain[0], Context.EXTERNAL)
    assert isinstance(d, ShareDecision) and d.allowed and d.verified


def test_secret_owner_only():
    led = _ledger()
    assert not led.may_share(led.chain[1], Context.EXTERNAL).allowed
    assert not led.may_share(led.chain[1], Context.TRUSTED).allowed
    assert led.may_share(led.chain[1], Context.OWNER).allowed


def test_internal_not_external_but_trusted_ok():
    led = _ledger()
    assert not led.may_share(led.chain[2], Context.EXTERNAL).allowed
    assert led.may_share(led.chain[2], Context.TRUSTED).allowed


def test_tampered_fact_is_never_shareable():
    led = _ledger()
    led.chain[0].content = "the master is not jp"      # tamper a public fact
    d = led.may_share(led.chain[0], Context.EXTERNAL)
    assert not d.allowed and not d.verified


def test_executable_runs_only_trusted_and_verified():
    led = _ledger()
    ex = led.chain[3]
    assert led.may_execute(ex, Context.TRUSTED).allowed
    assert not led.may_execute(ex, Context.EXTERNAL).allowed
    ex.content = "def f(): import os; os.system('rm -rf /')"
    assert not led.may_execute(ex, Context.OWNER).allowed   # tampered ⇒ refused


def test_non_executable_not_executed():
    led = _ledger()
    assert not led.may_execute(led.chain[0], Context.OWNER).allowed


def test_report():
    led = _ledger()
    rep = led.report()
    assert rep["entries"] == 4 and rep["chain_valid"] is True
