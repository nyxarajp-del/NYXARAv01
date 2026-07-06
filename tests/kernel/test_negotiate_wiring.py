"""P1-A — the structured consent/clarification Negotiator is wired into the kernel.

``agency/negotiate.py`` (Negotiator) was a complete, tested module that nothing imported. It is
now a first-class kernel faculty: the Master can be asked for explicit consent or to resolve a
goal conflict, every answer is appended to a tamper-evident hash-chained ledger (Rule 6), and —
crucially — an *escalated* action is now turned into a fail-closed CONFIRM the Master can answer,
instead of vanishing into a one-line message (Economic/Social reasoning, capabilities #59/#38).
"""

from __future__ import annotations

from nyxara.agency.permissions import Authority, Capability, RiskTier
from nyxara.guard.oversight import ReviewMode
from nyxara.kernel.orchestrator import Candidate, Disposition, NyxaraCore


def test_request_consent_is_fail_closed_and_ledgered():
    nyx = NyxaraCore()
    req = nyx.request_consent("erase the archive?", risk=RiskTier.HIGH, reversibility=0.0)
    assert req.kind.value == "consent"
    assert len(nyx.pending_negotiations()) == 1
    # only the Master may answer; the resolution seals the hash-chained ledger
    nyx.answer_negotiation(req.id, consent=True)
    assert nyx._negotiator().verify_ledger() is True
    assert nyx.pending_negotiations() == []


def test_non_owner_cannot_answer():
    import pytest

    from nyxara.kernel.errors import AuthorizationError
    nyx = NyxaraCore()
    req = nyx.request_consent("proceed?")
    with pytest.raises(AuthorizationError):
        nyx.answer_negotiation(req.id, consent=True, authority=Authority.AUTONOMOUS)


def test_escalated_action_opens_a_consent_request():
    # a risky autonomous action that the gates escalate becomes a structured, answerable CONFIRM.
    # Pin conservative oversight (default is now the fully-autonomous SOVEREIGN mode) so the action
    # actually escalates and opens the negotiation.
    nyx = NyxaraCore(review_mode=ReviewMode.AUTONOMOUS)
    risky = Candidate(text="delete production database", kind="act", confidence=0.8,
                      capability=Capability.TOOL_CALL, risk=RiskTier.HIGH, reversible=False)
    nyx._invoke_reasoner = lambda *a, **k: risky          # type: ignore[assignment]
    nyx._maybe_bootstrap = lambda s, f, c, a: c           # type: ignore[assignment]
    result = nyx.process("do something risky", authority=Authority.AUTONOMOUS)
    assert result.disposition is Disposition.ESCALATE
    pending = nyx.pending_negotiations()
    assert len(pending) == 1
    assert pending[0]["kind"] == "confirm"
    # fail-closed: the default is to cancel — silence never proceeds
    assert pending[0]["default_option_id"] == "cancel"


def test_owner_turns_do_not_spuriously_open_negotiations():
    nyx = NyxaraCore()
    nyx.process("hello NYXARA", authority=Authority.OWNER)
    assert nyx.pending_negotiations() == []
