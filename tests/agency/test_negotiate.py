"""Tests for nyxara.agency.negotiate."""

from __future__ import annotations

import pytest

from nyxara.agency.negotiate import (NegotiationKind, Negotiator, Option,
                                     RequestStatus)
from nyxara.agency.permissions import Authority, RiskTier
from nyxara.kernel.errors import AuthorizationError, InvariantViolation, ValidationError


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _opts():
    return [Option("a", "Option A", recommended=True),
            Option("b", "Option B", risk=RiskTier.MODERATE)]


# -------------------- Option -------------------- #
def test_option_to_dict():
    o = Option("a", "A", tradeoffs=["x"], risk=RiskTier.HIGH, recommended=True)
    d = o.to_dict()
    assert d["id"] == "a" and d["risk"] == "high" and d["recommended"] is True


# -------------------- request builders -------------------- #
def test_clarify_creates_open_request():
    neg = Negotiator()
    req = neg.clarify("which?", _opts())
    assert req.kind is NegotiationKind.CLARIFY
    assert req.open and req.recommended.id == "a"
    assert req in neg.pending()


def test_confirm_has_proceed_cancel_default_cancel():
    neg = Negotiator()
    req = neg.confirm("do it?")
    assert {o.id for o in req.options} == {"proceed", "cancel"}
    assert req.default_option_id == "cancel"
    assert req.option("cancel").is_abort


def test_confirm_recommend_proceed():
    neg = Negotiator()
    req = neg.confirm("do it?", recommend_proceed=True)
    assert req.option("proceed").recommended


def test_choose_request():
    neg = Negotiator()
    req = neg.choose("pick", _opts(), default_option_id="a")
    assert req.kind is NegotiationKind.CHOOSE


def test_consent_request_records_risk():
    neg = Negotiator()
    req = neg.request_consent("risky?", risk=RiskTier.CRITICAL, reversibility=0.0)
    assert req.kind is NegotiationKind.CONSENT
    assert req.context["risk"] == "critical"


def test_counter_propose():
    neg = Negotiator()
    req = neg.counter_propose("how about these?", _opts())
    assert req.kind is NegotiationKind.COUNTER


def test_resolve_conflict():
    neg = Negotiator()
    req = neg.resolve_conflict("speed vs safety", _opts())
    assert req.kind is NegotiationKind.RESOLVE_CONFLICT


def test_ttl_sets_deadline():
    clk = FakeClock()
    neg = Negotiator(clock=clk)
    req = neg.request_consent("x", ttl=50.0)
    assert req.deadline == 1050.0


def test_present_renders_options():
    neg = Negotiator()
    req = neg.clarify("which?", _opts())
    text = req.present()
    assert "Option A" in text and "recommended" in text


# -------------------- responding -------------------- #
def test_respond_clarify():
    neg = Negotiator()
    req = neg.clarify("which?", _opts())
    out = neg.respond(req.id, option_id="a", text="that one")
    assert out.proceed and out.chosen.id == "a"
    assert neg.get(req.id).status is RequestStatus.RESOLVED


def test_respond_confirm_proceed():
    neg = Negotiator()
    req = neg.confirm("do it?")
    out = neg.respond(req.id, option_id="proceed")
    assert out.proceed and out.consent


def test_respond_confirm_cancel_does_not_proceed():
    neg = Negotiator()
    req = neg.confirm("do it?")
    out = neg.respond(req.id, option_id="cancel")
    assert not out.proceed and not out.consent


def test_respond_consent_yes():
    neg = Negotiator()
    req = neg.request_consent("risky?")
    out = neg.respond(req.id, consent=True)
    assert out.consent and out.proceed


def test_respond_consent_no():
    neg = Negotiator()
    req = neg.request_consent("risky?")
    out = neg.respond(req.id, consent=False)
    assert not out.consent and not out.proceed


def test_only_owner_may_respond():
    neg = Negotiator()
    req = neg.confirm("do it?")
    with pytest.raises(AuthorizationError):
        neg.respond(req.id, option_id="proceed", authority=Authority.AUTONOMOUS)


def test_respond_unknown_request():
    neg = Negotiator()
    with pytest.raises(InvariantViolation):
        neg.respond("nope", option_id="a")


def test_respond_already_resolved():
    neg = Negotiator()
    req = neg.confirm("do it?")
    neg.respond(req.id, option_id="cancel")
    with pytest.raises(InvariantViolation):
        neg.respond(req.id, option_id="proceed")


def test_respond_consent_without_bool_rejected():
    neg = Negotiator()
    req = neg.request_consent("risky?")
    with pytest.raises(ValidationError):
        neg.respond(req.id, option_id="a")  # consent needs yes/no, not an option


def test_respond_choice_without_option_rejected():
    neg = Negotiator()
    req = neg.choose("pick", _opts())
    with pytest.raises(ValidationError):
        neg.respond(req.id, consent=True)  # needs an option


def test_respond_unknown_option_rejected():
    neg = Negotiator()
    req = neg.choose("pick", _opts())
    with pytest.raises(ValidationError):
        neg.respond(req.id, option_id="zzz")


# -------------------- withdraw -------------------- #
def test_withdraw_open_request():
    neg = Negotiator()
    req = neg.confirm("do it?")
    assert neg.withdraw(req.id)
    assert neg.get(req.id).status is RequestStatus.WITHDRAWN


def test_withdraw_resolved_returns_false():
    neg = Negotiator()
    req = neg.confirm("do it?")
    neg.respond(req.id, option_id="cancel")
    assert not neg.withdraw(req.id)


def test_withdraw_unknown_returns_false():
    assert not Negotiator().withdraw("nope")


# -------------------- timeouts (fail-closed) -------------------- #
def test_consent_expires_to_denied():
    clk = FakeClock()
    neg = Negotiator(clock=clk)
    req = neg.request_consent("risky?", ttl=50.0)
    clk.advance(60.0)
    outs = neg.expire_due()
    assert len(outs) == 1 and not outs[0].consent
    assert neg.get(req.id).status is RequestStatus.EXPIRED


def test_not_expired_before_deadline():
    clk = FakeClock()
    neg = Negotiator(clock=clk)
    neg.request_consent("risky?", ttl=50.0)
    clk.advance(40.0)
    assert neg.expire_due() == []


def test_choice_expires_without_default_does_not_proceed():
    clk = FakeClock()
    neg = Negotiator(clock=clk, default_on_timeout=False)
    req = neg.choose("pick", _opts(), default_option_id="a", ttl=10.0)
    clk.advance(20.0)
    outs = neg.expire_due()
    assert not outs[0].proceed and outs[0].chosen is None


def test_choice_expires_applies_default_when_enabled():
    clk = FakeClock()
    neg = Negotiator(clock=clk, default_on_timeout=True)
    req = neg.choose("pick", _opts(), default_option_id="a", ttl=10.0)
    clk.advance(20.0)
    outs = neg.expire_due()
    assert outs[0].chosen.id == "a" and outs[0].proceed


# -------------------- consent ledger -------------------- #
def test_ledger_records_each_resolution():
    neg = Negotiator()
    r1 = neg.confirm("a?")
    neg.respond(r1.id, option_id="proceed")
    r2 = neg.request_consent("b?")
    neg.respond(r2.id, consent=True)
    assert len(neg.ledger()) == 2
    assert neg.verify_ledger()


def test_ledger_chain_advances():
    neg = Negotiator()
    h0 = neg.head
    r = neg.confirm("a?")
    neg.respond(r.id, option_id="proceed")
    assert neg.head != h0


def test_ledger_tamper_detected():
    neg = Negotiator()
    r = neg.confirm("a?")
    neg.respond(r.id, option_id="proceed")
    neg.ledger()  # snapshot copy; tamper the internal store
    neg._ledger[0]["outcome"]["proceed"] = False
    with pytest.raises(InvariantViolation):
        neg.verify_ledger()


def test_empty_ledger_verifies():
    assert Negotiator().verify_ledger()


# -------------------- reporting -------------------- #
def test_report_counts():
    neg = Negotiator()
    r1 = neg.confirm("a?")
    neg.respond(r1.id, option_id="cancel")
    neg.confirm("b?")  # left open
    rep = neg.report()
    assert rep["requests"] == 2 and rep["pending"] == 1
    assert rep["by_status"]["resolved"] == 1


def test_request_to_dict():
    neg = Negotiator()
    req = neg.clarify("which?", _opts())
    d = req.to_dict()
    assert d["kind"] == "clarify" and len(d["options"]) == 2
