"""Tests for nyxara.guard.auth."""

from __future__ import annotations

import pytest

from nyxara.agency.permissions import Authority
from nyxara.guard.auth import (AuthGuard, AuthResult, Challenge, ContinuityToken, Factor,
                               FactorEvidence, Session)
from nyxara.kernel.config import OWNER
from nyxara.kernel.errors import AuthorizationError


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _guard(clk=None, **kw):
    return AuthGuard.from_passphrase("test passphrase", clock=clk or FakeClock(), **kw)


def _full_auth(guard, factors=None):
    ch = guard.issue_challenge()
    resp = guard.respond(ch.nonce)
    factors = factors or [FactorEvidence(Factor.BIOMETRIC, 0.9),
                          FactorEvidence(Factor.BEHAVIORAL, 0.8)]
    return guard.authenticate(factors=factors, challenge_id=ch.id, response=resp)


# -------------------- challenge / response -------------------- #
def test_issue_challenge():
    g = _guard()
    ch = g.issue_challenge()
    assert isinstance(ch, Challenge) and ch.nonce and ch.id


def test_verify_correct_response():
    g = _guard()
    ch = g.issue_challenge()
    assert g.verify_response(ch.id, g.respond(ch.nonce))


def test_verify_wrong_response():
    g = _guard()
    ch = g.issue_challenge()
    assert not g.verify_response(ch.id, "deadbeef")


def test_response_single_use():
    g = _guard()
    ch = g.issue_challenge()
    resp = g.respond(ch.nonce)
    assert g.verify_response(ch.id, resp)
    assert not g.verify_response(ch.id, resp)  # consumed


def test_challenge_expiry():
    clk = FakeClock()
    g = _guard(clk, challenge_ttl=100)
    ch = g.issue_challenge()
    clk.advance(150)
    assert not g.verify_response(ch.id, g.respond(ch.nonce))


def test_verify_unknown_challenge():
    assert not _guard().verify_response("nope", "x")


# -------------------- multi-factor authentication -------------------- #
def test_full_auth_succeeds():
    g = _guard()
    res = _full_auth(g)
    assert res.authenticated and res.authority is Authority.OWNER
    assert Factor.CRYPTOGRAPHIC in res.factors_passed
    assert res.session is not None and res.continuity_token


def test_auth_without_crypto_fails():
    g = _guard()
    res = g.authenticate(factors=[FactorEvidence(Factor.BIOMETRIC, 0.99),
                                  FactorEvidence(Factor.BEHAVIORAL, 0.99)])
    assert not res.authenticated
    assert res.authority is Authority.UNTRUSTED
    assert "cryptographic" in res.reason


def test_auth_wrong_signature_fails():
    g = _guard()
    ch = g.issue_challenge()
    res = g.authenticate(factors=[FactorEvidence(Factor.BIOMETRIC, 0.9),
                                  FactorEvidence(Factor.BEHAVIORAL, 0.9)],
                         challenge_id=ch.id, response="bad" * 10)
    assert not res.authenticated


def test_auth_insufficient_factors():
    g = _guard()
    ch = g.issue_challenge()
    # crypto only, but OWNER.required_factors == 2
    res = g.authenticate(challenge_id=ch.id, response=g.respond(ch.nonce))
    assert not res.authenticated and "insufficient" in res.reason


def test_auth_low_confidence_factor_ignored():
    g = _guard()
    ch = g.issue_challenge()
    res = g.authenticate(
        factors=[FactorEvidence(Factor.BIOMETRIC, 0.3)],  # below threshold
        challenge_id=ch.id, response=g.respond(ch.nonce))
    assert not res.authenticated  # only crypto passed -> 1/2


def test_auth_wrong_identity_rejected():
    g = _guard()
    ch = g.issue_challenge()
    res = g.authenticate(challenge_id=ch.id, response=g.respond(ch.nonce),
                         claimed_handle="EVE")
    assert not res.authenticated and "not the Master" in res.reason


def test_crypto_factor_cannot_be_faked_via_evidence():
    g = _guard()
    # supplying a CRYPTOGRAPHIC FactorEvidence directly must not count
    res = g.authenticate(factors=[FactorEvidence(Factor.CRYPTOGRAPHIC, 1.0),
                                  FactorEvidence(Factor.BIOMETRIC, 0.9)])
    assert not res.authenticated


def test_auth_score_computed():
    g = _guard()
    res = _full_auth(g)
    assert 0.0 < res.score <= 1.0


# -------------------- lockout -------------------- #
def test_lockout_after_failures():
    g = _guard(max_failures=3)
    for _ in range(3):
        g.authenticate(factors=[FactorEvidence(Factor.BIOMETRIC, 0.9)])  # no crypto -> fail
    res = _full_auth(g)  # even a valid attempt is locked out
    assert not res.authenticated and "locked" in res.reason


def test_success_resets_failures():
    g = _guard(max_failures=5)
    g.authenticate(factors=[FactorEvidence(Factor.BIOMETRIC, 0.9)])  # 1 failure
    _full_auth(g)  # success resets
    assert g.report()["failures"] == 0


# -------------------- continuity tokens -------------------- #
def test_continuity_restores_authority():
    clk = FakeClock()
    g = _guard(clk)
    token = _full_auth(g).continuity_token
    # a fresh guard with the same key
    g2 = AuthGuard.from_passphrase("test passphrase", clock=clk)
    res = g2.verify_continuity_token(token)
    assert res.authenticated and res.authority is Authority.OWNER
    assert res.session.via_continuity


def test_continuity_wrong_key_rejected():
    clk = FakeClock()
    token = _full_auth(_guard(clk)).continuity_token
    attacker = AuthGuard.from_passphrase("different key", clock=clk)
    assert not attacker.verify_continuity_token(token).authenticated


def test_continuity_malformed_token():
    assert not _guard().verify_continuity_token("not-a-valid-token").authenticated


def test_continuity_expired():
    clk = FakeClock()
    g = _guard(clk, continuity_ttl=100)
    token = _full_auth(g).continuity_token
    g2 = AuthGuard.from_passphrase("test passphrase", clock=clk)
    clk.advance(200)
    assert not g2.verify_continuity_token(token).authenticated


def test_continuity_replay_downgrade_rejected():
    clk = FakeClock()
    g = _guard(clk)
    old = _full_auth(g).continuity_token
    new = _full_auth(g).continuity_token
    verifier = AuthGuard.from_passphrase("test passphrase", clock=clk)
    assert verifier.verify_continuity_token(new).authenticated
    assert not verifier.verify_continuity_token(old).authenticated  # downgrade


def test_continuity_seq_increments():
    g = _guard()
    _full_auth(g)
    _full_auth(g)
    assert g.report()["last_seq"] == 2


def test_continuity_wrong_namespace_rejected():
    clk = FakeClock()
    g = _guard(clk)
    token = _full_auth(g).continuity_token
    # tamper: re-sign a token with a different namespace using the same key
    import base64
    import json
    b64, sig = token.split(".", 1)
    payload = json.loads(base64.urlsafe_b64decode(b64))
    payload["namespace"] = "evil.namespace"
    raw = json.dumps(payload, sort_keys=True).encode()
    forged = base64.urlsafe_b64encode(raw).decode() + "." + g._sign(raw)
    assert not g.verify_continuity_token(forged).authenticated


# -------------------- sessions -------------------- #
def test_authority_of_valid_session():
    g = _guard()
    res = _full_auth(g)
    assert g.authority_of(res.session.id) is Authority.OWNER
    assert g.is_master(res.session.id)


def test_authority_of_unknown_session():
    assert _guard().authority_of("nope") is Authority.UNTRUSTED


def test_session_expiry():
    clk = FakeClock()
    g = _guard(clk, session_ttl=100)
    res = _full_auth(g)
    clk.advance(200)
    assert g.authority_of(res.session.id) is Authority.UNTRUSTED


def test_revoke_session():
    g = _guard()
    res = _full_auth(g)
    assert g.revoke(res.session.id)
    assert not g.is_master(res.session.id)


def test_revoke_unknown():
    assert not _guard().revoke("nope")


def test_require_master_passes():
    g = _guard()
    res = _full_auth(g)
    g.require_master(res.session.id)  # no raise


def test_require_master_raises():
    g = _guard()
    with pytest.raises(AuthorizationError):
        g.require_master("nope")


# -------------------- helpers / shapes -------------------- #
def test_continuity_token_payload_roundtrip():
    tok = ContinuityToken("JP", "ns", 3, 1000.0, 2000.0)
    assert ContinuityToken.from_payload(tok.payload()).seq == 3


def test_factor_evidence_to_dict():
    assert FactorEvidence(Factor.BIOMETRIC, 0.9).to_dict()["factor"] == "biometric"


def test_auth_result_to_dict():
    res = _full_auth(_guard())
    d = res.to_dict()
    assert d["authenticated"] is True and d["authority"] == "owner"


def test_session_to_dict():
    res = _full_auth(_guard())
    assert res.session.to_dict()["authority"] == "owner"


def test_report():
    g = _guard()
    _full_auth(g)
    r = g.report()
    assert r["sessions"] == 1 and r["active"] == 1


def test_from_passphrase_deterministic():
    # same passphrase -> same key -> cross-instance token verification works
    clk = FakeClock()
    a = AuthGuard.from_passphrase("shared secret", clock=clk)
    token = _full_auth(a).continuity_token
    b = AuthGuard.from_passphrase("shared secret", clock=clk)
    assert b.verify_continuity_token(token).authenticated
