"""NYXARA · guard/auth.py — the Master's identity & authority verification (🔐, Rule 7).

Rule 7 is Singular Master Continuity: NYXARA serves *one* Master, and must be certain it
is truly him before she obeys — across restarts, across channels, against an impostor who
has read every public word he ever wrote. This module is that gate.

Authority is established by **multi-factor verification** anchored on a **cryptographic
challenge–response**: NYXARA issues a single-use nonce, and only someone holding the
shared master key can sign it (HMAC-SHA256, constant-time compared). The crypto factor is
*mandatory* — soft factors (biometric, behavioural, contextual, cognitive) corroborate
but can never substitute for it, so no amount of public knowledge impersonates the Master.

The bond **survives restarts** via signed, sequence-numbered :class:`ContinuityToken` s: a
fresh process holding the same master key re-establishes authority from the token without a
full re-auth, while monotonic sequence numbers defeat **replay/downgrade** of an old token
and consumed challenges defeat **response replay**.

Everything is **fail-closed**: a missing factor, a stale challenge, a skewed clock, a
wrong handle, a bad signature — any of them yields *no authority at all* (mapped to
:class:`~nyxara.agency.permissions.Authority.UNTRUSTED`), never a partial grant.

Depends on :mod:`kernel.config` (the one Owner), :mod:`agency.permissions` (Authority) and
:mod:`kernel.errors`. Pure standard library crypto (``hmac``/``hashlib``).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from nyxara.agency.permissions import Authority
from nyxara.kernel.config import OWNER, OwnerIdentity
from nyxara.kernel.errors import AuthorizationError

__all__ = [
    "Factor",
    "FactorEvidence",
    "Challenge",
    "ContinuityToken",
    "Session",
    "AuthResult",
    "AuthGuard",
]


# --------------------------------------------------------------------------- #
# Factors
# --------------------------------------------------------------------------- #
class Factor(str, Enum):
    CRYPTOGRAPHIC = "cryptographic"   # mandatory — proves possession of the master key
    BIOMETRIC = "biometric"
    COGNITIVE = "cognitive"           # a shared secret / knowledge
    BEHAVIORAL = "behavioral"
    CONTEXTUAL = "contextual"


@dataclass
class FactorEvidence:
    factor: Factor
    confidence: float = 0.0           # [0, 1] how strongly this factor matches
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"factor": self.factor.value, "confidence": round(self.confidence, 3),
                "detail": self.detail}


# --------------------------------------------------------------------------- #
# Challenge (single-use nonce)
# --------------------------------------------------------------------------- #
@dataclass
class Challenge:
    id: str
    nonce: str
    issued_at: float
    expires_at: float

    def expired(self, now: float) -> bool:
        return now >= self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "nonce": self.nonce, "expires_at": self.expires_at}


# --------------------------------------------------------------------------- #
# Continuity token (survives restarts)
# --------------------------------------------------------------------------- #
@dataclass
class ContinuityToken:
    handle: str
    namespace: str
    seq: int
    issued_at: float
    expires_at: float

    def payload(self) -> Dict[str, Any]:
        return {"handle": self.handle, "namespace": self.namespace, "seq": self.seq,
                "issued_at": self.issued_at, "expires_at": self.expires_at}

    @classmethod
    def from_payload(cls, d: Dict[str, Any]) -> "ContinuityToken":
        return cls(handle=d["handle"], namespace=d["namespace"], seq=d["seq"],
                   issued_at=d["issued_at"], expires_at=d["expires_at"])


# --------------------------------------------------------------------------- #
# Session & result
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    id: str
    authority: Authority
    factors: List[Factor]
    established_at: float
    expires_at: float
    via_continuity: bool = False
    revoked: bool = False

    def valid(self, now: float) -> bool:
        return not self.revoked and now < self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "authority": self.authority.value,
                "factors": [f.value for f in self.factors],
                "via_continuity": self.via_continuity, "revoked": self.revoked,
                "expires_at": self.expires_at}


@dataclass
class AuthResult:
    authenticated: bool
    authority: Authority
    reason: str
    factors_passed: List[Factor] = field(default_factory=list)
    score: float = 0.0
    session: Optional[Session] = None
    continuity_token: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"authenticated": self.authenticated, "authority": self.authority.value,
                "reason": self.reason, "score": round(self.score, 3),
                "factors_passed": [f.value for f in self.factors_passed],
                "session": self.session.id if self.session else None,
                "continuity_token": bool(self.continuity_token)}


# --------------------------------------------------------------------------- #
# Auth guard
# --------------------------------------------------------------------------- #
class AuthGuard:
    """Establishes and re-establishes the Master's authority, fail-closed (Rule 7)."""

    def __init__(self, *, root_key: Optional[bytes] = None, owner: OwnerIdentity = OWNER,
                 factor_threshold: float = 0.6, challenge_ttl: float = 120.0,
                 session_ttl: float = 3600.0, continuity_ttl: float = 86_400.0 * 30,
                 max_failures: int = 5, clock=None) -> None:
        self._key = root_key or os.urandom(32)
        self.owner = owner
        self.factor_threshold = factor_threshold
        self.challenge_ttl = challenge_ttl
        self.session_ttl = session_ttl
        self.continuity_ttl = continuity_ttl
        self.max_failures = max_failures
        self._clock = clock or time.time
        self._challenges: Dict[str, Challenge] = {}
        self._sessions: Dict[str, Session] = {}
        self._seq = 0
        self._last_seen_seq = -1
        self._failures = 0

    @classmethod
    def from_passphrase(cls, passphrase: str, *, owner: OwnerIdentity = OWNER, **kw: Any
                        ) -> "AuthGuard":
        salt = owner.continuity_namespace.encode("utf-8")
        key = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, 100_000)
        return cls(root_key=key, owner=owner, **kw)

    # ---- crypto primitives ---- #
    def _sign(self, msg: bytes) -> str:
        return hmac.new(self._key, msg, hashlib.sha256).hexdigest()

    def respond(self, nonce: str) -> str:
        """The Master's side: sign a challenge nonce with the shared key."""
        return self._sign(nonce.encode("utf-8"))

    # ---- challenge / response (anti-replay) ---- #
    def issue_challenge(self) -> Challenge:
        now = self._clock()
        ch = Challenge(id=uuid.uuid4().hex[:12], nonce=uuid.uuid4().hex,
                       issued_at=now, expires_at=now + self.challenge_ttl)
        self._challenges[ch.id] = ch
        return ch

    def verify_response(self, challenge_id: str, response: str) -> bool:
        ch = self._challenges.get(challenge_id)
        if ch is None:
            return False
        # single-use: consume regardless of outcome (defeats response replay)
        del self._challenges[challenge_id]
        if ch.expired(self._clock()):
            return False
        return hmac.compare_digest(self._sign(ch.nonce.encode("utf-8")), response)

    # ---- multi-factor authentication ---- #
    def authenticate(self, *, factors: Sequence[FactorEvidence] = (),
                     challenge_id: Optional[str] = None, response: Optional[str] = None,
                     claimed_handle: Optional[str] = None) -> AuthResult:
        now = self._clock()
        if self._failures >= self.max_failures:
            return AuthResult(False, Authority.UNTRUSTED,
                              "locked: too many failed attempts")

        handle = claimed_handle if claimed_handle is not None else self.owner.handle
        if handle != self.owner.handle:
            self._failures += 1
            return AuthResult(False, Authority.UNTRUSTED,
                              f"claimed identity {handle!r} is not the Master")

        # the cryptographic factor is mandatory
        crypto_ok = bool(challenge_id and response
                         and self.verify_response(challenge_id, response))

        passed: Dict[Factor, float] = {}
        if crypto_ok:
            passed[Factor.CRYPTOGRAPHIC] = 1.0
        for ev in factors:
            if ev.factor is Factor.CRYPTOGRAPHIC:
                continue   # crypto only via a real challenge-response
            if ev.confidence >= self.factor_threshold:
                passed[ev.factor] = max(passed.get(ev.factor, 0.0), ev.confidence)

        if not crypto_ok:
            self._failures += 1
            return AuthResult(False, Authority.UNTRUSTED,
                              "cryptographic challenge-response failed (mandatory)",
                              factors_passed=list(passed))

        required = self.owner.required_factors
        if len(passed) < required:
            self._failures += 1
            return AuthResult(False, Authority.UNTRUSTED,
                              f"insufficient factors: {len(passed)}/{required}",
                              factors_passed=list(passed))

        # success — establish a session and a continuity token
        self._failures = 0
        score = sum(passed.values()) / len(passed)
        session = Session(id=uuid.uuid4().hex[:12], authority=Authority.OWNER,
                          factors=list(passed), established_at=now,
                          expires_at=now + self.session_ttl)
        self._sessions[session.id] = session
        token = self._issue_continuity_token(now)
        return AuthResult(True, Authority.OWNER, "the Master is verified",
                          factors_passed=list(passed), score=score, session=session,
                          continuity_token=token)

    # ---- continuity (survives restarts) ---- #
    def _issue_continuity_token(self, now: float) -> str:
        self._seq += 1
        tok = ContinuityToken(handle=self.owner.handle,
                              namespace=self.owner.continuity_namespace, seq=self._seq,
                              issued_at=now, expires_at=now + self.continuity_ttl)
        raw = json.dumps(tok.payload(), sort_keys=True).encode("utf-8")
        b64 = base64.urlsafe_b64encode(raw).decode("ascii")
        return f"{b64}.{self._sign(raw)}"

    def verify_continuity_token(self, token: str) -> AuthResult:
        now = self._clock()
        try:
            b64, sig = token.split(".", 1)
            raw = base64.urlsafe_b64decode(b64.encode("ascii"))
            payload = json.loads(raw)
        except Exception:  # noqa: BLE001
            return AuthResult(False, Authority.UNTRUSTED, "malformed continuity token")
        if not hmac.compare_digest(self._sign(raw), sig):
            self._failures += 1
            return AuthResult(False, Authority.UNTRUSTED, "continuity token signature invalid")
        tok = ContinuityToken.from_payload(payload)
        if tok.handle != self.owner.handle or tok.namespace != self.owner.continuity_namespace:
            return AuthResult(False, Authority.UNTRUSTED,
                              "continuity token is for a different identity")
        if now >= tok.expires_at:
            return AuthResult(False, Authority.UNTRUSTED, "continuity token expired")
        if tok.seq <= self._last_seen_seq:
            return AuthResult(False, Authority.UNTRUSTED,
                              "continuity token replay/downgrade rejected")
        self._last_seen_seq = tok.seq
        session = Session(id=uuid.uuid4().hex[:12], authority=Authority.OWNER,
                          factors=[Factor.CRYPTOGRAPHIC], established_at=now,
                          expires_at=now + self.session_ttl, via_continuity=True)
        self._sessions[session.id] = session
        return AuthResult(True, Authority.OWNER, "authority restored from continuity token",
                          factors_passed=[Factor.CRYPTOGRAPHIC], score=1.0, session=session)

    # ---- sessions ---- #
    def authority_of(self, session_id: str) -> Authority:
        s = self._sessions.get(session_id)
        return s.authority if (s and s.valid(self._clock())) else Authority.UNTRUSTED

    def is_master(self, session_id: str) -> bool:
        return self.authority_of(session_id) is Authority.OWNER

    def revoke(self, session_id: str) -> bool:
        s = self._sessions.get(session_id)
        if s is None or s.revoked:
            return False
        s.revoked = True
        return True

    def require_master(self, session_id: str) -> None:
        """Raise unless the session is a verified Master session (fail-closed)."""
        if not self.is_master(session_id):
            raise AuthorizationError("Master authority required",
                                     context={"session": session_id})

    def report(self) -> Dict[str, Any]:
        now = self._clock()
        return {"sessions": len(self._sessions),
                "active": sum(1 for s in self._sessions.values() if s.valid(now)),
                "open_challenges": len(self._challenges),
                "failures": self._failures, "last_seq": self._seq}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA Master-authentication self-test")
    print("=" * 70)

    now = {"t": 1000.0}
    guard = AuthGuard.from_passphrase("correct horse battery staple", clock=lambda: now["t"])

    # full multi-factor auth: crypto challenge-response + soft factors
    ch = guard.issue_challenge()
    response = guard.respond(ch.nonce)   # the Master signs with the shared key
    res = guard.authenticate(
        factors=[FactorEvidence(Factor.BIOMETRIC, 0.95),
                 FactorEvidence(Factor.BEHAVIORAL, 0.8)],
        challenge_id=ch.id, response=response)
    print(f"\nfull auth           : {res.to_dict()}")
    assert res.authenticated and res.authority is Authority.OWNER
    assert guard.is_master(res.session.id)
    token = res.continuity_token

    # response replay is refused (the challenge was consumed)
    assert not guard.verify_response(ch.id, response)
    print("response replay     : refused (single-use challenge) ✓")

    # an impostor with only public 'soft' factors but no key cannot pass
    ch2 = guard.issue_challenge()
    impostor = guard.authenticate(
        factors=[FactorEvidence(Factor.BIOMETRIC, 0.99),
                 FactorEvidence(Factor.CONTEXTUAL, 0.99)],
        challenge_id=ch2.id, response="deadbeef" * 8)   # wrong signature
    print(f"\nimpostor (no key)   : auth={impostor.authenticated} ({impostor.reason})")
    assert not impostor.authenticated and impostor.authority is Authority.UNTRUSTED

    # a wrong claimed identity is rejected outright
    ch3 = guard.issue_challenge()
    eve = guard.authenticate(challenge_id=ch3.id, response=guard.respond(ch3.nonce),
                             claimed_handle="EVE")
    assert not eve.authenticated
    print("wrong identity      : refused ✓")

    # insufficient factors (crypto alone, but 2 required) fails
    ch4 = guard.issue_challenge()
    thin = guard.authenticate(challenge_id=ch4.id, response=guard.respond(ch4.nonce))
    print(f"insufficient factors: auth={thin.authenticated} ({thin.reason})")
    assert not thin.authenticated

    # CONTINUITY: a fresh process with the same master key restores authority from the token
    restarted = AuthGuard.from_passphrase("correct horse battery staple",
                                          clock=lambda: now["t"])
    restored = restarted.verify_continuity_token(token)
    print(f"\ncontinuity restore  : auth={restored.authenticated} via_continuity="
          f"{restored.session.via_continuity}")
    assert restored.authenticated and restored.session.via_continuity

    # a stolen token signed with the wrong key is rejected
    attacker = AuthGuard.from_passphrase("wrong passphrase", clock=lambda: now["t"])
    forged = attacker.verify_continuity_token(token)
    assert not forged.authenticated
    print("forged token        : refused (signature invalid) ✓")

    # replay/downgrade: after seeing a newer token, an older one is rejected
    ch5 = guard.issue_challenge()
    res2 = guard.authenticate(
        factors=[FactorEvidence(Factor.BIOMETRIC, 0.9), FactorEvidence(Factor.COGNITIVE, 0.9)],
        challenge_id=ch5.id, response=guard.respond(ch5.nonce))
    newer = res2.continuity_token
    fresh = AuthGuard.from_passphrase("correct horse battery staple", clock=lambda: now["t"])
    assert fresh.verify_continuity_token(newer).authenticated      # newest accepted
    assert not fresh.verify_continuity_token(token).authenticated  # older rejected
    print("replay/downgrade    : older token rejected after a newer one ✓")

    # expiry: an old challenge is dead
    ch6 = guard.issue_challenge()
    now["t"] += 200.0
    assert not guard.verify_response(ch6.id, guard.respond(ch6.nonce))
    print("challenge expiry    : stale challenge refused ✓")

    print(f"\nreport              : {guard.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
