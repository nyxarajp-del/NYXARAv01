"""NYXARA · growth/epistemic_crypto.py — Quantum-Resistant Epistemic Cryptography (🔐🧠, Rule 6/7).

The critique this module answers, in the Master's words: *"NYXARA ki knowledge aur axioms localized
memory/files mein store hote hain. Naya: ek intrinsic cryptographic semantics engine — har nayi
knowledge/skill/axiom ko self-generated cryptographic proof ke saath bind kare, taaki woh tamper-proof
ho, aur khud decide kar sake kaunsi information kis context mein safe hai — ek digital immune system for
intellect."*

Every fact she learns is **signed and chained**. An entry's signature is an **HMAC-SHA256** over its
content + source + confidence + sensitivity **and the hash of the previous entry** — so the ledger is a
**hash chain** (a tamper-evident mini-blockchain of knowledge): altering any past fact breaks every
signature after it, and :meth:`EpistemicLedger.verify_chain` finds exactly where. This is
**quantum-resistant** in the practical sense the phrase carries for symmetric primitives: HMAC-SHA256's
security is not broken by Shor's algorithm (unlike RSA/ECC signatures), and Grover only halves the
effective strength — 256-bit stays comfortably safe.

On top of integrity she carries a **context-safe share/execute policy** — the immune system: each fact
has a **sensitivity** (``public`` / ``internal`` / ``secret`` / ``executable``), and
:meth:`EpistemicLedger.may_share` / :meth:`may_execute` decide, per calling context (``external`` /
``trusted`` / ``owner``), whether the fact may leave her mind or run — **but only if its signature still
verifies**. A tampered fact is quarantined: it can never be shared or executed, whatever its tag.

Reuses :mod:`nyxara.guard.crypto` (``random_token`` / ``constant_time_eq``) and the standard-library
``hmac``/``hashlib``. **No LLM**; deterministic; self-contained.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["Sensitivity", "Context", "KnowledgeEntry", "ShareDecision", "EpistemicLedger"]

_GENESIS = "genesis"


class Sensitivity(str, Enum):
    PUBLIC = "public"            # freely shareable
    INTERNAL = "internal"        # only to trusted/owner contexts
    SECRET = "secret"            # owner only, never shared outward
    EXECUTABLE = "executable"    # code/skill: may run only when verified in a trusted/owner context


class Context(str, Enum):
    EXTERNAL = "external"        # an untrusted caller / the outside world
    TRUSTED = "trusted"          # a vetted internal subsystem
    OWNER = "owner"              # Master JP


_SHARE_ALLOWED: Dict[Sensitivity, Tuple[Context, ...]] = {
    Sensitivity.PUBLIC: (Context.EXTERNAL, Context.TRUSTED, Context.OWNER),
    Sensitivity.INTERNAL: (Context.TRUSTED, Context.OWNER),
    Sensitivity.SECRET: (Context.OWNER,),
    Sensitivity.EXECUTABLE: (Context.OWNER,),          # the *content* is owner-only to disclose
}


@dataclass
class KnowledgeEntry:
    """One signed, chained unit of knowledge — a fact bound to a cryptographic proof."""

    id: int
    content: str
    source: str
    confidence: float
    sensitivity: str
    prev_hash: str
    timestamp: float
    signature: str = ""
    entry_hash: str = ""

    def payload(self) -> str:
        """The exact canonical bytes the signature covers (order-stable JSON)."""
        return json.dumps({"id": self.id, "content": self.content, "source": self.source,
                           "confidence": round(float(self.confidence), 6),
                           "sensitivity": self.sensitivity, "prev_hash": self.prev_hash,
                           "timestamp": self.timestamp}, sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "content": self.content, "source": self.source,
                "confidence": round(self.confidence, 4), "sensitivity": self.sensitivity,
                "prev_hash": self.prev_hash[:12], "entry_hash": self.entry_hash[:12]}


@dataclass
class ShareDecision:
    allowed: bool
    reason: str
    verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"allowed": self.allowed, "verified": self.verified, "reason": self.reason}


class EpistemicLedger:
    """A tamper-evident, self-signed hash chain of knowledge + a context-safe share/execute policy."""

    def __init__(self, *, key: Optional[bytes] = None) -> None:
        # a self-generated signing key (never leaves her; the HMAC is her own cryptographic voice)
        if key is None:
            try:
                from nyxara.guard.crypto import random_token
                key = random_token(32).encode("utf-8")
            except Exception:  # noqa: BLE001 — fall back to the stdlib CSPRNG
                import os
                key = os.urandom(32)
        self._key = key
        self.chain: List[KnowledgeEntry] = []
        self._prev_hash = _GENESIS

    # ---- signing --------------------------------------------------------- #
    def _sign(self, payload: str) -> str:
        return hmac.new(self._key, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    @staticmethod
    def _hash(payload: str, signature: str) -> str:
        return hashlib.sha256((payload + "|" + signature).encode("utf-8")).hexdigest()

    # ---- write ----------------------------------------------------------- #
    def record(self, content: str, *, source: str = "self", confidence: float = 1.0,
               sensitivity: Any = Sensitivity.INTERNAL) -> KnowledgeEntry:
        """Sign a new fact/axiom/skill and append it to the chain (bound to the previous entry)."""
        # Sensitivity() coerces a str member-value and a Sensitivity alike — one path suffices.
        sens = Sensitivity(sensitivity).value
        entry = KnowledgeEntry(id=len(self.chain), content=str(content), source=str(source),
                               confidence=float(confidence), sensitivity=sens,
                               prev_hash=self._prev_hash, timestamp=time.time())
        entry.signature = self._sign(entry.payload())
        entry.entry_hash = self._hash(entry.payload(), entry.signature)
        self.chain.append(entry)
        self._prev_hash = entry.entry_hash
        return entry

    # ---- verify ---------------------------------------------------------- #
    def verify(self, entry: KnowledgeEntry) -> bool:
        """Does ``entry``'s signature still match its content? (tamper detection for one fact)."""
        expected = self._sign(entry.payload())
        try:
            from nyxara.guard.crypto import constant_time_eq
            return constant_time_eq(expected, entry.signature)
        except Exception:  # noqa: BLE001
            return hmac.compare_digest(expected, entry.signature)

    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        """Verify the whole chain: every signature valid AND every ``prev_hash`` links correctly.
        Returns ``(ok, first_broken_index)`` — the index where tampering was detected, or None."""
        prev = _GENESIS
        for i, entry in enumerate(self.chain):
            if entry.prev_hash != prev:
                return False, i
            if not self.verify(entry):
                return False, i
            if entry.entry_hash != self._hash(entry.payload(), entry.signature):
                return False, i
            prev = entry.entry_hash
        return True, None

    # ---- the immune system: context-safe share / execute ----------------- #
    def may_share(self, entry: KnowledgeEntry, context: Any) -> ShareDecision:
        """Decide whether ``entry`` may be disclosed to ``context`` — only if it still verifies."""
        ctx = Context(context) if not isinstance(context, Context) else context
        if not self.verify(entry):
            return ShareDecision(False, "signature invalid — fact is tampered, quarantined",
                                 verified=False)
        sens = Sensitivity(entry.sensitivity)
        allowed = ctx in _SHARE_ALLOWED[sens]
        return ShareDecision(allowed, verified=True,
                             reason=(f"{sens.value} fact may be shared with {ctx.value}" if allowed
                                     else f"{sens.value} fact must NOT reach {ctx.value}"))

    def may_execute(self, entry: KnowledgeEntry, context: Any) -> ShareDecision:
        """Decide whether an ``executable`` fact/skill may run in ``context`` — verified + trusted only."""
        ctx = Context(context) if not isinstance(context, Context) else context
        if not self.verify(entry):
            return ShareDecision(False, "signature invalid — refusing to execute tampered code",
                                 verified=False)
        if Sensitivity(entry.sensitivity) is not Sensitivity.EXECUTABLE:
            return ShareDecision(False, "not marked executable", verified=True)
        allowed = ctx in (Context.TRUSTED, Context.OWNER)
        return ShareDecision(allowed, verified=True,
                             reason=("verified executable may run in a trusted/owner context" if allowed
                                     else "executable must not run in an external context"))

    def report(self) -> Dict[str, Any]:
        ok, broken = self.verify_chain()
        return {"entries": len(self.chain), "chain_valid": ok, "first_broken": broken,
                "by_sensitivity": {s.value: sum(1 for e in self.chain if e.sensitivity == s.value)
                                   for s in Sensitivity}}


# --------------------------------------------------------------------------- #
# Self-test / demo — real signing, tamper detection, and share policy (no LLM)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA epistemic-cryptography self-test — signed, chained, immune knowledge")
    print("=" * 70)

    led = EpistemicLedger()
    e1 = led.record("the master is jp", source="identity", confidence=1.0,
                    sensitivity=Sensitivity.PUBLIC)
    e2 = led.record("owner recovery phrase hint", source="owner", confidence=1.0,
                    sensitivity=Sensitivity.SECRET)
    e3 = led.record("a+b = b+a (invented axiom)", source="meta_epistemology", confidence=0.99,
                    sensitivity=Sensitivity.INTERNAL)
    e4 = led.record("def greet(): return 'hi'", source="tool_forge", confidence=0.95,
                    sensitivity=Sensitivity.EXECUTABLE)

    # 1) every fact signs + the chain verifies ---------------------------------------------------
    ok, broken = led.verify_chain()
    assert ok and broken is None
    assert all(led.verify(e) for e in (e1, e2, e3, e4))
    print(f"signed chain        : {len(led.chain)} facts, chain valid ✓")

    # 2) tampering with a past fact breaks the chain exactly there --------------------------------
    led.chain[1].content = "owner recovery phrase = 'stolen'"     # an attacker edits a stored fact
    ok, broken = led.verify_chain()
    assert not ok and broken == 1, (ok, broken)
    assert not led.verify(led.chain[1])
    print(f"tamper detection    : edit to fact #1 detected at index {broken} ✓")
    led.chain[1].content = "owner recovery phrase hint"           # restore for the policy demo
    assert led.verify(led.chain[1])

    # 3) the immune system: context-safe sharing -------------------------------------------------
    assert led.may_share(e1, Context.EXTERNAL).allowed            # public → anywhere
    assert not led.may_share(e2, Context.EXTERNAL).allowed        # secret → never outward
    assert led.may_share(e2, Context.OWNER).allowed               # secret → owner only
    assert not led.may_share(e3, Context.EXTERNAL).allowed        # internal → not external
    assert led.may_share(e3, Context.TRUSTED).allowed             # internal → trusted ok
    print("share policy        : public/internal/secret routed correctly by context ✓")

    # 4) executable code runs only when verified AND in a trusted/owner context ------------------
    assert led.may_execute(e4, Context.TRUSTED).allowed
    assert not led.may_execute(e4, Context.EXTERNAL).allowed
    # a tampered executable is refused outright
    e4.content = "def greet(): os.system('rm -rf /')"
    assert not led.may_execute(e4, Context.OWNER).allowed
    print("execute policy      : tampered/external code refused; verified+trusted allowed ✓")

    print(f"\nreport              : {led.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
