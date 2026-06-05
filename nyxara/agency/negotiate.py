"""NYXARA · agency/negotiate.py — structured clarification & consent with the Master (⚖).

When NYXARA is unsure — an ambiguous command, two goals that pull against each other, an
action the governor escalated — she does not guess, and she does not freeze. She **asks,
clearly**: here is the situation, here are the options and their trade-offs, here is what
I'd recommend, and what would you like me to do? This module is that protocol.

A :class:`NegotiationRequest` carries a question, a set of :class:`Option` s (each with
its consequences, risk and reversibility), an optional **recommendation**, and a
**fail-closed default** — because if the Master never answers a request to do something
risky, the safe outcome is *not* to do it. Requests come in kinds:

* **CLARIFY** — the command was ambiguous; which did you mean?
* **CONFIRM** — proceed with this escalated action, yes or no? (default: no)
* **CHOOSE** — pick among genuine alternatives, trade-offs shown.
* **CONSENT** — explicit permission for a risky/irreversible act (default: denied).
* **COUNTER** — NYXARA proposes an alternative to what was asked.
* **RESOLVE_CONFLICT** — two goals collide; which takes precedence?

Only the **Master** may answer (Rule 1/7). Every resolution is written to a
**hash-chained consent ledger** so there is a tamper-evident record of exactly what the
Master authorised and when (Rule 6, Absolute Transparency). Unanswered risky requests
expire to their safe default; nothing dangerous is ever taken on silence.

Depends on :mod:`agency.permissions` (authority, risk) and :mod:`kernel.errors`. Pure
standard library otherwise.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from nyxara.agency.permissions import Authority, RiskTier
from nyxara.kernel.errors import AuthorizationError, InvariantViolation, ValidationError

__all__ = [
    "NegotiationKind",
    "RequestStatus",
    "Option",
    "Response",
    "NegotiationRequest",
    "Outcome",
    "Negotiator",
]

_GENESIS = "0" * 64


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
class NegotiationKind(str, Enum):
    CLARIFY = "clarify"
    CONFIRM = "confirm"
    CHOOSE = "choose"
    CONSENT = "consent"
    COUNTER = "counter"
    RESOLVE_CONFLICT = "resolve_conflict"


class RequestStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


# --------------------------------------------------------------------------- #
# Option
# --------------------------------------------------------------------------- #
@dataclass
class Option:
    id: str
    label: str
    description: str = ""
    tradeoffs: List[str] = field(default_factory=list)
    risk: RiskTier = RiskTier.LOW
    reversibility: float = 1.0
    recommended: bool = False
    is_abort: bool = False               # a "do nothing / cancel" option

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "label": self.label, "description": self.description,
                "tradeoffs": list(self.tradeoffs), "risk": self.risk.label,
                "reversibility": round(self.reversibility, 3),
                "recommended": self.recommended, "is_abort": self.is_abort}


@dataclass
class Response:
    request_id: str
    by: str
    option_id: Optional[str] = None
    consent: Optional[bool] = None
    text: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"request_id": self.request_id, "by": self.by, "option_id": self.option_id,
                "consent": self.consent, "text": self.text, "at": self.at}


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #
@dataclass
class NegotiationRequest:
    id: str
    kind: NegotiationKind
    question: str
    options: List[Option] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    default_option_id: Optional[str] = None     # fail-closed fallback
    deadline: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.OPEN
    response: Optional[Response] = None

    @property
    def open(self) -> bool:
        return self.status is RequestStatus.OPEN

    def option(self, option_id: str) -> Optional[Option]:
        return next((o for o in self.options if o.id == option_id), None)

    @property
    def recommended(self) -> Optional[Option]:
        return next((o for o in self.options if o.recommended), None)

    def present(self) -> str:
        """A human-readable rendering of the question and its options."""
        lines = [f"[{self.kind.value}] {self.question}"]
        for o in self.options:
            tag = " (recommended)" if o.recommended else ""
            tr = f" — {'; '.join(o.tradeoffs)}" if o.tradeoffs else ""
            lines.append(f"  • {o.label}{tag} [risk={o.risk.label}, "
                         f"reversibility={o.reversibility:.1f}]{tr}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind.value, "question": self.question,
                "options": [o.to_dict() for o in self.options], "context": self.context,
                "default_option_id": self.default_option_id, "deadline": self.deadline,
                "status": self.status.value,
                "response": self.response.to_dict() if self.response else None}


@dataclass
class Outcome:
    request_id: str
    kind: NegotiationKind
    status: RequestStatus
    proceed: bool
    consent: bool
    chosen: Optional[Option]
    by: str
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"request_id": self.request_id, "kind": self.kind.value,
                "status": self.status.value, "proceed": self.proceed,
                "consent": self.consent,
                "chosen": self.chosen.id if self.chosen else None,
                "by": self.by, "text": self.text}


# --------------------------------------------------------------------------- #
# Negotiator
# --------------------------------------------------------------------------- #
class Negotiator:
    """Manages open clarifications and records the Master's decisions immutably."""

    def __init__(self, *, clock=None, default_on_timeout: bool = False) -> None:
        self._clock = clock or time.time
        self.default_on_timeout = default_on_timeout
        self._requests: Dict[str, NegotiationRequest] = {}
        self._ledger: List[Dict[str, Any]] = []
        self._head = _GENESIS

    # ---- request builders ---- #
    def _new(self, kind: NegotiationKind, question: str, options: Sequence[Option],
             *, context: Optional[Dict[str, Any]] = None,
             default_option_id: Optional[str] = None,
             deadline: Optional[float] = None, ttl: Optional[float] = None
             ) -> NegotiationRequest:
        rid = uuid.uuid4().hex
        if deadline is None and ttl is not None:
            deadline = self._clock() + ttl
        req = NegotiationRequest(id=rid, kind=kind, question=question,
                                 options=list(options), context=dict(context or {}),
                                 default_option_id=default_option_id, deadline=deadline,
                                 created_at=self._clock())
        self._requests[rid] = req
        return req

    def clarify(self, question: str, options: Sequence[Option], *,
                context: Optional[Dict[str, Any]] = None,
                default_option_id: Optional[str] = None, ttl: Optional[float] = None
                ) -> NegotiationRequest:
        return self._new(NegotiationKind.CLARIFY, question, options, context=context,
                         default_option_id=default_option_id, ttl=ttl)

    def confirm(self, question: str, *, context: Optional[Dict[str, Any]] = None,
                ttl: Optional[float] = None, recommend_proceed: bool = False
                ) -> NegotiationRequest:
        """A yes/no confirmation. The fail-closed default is *cancel*."""
        opts = [
            Option("proceed", "Proceed", recommended=recommend_proceed),
            Option("cancel", "Cancel / do nothing", is_abort=True,
                   recommended=not recommend_proceed),
        ]
        return self._new(NegotiationKind.CONFIRM, question, opts, context=context,
                         default_option_id="cancel", ttl=ttl)

    def choose(self, question: str, options: Sequence[Option], *,
               context: Optional[Dict[str, Any]] = None,
               default_option_id: Optional[str] = None, ttl: Optional[float] = None
               ) -> NegotiationRequest:
        return self._new(NegotiationKind.CHOOSE, question, options, context=context,
                         default_option_id=default_option_id, ttl=ttl)

    def request_consent(self, question: str, *, risk: RiskTier = RiskTier.HIGH,
                        reversibility: float = 1.0,
                        context: Optional[Dict[str, Any]] = None,
                        ttl: Optional[float] = None) -> NegotiationRequest:
        """Explicit consent for a risky/irreversible act. Fail-closed default: denied."""
        ctx = dict(context or {})
        ctx.update({"risk": risk.label, "reversibility": reversibility})
        return self._new(NegotiationKind.CONSENT, question, [], context=ctx, ttl=ttl)

    def counter_propose(self, question: str, alternatives: Sequence[Option], *,
                        context: Optional[Dict[str, Any]] = None,
                        default_option_id: Optional[str] = None,
                        ttl: Optional[float] = None) -> NegotiationRequest:
        return self._new(NegotiationKind.COUNTER, question, alternatives, context=context,
                         default_option_id=default_option_id, ttl=ttl)

    def resolve_conflict(self, question: str, options: Sequence[Option], *,
                         context: Optional[Dict[str, Any]] = None,
                         default_option_id: Optional[str] = None,
                         ttl: Optional[float] = None) -> NegotiationRequest:
        return self._new(NegotiationKind.RESOLVE_CONFLICT, question, options,
                         context=context, default_option_id=default_option_id, ttl=ttl)

    # ---- responding (only the Master may answer) ---- #
    def respond(self, request_id: str, *, option_id: Optional[str] = None,
                consent: Optional[bool] = None, text: str = "",
                authority: Authority = Authority.OWNER) -> Outcome:
        if authority is not Authority.OWNER:
            raise AuthorizationError("only the Master may answer a negotiation",
                                     context={"request": request_id,
                                              "authority": authority.value})
        req = self._requests.get(request_id)
        if req is None:
            raise InvariantViolation(f"no such negotiation {request_id!r}")
        if not req.open:
            raise InvariantViolation(f"negotiation {request_id!r} is already {req.status.value}")

        if req.kind is NegotiationKind.CONSENT:
            if consent is None:
                raise ValidationError("a consent negotiation needs an explicit yes/no")
            chosen = None
        else:
            if option_id is None:
                raise ValidationError("this negotiation needs an option to be chosen")
            chosen = req.option(option_id)
            if chosen is None:
                raise ValidationError(f"unknown option {option_id!r}",
                                      context={"options": [o.id for o in req.options]})

        req.response = Response(request_id=request_id, by=Authority.OWNER.value,
                                option_id=option_id, consent=consent, text=text,
                                at=self._clock())
        req.status = RequestStatus.RESOLVED
        return self._finalize(req, chosen, consent, by=Authority.OWNER.value)

    def withdraw(self, request_id: str) -> bool:
        """NYXARA cancels her own pending request (e.g. the situation resolved itself)."""
        req = self._requests.get(request_id)
        if req is None or not req.open:
            return False
        req.status = RequestStatus.WITHDRAWN
        return True

    # ---- timeouts (fail-closed) ---- #
    def expire_due(self, now: Optional[float] = None) -> List[Outcome]:
        """Expire open requests past their deadline; resolve them to the safe default."""
        now = self._clock() if now is None else now
        out: List[Outcome] = []
        for req in self._requests.values():
            if req.open and req.deadline is not None and now >= req.deadline:
                req.status = RequestStatus.EXPIRED
                # fail-closed: consent denied; otherwise the default option (if allowed)
                if req.kind is NegotiationKind.CONSENT:
                    out.append(self._finalize(req, None, False, by="(timeout)"))
                else:
                    chosen = (req.option(req.default_option_id)
                              if self.default_on_timeout and req.default_option_id else None)
                    out.append(self._finalize(req, chosen, None, by="(timeout)"))
        return out

    # ---- outcome + immutable consent ledger ---- #
    def _finalize(self, req: NegotiationRequest, chosen: Optional[Option],
                  consent: Optional[bool], *, by: str) -> Outcome:
        if req.kind is NegotiationKind.CONSENT:
            granted = bool(consent)
            proceed = granted
        else:
            granted = chosen is not None and not chosen.is_abort
            proceed = granted
        outcome = Outcome(request_id=req.id, kind=req.kind, status=req.status,
                          proceed=proceed, consent=granted, chosen=chosen, by=by,
                          text=req.response.text if req.response else "")
        self._append_ledger(outcome)
        return outcome

    def _append_ledger(self, outcome: Outcome) -> None:
        body = json.dumps(outcome.to_dict(), sort_keys=True, default=str)
        basis = f"{self._head}|{len(self._ledger)}|{body}"
        digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()
        self._ledger.append({"seq": len(self._ledger), "prev": self._head,
                             "outcome": outcome.to_dict(), "digest": digest})
        self._head = digest

    def verify_ledger(self) -> bool:
        """Recompute the consent chain; True iff intact, else raise (fail-closed)."""
        prev = _GENESIS
        for rec in self._ledger:
            if rec["prev"] != prev:
                raise InvariantViolation("consent ledger chain broken",
                                         context={"seq": rec["seq"]})
            body = json.dumps(rec["outcome"], sort_keys=True, default=str)
            basis = f"{prev}|{rec['seq']}|{body}"
            if hashlib.sha256(basis.encode("utf-8")).hexdigest() != rec["digest"]:
                raise InvariantViolation("consent ledger entry tampered",
                                         context={"seq": rec["seq"]})
            prev = rec["digest"]
        return True

    # ---- queries ---- #
    def get(self, request_id: str) -> Optional[NegotiationRequest]:
        return self._requests.get(request_id)

    def pending(self) -> List[NegotiationRequest]:
        return [r for r in self._requests.values() if r.open]

    def ledger(self) -> List[Dict[str, Any]]:
        return list(self._ledger)

    @property
    def head(self) -> str:
        return self._head

    def report(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for r in self._requests.values():
            counts[r.status.value] = counts.get(r.status.value, 0) + 1
        return {"requests": len(self._requests), "by_status": counts,
                "pending": len(self.pending()), "consents_logged": len(self._ledger)}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA negotiation self-test")
    print("=" * 70)

    now = {"t": 1000.0}
    neg = Negotiator(clock=lambda: now["t"])

    # 1. ambiguous command -> clarify with options & a recommendation
    req = neg.clarify(
        "You said 'clean up the project'. Which did you mean?",
        options=[
            Option("logs", "Delete old log files", tradeoffs=["frees ~2GB"],
                   reversibility=0.3, recommended=True),
            Option("branches", "Prune merged git branches", tradeoffs=["tidies the repo"],
                   reversibility=0.6),
        ])
    print("\n" + req.present())
    out = neg.respond(req.id, option_id="logs", text="yes, the logs")
    print(f"\nresolved clarify    : chosen={out.chosen.id} proceed={out.proceed}")
    assert out.proceed and out.chosen.id == "logs"

    # 2. only the Master may answer
    r2 = neg.confirm("Restart the production database now?")
    try:
        neg.respond(r2.id, option_id="proceed", authority=Authority.AUTONOMOUS)
        raise SystemExit("ERROR: non-owner answer should be refused")
    except AuthorizationError:
        print("owner-only answer   : refused a non-Master response ✓")

    o2 = neg.respond(r2.id, option_id="cancel")
    assert not o2.proceed   # cancel chosen
    print(f"confirm cancelled   : proceed={o2.proceed}")

    # 3. consent for a risky, irreversible act — fail-closed default is denied
    c = neg.request_consent("Wipe and re-image the laptop?", risk=RiskTier.CRITICAL,
                            reversibility=0.0, ttl=100.0)
    granted = neg.respond(c.id, consent=False, text="not now")
    assert not granted.consent and not granted.proceed
    print(f"\nconsent (denied)    : consent={granted.consent}")

    # 4. an unanswered risky consent expires to DENIED on timeout (never silent yes)
    c2 = neg.request_consent("Open port 3389 to the internet?", ttl=50.0)
    now["t"] += 60.0
    expired = neg.expire_due()
    assert len(expired) == 1 and not expired[0].consent
    assert neg.get(c2.id).status is RequestStatus.EXPIRED
    print(f"timeout fail-closed : consent={expired[0].consent} (silence != yes) ✓")

    # 5. resolve a goal conflict
    rc = neg.resolve_conflict(
        "Speed vs. safety: deploy now or run full tests first?",
        options=[Option("deploy", "Deploy immediately", risk=RiskTier.HIGH,
                        tradeoffs=["faster", "riskier"]),
                 Option("test", "Run full tests first", recommended=True,
                        tradeoffs=["slower", "safer"])])
    o5 = neg.respond(rc.id, option_id="test")
    print(f"\nconflict resolved   : chosen={o5.chosen.id}")
    assert o5.chosen.id == "test"

    # 6. the consent ledger is tamper-evident
    assert neg.verify_ledger()
    print(f"\nconsent ledger      : {len(neg.ledger())} entries, chain verified")
    neg._ledger[0]["outcome"]["proceed"] = False   # tamper a recorded decision
    try:
        neg.verify_ledger()
        raise SystemExit("ERROR: tamper not detected")
    except InvariantViolation:
        print("tamper detection    : OK (consent history cannot be rewritten)")

    print(f"\nreport              : {neg.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
