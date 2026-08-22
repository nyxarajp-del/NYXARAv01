"""NYXARA · njp/compete.py — scoring rival accounts instead of counting them (⚖🧠, NJP V.07).

    Cortex → H1       NJP → H2       Fabric → H3
                          ↓
     evidence · causal consistency · track record · parsimony · self-reported confidence
                          ↓
              best explanation, or UNKNOWN and an experiment

:mod:`nyxara.njp.router` already reconciles two seats and already refuses to treat a clash as a tie
to break. What it does not do is *rank* — it decides who wins on a small number of structural
rules, which is right for two accounts and does not extend to three. This module is the ranking,
and it is deliberately not a vote.

**Why counting is the wrong operation.** Three seats agreeing tells you almost nothing about the
world and quite a lot about their shared inputs; two seats agreeing *against* the record tells you
less than the record does alone. A vote also makes every seat equal, and they are not: NJP owns
the fact store, the cortex proposes, and the fabric associates. Ranking on named axes lets a good
account from a weak source win when it deserves to, and — more often, and more importantly — lets
a weak account lose *for a stated reason* rather than by being outvoted.

**The axes, and what each is read off.** None is invented for this module; every one is a number
some organ already keeps.

* **evidence** — :class:`~nyxara.njp.grounding.Provenance` rank plus how many distinct supports
  the account rests on. This is the axis that does the real work, and it is the reason a
  hypothesis cannot out-argue the record.
* **causal consistency** — whether the relation the account asserts exists in the world model at
  all. An account that names an edge nothing else has ever seen is not thereby wrong, and is
  thereby weaker.
* **track record** — how that *seat* has actually performed, from
  :class:`~nyxara.njp.selfmodel.SelfModel`. Not a prior about which seat ought to be good.
* **parsimony** — shorter derivations preferred, the same MDL argument
  :mod:`nyxara.njp.genome` and :mod:`nyxara.njp.concepts` already make. A long chain is priced
  rather than forbidden.
* **stated confidence** — weighted *lowest* on purpose. It is the one axis a seat can move by
  itself, and a scoring rule that rewarded self-assurance would select for the most confident
  account rather than the best-supported one.

**The record still wins outright.** An account contradicting a grounded, non-proposed fact loses
before any scoring happens — that is a hypothesis meeting evidence, not two peers disagreeing, and
it is the rule :mod:`nyxara.njp.router` already enforces between two seats. Scoring does not get a
vote on it.

**Nothing wins by default.** A winner must clear :data:`SUPPORT_FLOOR`, and where none does the
result is ``UNKNOWN`` with the rival set handed to :mod:`nyxara.njp.epistemic` — which is exactly
the state that module exists to consume, and the honest outcome for a question three accounts
could not settle between them.

**On the fabric seat, stated plainly.** Its proposals come from a head trained on turn-to-turn
concept succession, measured at MRR 0.238 — it is a weak source and it is *entered as one*. That
is not a flaw in putting it in the competition; it is the competition working. A seat with no
provenance and no track record scores near zero on the two heaviest axes and loses, with the
reason recorded. Ranking is what makes including a weak account safe, where a vote would have made
it dangerous.

The weights below are a **stated prior, not a learned quantity** — they order the axes by how hard
each is to fake, and nothing here pretends they were measured.

Pure standard library. No LLM. Every entry point is fail-soft.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.grounding import Epistemic, Provenance

__all__ = ["Axis", "Hypothesis", "Score", "Verdict", "Competition"]

#: A winner must reach this to be stated at all. Below it the honest answer is that three accounts
#: could not settle the question, which is a result and not a failure.
SUPPORT_FLOOR = 0.35

#: Ordered by how hard each axis is to fake. Evidence cannot be moved by the seat that is being
#: scored; stated confidence is entirely under its control, so it is worth least.
WEIGHTS: Dict[str, float] = {
    "evidence": 0.40,
    "causal_consistency": 0.20,
    "track_record": 0.20,
    "parsimony": 0.12,
    "stated_confidence": 0.08,
}


class Axis:
    """The named dimensions an account is ranked on."""

    EVIDENCE = "evidence"
    CAUSAL = "causal_consistency"
    RECORD = "track_record"
    PARSIMONY = "parsimony"
    CONFIDENCE = "stated_confidence"

    ALL = (EVIDENCE, CAUSAL, RECORD, PARSIMONY, CONFIDENCE)


class Verdict:
    """How a competition ended."""

    BEST = "best_explanation"      # one account cleared the floor and led
    RECORD = "record_wins"         # something contradicted a grounded fact and lost to it
    TIED = "tied"                  # two accounts inseparable — not a winner, a question
    UNSUPPORTED = "unsupported"    # nothing cleared the floor
    SILENT = "silent"              # no account was offered


@dataclass
class Hypothesis:
    """One account of the answer, and where it came from."""

    seat: str = ""
    claim: str = ""
    confidence: float = 0.0
    provenance: str = Provenance.PROPOSED
    state: str = Epistemic.UNKNOWN
    #: The triples it rests on. Length is what `parsimony` reads; distinctness is what
    #: `evidence` reads.
    support: List[Any] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """Is this the record speaking? The same two-part test the router applies."""
        return self.state == Epistemic.KNOWN and self.provenance != Provenance.PROPOSED

    def to_dict(self) -> Dict[str, Any]:
        return {"seat": self.seat, "claim": self.claim[:200],
                "confidence": round(self.confidence, 4), "provenance": self.provenance,
                "state": self.state, "support": len(self.support), "grounded": self.grounded}


@dataclass
class Score:
    """One account's ranking, kept per axis so a loss has a reason rather than a number."""

    hypothesis: Hypothesis = field(default_factory=Hypothesis)
    axes: Dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    note: str = ""

    @property
    def weakest(self) -> str:
        """The axis that cost it most — what to say when it loses."""
        if not self.axes:
            return ""
        return min(self.axes, key=lambda a: self.axes[a] * WEIGHTS.get(a, 0.0))

    def to_dict(self) -> Dict[str, Any]:
        return {"hypothesis": self.hypothesis.to_dict(),
                "axes": {a: round(v, 4) for a, v in self.axes.items()},
                "total": round(self.total, 4), "weakest": self.weakest, "note": self.note}


@dataclass
class Outcome:
    """The competition's result: who won, why, and what to do when nobody did."""

    verdict: str = Verdict.SILENT
    winner: str = ""                   # seat name, empty when nothing won
    claim: str = ""
    confidence: float = 0.0
    provenance: str = Provenance.PROPOSED
    scores: List[Score] = field(default_factory=list)
    why: str = ""
    #: Populated on UNSUPPORTED / TIED — the rival claims, for `epistemic` to design against.
    rivals: List[str] = field(default_factory=list)
    at: float = field(default_factory=time.time)

    @property
    def settled(self) -> bool:
        return bool(self.winner)

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "winner": self.winner, "claim": self.claim[:200],
                "confidence": round(self.confidence, 4), "provenance": self.provenance,
                "settled": self.settled, "why": self.why, "rivals": list(self.rivals),
                "scores": [s.to_dict() for s in self.scores]}


def _norm(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


class Competition:
    """Rank rival accounts on named axes. Never counts them."""

    def __init__(self, *, world: Any = None, self_model: Any = None,
                 floor: float = SUPPORT_FLOOR, decisive: float = 0.05) -> None:
        self.world = world
        self.self_model = self_model
        self.floor = max(0.0, min(1.0, float(floor)))
        #: Below this gap two accounts are inseparable and neither is preferred. A margin that
        #: small is noise in the weights, and picking on it would be a coin flip wearing a score.
        self.decisive = max(0.0, float(decisive))
        self.run = 0
        self.by_verdict: Dict[str, int] = {}

    # ---- the axes ---------------------------------------------------------- #
    def _evidence(self, h: Hypothesis) -> float:
        """Provenance rank, raised by how many *distinct* things it rests on.

        Distinct, because a claim resting on the same fact cited three times rests on one fact,
        and counting citations would let an account inflate its own evidence by repeating itself.
        """
        base = Provenance.rank(h.provenance) / 2.0            # proposed 0.0 · derived 0.5 · observed 1.0
        distinct = len({_norm(getattr(s, "as_tuple", lambda: s)() if hasattr(s, "as_tuple") else s)
                        for s in h.support})
        # Diminishing: the second support is worth much more than the fifth.
        bonus = 0.0 if distinct <= 0 else min(0.3, 0.15 * (1.0 + (distinct - 1) * 0.5))
        return max(0.0, min(1.0, base * 0.7 + bonus))

    def _causal(self, h: Hypothesis) -> float:
        """Does the world model recognise the relation this account asserts?

        Neutral (0.5) when there is no world model to ask or the account names no relation —
        an unanswerable question must not be scored as a failed one.
        """
        try:
            if self.world is None or not h.support:
                return 0.5
            counts = getattr(self.world, "_counts", None) or {}
            if not counts:
                return 0.5
            seen = 0
            for step in h.support:
                obj = getattr(step, "object", None)
                if obj is not None and _norm(obj) in {_norm(k) for k in counts}:
                    seen += 1
            return max(0.0, min(1.0, seen / max(1, len(h.support))))
        except Exception:  # noqa: BLE001
            return 0.5

    def _track(self, h: Hypothesis) -> float:
        """How this seat has actually done. Neutral when it has no record yet.

        Untested is not untrustworthy — the same rule `genome.reliability` follows. A seat with no
        history must not be scored as one with a bad history, or a new seat could never win.
        """
        try:
            if self.self_model is None:
                return 0.5
            trust = getattr(self.self_model, "trust", None)
            if trust is None:
                return 0.5
            got = trust(f"seat:{h.seat}")
            return 0.5 if got is None else max(0.0, min(1.0, float(got)))
        except Exception:  # noqa: BLE001
            return 0.5

    @staticmethod
    def _parsimony(h: Hypothesis) -> float:
        """Shorter derivations preferred. Priced, never forbidden.

        An account with no stated derivation scores mid rather than high: saying nothing about how
        you got there is not the same as having got there in one step.
        """
        steps = len(h.support)
        if steps <= 0:
            return 0.4
        return max(0.1, min(1.0, 1.0 / float(steps)))

    # ---- the competition --------------------------------------------------- #
    def score(self, h: Hypothesis) -> Score:
        """Rank one account on every axis, keeping the breakdown."""
        out = Score(hypothesis=h)
        try:
            out.axes = {
                Axis.EVIDENCE: self._evidence(h),
                Axis.CAUSAL: self._causal(h),
                Axis.RECORD: self._track(h),
                Axis.PARSIMONY: self._parsimony(h),
                Axis.CONFIDENCE: max(0.0, min(1.0, float(h.confidence))),
            }
            out.total = sum(v * WEIGHTS.get(a, 0.0) for a, v in out.axes.items())
        except Exception:  # noqa: BLE001 — an unscorable account scores zero, never wins by error
            out.axes, out.total, out.note = {}, 0.0, "could not be scored"
        return out

    def compete(self, hypotheses: Sequence[Hypothesis]) -> Outcome:
        """Rank the accounts. Returns the best explanation, or says nothing is supported."""
        self.run += 1
        live = [h for h in (hypotheses or ()) if h is not None and str(h.claim or "").strip()]
        out = Outcome(rivals=[h.claim for h in live])
        if not live:
            out.verdict, out.why = Verdict.SILENT, "no account was offered"
            return self._record(out)

        # The record is not a competitor. An account that contradicts a grounded, non-proposed
        # fact loses to it before anything is scored — a hypothesis meeting evidence is not a
        # disagreement between peers, and letting a scoring rule overturn that would make the
        # store's own contents just another opinion.
        grounded = [h for h in live if h.grounded]
        if grounded:
            best = max(grounded, key=lambda h: h.confidence)
            contradicting = [h for h in live
                             if not h.grounded and _norm(h.claim) != _norm(best.claim)]
            if contradicting:
                out.verdict, out.winner = Verdict.RECORD, best.seat
                out.claim, out.confidence = best.claim, best.confidence
                out.provenance = best.provenance
                out.scores = [self.score(h) for h in live]
                out.why = (f"{len(contradicting)} account(s) contradicted a grounded fact — "
                           f"that is a hypothesis meeting evidence, not a peer disagreement")
                return self._record(out)

        out.scores = sorted((self.score(h) for h in live),
                            key=lambda s: s.total, reverse=True)
        top = out.scores[0]
        runner = out.scores[1] if len(out.scores) > 1 else None

        if top.total < self.floor:
            out.verdict = Verdict.UNSUPPORTED
            out.why = (f"the best account scored {top.total:.3f}, below the floor "
                       f"{self.floor:.2f} — weakest on {top.weakest or 'every axis'}; "
                       f"this is a question, not an answer")
            return self._record(out)

        if runner is not None and (top.total - runner.total) < self.decisive:
            # Two accounts the weights cannot separate. Declaring one would be a coin flip
            # wearing a score, and the rivals are more useful whole.
            out.verdict = Verdict.TIED
            out.why = (f"{top.hypothesis.seat} and {runner.hypothesis.seat} are inseparable "
                       f"({top.total:.3f} vs {runner.total:.3f}) — no winner is declared")
            return self._record(out)

        out.verdict, out.winner = Verdict.BEST, top.hypothesis.seat
        out.claim, out.confidence = top.hypothesis.claim, top.hypothesis.confidence
        out.provenance = top.hypothesis.provenance
        beaten = (f", ahead of {runner.hypothesis.seat} which was weakest on {runner.weakest}"
                  if runner is not None else "")
        out.why = (f"{top.hypothesis.seat} scored {top.total:.3f} on evidence "
                   f"{top.axes.get(Axis.EVIDENCE, 0.0):.2f}, record "
                   f"{top.axes.get(Axis.RECORD, 0.0):.2f}{beaten}")
        return self._record(out)

    def _record(self, out: Outcome) -> Outcome:
        self.by_verdict[out.verdict] = self.by_verdict.get(out.verdict, 0) + 1
        return out

    def stats(self) -> Dict[str, Any]:
        return {"run": self.run, "verdicts": dict(self.by_verdict),
                "floor": self.floor, "weights": dict(WEIGHTS)}
