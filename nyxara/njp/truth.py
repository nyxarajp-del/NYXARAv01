"""NYXARA · njp/truth.py — the Truth-Seeking Gauntlet (⚖, NJP V.01).

Nothing NJP produces becomes *established* by being confident. It becomes established by
surviving this: several **independent** sources of evidence, at least one of which can actually be
wrong in a checkable way, with any single refutation ending the matter.

The failure this exists to prevent is the ordinary one — a system trained on skewed evidence
restating its skew fluently and calling it knowledge. The defence is not a better generator; it is
a gate that a plausible-sounding claim cannot pass without something outside itself agreeing.

**The sources, and what each can actually establish.** They are deliberately unlike each other,
because three sources that fail the same way are one source:

* :class:`FormalSource` — hands the claim to the proof core (z3 / sympy). Can return **PROVED**
  or **REFUTED with a counter-model**. Most claims are not formally expressible, and it says
  ``INEXPRESSIBLE`` rather than guessing — that refusal is the point of it.
* :class:`PredictiveSource` — the only source that can be **surprised**. It asks whether the claim
  predicts observations that were **withheld** from whatever produced it. Fit is not transfer, and
  this is the difference.
* :class:`ConsistencySource` — does it contradict something already established? A contradiction
  is a genuine refutation; mere absence of contradiction is the *weakest* possible support and is
  scored as such.
* :class:`LedgerSource` — has NJP been wrong about something shaped like this before? Reads
  :class:`~nyxara.njp.ledger.ErrorMemory`. This is the "learn from your own errors" path, and it
  can only ever count **against** a claim, never for it.
* :class:`ObservationSource` — independent recorded observations that bear on the claim.

**The verdict rules**, and they are fail-closed:

1. Any source that **refutes** ⇒ ``REFUTED``. One refutation outranks any amount of agreement.
2. ``ESTABLISHED`` requires ``min_sources`` independent supports **including at least one hard
   source** (formal or predictive). Soft agreement alone never establishes anything — that is
   exactly how consensus becomes bias.
3. Otherwise ``SUPPORTED`` (real evidence, not enough) or ``CONJECTURE`` (none).
4. Nothing expressible produced ⇒ ``ABSTAINED``.

**What is honestly claimed.** This makes NJP *refuse to assert* what she cannot corroborate: an
unverified claim leaves here labelled ``CONJECTURE`` and the brain says so instead of stating it.
That is a real and strong property, and it is the mechanism people mean by "does not hallucinate".
What no gate can do is make the error rate a *constant* — the measured rate lives in
:mod:`nyxara.njp.ledger` and is whatever the evidence says it is, which is the only honest place
for it. A number written here would be decoration; a number written there is a measurement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = ["Verdict", "Evidence", "Judgement", "Source", "FormalSource", "PredictiveSource",
           "ConsistencySource", "LedgerSource", "ObservationSource", "TruthGauntlet"]


class Verdict:
    """The only labels a claim may carry out of the gauntlet."""

    ESTABLISHED = "established"     # multi-source verified, including a hard source
    SUPPORTED = "supported"        # real evidence, short of establishment
    CONJECTURE = "conjecture"      # nothing corroborates it — say so, do not assert it
    REFUTED = "refuted"
    ABSTAINED = "abstained"        # no source could speak to it at all


@dataclass
class Evidence:
    """One source's reading of one claim."""

    source: str = ""
    supports: bool = False
    refutes: bool = False
    hard: bool = False             # can this source be *wrong in a checkable way*?
    weight: float = 0.0            # 0..1 — how much this reading is worth
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "supports": self.supports, "refutes": self.refutes,
                "hard": self.hard, "weight": round(self.weight, 4), "detail": self.detail[:300]}


@dataclass
class Judgement:
    """What the gauntlet concluded, and every reading that produced it."""

    claim: str = ""
    verdict: str = Verdict.ABSTAINED
    confidence: float = 0.0
    evidence: List[Evidence] = field(default_factory=list)
    hard_supports: int = 0
    supports: int = 0
    t: float = field(default_factory=time.time)
    ms: float = 0.0

    @property
    def established(self) -> bool:
        return self.verdict == Verdict.ESTABLISHED

    @property
    def assertable(self) -> bool:
        """May NJP state this as fact? Only an established claim may be stated as one."""
        return self.verdict == Verdict.ESTABLISHED

    def to_dict(self) -> Dict[str, Any]:
        return {"claim": self.claim[:400], "verdict": self.verdict,
                "confidence": round(self.confidence, 4), "supports": self.supports,
                "hard_supports": self.hard_supports, "ms": round(self.ms, 3),
                "evidence": [e.to_dict() for e in self.evidence]}


class Source:
    """One independent way of being right or wrong about a claim."""

    name = "source"
    hard = False

    def check(self, claim: str, *, context: Any = None) -> Optional[Evidence]:  # pragma: no cover
        raise NotImplementedError


class FormalSource(Source):
    """The proof core. Returns a verdict only where the claim is formally expressible."""

    name = "formal"
    hard = True

    def check(self, claim: str, *, context: Any = None) -> Optional[Evidence]:
        try:
            from nyxara.njp.prove import ProofCore
        except Exception:  # noqa: BLE001 — no proof core installed ⇒ this source is simply absent
            return None
        try:
            core = ProofCore()
            cert = core.prove(claim)
            status = str(getattr(cert, "status", "") or "").lower()
            if status in ("proved", "proven", "valid"):
                return Evidence(source=self.name, supports=True, hard=True, weight=1.0,
                                detail=str(getattr(cert, "certificate", "") or "proved")[:300])
            if status in ("refuted", "invalid", "unsat_negation_failed"):
                return Evidence(source=self.name, refutes=True, hard=True, weight=1.0,
                                detail=str(getattr(cert, "counter_model", "") or "refuted")[:300])
            # contingent / unknown / inexpressible — an honest silence, not a vote
            return None
        except Exception:  # noqa: BLE001
            return None


class PredictiveSource(Source):
    """Does the claim predict data that was **held out** from whatever produced it?

    This is the only source that can be genuinely surprised, which is why it counts as hard. The
    caller supplies ``predictor`` (claim, sample) -> bool; the gauntlet supplies the withheld
    samples. With no held-out samples this source abstains rather than passing the claim.
    """

    name = "predictive"
    hard = True

    def __init__(self, predictor: Optional[Callable[[str, Any], bool]] = None,
                 holdout: Optional[Sequence[Any]] = None, *,
                 min_samples: int = 3, pass_ratio: float = 0.8) -> None:
        self.predictor = predictor
        self.holdout = list(holdout or [])
        self.min_samples = max(1, int(min_samples))
        self.pass_ratio = float(pass_ratio)

    def check(self, claim: str, *, context: Any = None) -> Optional[Evidence]:
        try:
            samples = list(self.holdout)
            if context is not None and not samples:
                samples = list(getattr(context, "holdout", None) or [])
            if self.predictor is None or len(samples) < self.min_samples:
                return None                        # cannot speak — correctly says nothing
            ok = 0
            for s in samples:
                try:
                    if self.predictor(claim, s):
                        ok += 1
                except Exception:  # noqa: BLE001 — a sample that errors is not a pass
                    continue
            ratio = ok / float(len(samples))
            if ratio >= self.pass_ratio:
                return Evidence(source=self.name, supports=True, hard=True, weight=ratio,
                                detail=f"predicted {ok}/{len(samples)} held-out samples")
            if ratio <= (1.0 - self.pass_ratio):
                return Evidence(source=self.name, refutes=True, hard=True, weight=1.0 - ratio,
                                detail=f"failed held-out prediction ({ok}/{len(samples)})")
            return None
        except Exception:  # noqa: BLE001
            return None


class ConsistencySource(Source):
    """Does this contradict something already established? Contradiction refutes; silence is weak."""

    name = "consistency"
    hard = False

    def __init__(self, established: Optional[Sequence[str]] = None) -> None:
        self.established: List[str] = list(established or [])

    @staticmethod
    def _negates(a: str, b: str) -> bool:
        """A deliberately narrow surface check for direct negation.

        Narrow on purpose: a loose contradiction detector that fires on unrelated sentences would
        refute true claims, which is a far worse failure than missing a subtle contradiction.
        """
        try:
            a_l, b_l = a.lower().strip(), b.lower().strip()
            if not a_l or not b_l:
                return False
            for neg in (" not ", " never ", " no "):
                if a_l.replace(neg, " ") == b_l.replace(neg, " ") and a_l != b_l:
                    return True
            return False
        except Exception:  # noqa: BLE001
            return False

    def check(self, claim: str, *, context: Any = None) -> Optional[Evidence]:
        try:
            pool = list(self.established)
            if context is not None:
                pool += list(getattr(context, "established", None) or [])
            for known in pool:
                if self._negates(claim, str(known)):
                    return Evidence(source=self.name, refutes=True, weight=0.9,
                                    detail=f"contradicts established: {str(known)[:120]}")
            if not pool:
                return None
            return Evidence(source=self.name, supports=True, weight=0.2,
                            detail="consistent with what is already established")
        except Exception:  # noqa: BLE001
            return None


class LedgerSource(Source):
    """Has NJP been wrong about something shaped like this before?

    This source can only count **against** a claim. A past error resembling the present claim is
    real evidence for caution; the *absence* of a past error is not evidence of correctness, and
    treating it as such would let a fresh topic pass on ignorance alone.
    """

    name = "ledger"
    hard = False

    def __init__(self, ledger: Any = None) -> None:
        self.ledger = ledger

    def check(self, claim: str, *, context: Any = None) -> Optional[Evidence]:
        try:
            led = self.ledger if self.ledger is not None else getattr(context, "ledger", None)
            errors = getattr(led, "errors", None)
            if errors is None:
                return None
            hits = errors.similar(claim)
            if not hits:
                return None
            return Evidence(source=self.name, refutes=False, supports=False, weight=0.5,
                            detail=(f"{len(hits)} similar past error(s); closest: "
                                    f"{hits[0].claim[:120]}"))
        except Exception:  # noqa: BLE001
            return None


class ObservationSource(Source):
    """Independent recorded observations bearing on the claim."""

    name = "observation"
    hard = False

    def __init__(self, observations: Optional[Sequence[str]] = None, *,
                 min_overlap: float = 0.34) -> None:
        self.observations: List[str] = list(observations or [])
        self.min_overlap = float(min_overlap)

    @staticmethod
    def _tok(text: str) -> set:
        return {w for w in "".join(c if c.isalnum() else " "
                                   for c in str(text or "").lower()).split() if len(w) > 2}

    def check(self, claim: str, *, context: Any = None) -> Optional[Evidence]:
        try:
            pool = list(self.observations)
            if context is not None:
                pool += list(getattr(context, "observations", None) or [])
            want = self._tok(claim)
            if not want or not pool:
                return None
            best, hits = 0.0, 0
            for obs in pool:
                have = self._tok(obs)
                if not have:
                    continue
                score = len(want & have) / float(len(want | have))
                if score >= self.min_overlap:
                    hits += 1
                    best = max(best, score)
            if not hits:
                return None
            return Evidence(source=self.name, supports=True, weight=min(1.0, best),
                            detail=f"{hits} corroborating observation(s), best overlap {best:.2f}")
        except Exception:  # noqa: BLE001
            return None


class TruthGauntlet:
    """Multi-source verification. Fail-closed: one refutation ends it."""

    def __init__(self, *, sources: Optional[Sequence[Source]] = None,
                 min_sources: int = 2, require_hard: bool = True,
                 ledger: Any = None) -> None:
        self.sources: List[Source] = list(sources) if sources is not None else [
            FormalSource(), PredictiveSource(), ConsistencySource(),
            LedgerSource(ledger=ledger), ObservationSource(),
        ]
        self.min_sources = max(1, int(min_sources))
        self.require_hard = bool(require_hard)
        self.ledger = ledger
        self.judged = 0
        self.established = 0
        self.refused = 0

    def judge(self, claim: str, *, context: Any = None) -> Judgement:
        """Put one claim through every source and return what may honestly be said about it."""
        out = Judgement(claim=str(claim or ""))
        t0 = time.perf_counter()
        try:
            if not out.claim.strip():
                out.ms = (time.perf_counter() - t0) * 1000.0
                return out
            self.judged += 1
            ctx = context if context is not None else self
            for src in self.sources:
                try:
                    ev = src.check(out.claim, context=ctx)
                except Exception:  # noqa: BLE001 — a broken source is a silent source
                    ev = None
                if ev is not None:
                    out.evidence.append(ev)

            if any(e.refutes for e in out.evidence):
                out.verdict = Verdict.REFUTED
                out.confidence = max((e.weight for e in out.evidence if e.refutes), default=1.0)
                self.refused += 1
                self._remember_error(out)
                out.ms = (time.perf_counter() - t0) * 1000.0
                return out

            supports = [e for e in out.evidence if e.supports]
            out.supports = len(supports)
            out.hard_supports = sum(1 for e in supports if e.hard)

            enough = out.supports >= self.min_sources
            hard_ok = (out.hard_supports >= 1) if self.require_hard else True
            if enough and hard_ok:
                out.verdict = Verdict.ESTABLISHED
                self.established += 1
            elif supports:
                out.verdict = Verdict.SUPPORTED
            elif out.evidence:
                out.verdict = Verdict.CONJECTURE
            else:
                out.verdict = Verdict.ABSTAINED

            if supports:
                out.confidence = min(1.0, sum(e.weight for e in supports) / float(len(supports)))
            # A claim that resembles a past error is penalised even when it otherwise stands.
            penalty = next((e for e in out.evidence if e.source == "ledger"), None)
            if penalty is not None:
                out.confidence = max(0.0, out.confidence - 0.5 * penalty.weight)
                if out.verdict == Verdict.ESTABLISHED and out.hard_supports < 2:
                    # Established, but it looks like something she has been wrong about before.
                    # Demote rather than assert: the whole point of keeping the error is to act on it.
                    out.verdict = Verdict.SUPPORTED
                    self.established -= 1
            out.ms = (time.perf_counter() - t0) * 1000.0
            return out
        except Exception:  # noqa: BLE001 — a failed gauntlet asserts nothing, which is correct
            out.verdict = Verdict.ABSTAINED
            out.ms = (time.perf_counter() - t0) * 1000.0
            return out

    def _remember_error(self, judgement: Judgement) -> None:
        """A refuted claim goes on the record, so the next one shaped like it has to answer for it."""
        try:
            errors = getattr(self.ledger, "errors", None)
            if errors is None:
                return
            why = next((e.detail for e in judgement.evidence if e.refutes), "refuted")
            errors.record(judgement.claim, verdict=Verdict.REFUTED, truth=why)
        except Exception:  # noqa: BLE001
            pass

    def stats(self) -> Dict[str, Any]:
        return {"sources": [s.name for s in self.sources], "judged": self.judged,
                "established": self.established, "refused": self.refused,
                "min_sources": self.min_sources, "require_hard": self.require_hard,
                "establish_rate": (round(self.established / self.judged, 4)
                                   if self.judged else None)}
