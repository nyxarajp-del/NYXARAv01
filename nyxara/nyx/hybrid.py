"""NYXARA · nyx/hybrid.py — symbolic ∧ sub-symbolic (⚖, NYX V.01 pillar 2).

Pattern-matching guesses well and cannot prove anything; symbolic reasoning proves things and
cannot guess. A mind needs both, *and a rule for who wins when they disagree*. This module is
that rule, and the machinery that makes it more than a preference.

Two jobs:

**1. Adjudicate.** When one candidate is *derived and checked* and another is merely confident,
the derived one wins — the repo's own law (``mind/faculties.py``: a verifiable engine that fits
beats a probabilistic one). Not a weighting that a loud enough guess can out-shout: a rule.

**2. Close the feedback loop without a human.** This is the part that matters for learning.
:meth:`SymbolicSubsymbolicFusion.verify` takes what she is about to say and tries to *check* it —
against a derivation she can carry out, and against the evidence she actually retrieved. The
verdict (``SUPPORTED`` / ``CONTRADICTED`` / ``UNVERIFIABLE``) becomes the outcome signal that
:mod:`nyxara.nyx.metacog` learns from. So a faculty that keeps asserting things her own symbolic
engines contradict gets demoted **by measurement, with nobody telling her**.

Honest framing:

* ``UNVERIFIABLE`` is the common case and is not a failure — most of what anyone says is not
  mechanically checkable. It is reported as its own verdict and deliberately does **not** move
  reliability, because "I could not check" is not evidence of being wrong.
* A ``SUPPORTED`` verdict means *her* engines agreed, on the evidence *she* holds. That is a
  real check with a real scope, not a claim of truth.

Composes :class:`~nyxara.mind.first_principles.FirstPrinciplesEngine` and, when available,
``mind/rag.check_grounding``. Pure standard library otherwise. Fail-soft throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["Verdict", "Verification", "SymbolicSubsymbolicFusion"]


class Verdict(str, Enum):
    """What checking actually established."""

    SUPPORTED = "supported"           # a derivation or her own evidence backs it
    CONTRADICTED = "contradicted"     # her symbolic engines say otherwise
    UNVERIFIABLE = "unverifiable"     # not mechanically checkable — the honest common case


@dataclass
class Verification:
    """The result of trying to check a claim, and what that is worth as a learning signal."""

    verdict: Verdict = Verdict.UNVERIFIABLE
    confidence: float = 0.0
    reason: str = ""
    derived: bool = False
    grounded_in: List[str] = field(default_factory=list)

    @property
    def outcome(self) -> Optional[float]:
        """The signal for :mod:`nyxara.nyx.metacog`, or ``None`` when nothing was learned.

        ``UNVERIFIABLE`` returns ``None`` on purpose: failing to check something is not
        evidence that it was wrong, and feeding it in as 0.0 would slowly condemn every
        faculty that speaks about un-checkable things.
        """
        if self.verdict is Verdict.SUPPORTED:
            return 1.0
        if self.verdict is Verdict.CONTRADICTED:
            return 0.0
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict.value, "confidence": round(self.confidence, 4),
                "reason": self.reason, "derived": self.derived,
                "grounded_in": self.grounded_in, "outcome": self.outcome}


class SymbolicSubsymbolicFusion:
    """Adjudicates between derived and guessed answers, and checks what she is about to say."""

    # Trace kinds that are records of her own speech rather than things she knows.
    _NOT_EVIDENCE = frozenset({"episode", "conjecture"})

    def __init__(self, *, engine: Any = None, grounding: bool = True,
                 min_grounding_overlap: float = 0.5, semantics: Any = None,
                 prover: Any = None) -> None:
        self._engine = engine
        self._tried = engine is not None
        self.grounding = bool(grounding)
        self.min_grounding_overlap = float(min_grounding_overlap)
        # NYX V.02: with a semantic space attached, evidence that says "automobile" counts as
        # covering a claim about a "car" — provided the space can *justify* the link. Without
        # one this is exact word overlap, exactly as V.01 did it.
        self.semantics = semantics
        # NYX V.02: and with a prover attached, a formally expressible claim can be *decided*
        # rather than merely corroborated. Most claims are not formally expressible; those go
        # down the unchanged path, which is why this is an addition and not a replacement.
        self.prover = prover

    def _get_engine(self) -> Any:
        if not self._tried:
            self._tried = True
            try:
                from nyxara.mind.first_principles import FirstPrinciplesEngine
                self._engine = FirstPrinciplesEngine()
            except Exception:  # noqa: BLE001 — no engine means everything is UNVERIFIABLE
                self._engine = None
        return self._engine

    def available(self) -> bool:
        """Honest capability report — False when no derivation engine could be built."""
        return self._get_engine() is not None

    # ---- 1. adjudication -------------------------------------------------- #
    def adjudicate(self, proposals: Sequence[Any]) -> Optional[Any]:
        """Pick the winner under the law: **verifiable beats probabilistic**.

        Among verified candidates the most confident wins; with none, the most confident overall.
        Returns ``None`` only when there is nothing to choose between.
        """
        try:
            live = [p for p in proposals if p is not None]
            if not live:
                return None
            verified = [p for p in live if bool(getattr(p, "verifiable", False))]
            pool = verified or live
            return max(pool, key=lambda p: float(getattr(p, "confidence", 0.0)))
        except Exception:  # noqa: BLE001
            return None

    # ---- 2. checking ------------------------------------------------------ #
    def verify(self, claim: str, *, stimulus: str = "", evidence: Sequence[Any] = ()
               ) -> Verification:
        """Try to check ``claim``. Derivation first, then grounding in her retrieved evidence."""
        out = Verification()
        try:
            claim = str(claim or "").strip()
            if not claim:
                return out
            derived = self._by_derivation(stimulus or claim)
            if derived is not None:
                return derived
            if self.grounding:
                grounded = self._by_grounding(claim, evidence, stimulus=stimulus)
                if grounded is not None:
                    return grounded
            out.reason = "no derivation applies and no retrieved evidence covers it"
            return out
        except Exception:  # noqa: BLE001 — a failed check is UNVERIFIABLE, never a crash
            return out

    def _by_derivation(self, text: str) -> Optional[Verification]:
        """Can she actually derive an answer here? If so, that settles it."""
        engine = self._get_engine()
        if engine is None:
            return None
        derivation = engine.derive(text)
        if derivation is None:
            return None
        verified = bool(getattr(derivation, "verified", False))
        domain = str(getattr(derivation, "domain", "?"))
        steps = len(getattr(derivation, "steps", []) or [])
        return Verification(
            verdict=Verdict.SUPPORTED if verified else Verdict.CONTRADICTED,
            confidence=float(getattr(derivation, "confidence", 0.5)),
            reason=(f"{domain} derivation over {steps} steps "
                    f"{'checked out' if verified else 'did not hold'}"),
            derived=True)

    def _by_grounding(self, claim: str, evidence: Sequence[Any], *, stimulus: str = ""
                      ) -> Optional[Verification]:
        """Is the claim actually carried by the evidence she retrieved, or is it free-floating?"""
        texts = []
        for item in evidence or ():
            text = getattr(item, "text", item)
            if not isinstance(text, str) or not text.strip():
                continue
            # Ground only in what she *knows*, never in what she previously *said*. An episode
            # is a record of her own utterance; checking a claim against it is circular, and
            # with any templated phrasing it grades a faculty against copies of its own output.
            if getattr(item, "kind", None) in self._NOT_EVIDENCE:
                continue
            if self._is_self_echo(claim, text):
                continue
            texts.append(text)
        if not texts:
            return None

        certificate = self._proved(claim)
        if certificate is not None:
            return Verification(
                verdict=Verdict.SUPPORTED if certificate.proved else Verdict.CONTRADICTED,
                confidence=1.0,
                reason=(f"machine-checked: {certificate.verdict} — {certificate.note}"),
                grounded_in=[certificate.formula] if certificate.formula else [])

        overlap, source = self._overlap(claim, texts, asked=stimulus)
        if overlap is None:
            # The claim says nothing the question did not already say. There is no assertion
            # here to ground — crediting it for echoing the prompt would be a free pass.
            return Verification(
                verdict=Verdict.UNVERIFIABLE, confidence=0.0,
                reason="the claim adds nothing beyond the question, so there is nothing to check")
        if overlap >= self.min_grounding_overlap:
            return Verification(
                verdict=Verdict.SUPPORTED, confidence=overlap,
                reason=f"{overlap:.0%} of the claim's content is carried by retrieved evidence",
                grounded_in=[source] if source else [])
        # Weak overlap is *not* a contradiction — it only means she cannot vouch for it here.
        return Verification(
            verdict=Verdict.UNVERIFIABLE, confidence=overlap,
            reason=f"only {overlap:.0%} of the claim is covered by what she retrieved")

    @staticmethod
    def _is_self_echo(claim: str, text: str) -> bool:
        """Is this 'evidence' just the claim itself, come back around?"""
        try:
            if claim.strip() == text.strip():
                return True
            from nyxara.nyx.graph import concepts_in
            a, b = set(concepts_in(claim)), set(concepts_in(text))
            return bool(a) and a == b
        except Exception:  # noqa: BLE001
            return False

    def _proved(self, claim: str) -> Any:
        """A machine-checkable verdict, when the claim admits one at all.

        Returns ``None`` for everything that is not formally expressible — which is most of
        what she says — so the unchanged evidence path still runs. A prover that returns
        ``contingent`` or ``unknown`` also returns ``None`` here: neither is a verdict.
        """
        if self.prover is None:
            return None
        try:
            got = self.prover.prove(claim)
            return got if getattr(got, "decisive", False) else None
        except Exception:  # noqa: BLE001 — a prover failure is never a proof
            return None

    def _covers(self, wanted: str, have: set) -> bool:
        """Is this one word of the claim carried by the evidence — as itself, or as a synonym?

        Only the **relational** rung counts. A distributional neighbour is not a substitute
        ("hot" sits next to "cold"), and letting one stand in for the other here would turn a
        coverage check into a generous one.
        """
        if wanted in have:
            return True
        space = self.semantics
        if space is None:
            return False
        try:
            for other, _weight in space.synonyms(wanted):
                if other in have:
                    return True
        except Exception:  # noqa: BLE001 — a missing space means exact overlap, not a crash
            return False
        return False

    def _overlap(self, claim: str, texts: Sequence[str], *, asked: str = "") -> tuple:
        """Fraction of the claim's *new* content that appears in the best supporting text.

        Words the question already contained are removed first: a reply that only restates the
        prompt would otherwise score near-perfect grounding against any evidence sharing those
        words, and be credited for asserting nothing. Returns ``(None, "")`` when nothing new
        is left — that is "no assertion to check", not "checked and fine".

        Deliberately simple and honest: this is a coverage check, not entailment. It catches a
        claim talking about something she has no evidence for; it cannot catch a subtle
        misstatement built from the right words.
        """
        try:
            from nyxara.nyx.graph import concepts_in
            wanted = set(concepts_in(claim)) - set(concepts_in(asked))
            if not wanted:
                return None, ""
            best, best_source = 0.0, ""
            for text in texts:
                have = set(concepts_in(text))
                covered = sum(1 for w in wanted if self._covers(w, have))
                score = covered / float(len(wanted))
                if score > best:
                    best, best_source = score, text[:80]
            return best, best_source
        except Exception:  # noqa: BLE001
            return 0.0, ""

    # ---- prefer the grounded phrasing -------------------------------------- #
    def check_and_learn(self, brain: Any, thought: Any) -> Optional[Verification]:
        """Check a thought and feed the verdict straight back into meta-cognition.

        This is the loop closing with no human in it: she says something, her own symbolic
        engines check it, and the specialist that said it is credited or debited accordingly.
        """
        try:
            if thought is None or not getattr(thought, "answer", ""):
                return None
            percept = getattr(thought, "percept", None)
            evidence = list(getattr(percept, "context", []) or [])
            got = self.verify(thought.answer, stimulus=thought.stimulus, evidence=evidence)
            outcome = got.outcome
            if outcome is not None and brain is not None:
                brain.resolve(thought, correct=outcome)
            return got
        except Exception:  # noqa: BLE001
            return None
