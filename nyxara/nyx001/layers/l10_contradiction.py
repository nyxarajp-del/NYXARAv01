"""NYXARA · nyx001/layers/l10_contradiction.py — Layer 10, self-attack (⚔️, belief rigidity).

The charter: "Actively searches for evidence that destroys its own hypotheses to reduce belief
rigidity."

The key word is *actively*. Every other layer here has an incentive to be right: the world model
lowers its error, the reasoning engine accumulates confirming evidence, curiosity goes where
progress is. Nothing in that loop is looking for reasons the whole thing is wrong. This layer's
job is to be the part of the system that wants to be wrong, and it does three things:

* **Falsification search.** For a claim "A precedes B", it reads Layer 4's succession counts and
  looks for what *else* followed A. The search is asymmetric on purpose: a symmetric search would
  reproduce the confirmation the rest of the system already provides.
* **Rigidity measurement.** :meth:`rigidity` is the fraction of held beliefs that are both
  high-confidence and rarely tested. A system whose confident beliefs are its least examined ones
  is rigid, and that is a number rather than an impression.
* **Forced re-examination.** :meth:`shake` takes the most rigid beliefs and marks them for
  retesting, lowering confidence toward the prior in proportion to how untested they are. This is
  deliberately uncomfortable: it makes the system *less* sure of things it has not earned being
  sure of.

Honest: falsification here is bounded by what the system has actually observed, and by one claim
shape — succession. It cannot find evidence never encountered, so it reduces rigidity about *seen*
things and does nothing about unknown unknowns. It also cannot tell a genuine counterexample from
a mislabelled one; a contradiction is recorded as evidence against, and the reasoning engine's
Beta posterior is what keeps one bad observation from destroying a well-supported claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

__all__ = ["Attack", "ContradictionEngine"]


@dataclass
class Attack:
    """One attempt to destroy a belief, and what it found."""

    claim: str = ""
    found: bool = False
    evidence: List[str] = field(default_factory=list)
    searched: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"claim": self.claim[:160], "found": self.found, "searched": self.searched,
                "evidence": [e[:120] for e in self.evidence[:5]]}


class ContradictionEngine:
    """Layer 10. Hunts for evidence against the system's own beliefs."""

    def __init__(self, substrate: Any, *, rigid_confidence: float = 0.8, rigid_tests: int = 3,
                 shake_rate: float = 0.25, search_k: int = 32,
                 disconfirm_ratio: float = 1.0) -> None:
        self.substrate = substrate
        self.rigid_confidence = float(rigid_confidence)
        self.rigid_tests = int(rigid_tests)
        self.shake_rate = float(shake_rate)
        self.search_k = int(search_k)
        self.disconfirm_ratio = float(disconfirm_ratio)
        self.attacks = 0
        self.destroyed = 0
        self.shaken = 0

    # ---- falsification ---- #
    def attack(self, hypothesis: Any, memory: Any, reasoning: Any = None,
               semantic: Any = None) -> Attack:
        """Search for evidence that disconfirms ``hypothesis``. Asymmetric by design.

        A succession claim "A precedes B" is disconfirmed by observing A followed by something
        that is *not* B. The evidence for that lives in Layer 4's succession counts, which record
        every ``X → Y`` transition between concepts — so the falsifier reads those directly.

        It used to search episode text for the concept names instead. Episodes store the raw
        observation ("alpha"), concepts are named ``c0``, and the two never matched: 732 attacks
        found exactly zero contradictions, and the engine looked like it was working. Searching
        the wrong store is worse than not searching, because the empty result reads as "nothing
        to contradict" rather than as "I cannot see".
        """
        claim = str(getattr(hypothesis, "claim", "") or "")
        out = Attack(claim=claim)
        if not claim:
            return out
        try:
            self.attacks += 1
            parts = claim.split(" precedes ")
            if len(parts) != 2:
                return out           # the only claim shape this layer can currently falsify
            antecedent, consequent = parts[0].strip(), parts[1].strip()
            succession = dict(getattr(semantic, "_succession", {}) or {})
            if not succession:
                return out
            confirming = 0.0
            disconfirming: List[Tuple[str, float]] = []
            for key, weight in succession.items():
                if "\x00" not in key:
                    continue
                src, dst = key.split("\x00", 1)
                if src != antecedent:
                    continue
                out.searched += 1
                if dst == consequent:
                    confirming += float(weight)
                else:
                    disconfirming.append((dst, float(weight)))
            total_dis = sum(w for _, w in disconfirming)
            # A single alternative is noise; an alternative that rivals the claimed consequent is
            # evidence against. The ratio, not the raw count, is what disconfirms.
            if total_dis > confirming * self.disconfirm_ratio:
                out.found = True
                disconfirming.sort(key=lambda t: t[1], reverse=True)
                for dst, w in disconfirming[:5]:
                    out.evidence.append(
                        f"{antecedent} → {dst} observed with weight {w:.2f} "
                        f"(vs {confirming:.2f} for {consequent})")
            if out.found and reasoning is not None:
                for e in out.evidence:
                    reasoning.contradict(claim, e)
                self.destroyed += 1
            return out
        except Exception:  # noqa: BLE001 — a failed attack found nothing, it did not refute
            return out

    # ---- rigidity ---- #
    def rigidity(self, hypotheses: List[Any]) -> Optional[float]:
        """Fraction of beliefs that are confident *and* barely tested. ``None`` when there are none."""
        items = [h for h in (hypotheses or []) if h is not None]
        if not items:
            return None
        rigid = sum(1 for h in items
                    if float(getattr(h, "confidence", 0.0)) >= self.rigid_confidence
                    and int(getattr(h, "tested", 0)) < self.rigid_tests)
        return float(rigid / len(items))

    def most_rigid(self, hypotheses: List[Any], *, k: int = 5) -> List[Any]:
        """Confident and untested, worst first — where the system is fooling itself."""
        items = [h for h in (hypotheses or []) if h is not None]
        items.sort(key=lambda h: (float(getattr(h, "confidence", 0.0))
                                  / (1.0 + int(getattr(h, "tested", 0)))), reverse=True)
        return items[: max(1, int(k))]

    def shake(self, hypotheses: List[Any], *, k: int = 5) -> List[str]:
        """Pull confidence back toward the prior on beliefs that have not earned it.

        Deliberately uncomfortable. A belief at 0.95 confidence after one test is not a belief at
        0.95 — it is a prior wearing a number, and this returns it toward 0.5 in proportion to how
        untested it is. Returns the claims that were shaken.
        """
        out: List[str] = []
        for h in self.most_rigid(hypotheses, k=k):
            try:
                tested = int(getattr(h, "tested", 0))
                conf = float(getattr(h, "confidence", 0.5))
                if conf < self.rigid_confidence or tested >= self.rigid_tests:
                    continue
                pull = self.shake_rate / (1.0 + tested)
                h.confidence = float(conf + (0.5 - conf) * pull)
                out.append(str(getattr(h, "claim", ""))[:120])
                self.shaken += 1
            except Exception:  # noqa: BLE001
                continue
        return out

    def sweep(self, reasoning: Any, memory: Any, *, k: int = 5,
              semantic: Any = None) -> Dict[str, Any]:
        """One full pass: attack the top hypotheses, then shake whatever stayed rigid."""
        try:
            hyps = reasoning.best(k=k) if reasoning is not None else []
            attacks = [self.attack(h, memory, reasoning, semantic) for h in hyps]
            shaken = self.shake(hyps, k=k)
            return {"attacked": len(attacks),
                    "destroyed": sum(1 for a in attacks if a.found),
                    "shaken": len(shaken),
                    "rigidity": self.rigidity(hyps)}
        except Exception:  # noqa: BLE001
            return {"attacked": 0, "destroyed": 0, "shaken": 0, "rigidity": None}

    def stats(self) -> Dict[str, Any]:
        return {"attacks": self.attacks, "destroyed": self.destroyed, "shaken": self.shaken,
                "destruction_rate": round(self.destroyed / self.attacks, 4)
                if self.attacks else None}

    def to_dict(self) -> Dict[str, Any]:
        return {"attacks": self.attacks, "destroyed": self.destroyed, "shaken": self.shaken}

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.attacks = int(d.get("attacks", 0))
            self.destroyed = int(d.get("destroyed", 0))
            self.shaken = int(d.get("shaken", 0))
        except Exception:  # noqa: BLE001
            pass
