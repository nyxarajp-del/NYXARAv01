"""NYXARA · growth/oracle.py — she poses her own problems, and proves them (♾️).

L-ORACLE and L-META-GAUNTLET, together, because they are one loop: she generates conjectures,
proves or refutes them, trains on what survived, and the difficulty of what she generates next
rises with what she has mastered.

**The pieces already exist and are good.** :mod:`~nyxara.growth.synthesis` generates arithmetic,
algebra, logic, number-theory and code items and keeps one **only** when
:class:`~nyxara.growth.prover.Prover` returns ``PROVEN`` — its ``RivalVerifier`` is built so that
generation is trusted by no one, and an LLM second opinion may *withhold* acceptance but can
never override a deterministic verdict. :mod:`~nyxara.growth.adversarial_self_play` has ``Elo``
and a ``StrategyBandit``; :mod:`~nyxara.growth.curriculum` has a frontier tier that promotes at
0.80 and demotes at 0.40. :mod:`~nyxara.void.heartbeat` and :mod:`~nyxara.kernel.autonomic` give
her the idle time to run in. What was missing was the loop that connects them.

**Where this differs from AlphaZero, which matters.** AlphaZero worked because Go's verifier is
*perfect, cheap and complete*: every position has a knowable outcome. Mathematics has a verifier
— z3 and sympy — but it is **incomplete**. It decides some statements and times out on most
interesting ones, and :class:`~nyxara.growth.prover.Prover` returns ``UNPROVABLE`` as an honest
abstention rather than a guess. So this loop produces true-but-shallow theorems abundantly and
deep ones rarely.

That is precisely why the value is not in generating conjectures at random. It is in the
**pressure toward hard-but-provable** — the Elo rating and the curriculum tier push difficulty up
only as fast as the prover can keep certifying. What comes out is verified maths and code data at
a scale no human dataset contains, in her own template. What does not come out is "the logic of
the universe, derived from first principles".

**The one rule the meta-loss must never break.** A loss that evolves alongside the model can
co-adapt with it: the number keeps improving while meaning drains out of it. "She never plateaus"
and "the ruler kept shrinking" look identical from inside. So the escalating difficulty here may
change *training*, and may never touch the **frozen held-out battery** that gates promotion.
:mod:`~nyxara.growth.causal_validation` is described in this repo as "the one place she cannot
grade her own homework"; :class:`FrozenRuler` applies the same principle, and
:meth:`MetaLoss.assert_ruler_untouched` makes it checkable.

Pure standard library at the core; the prover's ``z3``/``sympy`` are already dependencies.
Nothing imports back.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "OracleConfig",
    "OracleReport",
    "ProvenItem",
    "FrozenRuler",
    "MetaLoss",
    "SelfPlayOracle",
]


@dataclass
class OracleConfig:
    """How hard she pushes herself, and how long she is willing to spend."""

    batch: int = 32
    max_rounds: int = 8
    domains: Tuple[str, ...] = ("arithmetic", "algebra", "number_theory", "logic", "code")
    # Elo ratings for the proposer and the prover. When the proposer wins too often the
    # conjectures have run past what can be certified, and difficulty is pulled back.
    elo_k: float = 24.0
    target_prove_rate: float = 0.55      # the hard-but-provable band
    band: float = 0.15
    difficulty: int = 1
    max_difficulty: int = 8
    seconds_budget: float = 30.0
    seed: int = 0


@dataclass
class ProvenItem:
    """One conjecture that survived verification, with the certificate that carried it."""

    domain: str
    prompt: str
    answer: str
    certificate: str
    difficulty: int = 1

    def to_doc(self) -> str:
        """Rendered in her own template — train/inference parity, same as everything else."""
        try:
            from nyxara.mind.llm import format_self_training_doc
            return format_self_training_doc(self.prompt, self.answer)
        except Exception:  # noqa: BLE001
            return f"{self.prompt}\n\n{self.answer}\n"

    def to_dict(self) -> Dict[str, Any]:
        return {"domain": self.domain, "prompt": self.prompt, "answer": self.answer,
                "certificate": self.certificate[:300], "difficulty": self.difficulty}


@dataclass
class OracleReport:
    """What she proposed, what survived, and how the difficulty moved."""

    proposed: int = 0
    proven: int = 0
    refuted: int = 0
    unprovable: int = 0
    rounds: int = 0
    difficulty_start: int = 1
    difficulty_end: int = 1
    by_domain: Dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)

    @property
    def prove_rate(self) -> float:
        return self.proven / self.proposed if self.proposed else 0.0

    def summary(self) -> str:
        head = (f"· {self.proven}/{self.proposed} proven ({self.prove_rate:.0%}) over "
                f"{self.rounds} round(s); difficulty "
                f"{self.difficulty_start} -> {self.difficulty_end}")
        detail = (f"\n  refuted {self.refuted}, UNPROVABLE (abstained) {self.unprovable}"
                  f" — only proven items enter the corpus")
        domains = (("\n  " + ", ".join(f"{k} {v}" for k, v in sorted(self.by_domain.items())))
                   if self.by_domain else "")
        notes = ("\n  · " + "\n  · ".join(self.notes)) if self.notes else ""
        return head + detail + domains + notes

    def to_dict(self) -> Dict[str, Any]:
        return {"proposed": self.proposed, "proven": self.proven, "refuted": self.refuted,
                "unprovable": self.unprovable, "rounds": self.rounds,
                "prove_rate": round(self.prove_rate, 4),
                "difficulty_start": self.difficulty_start,
                "difficulty_end": self.difficulty_end,
                "by_domain": dict(self.by_domain), "seconds": round(self.seconds, 2),
                "notes": list(self.notes)}


# --------------------------------------------------------------------------- #
# L-META-GAUNTLET — the evolving loss, and the ruler it may not touch
# --------------------------------------------------------------------------- #
class FrozenRuler:
    """A held-out battery the meta-loss can never modify.

    Its contents are fingerprinted on construction, and :meth:`verify` re-checks that
    fingerprint. If a self-modifying training loop can also move the measuring stick, then
    "capability rose" and "the ruler shrank" become indistinguishable — which is the failure
    mode a self-evolving objective is most prone to and least able to notice.
    """

    def __init__(self, items: Sequence[Any]) -> None:
        self._items = tuple(items)
        self._fingerprint = self._hash(self._items)

    @staticmethod
    def _hash(items: Sequence[Any]) -> str:
        h = hashlib.sha256()
        for item in items:
            h.update(repr(item).encode("utf-8", "replace"))
        return h.hexdigest()

    @property
    def items(self) -> Tuple[Any, ...]:
        return self._items

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def verify(self) -> bool:
        """True iff the battery is byte-identical to the one this was constructed with."""
        return self._hash(self._items) == self._fingerprint


class MetaLoss:
    """Loss weights and problem difficulty that rise as she masters what she is given.

    Reuses the promotion/demotion thresholds :mod:`~nyxara.growth.curriculum` already uses (0.80
    up, 0.40 down), so the escalation policy is the one the rest of the system already follows.
    """

    PROMOTE = 0.80
    DEMOTE = 0.40

    def __init__(self, *, ruler: Optional[FrozenRuler] = None,
                 base_weights: Optional[Dict[str, float]] = None,
                 max_difficulty: int = 8) -> None:
        self.ruler = ruler
        self.max_difficulty = max_difficulty
        self.difficulty = 1
        self.weights: Dict[str, float] = dict(base_weights or {
            "token": 1.0, "counter_example": 0.1, "moe_aux": 0.01, "latent": 0.1,
        })
        self.history: List[Dict[str, Any]] = []

    def observe(self, mastery: Dict[str, float]) -> Dict[str, Any]:
        """Update difficulty and weights from measured per-domain mastery in [0, 1].

        Above ``PROMOTE`` the frontier rises and counter-examples are weighted more heavily;
        below ``DEMOTE`` it falls back. Between the two nothing moves — the point is to track
        real mastery, not to chase noise.
        """
        scored = {k: v for k, v in mastery.items() if k != "mean"}
        if not scored:
            return self.state()
        mean = sum(scored.values()) / len(scored)

        moved = 0
        if mean >= self.PROMOTE and self.difficulty < self.max_difficulty:
            self.difficulty += 1
            moved = 1
        elif mean <= self.DEMOTE and self.difficulty > 1:
            self.difficulty -= 1
            moved = -1

        # Mastered material teaches little; counter-examples at the frontier teach a lot.
        self.weights["counter_example"] = round(min(1.0, 0.1 + 0.15 * (self.difficulty - 1)), 4)
        # Re-weight toward the weakest domain, so training pressure follows the measurement.
        weakest = min(scored, key=lambda k: scored[k])
        self.weights[f"domain:{weakest}"] = round(1.0 + (1.0 - scored[weakest]), 4)

        record = {"mean_mastery": round(mean, 4), "difficulty": self.difficulty,
                  "moved": moved, "weakest": weakest, "at": time.time()}
        self.history.append(record)
        return record

    def assert_ruler_untouched(self) -> None:
        """Raise if the frozen battery has changed. Call before trusting any score.

        This is the check that keeps a rising number meaningful. Without it a self-evolving
        objective can improve indefinitely by making the test easier, and every metric
        downstream inherits the lie.
        """
        if self.ruler is not None and not self.ruler.verify():
            raise RuntimeError(
                "the frozen evaluation battery was modified — the meta-loss may raise "
                "difficulty but may NEVER touch the ruler that gates promotion")

    def state(self) -> Dict[str, Any]:
        return {"difficulty": self.difficulty, "weights": dict(self.weights),
                "ruler_fingerprint": self.ruler.fingerprint[:16] if self.ruler else None,
                "updates": len(self.history)}


# --------------------------------------------------------------------------- #
# L-ORACLE — the closed loop
# --------------------------------------------------------------------------- #
class SelfPlayOracle:
    """Generate → prove → keep, with difficulty tracked by Elo.

    The proposer and the prover are rated against each other. Too many conjectures surviving
    means she is asking herself easy questions; too few means she has run past what can be
    certified and is generating noise. Difficulty is steered to hold the proof rate inside a
    band, which is where the useful items live.
    """

    def __init__(self, *, config: Optional[OracleConfig] = None,
                 meta: Optional[MetaLoss] = None) -> None:
        self.cfg = config or OracleConfig()
        self.meta = meta
        self.rng = random.Random(self.cfg.seed)
        self.proposer_elo = 1000.0
        self.prover_elo = 1000.0
        self.difficulty = self.cfg.difficulty

    def _update_elo(self, proposer_won: bool) -> None:
        """Standard Elo, mirroring adversarial_self_play.Elo (K=24)."""
        expected = 1.0 / (1.0 + 10 ** ((self.prover_elo - self.proposer_elo) / 400.0))
        score = 1.0 if proposer_won else 0.0
        delta = self.cfg.elo_k * (score - expected)
        self.proposer_elo += delta
        self.prover_elo -= delta

    def _steer_difficulty(self, prove_rate: float) -> None:
        """Hold the proof rate in the hard-but-provable band.

        A verifier that certifies everything is being asked trivia; one that certifies nothing
        is being asked noise. Neither produces training data worth having.
        """
        low = self.cfg.target_prove_rate - self.cfg.band
        high = self.cfg.target_prove_rate + self.cfg.band
        if prove_rate > high and self.difficulty < self.cfg.max_difficulty:
            self.difficulty += 1
        elif prove_rate < low and self.difficulty > 1:
            self.difficulty -= 1

    def run(self, *, report: Optional[OracleReport] = None) -> Tuple[List[ProvenItem],
                                                                    OracleReport]:
        """Run the loop within its time budget. Only ``PROVEN`` items are returned."""
        report = report if report is not None else OracleReport()
        report.difficulty_start = self.difficulty
        start = time.monotonic()
        kept: List[ProvenItem] = []

        try:
            from nyxara.growth.synthesis import RivalVerifier, SyntheticGenerators
        except Exception as exc:  # noqa: BLE001
            report.note(f"synthesis unavailable ({exc}) — the oracle cannot run")
            return kept, report

        generators = SyntheticGenerators(seed=self.cfg.seed + self.difficulty)
        verifier = RivalVerifier()

        for round_index in range(self.cfg.max_rounds):
            if time.monotonic() - start > self.cfg.seconds_budget:
                report.note(f"stopped after {round_index} round(s) — time budget reached")
                break
            report.rounds += 1
            proven_this_round = 0

            for item in generators.batch(self.cfg.batch, self.cfg.domains):
                report.proposed += 1
                try:
                    ok, why = verifier.verify(item)
                except Exception as exc:  # noqa: BLE001 — an erroring verifier never accepts
                    ok, why = False, f"verifier error: {exc}"

                if ok:
                    report.proven += 1
                    proven_this_round += 1
                    report.by_domain[item.domain] = report.by_domain.get(item.domain, 0) + 1
                    kept.append(ProvenItem(domain=item.domain, prompt=item.prompt,
                                           answer=item.answer, certificate=why,
                                           difficulty=self.difficulty))
                elif "refut" in str(why).lower() or "mismatch" in str(why).lower():
                    report.refuted += 1
                else:
                    # Honest abstention: the prover could not decide. Discarded, never accepted
                    # as "probably true" — an unproven item in the corpus teaches her to assert
                    # things nothing checked.
                    report.unprovable += 1

            rate = proven_this_round / max(1, self.cfg.batch)
            self._update_elo(proposer_won=rate < self.cfg.target_prove_rate)
            self._steer_difficulty(rate)

        report.difficulty_end = self.difficulty
        report.seconds = time.monotonic() - start
        if self.meta is not None:
            self.meta.assert_ruler_untouched()
        if report.unprovable and not report.proven:
            report.note("nothing could be certified at this difficulty — the conjectures have "
                        "run past what the prover can decide")
        return kept, report

    def docs(self, items: Sequence[ProvenItem]) -> List[str]:
        """Proven items as training documents in her own template."""
        return [item.to_doc() for item in items]

    def save(self, items: Sequence[ProvenItem], path: Any) -> int:
        """Append proven items to a JSONL store. Returns how many were written."""
        store = Path(path)
        store.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with store.open("a", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
                written += 1
        return written
