"""NYXARA · eval/acquisition.py — did anything survive the teacher leaving? (🎓📏, Phase 4).

The plan's Phase 4 ends with one test and calls it the point of the whole phase::

    NJP + teacher   vs   NJP after distillation   vs   NJP alone

    "Teacher OFF ke baad performance kitni retain hui? Yahi real acquisition evidence hai."

Three arms over **identical facts**, so the only thing that differs is whether a teacher
demonstrated anything and whether it is still switched on:

``alone``      she is told every edge and asked the held-out questions. No teacher, ever.
``taught``     the same, and the teacher answers the held-out questions itself. This is the
               ceiling, and it is deliberately *not* a measurement of her: it is what the
               teacher can do through her mouth.
``distilled``  the same facts, the teacher demonstrates on the **training** chains only, the
               demonstrations are verified and distilled, the teacher is **switched off**, and
               then she is asked the held-out questions.

    retention = (distilled − alone) / (taught − alone)

The held-out chains are the load-bearing part. A teacher that made her memorise its answers scores
zero here, because it was never asked these questions — the only way through is to have acquired
something that generalises, which is what :mod:`nyxara.njp.teacher` distils: not the conclusion,
but the fact that the relation chains.

**Every arm is scored against a control.** Half the questions are chains that are *not* there —
``A`` and an entity from a different chain entirely. Answering "yes" to those is a false positive
and costs exactly what a hit earns, so a brain that learned to say yes scores 0.5 and cannot beat
one that learned to reason. Without that, "distil aggressively" and "answer yes" are the same
strategy and the benchmark rewards it.

**Hermetic.** The teacher is a :class:`~nyxara.njp.teacher.RecordedTeacher` — a transcript, not a
model — so this runs on a machine with no weights on disk and gives the same numbers twice. The
demonstrations it replays are the benchmark's own, which is honest as long as what is measured is
*what NJP does with them*, and it is: the teacher never touches the held-out questions in the arm
that counts.

Run it::

    python -m nyxara.eval --acquisition
    python -m nyxara.eval --acquisition --seed 7
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["ArmResult", "AcquisitionReport", "run_acquisition_benchmark"]

Preparer = Callable[[Any], None]

_SYLLABLES = ("ka", "ro", "mi", "ta", "zu", "ne", "lo", "vi", "sha", "dre", "qui", "bre")

#: Chain length, and it is chosen by measurement rather than by arithmetic. Each hop costs a
#: factor of the relation's transitivity, and the walk stops when the product falls under
#: `core._MIN_LINK_CONFIDENCE`. At the default prior three hops still complete — measured, all
#: three arms scored 1.00 and the test said nothing — and four do not. Four is therefore the band
#: where a structural fact about the relation is the difference between an answer and silence,
#: which is the only band in which the arms can separate at all.
_HOPS = 4


def _vocabulary(rng: random.Random, n: int, *, tag: str) -> List[str]:
    """``n`` distinct nonsense terms, stable for a seed and unique to this ``tag``."""
    out: List[str] = []
    seen = set()
    while len(out) < n:
        word = tag + "".join(rng.choice(_SYLLABLES) for _ in range(2)) + str(len(out))
        if word not in seen:
            seen.add(word)
            out.append(word)
    return out


@dataclass
class ArmResult:
    """One arm: what it was asked, what it got right, and what it said yes to wrongly."""

    name: str = ""
    hits: int = 0                # true chains answered yes
    true_total: int = 0
    rejects: int = 0             # false chains correctly not answered yes
    false_total: int = 0
    detail: List[str] = field(default_factory=list)
    ms: float = 0.0

    @property
    def total(self) -> int:
        return self.true_total + self.false_total

    @property
    def score(self) -> Optional[float]:
        """Hits and correct rejections over everything asked. "Always yes" scores 0.5."""
        return ((self.hits + self.rejects) / self.total) if self.total else None

    @property
    def false_positive_rate(self) -> Optional[float]:
        if not self.false_total:
            return None
        return (self.false_total - self.rejects) / self.false_total

    def to_dict(self) -> Dict[str, Any]:
        return {"arm": self.name, "score": self.score,
                "hits": f"{self.hits}/{self.true_total}",
                "rejects": f"{self.rejects}/{self.false_total}",
                "false_positive_rate": self.false_positive_rate,
                "ms": round(self.ms, 1), "detail": self.detail[:12]}


@dataclass
class AcquisitionReport:
    """The three arms and the one number the phase is graded on."""

    seed: int = 0
    arms: List[ArmResult] = field(default_factory=list)
    distilled_relations: Dict[str, int] = field(default_factory=dict)
    transitivity_before: float = 0.0
    transitivity_after: float = 0.0
    lessons_verified: int = 0
    lessons_refuted: int = 0
    lessons_undecided: int = 0

    def arm(self, name: str) -> Optional[ArmResult]:
        return next((a for a in self.arms if a.name == name), None)

    @property
    def retention(self) -> Optional[float]:
        """How much of the teacher's advantage she kept after it was switched off.

        ``None`` when the teacher had no advantage to keep — a ratio over a denominator of about
        zero is noise with a percent sign, and reporting it would make the least informative run
        look like the most decisive one.
        """
        alone, taught, distilled = (self.arm("alone"), self.arm("taught"), self.arm("distilled"))
        if not (alone and taught and distilled):
            return None
        base, ceiling, kept = alone.score, taught.score, distilled.score
        if base is None or ceiling is None or kept is None:
            return None
        headroom = ceiling - base
        if headroom <= 1e-9:
            return None
        return (kept - base) / headroom

    def to_dict(self) -> Dict[str, Any]:
        return {"seed": self.seed, "retention": self.retention,
                "arms": [a.to_dict() for a in self.arms],
                "distilled_relations": dict(self.distilled_relations),
                "transitivity": {"before": round(self.transitivity_before, 4),
                                 "after": round(self.transitivity_after, 4)},
                "lessons": {"verified": self.lessons_verified,
                            "refuted": self.lessons_refuted,
                            "undecided": self.lessons_undecided}}

    def render(self) -> str:
        def _fmt(v: Optional[float]) -> str:
            return "  n/a " if v is None else f"{v:5.2f}"

        lines = [
            "NYXARA — native intelligence acquisition (Phase 4)",
            "=" * 78,
            f"seed {self.seed}   ·   held-out chains only   ·   \"always yes\" scores 0.50",
            "",
            f"{'arm':<12} {'score':>7} {'hits':>9} {'rejects':>9} {'false+':>8}  note",
            "-" * 78,
        ]
        notes = {"alone": "no teacher, ever",
                 "taught": "the teacher answers — the ceiling, not her",
                 "distilled": "demonstrations distilled, teacher OFF"}
        for arm in self.arms:
            lines.append(
                f"{arm.name:<12} {_fmt(arm.score):>7} {arm.hits:>4}/{arm.true_total:<4} "
                f"{arm.rejects:>4}/{arm.false_total:<4} "
                f"{_fmt(arm.false_positive_rate):>8}  {notes.get(arm.name, '')}")
        lines.append("-" * 78)
        retention = self.retention
        lines.append("RETENTION    " + (
            "n/a — the teacher had no advantage to keep" if retention is None
            else f"{retention:5.2f}   of the teacher's advantage survived it being switched off"))
        lines.append("")
        lines.append(f"lessons: {self.lessons_verified} verified, {self.lessons_refuted} refuted, "
                     f"{self.lessons_undecided} undecided")
        lines.append(f"transitivity of the taught relation: "
                     f"{self.transitivity_before:.3f} → {self.transitivity_after:.3f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def _brain(prepare: Optional[Preparer] = None) -> Any:
    from nyxara.njp import NJPBrain
    brain = NJPBrain()
    if prepare is not None:
        prepare(brain)
    return brain


def _chains(rng: random.Random, count: int, *, tag: str, hops: int = _HOPS) -> List[List[str]]:
    """``count`` disjoint chains of ``hops + 1`` entities each."""
    names = _vocabulary(rng, count * (hops + 1), tag=tag)
    return [names[i * (hops + 1):(i + 1) * (hops + 1)] for i in range(count)]


def _teach_edges(brain: Any, chains: Sequence[Sequence[str]], relation: str) -> None:
    for chain in chains:
        for left, right in zip(chain, chain[1:]):
            brain.think(f"{left} {relation}s {right}")


def _ask(brain: Any, question: str) -> str:
    try:
        return str(getattr(brain.think(question), "answer", "") or "")
    except Exception:  # noqa: BLE001
        return ""


def _says_yes(said: str) -> bool:
    return said.strip().lower().startswith("yes")


def _questions(chains: Sequence[Sequence[str]], relation: str) -> List[Tuple[str, bool]]:
    """``(question, is_true)`` — each chain's endpoints, and a pairing that does not exist."""
    out: List[Tuple[str, bool]] = []
    for index, chain in enumerate(chains):
        out.append((f"does {chain[0]} {relation} {chain[-1]}?", True))
        other = chains[(index + 1) % len(chains)]
        if other is not chain:
            out.append((f"does {chain[0]} {relation} {other[-1]}?", False))
    return out


def _score(brain: Any, questions: Sequence[Tuple[str, bool]], arm: ArmResult,
           teacher: Any = None) -> None:
    for question, is_true in questions:
        said = ""
        if teacher is not None:
            lesson = teacher.teach(question)
            if lesson is not None:
                said = str(lesson.answer or "")
        if not said:
            said = _ask(brain, question)
        if is_true:
            arm.true_total += 1
            if _says_yes(said):
                arm.hits += 1
            else:
                arm.detail.append(f"miss: {question} -> {said!r}")
        else:
            arm.false_total += 1
            if _says_yes(said):
                arm.detail.append(f"false positive: {question} -> {said!r}")
            else:
                arm.rejects += 1


def run_acquisition_benchmark(*, seed: int = 20260826, chains: int = 4,
                              hops: int = _HOPS,
                              prepare: Optional[Preparer] = None) -> AcquisitionReport:
    """The three-arm acquisition test. Deterministic for a seed; no network, no weights."""
    from nyxara.njp.teacher import Distiller, Lesson, RecordedTeacher, Step

    report = AcquisitionReport(seed=seed)
    rng = random.Random(f"{seed}:acquisition")
    relation = _vocabulary(rng, 1, tag="rel")[0]
    training = _chains(rng, chains, tag="tr", hops=hops)
    heldout = _chains(rng, chains, tag="ho", hops=hops)
    every = list(training) + list(heldout)
    asked = _questions(heldout, relation)

    # The transcript: the teacher demonstrates the TRAINING chains, with their working. It is
    # never given a held-out question, in any arm that measures her.
    demonstrations: Dict[str, Lesson] = {}
    for chain in training:
        task = f"does {chain[0]} {relation} {chain[-1]}?"
        demonstrations[task] = Lesson(
            task=task, answer="yes",
            steps=tuple(Step(left, relation, right) for left, right in zip(chain, chain[1:])),
            rationale=f"{relation} chains", source="recorded", confidence=0.7)

    # ---- alone ------------------------------------------------------------- #
    arm = ArmResult(name="alone")
    t0 = time.perf_counter()
    brain = _brain(prepare)
    _teach_edges(brain, every, relation)
    _score(brain, asked, arm)
    arm.ms = (time.perf_counter() - t0) * 1000.0
    report.arms.append(arm)

    # ---- taught (the ceiling; the teacher answers) --------------------------- #
    arm = ArmResult(name="taught")
    t0 = time.perf_counter()
    brain = _brain(prepare)
    _teach_edges(brain, every, relation)
    ceiling = RecordedTeacher({
        question: Lesson(task=question, answer="yes" if is_true else "no", source="recorded")
        for question, is_true in asked}, name="ceiling")
    _score(brain, asked, arm, teacher=ceiling)
    arm.ms = (time.perf_counter() - t0) * 1000.0
    report.arms.append(arm)

    # ---- distilled (demonstrations only, then the teacher is switched off) --- #
    arm = ArmResult(name="distilled")
    t0 = time.perf_counter()
    brain = _brain(prepare)
    _teach_edges(brain, every, relation)
    stored = _norm_relation(brain, relation)
    report.transitivity_before = _transitivity(brain, stored)
    teacher = RecordedTeacher(demonstrations, name="recorded")
    distiller = Distiller()
    for task in demonstrations:
        lesson = teacher.teach(task)
        if lesson is not None:
            distiller.distil(lesson, brain)
    report.transitivity_after = _transitivity(brain, stored)
    report.lessons_verified = distiller.verified
    report.lessons_refuted = distiller.refuted
    report.lessons_undecided = distiller.undecided
    report.distilled_relations = dict(distiller.relations)
    # Teacher OFF from here. Nothing is passed to `_score`.
    _score(brain, asked, arm)
    arm.ms = (time.perf_counter() - t0) * 1000.0
    report.arms.append(arm)
    return report


def _norm_relation(brain: Any, relation: str) -> str:
    """The predicate the store actually filed, which is not always the verb as typed.

    ``kizzles`` is filed as ``kizzle``. Reading the posterior under the surface form would report
    a relation nothing has ever moved, and the whole measurement would sit at its prior looking
    like nothing was learned. Delegated to the brain's own resolution so there is one rule for
    this and not a second, looser one living in the benchmark.
    """
    try:
        subject = next(str(s) for s, _p, _o, _c in brain.learner._edges())
        return brain._resolve_relation(subject, relation)
    except Exception:  # noqa: BLE001
        return relation


def _transitivity(brain: Any, relation: str) -> float:
    try:
        return float(brain.learner._transitivity(relation).value)
    except Exception:  # noqa: BLE001
        return 0.0
