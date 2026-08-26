"""NYXARA · njp/teacher.py — the Teacher Council, and what survives the teacher leaving (🎓, Phase 4).

Phase 4 is *Native Intelligence Acquisition*, and its chain is written out in the plan::

    LLMs → tasks → solutions → reasoning structures → verification → distillation → cognitive programs

with the test that decides whether any of it happened::

    NJP + teacher   vs   NJP after distillation   vs   NJP alone

    "Teacher OFF ke baad performance kitni retain hui?"

**What is already here, and what was not.** :mod:`nyxara.njp.cortex` is the strong model as a
*proposer* — hypotheses, relations, causal laws — and :meth:`~nyxara.njp.cortex.Cortex.offer` puts
every proposal through the gates NJP already had. That is §18's ``A proposes, B reasons,
C verifies``, and it works. What it produces is **facts**: a surviving proposal becomes a
``DERIVED`` triple or a stated law. A fact is not a cognitive program. Switch the teacher off and
she keeps what it told her and none of what it *knew how to do*.

§17 says the distinction outright — *"Weights ko symbolic brain mein copy nahi karna. Behavior ko
structured knowledge/programs mein convert karna."* So this module distils **structure**:

* the teacher demonstrates a task by showing its steps, not just its answer;
* every step is checked against facts she already holds — a demonstration is a claim like any
  other, and an unverified one teaches nothing;
* what is written down is a **property of the relation**, never the demonstrated conclusion:
  a verified same-relation chain confirms that relation's
  :class:`~nyxara.njp.core.Transitivity` posterior.

That last line is the whole design. The conclusion of the demonstration is thrown away. What she
keeps is *"this relation chains"* — which she can then apply to entities the teacher never
mentioned, which is what makes retention measurable rather than a synonym for memorisation.

**Why transitivity is the thing worth teaching.** :meth:`~nyxara.njp.core.CognitiveLearningCore.reach`
and :meth:`~nyxara.njp.core.CognitiveLearningCore.connects` price every hop by that predicate's own
posterior, and an unlisted relation starts at ``_TRANSITIVE_DEFAULT`` — low, "deliberately not
zero: a relation nobody anticipated should be *allowed* to prove itself transitive, just not
assumed to be." At that prior a three-hop chain falls under ``_MIN_LINK_CONFIDENCE`` and is not
walked at all. Confirmed, it is. So the same brain, the same facts, and a structural property
learned from a verified demonstration is the difference between "I don't know" and a derivation
with its steps — and it generalises to every chain of that relation, including ones nobody taught.

**A council, not a vote.** §19 is explicit that specialisation is not majority rule, and nothing
here averages teachers. Every teacher that answers is recorded; where two disagree, both lessons
are kept and *both* are verified, because a disagreement between teachers is evidence about the
teachers and voting it away destroys exactly that.

**No teacher is trusted, including the good one.** A lesson whose steps she cannot corroborate is
``UNDECIDED`` and teaches nothing — not "probably right". A lesson contradicted by what she holds
is ``REFUTED`` and moves the posterior *down*. Only corroborated structure is distilled, and its
weight is one observation per demonstration, never certainty.

Pure standard library. The LLM appears only behind :class:`CortexTeacher`, and every entry point
here works with it absent — which is also what makes the acquisition benchmark hermetic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Step", "Lesson", "Verdict", "Verification", "Teacher", "RecordedTeacher",
    "CortexTeacher", "TeacherCouncil", "Distillation", "Distiller",
]

#: The most a single demonstration may move a relation's posterior. One lesson is one observation;
#: a teacher that could move it further would be a teacher whose word is evidence, which is the
#: thing the gates exist to refuse.
_MAX_WEIGHT = 1.0

#: A chain shorter than this is not a demonstration of transitivity — it is one fact.
_MIN_CHAIN = 2


def _norm(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


@dataclass(frozen=True)
class Step:
    """One link of a demonstrated chain: ``subject —relation→ object``."""

    subject: str = ""
    relation: str = ""
    object: str = ""

    @property
    def empty(self) -> bool:
        return not (self.subject and self.relation and self.object)

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "relation": self.relation, "object": self.object}


@dataclass
class Lesson:
    """What one teacher said, and — the part that matters — *how* it got there."""

    task: str = ""
    answer: str = ""
    steps: Tuple[Step, ...] = ()
    rationale: str = ""
    source: str = ""
    confidence: float = 0.5

    @property
    def relation(self) -> str:
        """The relation the chain is made of, when it is made of one.

        ``""`` when the steps mix relations. That is not a failure — a mixed chain is a *shape*,
        which :mod:`nyxara.njp.genome` already records — but it is not a transitivity claim, and
        conflating the two would let ``sparrow is_a bird``, ``bird needs water`` be read as
        evidence that ``is_a`` chains.
        """
        relations = {s.relation for s in self.steps if s.relation}
        return next(iter(relations)) if len(relations) == 1 else ""

    @property
    def demonstrates_transitivity(self) -> bool:
        return bool(self.relation) and len(self.steps) >= _MIN_CHAIN

    def to_dict(self) -> Dict[str, Any]:
        return {"task": self.task[:160], "answer": self.answer[:160],
                "steps": [s.to_dict() for s in self.steps], "source": self.source,
                "relation": self.relation, "confidence": round(self.confidence, 4)}


class Verdict:
    """What her own facts say about a demonstration."""

    SURVIVED = "survived"        # every step corroborated by something she already holds
    UNDECIDED = "undecided"      # she cannot check it — teaches nothing
    REFUTED = "refuted"          # a step contradicts what she holds


@dataclass
class Verification:
    """The check, with the counts that produced the verdict rather than a bare label."""

    verdict: str = Verdict.UNDECIDED
    corroborated: int = 0
    unknown: int = 0
    contradicted: int = 0
    why: str = ""

    @property
    def steps(self) -> int:
        return self.corroborated + self.unknown + self.contradicted

    @property
    def weight(self) -> float:
        """How much of the demonstration she could actually check, in [0, 1]."""
        return (self.corroborated / self.steps) if self.steps else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "corroborated": self.corroborated,
                "unknown": self.unknown, "contradicted": self.contradicted,
                "weight": round(self.weight, 4), "why": self.why[:200]}


# --------------------------------------------------------------------------- #
# Teachers
# --------------------------------------------------------------------------- #
class Teacher:
    """A named source that can attempt a task and show its working."""

    name: str = "teacher"

    def available(self) -> bool:  # pragma: no cover - interface
        return False

    def teach(self, task: str, *, context: str = "") -> Optional[Lesson]:  # pragma: no cover
        return None


class RecordedTeacher(Teacher):
    """A teacher whose answers are fixed in advance — a transcript, not a model.

    Two honest uses, and it is important that neither is a simulation of intelligence. It replays
    what a real teacher said, so a session can be re-measured without paying for the model again;
    and it makes the acquisition benchmark **hermetic**, so the three-arm test runs on a machine
    with no weights on disk. What it cannot do is invent, and nothing here asks it to: the
    benchmark supplies the demonstrations and measures what NJP does with them.
    """

    def __init__(self, lessons: Dict[str, Lesson], *, name: str = "recorded") -> None:
        self.name = str(name)
        self.lessons = {_norm(k): v for k, v in (lessons or {}).items()}
        self.asked = 0
        self.answered = 0

    def available(self) -> bool:
        return bool(self.lessons)

    def teach(self, task: str, *, context: str = "") -> Optional[Lesson]:
        self.asked += 1
        lesson = self.lessons.get(_norm(task))
        if lesson is None:
            return None
        self.answered += 1
        return Lesson(task=task, answer=lesson.answer, steps=lesson.steps,
                      rationale=lesson.rationale, source=self.name,
                      confidence=lesson.confidence)


class CortexTeacher(Teacher):
    """The strong model as a teacher — the same cortex, asked for *working* rather than a claim.

    Deliberately thin. :class:`~nyxara.njp.cortex.Cortex` already owns the prompting, the
    injection screen, the strict-JSON extraction and the honest ``extraction_rate``; re-doing any
    of that here would give the acquisition path its own, less careful copy of the one surface in
    the package that touches untrusted model text.
    """

    def __init__(self, cortex: Any, *, name: str = "cortex") -> None:
        self.name = str(name)
        self.cortex = cortex
        self.asked = 0
        self.answered = 0

    def available(self) -> bool:
        try:
            return bool(self.cortex is not None and self.cortex.available())
        except Exception:  # noqa: BLE001
            return False

    def teach(self, task: str, *, context: str = "") -> Optional[Lesson]:
        self.asked += 1
        try:
            if not self.available():
                return None
            from nyxara.njp.cortex import CortexReport
            report = CortexReport()
            hypotheses = self.cortex.hypotheses(task, context=context, report=report) or []
            if not hypotheses:
                return None
            lead = max(hypotheses, key=lambda h: float(getattr(h, "confidence", 0.0) or 0.0))
            steps = tuple(_steps_of(lead))
            if not steps:
                return None
            self.answered += 1
            return Lesson(task=task, answer=str(getattr(lead, "claim", "") or ""),
                          steps=steps, rationale=str(getattr(lead, "why", "") or ""),
                          source=self.name,
                          confidence=float(getattr(lead, "confidence", 0.4) or 0.4))
        except Exception:  # noqa: BLE001 — a teacher that fails teaches nothing, never raises
            return None


def _steps_of(proposal: Any) -> List[Step]:
    """Read a chain off a cortex proposal, in whichever of its shapes carries one."""
    out: List[Step] = []
    for raw in (getattr(proposal, "steps", None) or getattr(proposal, "support", None) or ()):
        if isinstance(raw, Step):
            out.append(raw)
        elif isinstance(raw, (tuple, list)) and len(raw) >= 3:
            out.append(Step(_norm(raw[0]), _norm(raw[1]), _norm(raw[2])))
    return [s for s in out if not s.empty]


class TeacherCouncil:
    """Several teachers, asked in turn. Disagreement is recorded, never averaged away."""

    def __init__(self, teachers: Sequence[Teacher] = ()) -> None:
        self.teachers: List[Teacher] = list(teachers)
        self.asked = 0
        self.lessons = 0
        self.disagreements = 0

    def add(self, teacher: Teacher) -> None:
        self.teachers.append(teacher)

    @property
    def available(self) -> bool:
        return any(t.available() for t in self.teachers)

    def ask(self, task: str, *, context: str = "") -> List[Lesson]:
        """Every teacher's attempt at one task. A teacher that cannot answer contributes nothing."""
        self.asked += 1
        out: List[Lesson] = []
        for teacher in self.teachers:
            try:
                if not teacher.available():
                    continue
                lesson = teacher.teach(task, context=context)
            except Exception:  # noqa: BLE001
                continue
            if lesson is not None:
                out.append(lesson)
        self.lessons += len(out)
        if len({_norm(l.answer) for l in out}) > 1:
            # Kept as a count rather than resolved. Which teacher is right is a question for the
            # verification below, and it answers it per-lesson — a majority would answer it by
            # counting teachers, which is a fact about the council and not about the world.
            self.disagreements += 1
        return out

    def stats(self) -> Dict[str, Any]:
        return {"teachers": len(self.teachers), "available": self.available,
                "asked": self.asked, "lessons": self.lessons,
                "disagreements": self.disagreements}


# --------------------------------------------------------------------------- #
# Verification and distillation
# --------------------------------------------------------------------------- #
@dataclass
class Distillation:
    """What one lesson actually changed. Zero is a normal and frequent outcome."""

    lesson: Optional[Lesson] = None
    verification: Optional[Verification] = None
    relation: str = ""
    before: float = 0.0
    after: float = 0.0
    shape_recorded: bool = False
    why: str = ""

    @property
    def moved(self) -> float:
        return self.after - self.before

    def to_dict(self) -> Dict[str, Any]:
        return {"lesson": self.lesson.to_dict() if self.lesson else None,
                "verification": self.verification.to_dict() if self.verification else None,
                "relation": self.relation, "before": round(self.before, 4),
                "after": round(self.after, 4), "moved": round(self.moved, 4),
                "shape_recorded": self.shape_recorded, "why": self.why[:200]}


class Distiller:
    """Checks a demonstration against her own facts, and keeps only its structure."""

    def __init__(self, *, max_weight: float = _MAX_WEIGHT) -> None:
        self.max_weight = float(max_weight)
        self.verified = 0
        self.refuted = 0
        self.undecided = 0
        self.distilled = 0
        self.relations: Dict[str, int] = {}
        self.history: List[Distillation] = []

    # ---- verification (Brain C) --------------------------------------------- #
    def verify(self, lesson: Lesson, brain: Any) -> Verification:
        """Check every step against what she already holds. The answer itself is not checked.

        Deliberately. The conclusion is the one part of a demonstration that is *not* evidence:
        if she could already confirm it she would not have needed teaching, and confirming it
        against the teacher's own say-so is how a guess becomes a fact by being asked for twice —
        the failure :mod:`nyxara.njp.cortex` caps its confidences to prevent.
        """
        out = Verification()
        try:
            edges = self._edges(brain)
            if not lesson.steps:
                out.why = "the teacher gave an answer with no working"
                return out
            for step in lesson.steps:
                if step.empty:
                    out.unknown += 1
                    continue
                key = (_norm(step.subject), _norm(step.relation))
                held = edges.get(key)
                if held is None:
                    out.unknown += 1
                elif _norm(step.object) in held:
                    out.corroborated += 1
                else:
                    out.contradicted += 1
            if out.contradicted:
                out.verdict = Verdict.REFUTED
                out.why = (f"{out.contradicted} of {out.steps} steps name something she holds "
                           f"a different value for")
            elif out.corroborated == out.steps and out.corroborated >= _MIN_CHAIN:
                out.verdict = Verdict.SURVIVED
                out.why = f"all {out.corroborated} steps corroborated by facts she already holds"
            else:
                out.verdict = Verdict.UNDECIDED
                out.why = (f"only {out.corroborated} of {out.steps} steps are checkable against "
                           f"what she holds")
            return out
        except Exception:  # noqa: BLE001
            out.why = "verification failed"
            return out

    @staticmethod
    def _edges(brain: Any) -> Dict[Tuple[str, str], set]:
        """``(subject, relation) → {objects}`` from her own store, once per verification."""
        out: Dict[Tuple[str, str], set] = {}
        learner = getattr(brain, "learner", None)
        if learner is None:
            return out
        try:
            for subject, predicate, obj, _conf in learner._edges():
                out.setdefault((_norm(subject), _norm(predicate)), set()).add(_norm(obj))
        except Exception:  # noqa: BLE001
            return out
        return out

    # ---- distillation ------------------------------------------------------- #
    def distil(self, lesson: Lesson, brain: Any) -> Distillation:
        """Turn a verified demonstration into structure. Never into a fact.

        The relation's posterior moves, and that is the entire durable effect. What she was shown
        — that *these* three entities chain — is discarded; what she keeps is that this relation
        is the kind that chains, which applies to every entity she will ever hear about.
        """
        out = Distillation(lesson=lesson)
        try:
            verification = self.verify(lesson, brain)
            out.verification = verification
            learner = getattr(brain, "learner", None)
            if learner is None:
                out.why = "no learning core to distil into"
                return out

            relation = _norm(lesson.relation)
            out.relation = relation
            if verification.verdict == Verdict.REFUTED:
                self.refuted += 1
                if relation:
                    posterior = learner._transitivity(relation)
                    out.before = posterior.value
                    posterior.refute(min(self.max_weight, max(0.0, 1.0 - verification.weight)))
                    out.after = posterior.value
                out.why = verification.why
                return out
            if verification.verdict != Verdict.SURVIVED:
                self.undecided += 1
                out.why = verification.why
                return out

            self.verified += 1
            if not lesson.demonstrates_transitivity:
                # A mixed-relation chain is a *shape*, and the genome owns shapes. Recording it
                # here as a transitivity claim would teach her that `is_a` chains because
                # `sparrow is_a bird, bird needs water` worked once.
                out.why = "verified, but the chain is not one relation — no transitivity claim"
                out.shape_recorded = self._record_shape(lesson, brain)
                return out

            posterior = learner._transitivity(relation)
            out.before = posterior.value
            posterior.confirm(min(self.max_weight, verification.weight))
            out.after = posterior.value
            self.distilled += 1
            self.relations[relation] = self.relations.get(relation, 0) + 1
            out.shape_recorded = self._record_shape(lesson, brain)
            out.why = (f"{relation} chains: {out.before:.3f} → {out.after:.3f} on "
                       f"{verification.corroborated} corroborated steps")
            return out
        except Exception:  # noqa: BLE001 — a failed distillation teaches nothing
            out.why = "distillation failed"
            return out
        finally:
            self.history.append(out)
            del self.history[:-128]

    @staticmethod
    def _record_shape(lesson: Lesson, brain: Any) -> bool:
        """Let the genome see the form, so a demonstrated shape can be promoted like a derived one."""
        genome = getattr(brain, "genome", None)
        if genome is None or not lesson.steps:
            return False
        try:
            from nyxara.njp.core import Derivation
            derivation = Derivation(
                answer=lesson.answer, confidence=float(lesson.confidence),
                kind="composed",
                support=[(s.subject, s.relation, s.object) for s in lesson.steps],
                steps=[f"{s.subject} —{s.relation}→ {s.object}" for s in lesson.steps],
                why=f"demonstrated by {lesson.source}")
            return genome.record(derivation, question=lesson.task) is not None
        except Exception:  # noqa: BLE001
            return False

    def stats(self) -> Dict[str, Any]:
        return {"verified": self.verified, "refuted": self.refuted,
                "undecided": self.undecided, "distilled": self.distilled,
                "relations": dict(self.relations)}
