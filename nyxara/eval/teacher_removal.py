"""NYXARA · eval/teacher_removal.py — can the teacher be taken away again? (✂️🎓)

Every other measurement around the graft asks whether something worked *while it is present*:
:mod:`nyxara.njp.graft` measures whether a converted layer reproduces the arithmetic it replaced,
:mod:`nyxara.njp.shadow` counts the gaps it found, the batteries in this package measure what she
can answer. None of them can answer the question that decides whether any of it was **learning**:

    *Remove Qwythos. Is she still better than she was before it arrived?*

The distinction is not academic, and it is the one a distillation pipeline is structurally
inclined to get wrong. A system trained against a teacher's outputs improves on every metric that
is measured with the teacher attached — while what it has actually acquired may be a **dependency**
rather than a capability. Those two outcomes are indistinguishable until the teacher is removed,
and they call for opposite next steps.

**Three arms, and the two comparisons that carry the answer**::

    A = NJP alone, before any distillation      — the baseline she started from
    B = NJP + Qwythos                           — the scaffolded ceiling
    C = NJP after distillation, Qwythos REMOVED — the only number that means anything

    C vs A   did the distillation leave anything behind?
    C vs B   what did removing the teacher cost?

    C > A and C ≈ B  ->  distilled   — capability was transferred
    C ≈ A            ->  dependent   — she acquired a dependency, not an ability
    A < C < B        ->  partial     — some transferred, some was the teacher all along

**Paired, and exact.** All three arms see identical tasks, so only the tasks whose outcome
*changed* carry information, and :func:`~nyxara.eval.ablation.mcnemar_exact` needs no assumption
about task difficulty. That function is reused rather than reimplemented — this module is the same
instrument as :mod:`nyxara.eval.ablation` pointed at a different question.

**The failure mode this is mostly built to avoid**, inherited from that module: *an ablation that
silently fails to ablate reports zero difference and reads exactly like proof that the faculty was
worthless*. Here it is worse in both directions. A removal that did not remove makes arm C a
second copy of arm B, and the verdict comes back ``distilled`` — the strongest possible claim,
from an experiment that never ran. So :meth:`TeacherRemoval.check_removed` verifies structurally
that arm C has no teacher, and a report that cannot establish it is ``broken``, never ``distilled``.

**Held out, or it measures nothing.** The batteries must be tasks the distillation never trained
on (``eval/general_novel.py``, ``eval/generalization.py``, ``eval/continual.py``, or a
``Pair.held_out`` split from :mod:`nyxara.njp.study`). Scored on its training set, every arm looks
excellent and ``C ≈ B`` is guaranteed — which would make this instrument a machine for producing
the answer it was hoping for.

**"No evidence of transfer" is not "evidence of no transfer."** With a small battery, a real but
modest gain is invisible. ``underpowered`` is reported as its own verdict for exactly that reason,
and it is not permission to conclude the distillation failed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.eval.ablation import mcnemar_exact

__all__ = ["Arm", "ArmRun", "TeacherRemovalResult", "TeacherRemoval", "run_teacher_removal",
           "score_benchmark_task"]

def score_benchmark_task(task: Any, answer: str, *, threshold: float = 1.0) -> bool:
    """Did ``answer`` pass ``task``? Use this rather than hand-rolling the grading.

    ``Benchmark`` tasks grade to a **tuple** ``(score, reason)``, and every non-empty tuple is
    truthy. A caller who writes the obvious ``bool(task.grade(answer))`` therefore scores *every*
    task as passed — failures included — and the arms of this experiment all come back perfect.
    That is not a hypothetical: the first live run of arm A scored 20/20 on a battery where
    nineteen of the twenty answers were the empty string, and the verdict would have been a
    confident ``inert``.

    A measurement instrument whose scoring contract is easy to get wrong is an instrument that
    reports numbers nobody checked, so the adapter lives here beside the harness rather than in
    each caller.
    """
    try:
        verdict = task.grade(answer)
    except Exception:  # noqa: BLE001 — a grader that raises is a failed task, not a crash
        return False
    if isinstance(verdict, tuple):
        return bool(float(verdict[0]) >= threshold)
    passed = getattr(verdict, "passed", None)
    if passed is not None:
        return bool(passed)
    score = getattr(verdict, "score", None)
    if score is not None:
        return bool(float(score) >= threshold)
    if isinstance(verdict, (int, float)):
        return bool(float(verdict) >= threshold)
    return bool(verdict)


#: Fewer discordant pairs than this and the comparison is reported ``underpowered`` rather than
#: given a verdict. Mirrors the floor `eval/ablation.py` uses, and for the same reason: below it,
#: an exact test cannot reach significance at any split, so a p-value would be theatre.
MIN_DISCORDANT = 6

#: Arm C is allowed to be this much worse than arm B and still count as "≈". A distillation that
#: retains all but a couple of points on a held-out battery has transferred the capability;
#: demanding exact parity would fail every real result.
PARITY_SLACK = 0.05


#: The three conditions, named once so nothing downstream spells them differently.
class Arm:
    ALONE = "A"        # NJP before distillation, no teacher
    SCAFFOLDED = "B"   # NJP with the teacher attached
    DISTILLED = "C"    # NJP after distillation, teacher removed


@dataclass
class ArmRun:
    """One arm's outcomes on the battery, task by task, in the battery's order."""

    arm: str
    passed: Tuple[bool, ...] = ()
    answers: Tuple[str, ...] = ()
    teacher_attached: Optional[bool] = None    # None = could not be established
    latency_s: float = 0.0

    @property
    def n(self) -> int:
        return len(self.passed)

    @property
    def score(self) -> float:
        return (sum(self.passed) / self.n) if self.n else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"arm": self.arm, "n": self.n, "passed": sum(self.passed),
                "score": round(self.score, 4), "teacher_attached": self.teacher_attached,
                "latency_s": round(self.latency_s, 3)}


@dataclass
class TeacherRemovalResult:
    """The verdict, and every number it was derived from."""

    battery: str = ""
    a: ArmRun = field(default_factory=lambda: ArmRun(Arm.ALONE))
    b: ArmRun = field(default_factory=lambda: ArmRun(Arm.SCAFFOLDED))
    c: ArmRun = field(default_factory=lambda: ArmRun(Arm.DISTILLED))
    removal_verified: bool = False
    removal_detail: str = ""
    min_discordant: int = MIN_DISCORDANT
    parity_slack: float = PARITY_SLACK

    # ---- the two comparisons ---- #
    def _discordant(self, x: ArmRun, y: ArmRun) -> Tuple[int, int]:
        """``(x_only, y_only)`` — tasks exactly one of them got right."""
        n = min(x.n, y.n)
        x_only = sum(1 for i in range(n) if x.passed[i] and not y.passed[i])
        y_only = sum(1 for i in range(n) if y.passed[i] and not x.passed[i])
        return x_only, y_only

    @property
    def c_vs_a(self) -> Dict[str, Any]:
        """Did the distillation leave anything behind once the teacher was gone?"""
        gained, lost = self._discordant(self.c, self.a)
        return {"gained": gained, "lost": lost, "delta": round(self.c.score - self.a.score, 4),
                "p_value": round(mcnemar_exact(gained, lost), 5)}

    @property
    def c_vs_b(self) -> Dict[str, Any]:
        """What did removing the teacher cost?"""
        kept, lost = self._discordant(self.c, self.b)
        return {"kept": kept, "lost": lost, "delta": round(self.c.score - self.b.score, 4),
                "p_value": round(mcnemar_exact(kept, lost), 5)}

    @property
    def responses_differed(self) -> int:
        """Tasks where the arms did not all answer identically.

        Zero means the teacher never entered the loop on these tasks — a statement about the
        *battery*, not about the distillation.
        """
        n = min(self.a.n, self.b.n, self.c.n)
        if not n or not (self.a.answers and self.b.answers and self.c.answers):
            return 0
        return sum(1 for i in range(n)
                   if not (self.a.answers[i] == self.b.answers[i] == self.c.answers[i]))

    @property
    def verdict(self) -> str:
        """``broken`` / ``inert`` / ``underpowered`` / ``distilled`` / ``partial`` / ``dependent``.

        Ordered so that every way the experiment can be invalid is checked *before* any way it can
        succeed. A removal that did not remove, or a battery the teacher never touched, must not be
        able to produce ``distilled`` — those are the two paths by which this instrument would
        report the strongest possible result from no evidence at all.
        """
        if not self.removal_verified:
            return "broken"
        if not (self.a.n and self.b.n and self.c.n):
            return "broken"
        if self.responses_differed == 0:
            return "inert"

        ca = self.c_vs_a
        gain_discordant = ca["gained"] + ca["lost"]
        teacher_discordant = sum(self._discordant(self.b, self.a))

        if gain_discordant == 0:
            # C answered every task exactly as A did. That is not an absence of evidence — it is
            # the sharpest possible evidence, PROVIDED the teacher could have changed something
            # here. `teacher_discordant` is what establishes that: with the teacher attached the
            # answers moved on that many tasks, and after distillation not one of them stayed
            # moved. Reporting this as `underpowered` (which an earlier ordering did, because zero
            # discordant pairs trivially falls below any floor) would file the cleanest dependency
            # result under "battery too small" — the one misreading that would let a failed
            # distillation look like an inconclusive one.
            if teacher_discordant >= self.min_discordant:
                return "dependent"
            return "underpowered"
        if gain_discordant < self.min_discordant:
            return "underpowered"

        gained = ca["p_value"] < 0.05 and ca["gained"] > ca["lost"]
        if not gained:
            # C is not measurably better than the baseline: whatever improved while the teacher
            # was attached did not survive its removal. This is the honest reading of a
            # scaffold that was load-bearing, and it is reported as plainly as a success.
            return "dependent"
        near_b = (self.b.score - self.c.score) <= self.parity_slack
        return "distilled" if near_b else "partial"

    @property
    def summary(self) -> str:
        return (f"A={self.a.score:.3f}  B={self.b.score:.3f}  C={self.c.score:.3f}  "
                f"-> {self.verdict}")

    def to_dict(self) -> Dict[str, Any]:
        return {"battery": self.battery, "verdict": self.verdict,
                "arms": {"A": self.a.to_dict(), "B": self.b.to_dict(), "C": self.c.to_dict()},
                "c_vs_a": self.c_vs_a, "c_vs_b": self.c_vs_b,
                "responses_differed": self.responses_differed,
                "removal_verified": self.removal_verified,
                "removal_detail": self.removal_detail,
                "min_discordant": self.min_discordant,
                "parity_slack": self.parity_slack,
                "summary": self.summary}

    def explain(self) -> str:
        """A paragraph a person can act on, including when the answer is unwelcome."""
        v = self.verdict
        ca, cb = self.c_vs_a, self.c_vs_b
        head = f"[{self.battery or 'battery'}] {self.summary}"
        body = {
            "broken": ("The teacher was not verifiably removed from arm C, so arm C is not an "
                       "independent measurement. No conclusion about transfer can be drawn: "
                       f"{self.removal_detail or 'removal unverified'}."),
            "inert": ("All three arms answered every task identically — the teacher never entered "
                      "the loop here. That is a finding about the battery, not about the "
                      "distillation; run one whose tasks the teacher would actually change."),
            "underpowered": (f"Only {ca['gained'] + ca['lost']} tasks changed between A and C, "
                             f"below the floor of {self.min_discordant}. A real but modest "
                             "transfer would be invisible at this size. This is not evidence "
                             "that the distillation failed."),
            "dependent": (f"C is not measurably better than A (delta {ca['delta']:+.3f}, "
                          f"p={ca['p_value']}; {ca['gained']} gained / {ca['lost']} lost, against "
                          f"{sum(self._discordant(self.b, self.a))} tasks the teacher did move). "
                          "What improved with the teacher attached did not survive its removal — "
                          "she acquired a dependency, not an ability. The scaffold is still "
                          "load-bearing."),
            "partial": (f"C beats A (delta {ca['delta']:+.3f}, p={ca['p_value']}) but stays "
                        f"{-cb['delta']:.3f} below B, past the {self.parity_slack:.2f} parity "
                        "slack. Some capability transferred and some of the scaffolded score was "
                        "the teacher itself."),
            "distilled": (f"C beats A (delta {ca['delta']:+.3f}, p={ca['p_value']}) and holds "
                          f"within {self.parity_slack:.2f} of B. Capability transferred: she "
                          "retains it with the teacher gone."),
        }.get(v, "")
        return f"{head}\n  {body}"


class TeacherRemoval:
    """Runs the three arms over one battery and verifies that the removal actually removed."""

    def __init__(self, *, min_discordant: int = MIN_DISCORDANT,
                 parity_slack: float = PARITY_SLACK) -> None:
        self.min_discordant = int(min_discordant)
        self.parity_slack = float(parity_slack)

    @staticmethod
    def check_removed(core: Any) -> Tuple[bool, str]:
        """Is the teacher genuinely gone from this core? ``(verified, detail)``.

        Structural, not trusting. Both the organ and its weights are checked, because either alone
        can be defeated: an organ can be detached while a grafted region still carries the same
        computation, and a graft can be dropped while ``cortex.ask`` still answers. A core that
        cannot be inspected returns ``False`` — unverified is not the same as removed, and this
        is the check that stops ``distilled`` being reported from an experiment that never ran.
        """
        try:
            njp = getattr(core, "njp", core)
            cortex = getattr(njp, "cortex", None)
            if cortex is not None:
                try:
                    if cortex.available():
                        return False, "cortex is still available on this core"
                except Exception:  # noqa: BLE001
                    return False, "cortex present and could not be interrogated"
            # `shadow_cognition`, not `shadow`: `NJPBrain.shadow()` is a method about her own
            # ignorance and has nothing to do with the teacher. Reading it here would find a bound
            # method with no `teacher` attribute, conclude the teacher was gone, and verify a
            # removal that never happened — which is the one path to a false `distilled`.
            shadow = getattr(njp, "shadow_cognition", None)
            if shadow is not None and getattr(shadow, "teacher", None) is not None:
                return False, "shadow cognition still holds a teacher reference"
            fabric = getattr(njp, "fabric", None)
            grafts = list(getattr(fabric, "grafts", []) or []) if fabric is not None else []
            if grafts:
                return False, f"{len(grafts)} grafted region(s) still attached to the fabric"
            return True, "no cortex, no teacher reference, no grafted regions"
        except Exception as exc:  # noqa: BLE001
            return False, f"could not inspect the core: {exc}"

    def run(self, tasks: Sequence[Any], *, battery: str = "",
            solve_a: Callable[[Any], str], solve_b: Callable[[Any], str],
            solve_c: Callable[[Any], str], score: Callable[[Any, str], bool],
            core_c: Any = None) -> TeacherRemovalResult:
        """Run all three arms over the same tasks, in the same order."""
        res = TeacherRemovalResult(battery=str(battery),
                                   min_discordant=self.min_discordant,
                                   parity_slack=self.parity_slack)
        for arm, solver, slot in ((Arm.ALONE, solve_a, "a"),
                                  (Arm.SCAFFOLDED, solve_b, "b"),
                                  (Arm.DISTILLED, solve_c, "c")):
            passed: List[bool] = []
            answers: List[str] = []
            t0 = time.perf_counter()
            for task in tasks:
                try:
                    answer = str(solver(task) or "")
                except Exception:  # noqa: BLE001 — a crashed solver is a failed task, not a
                    answer = ""     # crashed experiment; the arm keeps its shape either way
                answers.append(answer)
                try:
                    passed.append(bool(score(task, answer)))
                except Exception:  # noqa: BLE001
                    passed.append(False)
            setattr(res, slot, ArmRun(arm=arm, passed=tuple(passed), answers=tuple(answers),
                                      latency_s=time.perf_counter() - t0))
        if core_c is not None:
            res.removal_verified, res.removal_detail = self.check_removed(core_c)
        else:
            res.removal_verified = False
            res.removal_detail = ("no core supplied for arm C, so the removal could not be "
                                  "verified structurally")
        return res


def run_teacher_removal(tasks: Sequence[Any], **kwargs: Any) -> TeacherRemovalResult:
    """Convenience wrapper around :meth:`TeacherRemoval.run`."""
    return TeacherRemoval().run(tasks, **kwargs)
