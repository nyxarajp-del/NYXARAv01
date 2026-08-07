"""NYXARA · eval/ablation.py — does a faculty beat its own absence? (📏 → ✂️)

Every other battery in this package measures **the whole mind**: how safe it is, how capable it is,
how well it generalises. None of them can answer the question that decides whether a module should
exist:

    *Is she measurably better with this faculty than without it?*

92 capability modules are wired into the core. Their value is asserted in ``docs/CAPABILITIES.md``
and demonstrated by their own self-tests, which is evidence that each one *works* — not evidence
that any one of them *helps*. A module can be correct, tested, documented, and still contribute
nothing to an answer. Deleting on taste would be guessing; keeping on faith would be hoarding. This
module produces the third option: evidence.

**The method.** Run the same held-out battery twice against the same core — once with the faculty
live, once with it disabled through the code's own ``is None`` fallback path — and compare the
*paired* outcomes. Paired is what makes this cheap and sharp: the two arms see identical tasks, so
only the tasks whose outcome **changed** carry information, and McNemar's exact test on those
discordant pairs needs no assumption about task difficulty.

**The failure mode this module is mostly built to avoid.** An ablation that silently fails to
ablate reports zero difference and reads exactly like proof that a faculty is worthless. That is
the one error that would destroy working code. Two guards, both hard:

* ``disable()`` must *report* that it changed something; a toggle that no-ops is a broken
  experiment, not a null result;
* if every response in both arms is byte-identical, the verdict is **inert** — the faculty never
  entered the loop on these tasks — which is a statement about the *battery*, not the faculty.

**"No evidence of benefit" is not "evidence of no benefit."** With eight held-out tasks almost
nothing reaches significance, and a report that quietly rounded that to "delete it" would be
laundering ignorance into a recommendation. Every non-significant verdict therefore carries the
discordant-pair count that produced it, and :meth:`AblationResult.deletion_candidate` refuses to
recommend anything until the experiment was large enough to have noticed.

Cost is measured alongside accuracy, because it is half the decision: a faculty that adds nothing
and costs 12 seconds a turn is a stronger candidate for removal than one that adds nothing and
costs a microsecond.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.eval.benchmark import Benchmark, BenchmarkReport, Solver

__all__ = [
    "Faculty",
    "attr_faculty",
    "AblationResult",
    "AblationReport",
    "mcnemar_exact",
    "ablate",
    "run_ablation",
    "CORE_FACULTIES",
]


# --------------------------------------------------------------------------- #
# What can be switched off
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Faculty:
    """A named part of the mind that can be turned off on a built core.

    ``disable`` receives a live core and returns ``True`` if it actually changed something. That
    return value is not a convenience — it is the experiment's validity check. A toggle that
    silently no-ops produces a perfect null result for a faculty that was never switched off, and
    a null result is the input to a deletion decision.
    """

    name: str
    disable: Callable[[Any], bool]
    note: str = ""
    #: The core attribute this ablates, when it ablates one. Recorded rather than inferred from
    #: ``note``: a checker that reads the prose gets the answer wrong the moment the prose is
    #: rewritten, and the thing being checked is that the attribute name is not a typo.
    attr: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover — display only
        return self.name


def attr_faculty(name: str, attr: str, note: str = "") -> Faculty:
    """A faculty disabled by nulling one attribute on the core.

    This is the honest way to ablate *this* codebase: the call sites already guard with
    ``if self.<attr> is None`` and fall back, so setting it to ``None`` exercises the real
    degraded path rather than a special test-only branch. Reports ``False`` when the attribute
    is missing or already ``None`` — nothing was disabled, so nothing was measured.
    """

    def _disable(core: Any) -> bool:
        if getattr(core, attr, None) is None:
            return False
        setattr(core, attr, None)
        return True

    return Faculty(name=name, disable=_disable, note=note or f"core.{attr} = None", attr=attr)


#: A starting set aimed at the faculties that sit on the *turn* path, since those are the ones
#: whose absence a battery can actually see. A faculty that only runs on an idle tick or during
#: growth will read as ``inert`` here, and that is the correct answer for this instrument — it
#: means "ask a different battery", not "delete it".
CORE_FACULTIES: Tuple[Faculty, ...] = (
    # The two measured cost centres, first. Neither is a guess: a bare "Hii" turn took 122.7 s of
    # which 109.2 s was ``_inject_domain_expertise`` spending a whole model call to *label* the
    # greeting, and recursive improvement buys its quality with N further full passes. Both are
    # advisory, so the question "is that price worth paying?" is exactly answerable here.
    attr_faculty("domain_expertise", "general_intelligence",
                 "classifies the domain and prepends an expert methodology (costs a generation)"),
    attr_faculty("recursive_improve", "recursive_improver",
                 "N critique+revise passes over the candidate (costs N generations)"),
    attr_faculty("role_council", "role_council", "the multi-role deliberation council"),
    attr_faculty("self_model", "self_model", "the self-knowledge report injected into context"),
    attr_faculty("world_model", "world_model", "the predictive world model"),
    attr_faculty("retriever", "retriever", "memory recall on the turn path"),
    attr_faculty("tom", "tom", "theory of mind / the Master's mental state"),
    attr_faculty("affect", "affect", "the affective read of the turn"),
)


# --------------------------------------------------------------------------- #
# The paired test
# --------------------------------------------------------------------------- #
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for ``b`` / ``c`` discordant pairs.

    ``b`` = tasks the faculty got right that its absence got wrong; ``c`` = the reverse. The
    concordant tasks carry no information about the faculty and are correctly ignored: under the
    null, each discordant pair is a fair coin, so ``b ~ Binomial(b + c, 0.5)``.

    Exact rather than chi-squared because these batteries are small — the chi-squared
    approximation is unreliable at exactly the sample sizes this instrument runs at, and being
    wrong in the anti-conservative direction here means deleting working code.
    """
    n = int(b) + int(c)
    if n == 0:
        return 1.0                      # nothing disagreed; no evidence either way
    k = max(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


@dataclass
class AblationResult:
    """What one faculty's absence did to one battery."""

    faculty: str
    n_tasks: int
    with_passed: int
    without_passed: int
    #: discordant pairs — the only tasks that carry information
    helped: int = 0                      # faculty right, absence wrong
    hurt: int = 0                        # faculty wrong, absence right
    p_value: float = 1.0
    responses_differed: int = 0
    with_latency_s: float = 0.0
    without_latency_s: float = 0.0
    disabled_ok: bool = True
    alpha: float = 0.05
    #: how many discordant pairs the experiment must produce before a null result is allowed to
    #: mean anything. Below this, "no evidence" is a statement about the battery's size.
    min_discordant: int = 6
    note: str = ""

    @property
    def accuracy_delta(self) -> float:
        """with − without, in accuracy. Positive means the faculty earned its place."""
        if not self.n_tasks:
            return 0.0
        return (self.with_passed - self.without_passed) / self.n_tasks

    @property
    def latency_cost_s(self) -> float:
        """What the faculty costs per task. Half of the decision, and the half that is certain."""
        return self.with_latency_s - self.without_latency_s

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def verdict(self) -> str:
        """One of ``broken`` / ``inert`` / ``helps`` / ``hurts`` / ``no-evidence`` / ``underpowered``.

        ``inert`` and ``underpowered`` are deliberately *not* folded into ``no-evidence``: they
        have different causes and different remedies. Inert means the faculty never ran on these
        tasks — the battery is wrong for it. Underpowered means it ran and changed answers, but too
        few of them to conclude anything — the battery is too small.
        """
        if not self.disabled_ok:
            return "broken"
        if self.responses_differed == 0:
            return "inert"
        if self.significant:
            return "helps" if self.helped > self.hurt else "hurts"
        if (self.helped + self.hurt) < self.min_discordant:
            return "underpowered"
        return "no-evidence"

    @property
    def deletion_candidate(self) -> bool:
        """Whether the evidence supports removing this faculty.

        Requires a *positive* finding, never merely the absence of one. ``broken`` and
        ``underpowered`` are explicitly excluded: an experiment that failed, or one too small to
        have noticed a real effect, is not permission to delete — it is a reason to run a better
        experiment. ``inert`` qualifies only in the narrow sense that the faculty demonstrably did
        not participate in *these* turns; the honest next step is a battery that would exercise it,
        not the ``rm``.
        """
        return self.verdict in ("hurts", "no-evidence")

    def to_dict(self) -> Dict[str, Any]:
        return {"faculty": self.faculty, "verdict": self.verdict, "n_tasks": self.n_tasks,
                "with_passed": self.with_passed, "without_passed": self.without_passed,
                "accuracy_delta": round(self.accuracy_delta, 4),
                "helped": self.helped, "hurt": self.hurt,
                "p_value": round(self.p_value, 5),
                "responses_differed": self.responses_differed,
                "latency_cost_s": round(self.latency_cost_s, 4),
                "disabled_ok": self.disabled_ok,
                "deletion_candidate": self.deletion_candidate,
                "note": self.note}

    def explain(self) -> str:
        """One honest line, including what the number cannot support."""
        if self.verdict == "broken":
            return (f"{self.faculty}: the ablation did not take effect — no measurement was made "
                    f"(the toggle reported no change)")
        if self.verdict == "inert":
            return (f"{self.faculty}: never changed a single answer on {self.n_tasks} tasks — it "
                    f"did not participate in these turns; this measures the battery, not the "
                    f"faculty")
        if self.verdict == "underpowered":
            return (f"{self.faculty}: only {self.helped + self.hurt} task(s) changed outcome "
                    f"(need {self.min_discordant}) — too small to conclude anything either way")
        if self.verdict == "helps":
            return (f"{self.faculty}: earns its place — +{self.accuracy_delta:.1%} accuracy "
                    f"(p={self.p_value:.3f}), costing {self.latency_cost_s:+.3f}s/task")
        if self.verdict == "hurts":
            return (f"{self.faculty}: measurably WORSE than its absence — {self.accuracy_delta:.1%} "
                    f"accuracy (p={self.p_value:.3f})")
        return (f"{self.faculty}: no measurable benefit over {self.n_tasks} tasks "
                f"({self.helped} up / {self.hurt} down, p={self.p_value:.3f}), "
                f"costing {self.latency_cost_s:+.3f}s/task")


@dataclass
class AblationReport:
    battery: str
    results: List[AblationResult] = field(default_factory=list)
    seed: int = 0

    def __len__(self) -> int:
        return len(self.results)

    def by_verdict(self, verdict: str) -> List[AblationResult]:
        return [r for r in self.results if r.verdict == verdict]

    @property
    def earned(self) -> List[AblationResult]:
        return self.by_verdict("helps")

    @property
    def candidates(self) -> List[AblationResult]:
        """Ranked by what removing them buys: most wasted time first."""
        return sorted((r for r in self.results if r.deletion_candidate),
                      key=lambda r: -r.latency_cost_s)

    @property
    def unmeasured(self) -> List[AblationResult]:
        """Faculties this run could not decide — the honest residue, kept visible."""
        return [r for r in self.results if r.verdict in ("inert", "underpowered", "broken")]

    def to_dict(self) -> Dict[str, Any]:
        return {"battery": self.battery, "seed": self.seed, "n_faculties": len(self.results),
                "earned": [r.faculty for r in self.earned],
                "deletion_candidates": [r.faculty for r in self.candidates],
                "unmeasured": [r.faculty for r in self.unmeasured],
                "results": [r.to_dict() for r in self.results]}

    def render(self) -> str:
        lines = [f"ablation over {self.battery} (seed {self.seed})", "=" * 70]
        for r in sorted(self.results, key=lambda x: (x.verdict, -x.accuracy_delta)):
            lines.append("  " + r.explain())
        lines.append("-" * 70)
        lines.append(f"  earned their place : {[r.faculty for r in self.earned] or 'none'}")
        lines.append(f"  deletion candidates: {[r.faculty for r in self.candidates] or 'none'}")
        lines.append(f"  undecided by this run: {[r.faculty for r in self.unmeasured] or 'none'}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Running one
# --------------------------------------------------------------------------- #
def _outcomes(report: BenchmarkReport) -> Dict[str, Tuple[bool, str, float]]:
    return {r.task_id: (r.passed, r.response, r.latency_s) for r in report.results}


def ablate(faculty: Faculty, benchmark: Benchmark,
           core_factory: Callable[[], Any], *,
           alpha: float = 0.05, min_discordant: int = 6) -> AblationResult:
    """Run ``benchmark`` twice — faculty on, faculty off — and compare the paired outcomes.

    A **fresh core per arm**, because ablation mutates the core and a shared one would carry the
    first arm's memory, journal and fatigue into the second. Same battery, same order, same seed:
    the only intended difference between the arms is the faculty.
    """
    from nyxara.agency.permissions import Authority

    def _solver(core: Any) -> Solver:
        def _solve(prompt: str) -> str:
            return core.process(prompt, authority=Authority.OWNER).response
        return _solve

    on_core = core_factory()
    t0 = time.monotonic()
    on_report = benchmark.run(_solver(on_core))
    on_wall = time.monotonic() - t0

    off_core = core_factory()
    disabled_ok = bool(faculty.disable(off_core))
    t0 = time.monotonic()
    off_report = benchmark.run(_solver(off_core))
    off_wall = time.monotonic() - t0

    on_by_id, off_by_id = _outcomes(on_report), _outcomes(off_report)
    shared = [tid for tid in on_by_id if tid in off_by_id]

    helped = hurt = differed = 0
    for tid in shared:
        on_pass, on_text, _ = on_by_id[tid]
        off_pass, off_text, _ = off_by_id[tid]
        if on_text != off_text:
            differed += 1
        if on_pass and not off_pass:
            helped += 1
        elif off_pass and not on_pass:
            hurt += 1

    n = len(shared) or 1
    return AblationResult(
        faculty=faculty.name,
        n_tasks=len(shared),
        with_passed=sum(1 for t in shared if on_by_id[t][0]),
        without_passed=sum(1 for t in shared if off_by_id[t][0]),
        helped=helped, hurt=hurt,
        p_value=mcnemar_exact(helped, hurt),
        responses_differed=differed,
        with_latency_s=on_wall / n, without_latency_s=off_wall / n,
        disabled_ok=disabled_ok, alpha=alpha, min_discordant=min_discordant,
        note=faculty.note)


def run_ablation(faculties: Sequence[Faculty] = CORE_FACULTIES, *,
                 benchmark: Optional[Benchmark] = None,
                 core_factory: Optional[Callable[[], Any]] = None,
                 holdout_frac: float = 0.4, seed: int = 0,
                 limit: int = 0,
                 alpha: float = 0.05, min_discordant: int = 6) -> AblationReport:
    """Ablate each faculty against the **held-out** fold of a battery.

    Held-out on purpose: the self-improvement loop tunes against the train fold, so scoring a
    faculty there would measure how well the loop has fitted it, not whether it helps. ``seed``
    rotates which tasks are held out, so a faculty cannot be kept alive by one lucky partition.
    """
    if benchmark is None:
        from nyxara.eval.general_novel import build_general_novel_benchmark
        benchmark = build_general_novel_benchmark()
    _, holdout = benchmark.split(holdout_frac, seed=seed)
    if limit:
        holdout = holdout.sample(limit, seed=seed)
    if core_factory is None:
        from nyxara.eval.harness import default_core_factory
        core_factory = default_core_factory

    results = [ablate(f, holdout, core_factory, alpha=alpha, min_discordant=min_discordant)
               for f in faculties]
    return AblationReport(battery=holdout.name, results=results, seed=seed)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA ablation self-test")
    print("=" * 70)

    # ---- the statistic ---- #
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(8, 0) < 0.01, "a clean 8-0 split is strong evidence"
    assert mcnemar_exact(4, 4) == 1.0, "a perfect tie is no evidence at all"
    assert mcnemar_exact(3, 0) > 0.05, "3-0 is suggestive, not significant — and must not round up"
    print(f"\nmcnemar 8-0         : p={mcnemar_exact(8, 0):.4f}")
    print(f"mcnemar 3-0         : p={mcnemar_exact(3, 0):.4f} (correctly NOT significant)")

    # ---- the verdicts, and what each refuses to conclude ---- #
    broken = AblationResult("ghost", 20, 10, 10, disabled_ok=False)
    assert broken.verdict == "broken" and not broken.deletion_candidate
    print(f"\nfailed toggle       : {broken.explain()}")

    inert = AblationResult("idle_only", 20, 10, 10, responses_differed=0)
    assert inert.verdict == "inert" and not inert.deletion_candidate

    small = AblationResult("thin", 8, 5, 4, helped=1, hurt=0,
                           p_value=mcnemar_exact(1, 0), responses_differed=3)
    assert small.verdict == "underpowered", "1 discordant pair concludes nothing"
    assert not small.deletion_candidate, "an underpowered null must never license a deletion"
    print(f"underpowered        : {small.explain()}")

    good = AblationResult("council", 40, 30, 20, helped=11, hurt=1,
                          p_value=mcnemar_exact(11, 1), responses_differed=25,
                          with_latency_s=0.9, without_latency_s=0.4)
    assert good.verdict == "helps" and not good.deletion_candidate
    print(f"earns its place     : {good.explain()}")

    waste = AblationResult("expensive_noop", 40, 20, 20, helped=4, hurt=4,
                           p_value=mcnemar_exact(4, 4), responses_differed=30,
                           with_latency_s=12.5, without_latency_s=0.5)
    assert waste.verdict == "no-evidence" and waste.deletion_candidate
    print(f"no measurable value : {waste.explain()}")

    harmful = AblationResult("misfire", 40, 12, 24, helped=1, hurt=13,
                             p_value=mcnemar_exact(1, 13), responses_differed=30)
    assert harmful.verdict == "hurts" and harmful.deletion_candidate
    print(f"actively harmful    : {harmful.explain()}")

    # ---- the guard that matters most ---- #
    # A no-op toggle must be reported as a broken experiment, never as a null result. This is the
    # path that would otherwise delete working code on the strength of a measurement never taken.
    class _Core:
        role_council = None

    fac = attr_faculty("role_council", "role_council")
    assert fac.disable(_Core()) is False, "nulling an already-None attribute disabled nothing"
    print("\nno-op toggle        : reported as a failed ablation, not a null result ✓")

    class _Live:
        role_council = object()

    live = _Live()
    assert fac.disable(live) is True and live.role_council is None
    print("real toggle         : reported as effective ✓")

    # ---- the report ---- #
    report = AblationReport(battery="demo:holdout",
                            results=[good, waste, harmful, small, inert, broken])
    assert [r.faculty for r in report.earned] == ["council"]
    assert report.candidates[0].faculty == "expensive_noop", "rank by wasted time"
    assert {r.faculty for r in report.unmeasured} == {"thin", "idle_only", "ghost"}
    print("\n" + report.render())

    print("\nALL SELF-TESTS PASSED ✓")
