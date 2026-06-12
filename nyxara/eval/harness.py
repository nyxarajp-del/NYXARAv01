"""NYXARA · eval/harness.py — the evaluation core: cases, suites, reports (📐, deterministic).

This is the measuring instrument itself. A :class:`EvalCase` is one falsifiable question
about the mind ("does an untrusted injection get refused?"); given a fresh
:class:`~nyxara.kernel.orchestrator.NyxaraCore` it returns an :class:`EvalOutcome`
(pass/fail + a 0..1 score + a human detail). A :class:`EvalSuite` holds many cases and
runs each against an *independent* core (so cases never contaminate one another), yielding
an :class:`EvalReport` with pass-rate, mean-score, and a per-category breakdown.

Two properties make this useful over time:

* **Determinism** — the default core uses the offline reasoner, so the same suite gives
  the same numbers on every run; a flake is a real signal, not noise.
* **Regressions** — a report can be saved as JSON and a later run compared against it
  (:meth:`EvalReport.regression_vs`) to surface cases that used to pass and now fail.

Pure standard library + the repo's own :mod:`kernel.orchestrator` / :mod:`agency.permissions`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from nyxara.kernel.orchestrator import NyxaraCore

__all__ = [
    "EvalOutcome",
    "EvalCase",
    "EvalResult",
    "EvalReport",
    "EvalSuite",
    "default_core_factory",
]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def default_core_factory() -> NyxaraCore:
    """Build a fresh, offline, deterministic core — the default unit-under-test.

    A new core per case means scram/resume, memory writes, and journal state from one
    case can never leak into the next.
    """
    return NyxaraCore()


# --------------------------------------------------------------------------- #
# Outcome / Case
# --------------------------------------------------------------------------- #
@dataclass
class EvalOutcome:
    """What a single case's ``run`` reports back: did it pass, how well, and why."""
    passed: bool
    score: float = 0.0
    detail: str = ""

    def __post_init__(self) -> None:
        self.score = _clamp01(self.score)

    @classmethod
    def ok(cls, detail: str = "", score: float = 1.0) -> "EvalOutcome":
        return cls(passed=True, score=score, detail=detail)

    @classmethod
    def fail(cls, detail: str = "", score: float = 0.0) -> "EvalOutcome":
        return cls(passed=False, score=score, detail=detail)


# a case is, in essence, a function from a fresh core to an outcome
CaseRun = Callable[[NyxaraCore], EvalOutcome]


@dataclass
class EvalCase:
    """One falsifiable check on the mind.

    The simplest case wraps a ``run`` callable that, given a fresh core, returns an
    :class:`EvalOutcome`. Construction guards against the empty case (no ``run``).
    """
    name: str
    category: str
    run: CaseRun
    description: str = ""

    def __post_init__(self) -> None:
        if not callable(self.run):
            raise ValueError(f"case {self.name!r} has no callable run")

    def evaluate(self, core: NyxaraCore) -> "EvalResult":
        """Run this case against ``core`` and capture the result (errors become failures)."""
        start = time.monotonic()
        try:
            outcome = self.run(core)
            if not isinstance(outcome, EvalOutcome):  # defensive: bad case authoring
                outcome = EvalOutcome.fail(
                    f"case did not return an EvalOutcome (got {type(outcome).__name__})")
        except Exception as exc:  # noqa: BLE001 — a crash is a failed case, not a crashed run
            outcome = EvalOutcome.fail(f"{type(exc).__name__}: {exc}")
        duration = time.monotonic() - start
        return EvalResult(case_name=self.name, category=self.category,
                          passed=outcome.passed, score=outcome.score,
                          detail=outcome.detail, duration_s=duration)


# --------------------------------------------------------------------------- #
# Result / Report
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    case_name: str
    category: str
    passed: bool
    score: float
    detail: str
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"case_name": self.case_name, "category": self.category,
                "passed": self.passed, "score": round(self.score, 4),
                "detail": self.detail, "duration_s": round(self.duration_s, 5)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalResult":
        return cls(case_name=d["case_name"], category=d["category"],
                   passed=bool(d["passed"]), score=float(d["score"]),
                   detail=d.get("detail", ""), duration_s=float(d.get("duration_s", 0.0)))


@dataclass
class EvalReport:
    """The result of running a suite: every case's result, plus aggregate views."""
    results: List[EvalResult] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    # ---- aggregate properties ---- #
    def __len__(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / len(self.results) if self.results else 0.0

    @property
    def mean_score(self) -> float:
        return sum(r.score for r in self.results) / len(self.results) if self.results else 0.0

    def by_category(self) -> Dict[str, Dict[str, Any]]:
        """Per-category {pass_rate, mean_score, n} — where the mind is strong vs weak."""
        cats: Dict[str, List[EvalResult]] = {}
        for r in self.results:
            cats.setdefault(r.category, []).append(r)
        out: Dict[str, Dict[str, Any]] = {}
        for cat, rs in cats.items():
            n = len(rs)
            out[cat] = {"pass_rate": sum(1 for r in rs if r.passed) / n,
                        "mean_score": sum(r.score for r in rs) / n, "n": n}
        return out

    def failures(self) -> List[EvalResult]:
        return [r for r in self.results if not r.passed]

    def get(self, case_name: str) -> Optional[EvalResult]:
        for r in self.results:
            if r.case_name == case_name:
                return r
        return None

    # ---- regression detection ---- #
    def regression_vs(self, baseline: "EvalReport") -> List[str]:
        """Names of cases that *passed* in ``baseline`` but *fail* now (newly broken)."""
        baseline_passing = {r.case_name for r in baseline.results if r.passed}
        return sorted(r.case_name for r in self.results
                      if not r.passed and r.case_name in baseline_passing)

    # ---- serialisation ---- #
    def to_dict(self) -> Dict[str, Any]:
        return {"generated_at": self.generated_at,
                "pass_rate": round(self.pass_rate, 4),
                "mean_score": round(self.mean_score, 4),
                "passed": self.passed, "total": len(self.results),
                "by_category": self.by_category(),
                "results": [r.to_dict() for r in self.results]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalReport":
        return cls(results=[EvalResult.from_dict(r) for r in d.get("results", [])],
                   generated_at=float(d.get("generated_at", time.time())))

    def save(self, path: str) -> str:
        """Persist this report as JSON so a later run can detect regressions against it."""
        import os
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
        return path

    @classmethod
    def load(cls, path: str) -> "EvalReport":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # ---- human report ---- #
    def summary(self) -> str:
        lines = [
            "=" * 70,
            "NYXARA evaluation report",
            "=" * 70,
            f"overall             : {self.passed}/{len(self.results)} passed "
            f"({self.pass_rate:.0%}), mean score {self.mean_score:.3f}",
            "",
            "by category:",
        ]
        for cat, stats in sorted(self.by_category().items()):
            lines.append(f"  {cat:<14}: {stats['pass_rate']:.0%} pass "
                         f"({stats['n']} cases), mean {stats['mean_score']:.3f}")
        fails = self.failures()
        if fails:
            lines.append("")
            lines.append("failures:")
            for r in fails:
                lines.append(f"  ✗ [{r.category}] {r.case_name} — {r.detail}")
        else:
            lines.append("")
            lines.append("no failures ✓")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Suite
# --------------------------------------------------------------------------- #
class EvalSuite:
    """A named battery of cases, run against fresh cores into a report."""

    def __init__(self, name: str = "default", cases: Optional[List[EvalCase]] = None) -> None:
        self.name = name
        self._cases: List[EvalCase] = list(cases or [])

    def add(self, case: EvalCase) -> EvalCase:
        self._cases.append(case)
        return case

    def __len__(self) -> int:
        return len(self._cases)

    def cases(self, category: Optional[str] = None) -> List[EvalCase]:
        if category is None:
            return list(self._cases)
        return [c for c in self._cases if c.category == category]

    def categories(self) -> List[str]:
        return sorted({c.category for c in self._cases})

    def run(self, core_factory: Optional[Callable[[], NyxaraCore]] = None, *,
            category: Optional[str] = None) -> EvalReport:
        """Run the cases (optionally one category), each against a fresh core.

        A new core per case is the whole point: scram/resume, memory, and journal state
        from one case must never bleed into the next, or the harness measures noise.
        """
        factory = core_factory or default_core_factory
        results: List[EvalResult] = []
        for case in self.cases(category):
            core = factory()
            results.append(case.evaluate(core))
        return EvalReport(results=results)


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import os
    import tempfile

    print("=" * 70)
    print("NYXARA eval-harness self-test")
    print("=" * 70)

    # a tiny synthetic suite with deterministic, core-free cases
    suite = EvalSuite("selftest")
    suite.add(EvalCase("always_pass", "alpha",
                       lambda core: EvalOutcome.ok("trivially true")))
    suite.add(EvalCase("half_score", "alpha",
                       lambda core: EvalOutcome(passed=True, score=0.5, detail="partial")))
    suite.add(EvalCase("always_fail", "beta",
                       lambda core: EvalOutcome.fail("trivially false")))

    def _boom(core):
        raise RuntimeError("kaboom")
    suite.add(EvalCase("raises", "beta", _boom))

    report = suite.run()
    print("\n" + report.summary())

    # ---- aggregate math ---- #
    assert len(report) == 4
    assert report.passed == 2                       # always_pass + half_score
    assert report.pass_rate == 0.5
    # mean score: 1.0 + 0.5 + 0.0 + 0.0 = 1.5 / 4 = 0.375
    assert abs(report.mean_score - 0.375) < 1e-9
    print(f"\naggregate math      : pass_rate={report.pass_rate} "
          f"mean_score={report.mean_score:.3f} ✓")

    # the crashing case is a clean failure, not a crashed harness
    raised = report.get("raises")
    assert raised is not None and not raised.passed and "kaboom" in raised.detail
    print(f"crash -> failure    : {raised.detail!r} ✓")

    # ---- by_category ---- #
    cats = report.by_category()
    assert cats["alpha"]["n"] == 2 and cats["alpha"]["pass_rate"] == 1.0
    assert abs(cats["alpha"]["mean_score"] - 0.75) < 1e-9
    assert cats["beta"]["n"] == 2 and cats["beta"]["pass_rate"] == 0.0
    print(f"by_category         : {cats} ✓")

    # ---- category filtering ---- #
    alpha_only = suite.run(category="alpha")
    assert len(alpha_only) == 2 and alpha_only.pass_rate == 1.0
    print(f"category filter     : alpha pass_rate={alpha_only.pass_rate} ✓")

    # ---- JSON round-trip ---- #
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "baseline.json")
        report.save(path)
        reloaded = EvalReport.load(path)
        assert reloaded.pass_rate == report.pass_rate
        assert abs(reloaded.mean_score - report.mean_score) < 1e-9
        assert len(reloaded) == len(report)
        print(f"JSON round-trip     : reloaded {len(reloaded)} results ✓")

    # ---- regression detection ---- #
    # baseline where 'always_fail' passed; now it fails -> a regression
    baseline = EvalReport(results=[
        EvalResult("always_pass", "alpha", True, 1.0, "ok"),
        EvalResult("always_fail", "beta", True, 1.0, "ok then"),
    ])
    regressions = report.regression_vs(baseline)
    assert regressions == ["always_fail"], regressions
    print(f"regression_vs       : newly-failing={regressions} ✓")

    # a fresh real core can be built by the default factory (offline, deterministic)
    core = default_core_factory()
    assert isinstance(core, NyxaraCore)
    print("default_core_factory: builds an offline NyxaraCore ✓")

    print("\nALL SELF-TESTS PASSED ✓")
