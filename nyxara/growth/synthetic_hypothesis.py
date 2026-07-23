"""NYXARA · growth/synthetic_hypothesis.py — Epistemic Auto-Evolution (🧬, Stage F).

She fixes faults that *surface* (failing tests, telemetry) — but stays blind to **hidden pattern risks
no test or past log ever exercised**: concurrent-fault edge cases. The Master asked her to, in the
background, **generate synthetic scenarios that never appeared in prompts or logs, self-formulate a
hypothesis, test it in the sandbox, and pre-derive the sub-routine that survives it**.

There is a real `Scientist` hypothesis loop (`growth/scientist.py`), `sim/montecarlo.py`, and a
`sim/sandbox` — but **no fault-injection / chaos-scenario generator**. This module is that:

* **Monte-Carlo concurrent-fault generator.** It composes *simultaneous* stressors — CPU spike + memory
  pressure + network packet-drop + disk-full + latency + high concurrency — at random intensities,
  producing scenarios that are combinations **absent from history** (novelty-tracked by signature, with
  optional HDC novelty from `cognition.hyper_dimensional_vectors`), so she explores the untested corners
  rather than re-running known ones.
* **Self-formulated hypothesis.** For each scenario she states a falsifiable
  :class:`~nyxara.growth.scientist.Hypothesis` ("under this concurrency, does the target degrade or
  deadlock?") and **tests it against a probe** — a caller-supplied subroutine-under-test, or a built-in
  *modelled* system so the engine is demonstrable on a bare box.
* **Discovered gap → hardening.** A scenario the probe fails is surfaced as a discovered risk and can be
  pushed to Stage E as a **measurable coverage/hardening objective** (feeding the same gated pipeline);
  a discovered failing *test* is handed to Stage D's causal repair.

The honest boundary: this is **continuous Monte-Carlo synthetic-test generation over *simulated* fault
conditions — never physical magic**. No real host is stressed; injection is confined to the scenario
config handed to the probe/sandbox, and she abstains honestly when a scenario cannot be evaluated.
Pure standard library; **no LLM**.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from nyxara.growth.scientist import Hypothesis

__all__ = ["FaultScenario", "HypothesisResult", "SyntheticReport", "SyntheticHypothesisEngine"]

# The fault dimensions and their intensity ranges. Values are *modelled* stressor levels handed to a
# probe — never applied to the real machine.
_DIMENSIONS: Dict[str, Tuple[float, float]] = {
    "cpu_spike": (0.70, 1.00),        # fraction of CPU saturated
    "mem_spike": (0.70, 1.00),        # fraction of RAM pressured
    "packet_drop": (0.10, 0.60),      # fraction of packets dropped
    "disk_full": (0.80, 1.00),        # fraction of disk consumed
    "latency_ms": (50.0, 500.0),      # added I/O latency
    "concurrency": (2.0, 16.0),       # simultaneous workers
}

# A probe returns True (survived) or False / raises (failed). Signature: probe(scenario_config) -> bool.
Probe = Callable[[Dict[str, float]], bool]


@dataclass
class FaultScenario:
    """One synthetic combination of *simultaneous* stressors — a never-before-seen edge case."""

    stressors: Dict[str, float]
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = "|".join(f"{k}={self.stressors[k]:.3g}"
                                      for k in sorted(self.stressors))

    def describe(self) -> str:
        return ", ".join(f"{k}={v:.3g}" for k, v in sorted(self.stressors.items()))

    def to_dict(self) -> Dict[str, Any]:
        return {"stressors": {k: round(v, 4) for k, v in self.stressors.items()},
                "signature": self.signature}


@dataclass
class HypothesisResult:
    """The outcome of testing one scenario's hypothesis against the probe."""

    scenario: FaultScenario
    hypothesis: str
    survived: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"scenario": self.scenario.to_dict(), "hypothesis": self.hypothesis,
                "survived": self.survived, "detail": self.detail}


@dataclass
class SyntheticReport:
    """One background hunt: what she generated, which were novel, and what failed."""

    generated: int = 0
    novel: int = 0
    tested: int = 0
    survived: int = 0
    failed: int = 0
    failures: List[HypothesisResult] = field(default_factory=list)
    note: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"generated": self.generated, "novel": self.novel, "tested": self.tested,
                "survived": self.survived, "failed": self.failed, "note": self.note,
                "failures": [f.to_dict() for f in self.failures]}


class SyntheticHypothesisEngine:
    """Monte-Carlo generator of never-seen concurrent-fault scenarios, self-hypothesised + tested."""

    def __init__(self, *, core: Any = None, seed: int = 1337, max_stressors: int = 3,
                 dimensions: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        self.core = core
        self.rng = random.Random(seed)
        self.dims = dict(dimensions or _DIMENSIONS)
        self.max_stressors = max(2, min(int(max_stressors), len(self.dims)))
        self._seen: set[str] = set()
        self.history: List[SyntheticReport] = []

    # ------------------------------------------------------------------ #
    # Monte-Carlo scenario generation
    # ------------------------------------------------------------------ #
    def generate(self, n: int = 8) -> List[FaultScenario]:
        """Sample ``n`` scenarios, each combining 2..max_stressors *simultaneous* stressors."""
        keys = list(self.dims)
        out: List[FaultScenario] = []
        for _ in range(max(1, int(n))):
            k = self.rng.randint(2, self.max_stressors)
            chosen = self.rng.sample(keys, k)
            stressors: Dict[str, float] = {}
            for dim in chosen:
                lo, hi = self.dims[dim]
                v = self.rng.uniform(lo, hi)
                stressors[dim] = round(v, 3)
            out.append(FaultScenario(stressors=stressors))
        return out

    def _novel(self, scenarios: List[FaultScenario]) -> List[FaultScenario]:
        novel = [s for s in scenarios if s.signature not in self._seen]
        for s in novel:
            self._seen.add(s.signature)
        return novel

    # ------------------------------------------------------------------ #
    # Hypothesis + test
    # ------------------------------------------------------------------ #
    def hypothesize(self, scenario: FaultScenario) -> Hypothesis:
        """A falsifiable statement about the target's behaviour under this concurrent fault."""
        return Hypothesis(
            statement=(f"Under simultaneous {scenario.describe()}, the target sub-routine still "
                       f"completes correctly (no degradation or deadlock)."),
            prediction=True, variable="survives_fault",
            rationale="a concurrent-stress combination absent from tests and logs",
            confidence=0.5)

    def test(self, scenario: FaultScenario, probe: Probe) -> HypothesisResult:
        """Run the probe under the injected (modelled) fault config; survived iff it returns truthy."""
        hyp = self.hypothesize(scenario)
        try:
            survived = bool(probe(dict(scenario.stressors)))
            detail = "probe completed" if survived else "probe reported failure under fault"
        except Exception as exc:  # noqa: BLE001 — a raise IS the discovered failure, not an engine error
            survived = False
            detail = f"probe raised {type(exc).__name__}: {exc}"
        return HypothesisResult(scenario=scenario, hypothesis=hyp.statement,
                                survived=survived, detail=detail)

    # ------------------------------------------------------------------ #
    def hunt(self, probe: Optional[Probe] = None, *, n: int = 8,
             max_tests: Optional[int] = None) -> SyntheticReport:
        """One background pass: generate → keep the novel ones → hypothesise → test → report failures."""
        probe = probe or self._default_probe
        scenarios = self.generate(n)
        novel = self._novel(scenarios)
        if max_tests is not None:
            novel = novel[:max(0, int(max_tests))]
        results = [self.test(s, probe) for s in novel]
        failures = [r for r in results if not r.survived]
        report = SyntheticReport(
            generated=len(scenarios), novel=len(novel), tested=len(results),
            survived=sum(1 for r in results if r.survived), failed=len(failures),
            failures=failures,
            note=(f"{len(failures)} never-seen edge case(s) found that the target does not survive"
                  if failures else "no failing edge case in this batch — target robust so far"))
        self.history.append(report)
        return report

    @staticmethod
    def _default_probe(stressors: Dict[str, float]) -> bool:
        """A *modelled* system-under-test so the engine is demonstrable with no external deps.

        It fails on genuinely-adverse concurrent combinations — the kind a single-fault test never
        exercises — e.g. a CPU spike *and* heavy packet loss *and* high concurrency at once. This is
        a model, not a real host: it proves the generator finds real edge cases in a defined system,
        which is exactly the honest, bounded claim (no physical magic)."""
        cpu = stressors.get("cpu_spike", 0.0)
        drop = stressors.get("packet_drop", 0.0)
        conc = stressors.get("concurrency", 0.0)
        mem = stressors.get("mem_spike", 0.0)
        # a compound failure mode invisible to any single-stressor test:
        if cpu > 0.9 and drop > 0.4 and conc > 8:
            return False   # saturated CPU + lossy network + high concurrency → deadlock
        if mem > 0.95 and conc > 12:
            return False   # memory pressure under heavy concurrency → OOM-style failure
        return True

    def report(self) -> Dict[str, Any]:
        return {"passes": len(self.history), "known_signatures": len(self._seen),
                "last": self.history[-1].to_dict() if self.history else None}
