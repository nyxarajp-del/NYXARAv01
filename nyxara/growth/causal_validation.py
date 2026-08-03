"""NYXARA · growth/causal_validation.py — check the causal graph against a world that cannot lie (⚖).

Everything else in the causal stack is validated against something she inferred: a permutation
test on her own event stream, a NOTEARS fit cross-checking a symbolic verdict, a mechanism scored
by its own R². All of that is honest, and none of it answers the question that actually matters —
*is the graph right?* A self-improving system that only ever checks itself against itself is
grading its own homework, and :mod:`~nyxara.eval.datasets` already says so in as many words about
the benchmark side.

:mod:`~nyxara.sim` is the way out. Those worlds are deterministic and each obeys a **closed-form
law that is true by construction**: an RC circuit's time constant *is* ``R·C``, a wave on a string
*does* travel at ``√(T/μ)``, a gas's wall force *is* proportional to ``n·T``. So the structure is
known before she looks. Sample the inputs, run the world, hand the resulting table to
:func:`~nyxara.growth.dataset_causal.learn_causal_graph`, and compare what came back with what is
true. Nothing here is an opinion.

Both kinds of error are counted, and the second is the one worth having:

* a **missed** edge — a real dependence she did not find (she was too conservative);
* a **spurious** edge — a dependence she reported that the world does not have.

Spurious edges are the failure that matters, because that is what a confident wrong answer is made
of. The report keeps them separate rather than folding both into one accuracy number, since a
graph that finds everything and invents nothing is a very different object from one that scores
the same by doing both.

The inputs are sampled **independently** on purpose. That makes every input⊥input pair a real
negative control: any edge the learner draws between two inputs is, by construction, something it
made up.

Depends on ``sim`` (the worlds) and ``growth/dataset_causal`` (the learner). Nothing imports back.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["GroundTruthCase", "ValidationReport", "CASES", "validate_case", "validate_all"]


@dataclass
class GroundTruthCase:
    """One world with a known law: how to sample it, and which edges are genuinely there."""

    name: str
    law: str                                    # the closed-form truth, for the report
    inputs: Dict[str, Tuple[float, float]]      # variable -> (low, high) sampling range
    output: str
    run: Callable[[Dict[str, float]], Optional[float]]
    #: Inputs the output does not actually depend on. Empty for most worlds; when present these
    #: are the sharpest negative controls available, because the world provably ignores them.
    inert: Tuple[str, ...] = ()

    def sample(self, n: int, *, seed: int = 0) -> List[Dict[str, float]]:
        """Draw ``n`` independent parameter sets and run the world on each.

        Independence is the point: it makes every input⊥input pair a negative control the learner
        has to *not* find an edge in."""
        rng = random.Random(seed)
        rows: List[Dict[str, float]] = []
        for _ in range(max(0, n)):
            row = {name: rng.uniform(low, high) for name, (low, high) in self.inputs.items()}
            try:
                value = self.run(dict(row))
            except Exception:  # noqa: BLE001 — a world that refuses a sample simply skips it
                continue
            if value is None or not math.isfinite(float(value)):
                continue
            row[self.output] = float(value)
            rows.append(row)
        return rows

    @property
    def columns(self) -> List[str]:
        return list(self.inputs) + [self.output]

    @property
    def true_edges(self) -> set:
        """The edges the world genuinely has: every non-inert input drives the output."""
        return {(name, self.output) for name in self.inputs if name not in self.inert}


@dataclass
class ValidationReport:
    """What the learner got right, what it missed, and — the one that matters — what it invented."""

    case: str = ""
    law: str = ""
    rows: int = 0
    orientation: str = ""
    found: set = field(default_factory=set)
    expected: set = field(default_factory=set)
    missed: set = field(default_factory=set)
    spurious: set = field(default_factory=set)
    notes: List[str] = field(default_factory=list)

    @property
    def recovered(self) -> bool:
        """Exactly the true structure: nothing missed, and nothing invented.

        Stated as ``found == expected`` rather than "no missed and no spurious", because those two
        are also both empty when the run bailed before learning anything — which would report a
        world it never looked at as perfectly recovered."""
        return bool(self.expected) and self.found == self.expected

    @property
    def invented_nothing(self) -> bool:
        """The weaker but more important property — she may be shy, she may not fabricate."""
        return not self.spurious

    def summary(self) -> str:
        verdict = ("recovered exactly" if self.recovered
                   else "invented nothing, missed some" if self.invented_nothing
                   else "INVENTED EDGES")
        lines = [f"· {self.case} ({self.law}) — {verdict}; {self.rows} row(s), "
                 f"orientation {self.orientation or 'n/a'}"]
        if self.missed:
            lines.append("    missed:   " + ", ".join(f"{c}→{e}" for c, e in sorted(self.missed)))
        if self.spurious:
            lines.append("    INVENTED: " + ", ".join(f"{c}→{e}" for c, e in sorted(self.spurious)))
        lines.extend(f"    · {n}" for n in self.notes)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"case": self.case, "law": self.law, "rows": self.rows,
                "orientation": self.orientation,
                "found": sorted(list(self.found)), "expected": sorted(list(self.expected)),
                "missed": sorted(list(self.missed)), "spurious": sorted(list(self.spurious)),
                "recovered": self.recovered, "invented_nothing": self.invented_nothing,
                "notes": list(self.notes)}


# --------------------------------------------------------------------------- #
# the cases — each law is true by construction, not by anything she inferred
# --------------------------------------------------------------------------- #
def _rc(row: Dict[str, float]) -> Optional[float]:
    from nyxara.sim import RCCircuit
    return RCCircuit().measure_time_constant(row["resistance"], row["capacitance"])


def _wave(row: Dict[str, float]) -> Optional[float]:
    from nyxara.sim import WaveString
    return WaveString().measure_speed(row["tension"], row["mu"])


def _gas(row: Dict[str, float]) -> Optional[float]:
    from nyxara.sim import GasBox
    return GasBox().measure_wall_force(int(row["n_particles"]), row["temperature"], steps=600)


CASES: Tuple[GroundTruthCase, ...] = (
    GroundTruthCase(
        name="rc_circuit", law="tau = R * C",
        inputs={"resistance": (1.0, 10.0), "capacitance": (1.0, 10.0)},
        output="tau", run=_rc),
    GroundTruthCase(
        name="wave_string", law="v = sqrt(T / mu)",
        inputs={"tension": (1.0, 20.0), "mu": (0.5, 5.0)},
        output="speed", run=_wave),
    GroundTruthCase(
        name="gas_box", law="F ~ n * T",
        inputs={"n_particles": (20.0, 120.0), "temperature": (0.5, 5.0)},
        output="wall_force", run=_gas),
)


def validate_case(case: GroundTruthCase, *, n: int = 120, seed: int = 0,
                  declare_order: bool = True) -> ValidationReport:
    """Run one world, learn its graph from the numbers, and compare with the law. Never raises.

    ``declare_order`` supplies the inputs-before-output ordering, which is not cheating: it is the
    one thing a static table genuinely cannot know, and it is exactly what the Master supplies with
    ``--order`` in real use. Turn it off to test the unaided NOTEARS path."""
    from nyxara.growth.dataset_causal import learn_causal_graph

    report = ValidationReport(case=case.name, law=case.law, expected=case.true_edges)
    rows = case.sample(n, seed=seed)
    report.rows = len(rows)
    if len(rows) < 8:
        report.notes.append("world produced too few usable samples to judge")
        return report

    order = case.columns if declare_order else None
    learned = learn_causal_graph(case.columns, rows, order=order, seed=seed)
    report.orientation = learned.orientation
    report.found = {(cause, effect) for cause, effect, _w in learned.edges}
    report.missed = report.expected - report.found
    report.spurious = report.found - report.expected
    for note in learned.notes:
        report.notes.append(note)
    return report


def validate_all(*, n: int = 120, seed: int = 0,
                 cases: Optional[Sequence[GroundTruthCase]] = None) -> List[ValidationReport]:
    """Validate every ground-truth world and return one report each."""
    return [validate_case(case, n=n, seed=seed) for case in (cases if cases is not None else CASES)]
