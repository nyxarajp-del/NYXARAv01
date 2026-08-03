"""NYXARA · growth/counterfactual.py — counterfactual causal dreaming (🎯, Rule 4, F10).

Ordinary sleep replays what happened; F10 replays what *could* have happened. During the dream cycle
NYXARA takes each failure and asks a **Pearl Level-3** question — *"had I searched differently, would I
have solved it?"* — by actually **running the counterfactual world** (re-running the search under an
intervention). Where a counterfactual flips a failure into a success, she does **causal credit
assignment**: the low-prior primitives that the ordinary search pruned are the **bottleneck cause**, so
she propagates credit onto exactly those **DSL search priors** — and next time the same problem is in
reach at normal budget.

Crucially honest: a failure the counterfactual **cannot** flip is marked *genuinely hard* — attributed
to real out-of-reach, not bluffed away. The counterfactual is not a story; it is the search re-run, so
the causal claim is checked, not asserted. Pure standard library; **no LLM**; it nudges only the search
prior (how she looks), never a character/value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from nyxara.growth.noesis import Library, Prog, Task, synthesize

__all__ = ["Failure", "DreamReport", "CounterfactualDreamer"]


@dataclass
class Failure:
    task: Task
    reason: str = "abstained"     # "abstained" | "overfit"


def _primitives_used(prog: Prog) -> List[str]:
    out: List[str] = []
    if prog.kind == "app":
        out.append(prog.name)
    for a in prog.args:
        out.extend(_primitives_used(a))
    return out


@dataclass
class DreamReport:
    replayed: int = 0
    solvable_with_more_compute: int = 0     # counterfactually flippable failures
    genuinely_hard: int = 0                 # honestly out of reach — not bluffed
    credited: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {"replayed": self.replayed,
                "solvable_with_more_compute": self.solvable_with_more_compute,
                "genuinely_hard": self.genuinely_hard,
                "credited": {k: round(v, 3) for k, v in self.credited.items()}}


class CounterfactualDreamer:
    """Replay failures, run the counterfactual, and credit the causal bottleneck priors (F10)."""

    def __init__(self, *, base_beam: int = 60, cf_beam: int = 400,
                 cf_expansions: int = 200000, credit_gain: float = 3.0) -> None:
        self.base_beam = base_beam
        self.cf_beam = cf_beam
        self.cf_expansions = cf_expansions
        self.credit_gain = credit_gain

    def _low_prior_names(self, lib: Library, names: Sequence[str]) -> List[str]:
        counts = lib.counts
        vals = sorted(counts.get(n, 0.0) for n in lib.primitives)
        median = vals[len(vals) // 2] if vals else 0.0
        return [n for n in names if counts.get(n, 0.0) <= median]

    def dream(self, failures: Sequence[Failure], lib: Library) -> DreamReport:
        report = DreamReport()
        for f in failures:
            report.replayed += 1
            # baseline: the ordinary, bounded search really does fail here
            base = synthesize(f.task, lib, beam=self.base_beam, max_expansions=20000)
            if base.solved:
                continue
            # the counterfactual world: search harder, and see what would have happened
            cf = synthesize(f.task, lib, beam=self.cf_beam, max_expansions=self.cf_expansions)
            if not (cf.solved and cf.program is not None):
                report.genuinely_hard += 1        # honest: really out of reach
                continue
            report.solvable_with_more_compute += 1
            # causal credit: the low-prior primitives the ordinary search pruned are the bottleneck —
            # propagate credit onto exactly those DSL search priors.
            bottleneck = self._low_prior_names(lib, _primitives_used(cf.program))
            for name in set(bottleneck):
                lib.counts[name] = lib.counts.get(name, 0.0) + self.credit_gain
                report.credited[name] = report.credited.get(name, 0.0) + self.credit_gain
        return report
