"""NYXARA · eval/generalization.py — does she generalize *herself*, or only via the LLM? (📐)

The critique this benchmark answers: *"NYXARA can only generalize as far as her 9B base model;
the scaffolding polishes output but never breaks the base model's ceiling."* The honest reply is
not a claim that the ceiling is gone — it is a **measurement**: on structurally-transferable
tasks, NYXARA's *own* faculties (the relational structure-mapper of :mod:`nyxara.mind.transfer`
and the first-principles law inducer of :mod:`nyxara.growth.open_world`) recover the answer
**with no LLM in the loop at all**. This module measures exactly that, so the claim "she
generalizes herself" is backed by a number, not an adjective.

Two graded suites, both fully offline (they run green under the CI mock — the whole point is that
they do not need a language model):

* **Structure transfer** — a held-out target domain is given only its *surface* relations; the
  engine must project the deeper (higher-order) relation by mapping from a known base domain. The
  projected candidate inference is graded against the truth the target was never told.
* **Law induction** — an unseen black-box "machine" is confronted cold; the open-world generalizer
  must recover its hidden law and validate on **fresh, never-probed inputs** (held-out accuracy).

The report also records a *base-only* baseline: the same tasks answered with no own-faculty help.
With the mock model that baseline is ~0, so the delta *is* the own-faculty contribution — the
generalization the scaffolding-only critique says cannot exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.mind.analogy import entity, relation
from nyxara.mind.lot import Predicate
from nyxara.mind.transfer import RelationalTransferEngine

__all__ = [
    "TransferTask",
    "GeneralizationResult",
    "GeneralizationReport",
    "build_structure_transfer_suite",
    "run_structure_transfer",
    "run_law_induction",
    "run_generalization_benchmark",
]


# --------------------------------------------------------------------------- #
# Structure-transfer suite
# --------------------------------------------------------------------------- #
@dataclass
class TransferTask:
    """A held-out target domain plus the deep inference the engine must *project* into it.

    ``target_relations`` are the only facts the engine is given; ``expected_tokens`` are
    substrings that a correct projected candidate inference must contain (e.g. the higher-order
    ``causes`` relation and the two target entities). The target is never told this inference —
    recovering it *is* the generalization."""

    name: str
    target_relations: List[Predicate]
    expected_tokens: Sequence[str]
    expected_base: Optional[str] = None


@dataclass
class GeneralizationResult:
    task: str
    solved: bool
    detail: str = ""


@dataclass
class GeneralizationReport:
    results: List[GeneralizationResult] = field(default_factory=list)
    baseline_solved: int = 0
    holdout_accuracy: Optional[float] = None

    @property
    def solved(self) -> int:
        return sum(1 for r in self.results if r.solved)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def accuracy(self) -> float:
        return self.solved / self.total if self.total else 0.0

    @property
    def own_faculty_delta(self) -> float:
        """Own-faculty accuracy minus the base-only baseline — the contribution the
        scaffolding-only critique says is impossible."""
        base = self.baseline_solved / self.total if self.total else 0.0
        return self.accuracy - base

    def to_dict(self) -> Dict[str, object]:
        return {
            "solved": self.solved, "total": self.total,
            "accuracy": round(self.accuracy, 3),
            "baseline_solved": self.baseline_solved,
            "own_faculty_delta": round(self.own_faculty_delta, 3),
            "holdout_accuracy": (None if self.holdout_accuracy is None
                                 else round(self.holdout_accuracy, 3)),
            "results": [{"task": r.task, "solved": r.solved, "detail": r.detail}
                        for r in self.results],
        }

    def render(self) -> str:
        lines = [f"Generalization: {self.solved}/{self.total} solved by her OWN faculties "
                 f"(accuracy {self.accuracy:.0%}), base-only baseline {self.baseline_solved}"
                 f"/{self.total} → own-faculty delta {self.own_faculty_delta:+.0%}"]
        if self.holdout_accuracy is not None:
            lines.append(f"Law-induction held-out accuracy: {self.holdout_accuracy:.0%}")
        for r in self.results:
            mark = "✓" if r.solved else "✗"
            lines.append(f"  {mark} {r.task}: {r.detail}")
        return "\n".join(lines)


def build_structure_transfer_suite() -> List[TransferTask]:
    """Held-out target domains, each solvable only by projecting structure from a seeded base."""
    e, r = entity, relation
    nucleus, electron = e("nucleus"), e("electron")
    charge = e("charge")
    borrower, lender = e("borrower"), e("lender")
    return [
        # atom ← solar_system: project the higher-order CAUSE the atom was never told
        TransferTask(
            "atom",
            [r("attracts", nucleus, electron), r("revolves_around", electron, nucleus)],
            expected_tokens=["causes", "nucleus", "electron"],
            expected_base="solar_system",
        ),
        # market credit ← supply_demand: project the clearing relation
        TransferTask(
            "credit_market",
            [r("raises", lender, charge), r("lowers", borrower, charge)],
            expected_tokens=["causes", "clears"],
            expected_base="supply_demand",
        ),
    ]


def run_structure_transfer(engine: Optional[RelationalTransferEngine] = None,
                           tasks: Optional[Sequence[TransferTask]] = None
                           ) -> GeneralizationReport:
    """Grade the engine's projected inferences against the held-out truth."""
    eng = engine if engine is not None else RelationalTransferEngine()
    suite = list(tasks) if tasks is not None else build_structure_transfer_suite()
    report = GeneralizationReport()
    for task in suite:
        tr = eng.generalize(task.name, target_relations=task.target_relations)
        solved = False
        detail = "no structure mapped (would defer to the LLM)"
        if tr is not None:
            inferred = " ".join(str(p) for p in tr.candidate_inferences).lower()
            solved = all(tok.lower() in inferred for tok in task.expected_tokens)
            if task.expected_base is not None and tr.base_domain != task.expected_base:
                solved = False
            detail = (f"base={tr.base_domain} score={tr.structural_score:.2f} "
                      f"projected={[str(p) for p in tr.candidate_inferences]}")
        report.results.append(GeneralizationResult(task.name, solved, detail))
    # base-only baseline: with no known-domain structure to map from, nothing is projected.
    empty = RelationalTransferEngine(store=_empty_store())
    for task in suite:
        tr = empty.generalize(task.name, target_relations=task.target_relations)
        if tr is not None and tr.candidate_inferences:
            report.baseline_solved += 1
    return report


def _empty_store():
    from nyxara.mind.transfer import DomainSchemaStore
    return DomainSchemaStore(schemas=[])


# --------------------------------------------------------------------------- #
# Law-induction suite (first-principles, no LLM) — reuses the open-world generalizer
# --------------------------------------------------------------------------- #
def run_law_induction(machines: Optional[Sequence[Tuple[str, Callable[[float], float], object]]]
                      = None, *, budget: int = 48) -> GeneralizationReport:
    """Confront unseen black-box machines cold and recover each hidden law, scoring on the
    open-world generalizer's held-out accuracy (fresh, never-probed inputs). No LLM anywhere."""
    from nyxara.growth.open_world import DomainSpec, OpenWorldGeneralizer

    if machines is None:
        machines = [
            ("affine", lambda x: 3 * x + 4, DomainSpec(1, "real")),
            ("modular", lambda x: x % 7, DomainSpec(1, "int", -10, 10)),
            ("quadratic", lambda x: 2 * x * x - x, DomainSpec(1, "real")),
        ]
    report = GeneralizationReport()
    accs: List[float] = []
    for name, fn, spec in machines:
        gen = OpenWorldGeneralizer(seed=7)
        acc = 0.0
        detail = "unmodelled"
        try:
            rep = gen.understand(fn, domain=spec, budget=budget, label=name)
            acc = float(getattr(rep, "holdout_accuracy", 0.0) or 0.0)
            law = getattr(rep, "law", "?")
            detail = f"law={law} holdout={acc:.0%}"
        except Exception as exc:  # noqa: BLE001 — a machine that defeats her is an honest ✗
            detail = f"failed: {exc}"
        accs.append(acc)
        report.results.append(GeneralizationResult(name, acc >= 0.8, detail))
    report.holdout_accuracy = sum(accs) / len(accs) if accs else None
    return report


def run_generalization_benchmark() -> Dict[str, object]:
    """Run both suites and return a combined, auditable report."""
    st = run_structure_transfer()
    li = run_law_induction()
    return {"structure_transfer": st.to_dict(), "law_induction": li.to_dict()}


# --------------------------------------------------------------------------- #
# CLI / self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA generalization benchmark — does she generalize HERSELF? (no LLM)")
    print("=" * 70)

    st = run_structure_transfer()
    print("\n[structure transfer]")
    print(st.render())
    assert st.solved >= 1, "at least one held-out domain must be solved by transfer alone"
    assert st.own_faculty_delta > 0, "own faculties must beat the base-only baseline"

    li = run_law_induction()
    print("\n[law induction]")
    print(li.render())
    assert li.holdout_accuracy is not None and li.holdout_accuracy >= 0.8, \
        "she must recover unseen laws and validate on held-out inputs"

    print("\nALL SELF-TESTS PASSED ✓")
