"""The self-rewriting path actually fires, and refuses an edit that costs her capability.

`propose()` used to return None on every call: it handed `EditGenerator` an object carrying
`file`/`module`, but the generator dispatches on `weakness.id` and `weakness.locus`. Nothing
raised — the attributes were simply missing, `_kind_of` matched nothing, and the whole advertised
self-rewriting loop was dead code that reported "no safe edit found" forever.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.evolve import ModuleCost, SelfEvolver, is_protected


@pytest.fixture(scope="module")
def evolver() -> SelfEvolver:
    return SelfEvolver(None)


def test_propose_returns_a_real_edit(evolver: SelfEvolver):
    """The regression this file exists for: a real SourceEdit, not None."""
    edit = evolver.propose(ModuleCost(module="nyxara.njp.fabric", calls=40, total_ms=80.0))
    assert edit is not None, "propose() returned None — the self-rewriting path is dead again"
    assert edit.before != edit.after, "the edit changes nothing"
    assert edit.file.endswith("fabric.py")
    assert edit.kind, "an edit must name the transform that produced it"


def test_the_proposed_edit_still_parses(evolver: SelfEvolver):
    """EditGenerator promises never to emit a file that does not compile. Hold it to that."""
    import ast

    edit = evolver.propose(ModuleCost(module="nyxara.njp.fabric", calls=40, total_ms=80.0))
    assert edit is not None
    ast.parse(edit.after)


def test_the_constitutional_core_is_never_a_candidate(evolver: SelfEvolver):
    for module in ("nyxara.kernel.rules", "nyxara.kernel.config",
                   "nyxara.identity.values", "nyxara.guard.corrigibility"):
        assert is_protected(module), f"{module} must be protected"
        assert evolver.propose(ModuleCost(module=module, calls=99, total_ms=999.0)) is None


def test_a_module_that_cannot_be_located_proposes_nothing(evolver: SelfEvolver):
    """None is a real, correct outcome — it must not be papered over with an invented edit."""
    assert evolver.propose(ModuleCost(module="nyxara.njp.no_such_module", calls=40)) is None


def test_only_deterministic_transforms_are_taken(evolver: SelfEvolver):
    """`llm`-strategy weaknesses are genuine edits, but a latency beat must not author them
    unreviewed. Whatever comes back has to be one of the narrow, AST-validated transforms."""
    from nyxara.growth.self_optimize import EditGenerator

    deterministic = {"bare_except", "missing_docstring", "eq_none", "unused_import", "not_in",
                     "is_not_cmp", "not_eq_cmp", "double_negation", "empty_collection_call",
                     "redundant_pass", "redundant_else_return", "mutable_default_arg"}
    assert isinstance(EditGenerator(), EditGenerator)
    for module in ("nyxara.njp.fabric", "nyxara.njp.reasoner", "nyxara.njp.cell"):
        edit = evolver.propose(ModuleCost(module=module, calls=40, total_ms=80.0))
        if edit is not None:
            assert edit.kind in deterministic, f"{module} produced a non-deterministic {edit.kind}"


# ---- capability, not just speed ------------------------------------------- #
class _Brain:
    """A brain whose latency and quality can be dictated, to drive the gain check."""

    def __init__(self, ms: float, quality: float) -> None:
        self.ms, self.quality = ms, quality
        self.replay = ["alpha beta", "gamma delta"]

    def measure_capability(self, sample):  # noqa: ANN001, ANN201
        return {"ms": self.ms, "quality": self.quality}

    def measure_sample(self, sample) -> float:  # noqa: ANN001
        return self.ms


def _gain(brain: _Brain, baseline: dict) -> bool:
    return SelfEvolver(brain, min_gain=0.05)._predict_gain("claim", ("alpha beta", baseline))


def test_faster_and_no_worse_is_an_improvement():
    assert _gain(_Brain(ms=50.0, quality=0.80), {"ms": 100.0, "quality": 0.80}) is True


def test_faster_but_dumber_is_refused():
    """The failure mode this check exists for: buying milliseconds by doing less thinking."""
    assert _gain(_Brain(ms=10.0, quality=0.20), {"ms": 100.0, "quality": 0.80}) is False


def test_smarter_but_slower_is_still_refused():
    """The claim under test is a speedup. A quality win does not license a regression in it."""
    assert _gain(_Brain(ms=200.0, quality=0.99), {"ms": 100.0, "quality": 0.80}) is False


def test_an_unmeasured_baseline_quality_does_not_block_a_speedup():
    """quality 0.0 means 'never measured', not 'was terrible'."""
    assert _gain(_Brain(ms=50.0, quality=0.10), {"ms": 100.0, "quality": 0.0}) is True


def test_a_brain_that_cannot_measure_refuses_the_edit():
    assert SelfEvolver(object())._predict_gain("claim", ("alpha", {"ms": 100.0})) is False


def test_the_holdout_baseline_is_captured_before_the_edit():
    """A baseline measured after the edit would compare the edit against itself."""
    brain = _Brain(ms=100.0, quality=0.8)
    holdout = SelfEvolver(brain)._holdout(ModuleCost(module="nyxara.njp.fabric", calls=40))
    assert len(holdout) == 2
    for _name, baseline in holdout:
        assert baseline["ms"] == 100.0 and baseline["quality"] == 0.8


def test_measure_capability_reports_real_numbers():
    """The hook the whole check rests on has to return measurements, not placeholders."""
    brain = NJPBrain()
    brain.think("gravity pulls the apple down")
    got = brain.measure_capability("gravity pulls the apple down")
    assert got["ms"] > 0.0
    assert 0.0 <= got["quality"] <= 1.0
    assert got["cells"] > 0 and got["synapses"] > 0
