"""Two vetoes in front of the "provably better" gate — and what they exist to stop.

Every proof method in ``improvement_proof.py`` is guarded by "no regression", and ``regression``
reads exactly one battery. Counted::

    build_default_benchmark(): 29 tasks
    {'arithmetic': 12, 'logic': 4, 'algebra': 2, 'calculus': 2,
     'sequence': 2, 'percent': 2, 'date': 2, 'multi_step': 3}

That is the whole evidence base licensing a whole-file rewrite. A rewrite of the grounding layer —
extraction, canonicalisation, the epistemic states, the thing most of this package does — is
certified ``pareto-capability`` with the reason ``deterministic dominance proof`` on the strength of
29 sums it never touches. Meanwhile :mod:`nyxara.eval.intelligence` measures memorisation,
generalisation, recombination, causal prediction, self-correction, transfer and paraphrase, and the
prover was not reading it.

Both additions are **vetoes, not proof methods**: they withhold a certificate the methods would
otherwise grant and can never grant one themselves. Only A–D may say *better*. Adversarial
robustness has no battery here and no veto is invented for it.
"""
from __future__ import annotations

from nyxara.growth.improvement_proof import ImprovementProver


class _Result:
    def __init__(self, task_id: str, passed: bool) -> None:
        self.task_id, self.passed = task_id, passed


class _Report:
    """The duck-type the prover reads off a BenchmarkReport."""

    def __init__(self, tasks: dict) -> None:
        self._tasks = tasks
        self.results = [_Result(k, v) for k, v in tasks.items()]

    @property
    def mean_score(self) -> float:
        return sum(1.0 if v else 0.0 for v in self._tasks.values()) / max(1, len(self._tasks))

    def get(self, task_id: str):
        return next((r for r in self.results if r.task_id == task_id), None)

    def regression_vs(self, baseline: "_Report"):
        return [k for k, v in self._tasks.items() if baseline._tasks.get(k) and not v]


class _Stage:
    def __init__(self, stage: str, score: float) -> None:
        self.stage, self.score = stage, score


class _Curve:
    def __init__(self, stages: dict) -> None:
        self.stages = [_Stage(k, v) for k, v in stages.items()]


BEFORE = _Report({"t1": False, "t2": True})
AFTER = _Report({"t1": True, "t2": True})
SRC = dict(before_src="x = 1\n", after_src="x = 2\n", edit_kind="rewrite")
HELD = _Curve({"memorization": 1.0, "transfer": 1.0, "paraphrase": 1.0})


def _prove(**kw):
    return ImprovementProver().prove(before=BEFORE, after=AFTER, **SRC, **kw)


# --------------------------------------------------------------------------- #
# The gate still certifies what it should
# --------------------------------------------------------------------------- #
def test_a_real_capability_gain_still_certifies():
    certificate = _prove()
    assert certificate.better
    assert certificate.method == "pareto-capability"


def test_absent_evidence_vetoes_nothing():
    """A check that cannot be run must not read as a check that failed."""
    assert _prove(reasoning_before=None, reasoning_after=None,
                  teacher_before=None, teacher_after=None).better


# --------------------------------------------------------------------------- #
# Veto 1 — reasoning must not regress
# --------------------------------------------------------------------------- #
def test_an_arithmetic_gain_that_degrades_paraphrase_is_refused():
    worse = _Curve({"memorization": 1.0, "transfer": 1.0, "paraphrase": 0.4})
    certificate = _prove(reasoning_before=HELD, reasoning_after=worse)
    assert not certificate.better
    assert "paraphrase" in certificate.reason


def test_a_gain_that_degrades_transfer_is_refused():
    worse = _Curve({"memorization": 1.0, "transfer": 0.5, "paraphrase": 1.0})
    assert not _prove(reasoning_before=HELD, reasoning_after=worse).better


def test_reasoning_held_steady_does_not_block():
    certificate = _prove(reasoning_before=HELD, reasoning_after=HELD)
    assert certificate.better and certificate.method == "pareto-capability"


def test_reasoning_that_improves_does_not_block():
    better = _Curve({"memorization": 1.0, "transfer": 1.0, "paraphrase": 1.0})
    lower = _Curve({"memorization": 1.0, "transfer": 0.5, "paraphrase": 0.4})
    assert _prove(reasoning_before=lower, reasoning_after=better).better


def test_a_stage_that_disappeared_is_not_a_regression_in_it():
    """Comparing two different batteries is not a comparison."""
    fewer = _Curve({"memorization": 1.0})
    assert _prove(reasoning_before=HELD, reasoning_after=fewer).better


def test_an_unscored_stage_is_ignored():
    unscored = _Curve({"memorization": 1.0, "transfer": 1.0, "paraphrase": None})
    assert _prove(reasoning_before=HELD, reasoning_after=unscored).better


# --------------------------------------------------------------------------- #
# Veto 2 — the gain must survive the teacher being taken away
# --------------------------------------------------------------------------- #
def test_a_gain_that_exists_only_with_the_cortex_is_refused():
    certificate = _prove(teacher_before={"on": 0.5, "off": 0.5},
                         teacher_after={"on": 0.9, "off": 0.5})
    assert not certificate.better
    assert "cortex" in certificate.reason


def test_a_gain_that_survives_the_cortex_being_detached_passes():
    certificate = _prove(teacher_before={"on": 0.5, "off": 0.5},
                         teacher_after={"on": 0.9, "off": 0.8})
    assert certificate.better and certificate.method == "pareto-capability"


def test_an_improvement_without_the_cortex_at_all_passes():
    """The environment this usually runs in has no language model attached."""
    assert _prove(teacher_before={"on": 0.5, "off": 0.5},
                  teacher_after={"on": 0.5, "off": 0.8}).better


def test_a_half_supplied_teacher_pair_vetoes_nothing():
    assert _prove(teacher_before={"on": 0.5}, teacher_after={"on": 0.9}).better


# --------------------------------------------------------------------------- #
# A veto can only withhold, never grant
# --------------------------------------------------------------------------- #
def test_holding_reasoning_steady_cannot_certify_on_its_own():
    """No capability gain to begin with — the veto must not turn that into a pass."""
    flat = _Report({"t1": False, "t2": True})
    certificate = ImprovementProver().prove(
        before=flat, after=flat, **SRC, reasoning_before=HELD, reasoning_after=HELD,
        teacher_before={"on": 0.5, "off": 0.5}, teacher_after={"on": 0.9, "off": 0.9})
    assert not certificate.better
    assert certificate.method == "none"
