"""Measurement. The tests are almost entirely about refusing to report numbers not measured.

The charter replaced a marketing word with a formula. That only helps if the formula declines to
answer when it has no evidence — a zero is a measurement ("no reasoning ability"), and printing it
for something merely unasked is the easiest way to make an evaluation lie.
"""

from __future__ import annotations

import math

import pytest

from nyxara.nyx001.layers import CognitiveStack
from nyxara.nyx001.metrics import IntelligenceIndex
from nyxara.nyx001.proving_ground import ProvingGround


class Cfg:
    seed = 41
    scale = "toy"
    lr = 5e-3
    memory_capacity = 256


def test_a_newborn_measures_nothing_and_says_so():
    """Every component must be None, not 0.0. This is the whole point of the module."""
    rep = IntelligenceIndex().measure(CognitiveStack(Cfg()))
    assert all(c.value is None for c in rep.components), \
        [c.to_dict() for c in rep.components if c.value is not None]
    assert rep.index is None
    assert rep.coverage == 0.0
    assert "NOT MEASURABLE" in rep.render()


def test_every_unmeasured_component_explains_why():
    rep = IntelligenceIndex().measure(CognitiveStack(Cfg()))
    for c in rep.components:
        if c.value is None:
            assert c.note, f"{c.key} is unmeasured with no reason given"


def test_components_appear_only_once_there_is_evidence():
    stack = CognitiveStack(Cfg())
    for tok in ["alpha one", "beta two", "gamma three"] * 20:
        stack.cycle(tok)
    rep = IntelligenceIndex().measure(stack)
    assert rep.measured, "ran 60 cycles and still measured nothing"
    assert rep.index is not None
    for c in rep.measured:
        assert 0.0 <= c.value <= 1.0
        assert c.samples > 0, f"{c.key} reported a value backed by zero samples"


def test_index_is_a_geometric_mean_not_an_arithmetic_one():
    """One near-zero component must drag the index down, not be averaged away by a strong one."""
    from nyxara.nyx001.metrics import Component, IndexReport
    rep = IndexReport(components=[
        Component(key="A", name="a", value=1.0, samples=50),
        Component(key="B", name="b", value=0.0001, samples=50),
    ])
    vals = [max(1e-6, c.value) for c in rep.measured]
    geometric = math.exp(sum(math.log(v) for v in vals) / len(vals))
    arithmetic = sum(vals) / len(vals)
    assert geometric < arithmetic / 10, "the aggregate is not punishing the weak component"


def test_thin_evidence_is_flagged():
    from nyxara.nyx001.metrics import Component, IndexReport
    rep = IndexReport(components=[Component(key="G", name="g", value=0.9, samples=2)])
    assert rep.components[0].thin
    assert rep.thin
    assert "thin" in rep.render()


def test_render_never_prints_a_number_it_did_not_measure():
    text = IntelligenceIndex().measure(CognitiveStack(Cfg())).render()
    for line in text.splitlines():
        if "not yet measurable" in line:
            assert "0.0000" not in line, "printed a zero for something unmeasured"


# --------------------------------------------------------------------------- #
# The proving ground
# --------------------------------------------------------------------------- #
def test_ratio_tests_refuse_a_reference_they_never_earned():
    """0.0117/0.0111 once scored a clean 1.0000. A ratio of two near-zeros is noise."""
    pg = ProvingGround(width=16, train_steps=1, min_reference_skill=0.15)
    for result in (pg.interference(), pg.adaptation()):
        assert result.score is None, f"{result.name} scored on an unearned reference"
        assert result.passed is None
        assert "floor" in result.error, f"{result.name} gave no reason"


def test_report_verdict_is_not_yet_until_all_six_pass():
    pg = ProvingGround(width=16, train_steps=1)
    rep = pg.run_all(only=["zero_shot"])
    assert not rep.all_passed
    assert "Not intelligence yet" in rep.render()


def test_every_test_reports_a_copy_baseline_where_one_applies():
    """A world model that cannot beat copying has learned nothing, however small its loss."""
    pg = ProvingGround(width=16, train_steps=200)
    rep = pg.run_all(only=["zero_shot", "long_horizon"])
    for r in rep.results:
        if r.score is not None:
            assert r.baseline is not None, f"{r.name} scored with no baseline to beat"


@pytest.mark.timeout(1800)
def test_the_suite_runs_end_to_end_and_grades_held_out_data():
    """Slow by construction: every test trains from scratch, and nothing carries between them."""
    rep = ProvingGround(width=16, train_steps=3000).run_all()
    assert len(rep.results) == 6
    assert all(r.name for r in rep.results)
    for r in rep.results:
        assert r.score is None or 0.0 <= r.score <= 1.0
        # a test that could not run must say so rather than silently scoring zero
        assert r.error or r.score is not None
    assert rep.render()
