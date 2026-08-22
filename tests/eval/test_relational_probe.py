"""The control that turned a promising result into a negative one.

Training the gradient head on ``(subject, predicate) → object`` instead of turn-succession looks
like it works: trained pairs come back at rank 1 and held-out members score MRR 0.521, comfortably
above the 0.202 a uniform guess would give. Then the control runs — subjects the brain has never
seen, asked the same questions — and scores 0.500. The subject contributed 0.021.

Taught the ``is_a`` link as well, so two hops are composable in principle, it gets worse rather
than better: 0.233 against a control of 0.583. Teaching ``sparrow is_a bird`` trains that row to
predict *bird*, so asking ``(sparrow, requires)`` pushes toward ``bird`` and away from ``water``.
The head superimposes; it does not compose.

These tests pin the probe's *discipline* rather than the numbers, which move with the seed. What
must not change is that the control is always reported, and that the verdict is read off the gap
between the two rather than off the held-out score alone.
"""
from __future__ import annotations

import pytest

from nyxara.eval.relational import RelationalProbe, probe
from nyxara.njp.learn import available


pytestmark = pytest.mark.skipif(not available(), reason="gradients need numpy")


def test_the_verdict_is_read_off_the_gap_not_the_held_out_score():
    """A held-out score on its own is uninterpretable — the object prior alone looks respectable."""
    flattering = RelationalProbe(held_mrr=0.9, control_mrr=0.9)
    assert not flattering.generalises

    real = RelationalProbe(held_mrr=0.5, control_mrr=0.2)
    assert real.generalises


def test_a_small_gap_is_not_a_capability():
    """With a vocabulary this size a couple of rank positions is noise."""
    assert not RelationalProbe(held_mrr=0.52, control_mrr=0.50).generalises


def test_a_negative_gap_is_not_a_capability():
    assert not RelationalProbe(held_mrr=0.23, control_mrr=0.58).generalises


def test_the_probe_always_reports_the_control():
    report = probe(epochs=8)
    assert report.control_mrr >= 0.0
    assert "control" in report.render()


def test_it_memorises_what_it_was_trained_on():
    """The head is not broken — it fits the training pairs. That is what makes the result a
    result rather than a failure to train."""
    report = probe(epochs=40)
    assert report.trained_mrr > 0.8


def test_the_measured_conclusion_still_holds():
    """If this ever fails, the encoding changed and the docstring's reading needs rewriting."""
    report = probe(epochs=40)
    assert not report.generalises, (
        "the subject now contributes — re-run the is_a variant and update eval/relational.py")


def test_teaching_is_a_does_not_rescue_it():
    report = probe(epochs=40, teach_is_a=True)
    assert report.taught_is_a
    assert not report.generalises


def test_the_render_names_the_cause_when_it_fails():
    report = probe(epochs=8)
    if not report.generalises:
        rendered = report.render()
        assert "blake2b" in rendered
        assert "orthogonal representations" in rendered


def test_the_report_is_serialisable():
    row = probe(epochs=8).to_dict()
    assert set(row) >= {"held_mrr", "control_mrr", "gap", "generalises"}
