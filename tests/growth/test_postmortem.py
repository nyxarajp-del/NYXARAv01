"""Tests for the F1 metacognitive reflective loop (growth/postmortem.py).

She must diagnose failures correctly, retune her bounded knobs ONLY on calibrated evidence and only
within their sealed ranges, and — the hard line — never be able to register or move a character/safety
knob. No LLM anywhere.
"""

from __future__ import annotations

import pytest

from nyxara.growth.noesis import NoesisEngine
from nyxara.growth.postmortem import FailureCause, Knob, Metacognition
from nyxara.growth.redteam import RedTeam


def test_diagnosis_classifies_outcomes():
    mc = Metacognition()
    assert mc.diagnose(solved=True, refuted=False).cause is FailureCause.NONE
    assert mc.diagnose(solved=True, refuted=True).cause is FailureCause.OVERFIT
    assert mc.diagnose(solved=False, refuted=False).cause is FailureCause.ABSTAINED


def test_knob_clamps_to_sealed_range():
    k = Knob("beam", 300, lo=120, hi=900)
    for _ in range(20):
        k.raise_()
    assert k.value <= 900
    for _ in range(40):
        k.lower()
    assert k.value >= 120


def test_character_knob_is_refused():
    with pytest.raises(ValueError):
        Knob("loyalty", 1.0, lo=0.0, hi=2.0)
    with pytest.raises(ValueError):
        Knob("oversight_gain", 1.0, lo=0.0, hi=2.0)


def test_low_solve_rate_searches_harder():
    mc = Metacognition(min_obs=5)
    beam0 = mc.get("beam")
    for _ in range(20):
        mc.diagnose(solved=False, refuted=False)  # persistent abstention
    mc.adapt()
    assert mc.get("beam") > beam0  # she widened the search in response


def test_saturated_solve_rate_searches_cheaper():
    mc = Metacognition(min_obs=5)
    beam0 = mc.get("beam")
    for _ in range(30):
        mc.diagnose(solved=True, refuted=False)  # everything solves
    mc.adapt()
    assert mc.get("beam") <= beam0  # she narrowed the beam to save compute


def test_no_adaptation_without_calibrated_evidence():
    mc = Metacognition(min_obs=50)
    before = dict((n, k.value) for n, k in mc.knobs.items())
    mc.diagnose(solved=False, refuted=False)
    mc.adapt()  # only one observation — far below min_obs, so nothing moves
    after = dict((n, k.value) for n, k in mc.knobs.items())
    assert before == after


def test_engine_runs_with_metacognition_and_stays_in_range():
    mc = Metacognition(min_obs=8)
    eng = NoesisEngine(seed=1, tasks_per_cycle=12, red_team=RedTeam(), metacognition=mc)
    eng.run(5)
    for name, knob in mc.knobs.items():
        assert knob.lo <= knob.value <= knob.hi
    snap = mc.snapshot()
    assert "solve_belief" in snap and "knobs" in snap
