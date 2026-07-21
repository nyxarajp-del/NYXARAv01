"""Tests for F2 — predictive processing / active inference (mind/active_inference.py).

She must predict a stream and report surprise, prefer the LESS predictable source when choosing where
to look (epistemic action), and — wired into Noēsis — steer practice toward the family she understands
least. No LLM anywhere.
"""

from __future__ import annotations

from nyxara.growth.noesis import SYNTHETIC_FAMILIES, NoesisEngine
from nyxara.mind.active_inference import (
    ActiveInference,
    EpistemicScheduler,
    GenerativeModel,
    Predictor,
)


def test_predictor_surprise_falls_on_a_predictable_stream():
    p = Predictor()
    first = p.observe(5.0)
    for _ in range(10):
        p.observe(5.0)          # perfectly predictable
    settled = p.observe(5.0)
    assert settled < first      # once learned, a constant stream stops surprising her


def test_predictor_is_surprised_by_a_jump():
    p = Predictor()
    for _ in range(10):
        p.observe(5.0)
    jump = p.observe(100.0)     # a sudden break in the pattern
    assert jump > 0.5


def test_generative_model_information_gain_prefers_the_noisy_source():
    gm = GenerativeModel()
    for i in range(12):
        gm.observe("stable", 3.0)
        gm.observe("noisy", (i * 7) % 13 - 6)   # erratic
    assert gm.expected_information_gain("noisy") > gm.expected_information_gain("stable")


def test_active_inference_acts_toward_the_uncertain_source():
    ai = ActiveInference()
    for i in range(12):
        ai.perceive("known", 1.0)
        ai.perceive("mystery", (i * 5) % 11)
    best, gain, rec = ai.epistemic_action(["known", "mystery"])
    assert best == "mystery"
    assert 0.0 <= gain <= 1.0
    assert rec is not None


def test_epistemic_scheduler_upweights_the_least_understood_family():
    sched = EpistemicScheduler()
    # "solid" is always solved (predictable competence); "shaky" flips (unpredictable)
    for i in range(12):
        sched.observe("solid", True)
        sched.observe("shaky", i % 2 == 0)
    w = sched.weights(["solid", "shaky"])
    assert w["shaky"] > w["solid"]
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_engine_runs_with_curiosity_and_still_compounds():
    sched = EpistemicScheduler()
    eng = NoesisEngine(seed=1, tasks_per_cycle=12, curiosity=sched)
    reports = eng.run(5)
    assert reports[-1].avg_solution_size < reports[0].avg_solution_size
    # she has formed competence predictions over the real task families
    w = sched.weights(SYNTHETIC_FAMILIES)
    assert set(w) == set(SYNTHETIC_FAMILIES) and abs(sum(w.values()) - 1.0) < 1e-6
