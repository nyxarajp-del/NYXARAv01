"""P2-C — post-turn self-evaluation is validated against MEASURED outcomes.

The quality score used to be self-asserted (a turn that ACTed scored ~1.0 regardless of whether
its action actually worked). It is now anchored to the turn's measured ground truth when one
exists — a tool that ran and either succeeded or failed — so self-evaluation tracks real
correctness rather than the bare fact of acting (capabilities #21 Reflection, #22 Self-Eval, #12).
Conversational turns carry no ground truth, so the honest heuristic still stands for them.
"""

from __future__ import annotations

from nyxara.mind.meta_intelligence import MetaIntelligence


class _Cand:
    def __init__(self, tool="calculate"):
        self.text = "compute 2+2"
        self.kind = "act"
        self.confidence = 0.9
        self.risk = type("R", (), {"value": 1})()
        self.rationale = "arithmetic"
        self.tool = tool


class _Result:
    def __init__(self, disp="ACT"):
        self.disposition = disp


def test_measured_success_scores_higher_than_failure():
    mi = MetaIntelligence()
    good = mi.evaluate_turn("q", _Cand(), _Result("ACT"), None, outcome_correct=True)
    bad = mi.evaluate_turn("q", _Cand(), _Result("REFUSE"), None, outcome_correct=False)
    assert good.quality_score > bad.quality_score
    assert good.quality_score >= 0.7
    assert bad.quality_score <= 0.3


def test_failure_marks_process_not_optimal():
    mi = MetaIntelligence()
    ev = mi.evaluate_turn("q", _Cand(), _Result("REFUSE"), None, outcome_correct=False)
    assert ev.process_was_optimal is False
    assert any("measured outcome" in step for step in ev.reasoning_chain)


def test_quality_tracks_measured_accuracy_over_turns():
    # feed a run that is right 80% of the time; the mean quality should land near 0.8, not ~1.0
    mi = MetaIntelligence()
    scores = []
    for i in range(10):
        ev = mi.evaluate_turn("q", _Cand(), _Result("ACT"), None,
                              outcome_correct=(i < 8))
        scores.append(ev.quality_score)
    mean_q = sum(scores) / len(scores)
    assert 0.7 <= mean_q <= 0.9        # tracks the real 0.8 accuracy, not a flat 1.0


def test_no_ground_truth_falls_back_to_heuristic():
    # a conversational turn (no tool) has no measured outcome → heuristic estimate, never faked
    mi = MetaIntelligence()
    ev = mi.evaluate_turn("hi", _Cand(tool=""), _Result("ACT"), None, outcome_correct=None)
    assert 0.0 <= ev.quality_score <= 1.0
