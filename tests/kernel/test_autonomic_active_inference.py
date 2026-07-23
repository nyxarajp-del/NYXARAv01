"""Stage B — the continuous active-inference tick.

The free-energy machinery used to run only inside a turn. These tests prove the background loop now
predicts every tick, measures surprise + a real Shannon-entropy over the belief state, and pre-empts
(probes the most-informative channel) when uncertainty spikes — all LLM-free and degrading honestly.
"""

from __future__ import annotations

from nyxara.mind.active_inference_loop import ContinuousActiveInference, InferenceReading


def test_belief_entropy_is_normalised_shannon():
    ai = ContinuousActiveInference()
    # uniform surprise across two channels → maximal (normalised) entropy
    assert abs(ai._belief_entropy({"a": 0.5, "b": 0.5}) - 1.0) < 1e-9
    # all surprise on one channel → zero entropy (uncertainty is localised)
    assert ai._belief_entropy({"a": 1.0, "b": 0.0}) == 0.0
    # a single channel / empty is defined as 0 (no distribution to be uncertain over)
    assert ai._belief_entropy({"a": 0.9}) == 0.0
    assert ai._belief_entropy({}) == 0.0


def test_first_observation_is_surprising_then_stabilises():
    ai = ContinuousActiveInference(surprise_threshold=0.4)
    r0 = ai.observe({"x": 10.0, "y": 20.0})
    assert isinstance(r0, InferenceReading)
    assert r0.channels == 2
    assert r0.preemptive is True          # a never-seen world is surprising → probe
    assert r0.probe in {"x", "y"}
    # feed the same values — her forward model learns them and surprise collapses
    for _ in range(12):
        r = ai.observe({"x": 10.0, "y": 20.0})
    assert r.free_energy < 0.4
    assert r.preemptive is False


def test_a_spike_after_calm_triggers_preemption_on_the_right_channel():
    ai = ContinuousActiveInference(surprise_threshold=0.4)
    for _ in range(15):
        ai.observe({"x": 10.0, "y": 20.0})   # settle
    r = ai.observe({"x": 10.0, "y": 20.0})
    assert r.preemptive is False
    # a sudden jump on y is a genuine prediction error → she pre-empts and points at y
    spike = ai.observe({"x": 10.0, "y": 999.0})
    assert spike.preemptive is True
    assert spike.probe == "y"
    assert ai.preemptions >= 1


def test_no_channels_is_a_silent_noop():
    ai = ContinuousActiveInference()
    r = ai.observe({})
    assert r.free_energy == 0.0 and r.preemptive is False and r.channels == 0


def test_non_numeric_values_are_skipped_not_fatal():
    ai = ContinuousActiveInference(surprise_threshold=0.4)
    r = ai.observe({"good": 1.0, "bad": "not-a-number", "none": None})
    assert r.channels == 1  # only the numeric channel counts


# --------------------------------------------------------------------------- #
# Integration: the AutonomicLoop drives the tick and acts pre-emptively.
# --------------------------------------------------------------------------- #
class _StubCuriosity:
    def __init__(self):
        self.ticks = 0

    def tick(self):
        self.ticks += 1
        return {"question": "why?"}  # a real curiosity step happened


class _StubCore:
    """A minimal core: only the attributes the code-mode tick reaches for exist."""

    def __init__(self):
        self.active_curiosity = _StubCuriosity()


def _make_loop(core, ai):
    from nyxara.kernel.autonomic import AutonomicLoop
    return AutonomicLoop(core=core, decision_mode="code", active_inference=ai)


def test_loop_preemption_runs_a_gated_epistemic_action():
    core = _StubCore()
    ai = ContinuousActiveInference(surprise_threshold=0.4)  # first tick will pre-empt
    loop = _make_loop(core, ai)
    summary = loop._decide_and_act_once()
    assert summary["preemptive"] is True
    # the pre-emption produced a real self-directed action (active curiosity), not a no-op
    assert summary.get("preemptive_action") == "curiosity"
    assert core.active_curiosity.ticks >= 1
    assert loop.preemptions >= 1
    assert summary["productive"] is True
    # and it is observable in the loop report
    rep = loop.report()
    assert rep["preemptions"] >= 1
    assert "active_inference" in rep


def test_loop_stays_quiet_when_state_is_predictable():
    core = _StubCore()
    ai = ContinuousActiveInference(surprise_threshold=0.9)  # very high bar → rarely pre-empts
    loop = _make_loop(core, ai)
    # run several ticks; internal telemetry is stable so surprise stays under the high bar
    last = None
    for _ in range(6):
        last = loop._decide_and_act_once()
    assert last["preemptive"] is False
