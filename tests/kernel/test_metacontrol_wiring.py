"""First-class metacognition is wired into the live turn loop (mind/metacontrol.py).

Before this wiring the deep-reasoning ladder always climbed to its ceiling and nothing decided
*upfront* how much compute a turn deserved. These tests prove the loop is now real end to end:

* the core builds a MetacognitiveController and plans compute BEFORE the reasoner runs;
* an easy turn is allocated the 1-forward-pass fast path, a hard turn a deep budget;
* every finished turn records its outcome back into the calibrator (the loop closes);
* the reporter exposes the calibration state; the feature flag genuinely disables it all;
* the config validator rejects inverted difficulty bands.

All hermetic: TEST-suite environment, mock provider, no network.
"""

from __future__ import annotations

import pytest

from nyxara.kernel.orchestrator import NyxaraCore

EASY = "hello"
HARD = ("prove that every finite division ring is commutative, deriving the theorem "
        "from first principles with a rigorous argument")


def test_core_builds_metacontrol_and_plans_before_reasoning():
    nyx = NyxaraCore()
    assert nyx.metacontrol is not None
    nyx.process(EASY)
    plan = nyx.metacontrol.last_plan
    assert plan is not None and plan.estimate is not None


def test_outcome_recorded_every_turn():
    """One turn, one sample — asserted as a delta, because a fresh core is not a blank one.

    `MetacognitiveController` restores `metacontrol.json` on construction and saves it every
    `persist_every` records (5 by default), and every test in a run shares one `NYXARA_HOME`. So
    once any earlier test has taken enough turns, a newly-built core wakes up already calibrated
    — measured at 15 samples after `test_autonomic.py`. That is the behaviour this repo wants in
    production, where remembering how well-judged her own effort estimates were is the entire
    point of persisting them, and it means the absolute count says nothing about wiring. The
    increment does.
    """
    nyx = NyxaraCore()
    start = nyx.metacontrol.calibrator.samples
    nyx.process(EASY)
    assert nyx.metacontrol.calibrator.samples == start + 1
    nyx.process("what is the capital of France?")
    assert nyx.metacontrol.calibrator.samples == start + 2
    # the consumed plan never leaks into the next turn
    assert nyx._last_compute_plan is None


def test_easy_turn_allocates_less_than_hard_turn():
    nyx = NyxaraCore()
    nyx.process(EASY)
    easy_plan = nyx.metacontrol.last_plan
    nyx.process(HARD)
    hard_plan = nyx.metacontrol.last_plan
    assert easy_plan.entry_rung == 0            # one forward pass
    assert hard_plan.entry_rung > easy_plan.entry_rung
    assert hard_plan.max_seconds > easy_plan.max_seconds


def test_reporter_exposes_metacontrol_section():
    nyx = NyxaraCore()
    nyx.process(EASY)
    provider = nyx.reporter._providers.get("metacontrol")
    assert provider is not None                 # the live section is registered
    report = provider()
    assert report.get("calibration", {}).get("samples", 0) >= 1
    assert report.get("last_plan") is not None


def test_feature_flag_disables_metacontrol(monkeypatch):
    from nyxara.kernel.config import reload_settings
    monkeypatch.setenv("NYXARA_FEATURES__METACOGNITIVE_CONTROL", "false")
    reload_settings()
    try:
        nyx = NyxaraCore()
        assert nyx.metacontrol is None
        result = nyx.process(EASY)              # graceful degradation: the turn still completes
        assert result is not None
    finally:
        monkeypatch.delenv("NYXARA_FEATURES__METACOGNITIVE_CONTROL")
        reload_settings()


def test_env_override_disables_metacontrol(monkeypatch):
    from nyxara.kernel.config import NyxaraSettings
    monkeypatch.setenv("NYXARA_METACONTROL__ENABLED", "0")
    assert NyxaraSettings().metacontrol.enabled is False


def test_config_rejects_inverted_bands():
    from nyxara.kernel.config import MetaControlConfig
    with pytest.raises(ValueError):
        MetaControlConfig(easy_below=0.9, moderate_below=0.5, hard_below=0.8)


def test_recursive_improve_respects_fast_path():
    from nyxara.mind.metacontrol import ComputeBudget

    class _Improver:
        def __init__(self):
            self.called = False

        def improve(self, stimulus, candidate, n_iterations=5):
            self.called = True
            return candidate

    nyx = NyxaraCore()
    imp = _Improver()
    nyx.recursive_improver = imp
    from nyxara.kernel.orchestrator import Candidate
    cand = Candidate(text="hi", kind="respond")
    nyx._last_compute_plan = ComputeBudget(max_rung=1, samples=1, max_seconds=5.0, entry_rung=0)
    out = nyx._recursive_improve("hello", cand)
    assert out is cand and imp.called is False  # one forward pass means one forward pass
    nyx._last_compute_plan = None
    nyx._recursive_improve("hello", cand)
    assert imp.called is True                   # without a fast-path plan it runs as before
