"""Tests for the RSIE full-wire: the intelligence directive becomes a real cross-system action.

Hermetic and offline: a fake core (with research/investigation queues) and a fake growth engine
stand in for the live orchestrator, so dispatch is verified without running a real foundry forge
or applying any source edits.
"""

from __future__ import annotations

from typing import List

from nyxara.growth.recursive_improvement import RecursiveSelfImprovement, SelfImprovementReport
from nyxara.kernel.config import get_settings


class _FakeCore:
    """A minimal core exposing the two queues the idle scientist/researcher drain."""
    def __init__(self):
        self._research_queue: List[str] = []
        self._investigation_queue: List[str] = []
        self.oversight = None


class _FakeResult:
    def __init__(self, promoted: bool):
        self.promoted = promoted


class _FakeEngine:
    """Stands in for GrowthEngine.improve_self (the gauntlet-gated foundry forge)."""
    def __init__(self, results):
        self._results = results
        self.calls = 0

    def improve_self(self, *, generations: int = 1):
        self.calls += 1
        return self._results


def _rsi(core=None, engine=None) -> RecursiveSelfImprovement:
    s = get_settings().model_copy(deep=True)
    s.self_improvement.enable_directive_dispatch = True
    s.self_improvement.enable_improvement_ledger = False   # isolate dispatch from credit here
    rsi = RecursiveSelfImprovement(core=core, settings=s, growth_engine=engine)
    return rsi


# --------------------------------------------------------------------------- #
# acquire_knowledge → research/investigation queues
# --------------------------------------------------------------------------- #
def test_acquire_knowledge_enqueues_weakest_categories():
    core = _FakeCore()
    rsi = _rsi(core=core)
    rsi._bench = {"by_category": {"math": 0.2, "logic": 0.9, "code": 0.4, "text": 0.5}}
    rsi._directive = {"action": "acquire_knowledge"}
    report = SelfImprovementReport()
    rsi._enact_directive(wreport=None, report=report, enact=True)
    assert report.directive_action == "acquire_knowledge"
    assert report.directive_results["dispatched"] is True
    # only the weakest categories become research topics — the strongest (logic) is left alone
    assert any("math" in t for t in core._investigation_queue)
    assert any("code" in t for t in core._investigation_queue)
    assert not any("logic" in t for t in core._investigation_queue)


def test_acquire_knowledge_without_core_is_clean_noop():
    rsi = _rsi(core=None)
    rsi._bench = {"by_category": {"math": 0.2}}
    rsi._directive = {"action": "acquire_knowledge"}
    report = SelfImprovementReport()
    rsi._enact_directive(wreport=None, report=report, enact=True)
    assert report.directive_results["dispatched"] is False
    assert "no core handle" in report.directive_results["reason"]


# --------------------------------------------------------------------------- #
# train_self_model → gauntlet-gated foundry forge
# --------------------------------------------------------------------------- #
def test_train_self_model_drives_foundry_when_enabled():
    engine = _FakeEngine([_FakeResult(True), _FakeResult(False)])
    rsi = _rsi(core=_FakeCore(), engine=engine)
    rsi.settings.foundry.enabled = True
    rsi._directive = {"action": "train_self_model"}
    report = SelfImprovementReport()
    rsi._enact_directive(wreport=None, report=report, enact=True)
    assert engine.calls == 1
    assert report.directive_results["dispatched"] is True
    assert report.directive_results["promoted"] == 1
    assert report.directive_results["cycles"] == 2


def test_train_self_model_noop_when_foundry_disabled():
    engine = _FakeEngine([_FakeResult(True)])
    rsi = _rsi(core=_FakeCore(), engine=engine)
    rsi.settings.foundry.enabled = False
    rsi._directive = {"action": "train_self_model"}
    report = SelfImprovementReport()
    rsi._enact_directive(wreport=None, report=report, enact=True)
    assert engine.calls == 0
    assert report.directive_results["dispatched"] is False


def test_train_self_model_respects_closed_oversight_gate():
    class _ClosedGate:
        def gate(self):
            return False
    core = _FakeCore()
    core.oversight = _ClosedGate()
    engine = _FakeEngine([_FakeResult(True)])
    rsi = _rsi(core=core, engine=engine)
    rsi.settings.foundry.enabled = True
    rsi._directive = {"action": "train_self_model"}
    report = SelfImprovementReport()
    rsi._enact_directive(wreport=None, report=report, enact=True)
    assert engine.calls == 0
    assert report.directive_results["dispatched"] is False
    assert "oversight" in report.directive_results["reason"]


# --------------------------------------------------------------------------- #
# dry-run / disabled dispatch
# --------------------------------------------------------------------------- #
def test_dry_run_dispatches_nothing():
    engine = _FakeEngine([_FakeResult(True)])
    rsi = _rsi(core=_FakeCore(), engine=engine)
    rsi.settings.foundry.enabled = True
    rsi._directive = {"action": "train_self_model"}
    report = SelfImprovementReport()
    rsi._enact_directive(wreport=None, report=report, enact=False)
    assert engine.calls == 0
    assert report.directive_results["dispatched"] is False
    assert report.directive_action == "train_self_model"


def test_handled_elsewhere_actions_are_marked():
    rsi = _rsi(core=_FakeCore())
    for action in ("deepen_reasoning", "resolve_weaknesses"):
        rsi._directive = {"action": action}
        report = SelfImprovementReport()
        rsi._enact_directive(wreport=None, report=report, enact=True)
        assert report.directive_action == action
        assert report.directive_results["dispatched"] is False
        assert "source-edit" in report.directive_results["reason"]


# --------------------------------------------------------------------------- #
# the wire itself: GrowthEngine threads `core` into the RSIE
# --------------------------------------------------------------------------- #
def test_growth_engine_from_core_threads_core_into_rsi():
    from nyxara.growth.autolearn import GrowthEngine
    core = _FakeCore()
    eng = GrowthEngine.from_core(core)
    assert eng.core is core
    rsi = eng._rsi_obj()
    assert rsi.core is core
    assert rsi.growth_engine is eng
