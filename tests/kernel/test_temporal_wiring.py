"""Tests for the Step-8 temporal wiring — the orchestrator gives the reasoner a sense
of *when*: every turn is stamped, and idle reflection surfaces rhythms / precedence.
"""

from __future__ import annotations

from nyxara.agency.permissions import Authority
from nyxara.kernel.orchestrator import NyxaraCore
from nyxara.mind.temporal import DAY


def _core(**kw):
    return NyxaraCore(**kw)


def test_temporal_faculty_built():
    nyx = _core()
    assert nyx.temporal is not None


def test_turns_are_stamped_in_time():
    nyx = _core()
    before = len(nyx.temporal._events)
    nyx.process("hello there", authority=Authority.OWNER)
    nyx.process("how are you?", authority=Authority.OWNER)
    assert len(nyx.temporal._events) > before


def test_temporal_patterns_surface_a_rhythm():
    nyx = _core()
    # inject a clean weekly rhythm of a remembered event and confirm it is reported
    for k in range(5):
        nyx.temporal.observe("weekly-review", k * 7 * DAY)
    patterns = nyx.temporal_patterns()
    assert any(r["label"] == "weekly-review" for r in patterns["rhythms"])


def test_idle_maintenance_reports_temporal_finding():
    nyx = _core()
    for k in range(4):
        nyx.temporal.observe("standup", k * DAY)
    report = nyx.idle_maintenance()
    assert "temporal" in report
    assert "standup" in report["temporal"]


def test_temporal_safe_without_growth():
    nyx = _core(enable_growth=False)
    assert nyx.temporal is None
    assert nyx.temporal_patterns() == {"rhythms": [], "precedence": []}
