"""Idle work aims at measured ignorance, not at whatever happened to be noticed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from nyxara.growth.epistemic_drive import EpistemicDrive


@dataclass
class _FakeWeakness:
    title: str
    severity: float
    source: str = "benchmark"


class _FakeReport:
    def __init__(self, weaknesses: List[_FakeWeakness]) -> None:
        self.weaknesses = weaknesses

    def top(self, n: int = 10) -> List[_FakeWeakness]:
        return sorted(self.weaknesses, key=lambda w: w.severity, reverse=True)[:n]


class _Core:
    """The narrow slice of NyxaraCore the seeder actually touches."""

    memory = None

    def __init__(self) -> None:
        self._research_queue: List[str] = []
        self._investigation_queue: List[str] = []

    _seed_queues_from_drive = None      # bound below from the real implementation


def _core_with(drive) -> _Core:
    from nyxara.kernel.orchestrator import NyxaraCore

    core = _Core()
    core._epistemic_drive = drive
    # borrow the real method rather than reimplementing it in the test
    core._seed_queues_from_drive = NyxaraCore._seed_queues_from_drive.__get__(core, _Core)
    return core


# -------------------- knowledge_gaps is the symmetric partner -------------------- #
def test_knowledge_gaps_filters_before_capping():
    """top_gaps-then-filter is a trap: fifty architectural weaknesses fill the ranked list and
    every knowledge gap is truncated away before the filter ever runs."""
    weaknesses = [_FakeWeakness(f"layering violation {i}", 0.9, source="architecture")
                  for i in range(50)]
    weaknesses.append(_FakeWeakness("quantum error correction", 0.5, source="benchmark"))
    drive = EpistemicDrive(weakness_report=_FakeReport(weaknesses))

    assert [g.subject for g in drive.knowledge_gaps(5)] == ["quantum error correction"]
    assert all(g.searchable for g in drive.knowledge_gaps(5))
    assert all(not g.searchable for g in drive.engineering_gaps(5))


# -------------------- the seeding -------------------- #
def test_measured_gaps_become_queued_work():
    drive = EpistemicDrive(weakness_report=_FakeReport([
        _FakeWeakness("thermodynamics of open systems", 0.9),
        _FakeWeakness("graph colouring", 0.3),
    ]))
    core = _core_with(drive)

    assert core._seed_queues_from_drive(k=4) == 2
    queued = core._research_queue + core._investigation_queue
    assert "thermodynamics of open systems" in queued
    assert "graph colouring" in queued


def test_high_pressure_gaps_get_an_experiment_not_just_a_search():
    """A gap she is very unsure of is worth testing, not only reading about."""
    drive = EpistemicDrive(weakness_report=_FakeReport([
        _FakeWeakness("thermodynamics of open systems", 0.95),
    ]))
    core = _core_with(drive)
    core._seed_queues_from_drive(k=4)
    assert core._investigation_queue == ["thermodynamics of open systems"]
    assert core._research_queue == []


def test_low_pressure_gaps_only_get_read_about():
    drive = EpistemicDrive(weakness_report=_FakeReport([
        _FakeWeakness("graph colouring", 0.2),
    ]))
    core = _core_with(drive)
    core._seed_queues_from_drive(k=4)
    assert core._research_queue == ["graph colouring"]
    assert core._investigation_queue == []


def test_engineering_gaps_are_never_queued_as_research():
    """No amount of reading fixes an import cycle — those have their own path via self_evolving."""
    drive = EpistemicDrive(weakness_report=_FakeReport([
        _FakeWeakness("import cycle across 4 modules", 0.95, source="architecture"),
        _FakeWeakness("layering violation: a.b -> c.d", 0.9, source="architecture"),
    ]))
    core = _core_with(drive)
    assert core._seed_queues_from_drive(k=4) == 0
    assert core._research_queue == [] and core._investigation_queue == []


def test_seeding_is_idempotent():
    drive = EpistemicDrive(weakness_report=_FakeReport([
        _FakeWeakness("thermodynamics", 0.9), _FakeWeakness("graph colouring", 0.3),
    ]))
    core = _core_with(drive)
    first = core._seed_queues_from_drive(k=4)
    before = len(core._research_queue) + len(core._investigation_queue)
    core._seed_queues_from_drive(k=4)
    assert first == before
    assert len(core._research_queue) + len(core._investigation_queue) == before


def test_a_broken_drive_never_costs_the_idle_tick():
    class _Exploding:
        def knowledge_gaps(self, _k):
            raise RuntimeError("boom")

    core = _core_with(_Exploding())
    assert core._seed_queues_from_drive(k=4) == 0


def test_an_unavailable_drive_is_remembered_and_not_retried():
    core = _core_with(False)
    assert core._seed_queues_from_drive(k=4) == 0


def test_the_report_records_what_was_seeded():
    drive = EpistemicDrive(weakness_report=_FakeReport([_FakeWeakness("thermodynamics", 0.9)]))
    core = _core_with(drive)
    report: dict = {}
    core._seed_queues_from_drive(report, k=4)
    assert report["epistemic_seeded"] == 1
