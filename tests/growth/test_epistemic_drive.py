"""Tests for nyxara.growth.epistemic_drive — five uncertainty numbers become one hunger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from nyxara.growth.acquire import gap_topics
from nyxara.growth.epistemic_drive import DrivePressure, EpistemicDrive, Gap
from nyxara.growth.signal_bus import get_signal_bus


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


class _FakeWorldModel:
    def __init__(self, epistemic: float) -> None:
        self._epistemic = epistemic

    def mean_epistemic(self) -> float:
        return self._epistemic


def _drive(**kw) -> EpistemicDrive:
    report = kw.pop("weakness_report", _FakeReport([
        _FakeWeakness("arithmetic word problems", 0.9),
        _FakeWeakness("unit conversion", 0.6),
    ]))
    return EpistemicDrive(weakness_report=report, **kw)


# -------------------- fusion -------------------- #
def test_pressure_reports_every_contributing_signal():
    drive = _drive(world_model=_FakeWorldModel(0.8))
    pressure = drive.pressure()
    assert "weakness" in pressure.contributors
    assert "world_model" in pressure.contributors
    assert 0.0 < pressure.total <= 1.0


def test_pressure_is_zero_when_nothing_reports():
    drive = EpistemicDrive(weakness_report=_FakeReport([]))
    pressure = drive.pressure()
    assert pressure.total == 0.0
    assert "no signal available" in pressure.summary()


def test_total_averages_only_reporting_signals():
    """Averaging over all five would dilute a single live faculty into near-zero and report
    contentment on a machine that is in fact very ignorant."""
    pressure = DrivePressure(weakness=1.0, contributors=["weakness"])
    assert pressure.total == 1.0


def test_a_severe_weakness_outranks_an_ambient_signal():
    """The ranking is pressure x source weight, so signals on different scales stay comparable."""
    bus = get_signal_bus()
    bus.post("reflection_focus", "some passing thought", weight=1.0)
    drive = _drive()
    gaps = drive.top_gaps(5)
    assert gaps[0].source == "weakness"
    assert gaps[0].subject == "arithmetic word problems"


# -------------------- knowledge vs engineering -------------------- #
def test_engineering_weaknesses_never_reach_the_acquirer():
    """"layering violation: nyxara.agency.llm_tool -> nyxara.growth.foundry" is a real, correctly
    ranked weakness and a useless web search. Sending it burns the acquisition budget on her own
    module names."""
    report = _FakeReport([
        _FakeWeakness("layering violation: a.b -> c.d", 0.95, source="architecture"),
        _FakeWeakness("import cycle across 4 modules", 0.9, source="architecture"),
        _FakeWeakness("thermodynamics", 0.5, source="benchmark"),
    ])
    drive = EpistemicDrive(weakness_report=report)

    assert drive.topics(limit=5) == ["thermodynamics"]
    engineering = [g.subject for g in drive.engineering_gaps(5)]
    assert "layering violation: a.b -> c.d" in engineering


def test_knowledge_gaps_are_not_truncated_away_by_engineering_ones():
    """Fifty architectural weaknesses used to fill the ranked list entirely, so every knowledge
    gap was cut before anything could select it. Filtering must happen before the cap."""
    weaknesses = [_FakeWeakness(f"layering violation {i}", 0.9, source="architecture")
                  for i in range(50)]
    weaknesses.append(_FakeWeakness("quantum error correction", 0.4, source="benchmark"))
    drive = EpistemicDrive(weakness_report=_FakeReport(weaknesses))
    assert "quantum error correction" in drive.topics(limit=5)


def test_gap_kind_defaults_to_knowledge():
    assert Gap(subject="x").searchable is True
    assert Gap(subject="x", kind="engineering").searchable is False


# -------------------- signal payloads -------------------- #
def test_signal_subjects_are_whole_phrases_not_tokens():
    """focus_terms() tokenises, which is right for steering edits and wrong here: "protein
    folding dynamics" must not reach the acquirer as three one-word searches."""
    bus = get_signal_bus()
    bus.post("world_model_gap", "protein folding dynamics", weight=0.9)
    drive = EpistemicDrive(weakness_report=_FakeReport([]))
    assert "protein folding dynamics" in drive.topics(limit=5)


def test_non_string_payloads_are_skipped():
    bus = get_signal_bus()
    bus.post("world_model_gap", {"not": "a string"}, weight=0.9)
    drive = EpistemicDrive(weakness_report=_FakeReport([]))
    assert drive.topics(limit=5) == []


# -------------------- the wire into acquisition -------------------- #
def test_gap_topics_now_reaches_measured_ignorance():
    """The whole point: gap_topics listed weakness_report as its top input and no caller ever
    passed one, so the strongest signal she had was dead on arrival."""
    bus = get_signal_bus()
    bus.post("weak_category", "thermodynamics of open systems", weight=0.8)
    topics = gap_topics(limit=4)
    assert "thermodynamics of open systems" in topics


def test_gap_topics_falls_back_to_the_static_battery_when_asked():
    topics = gap_topics(limit=4, use_drive=False)
    assert "general knowledge" in topics


def test_gap_topics_prefers_an_explicit_weakness_report():
    report = _FakeReport([_FakeWeakness("explicitly supplied gap", 0.9)])
    assert gap_topics(limit=3, weakness_report=report)[0] == "explicitly supplied gap"


def test_gap_topics_never_raises_on_a_broken_drive():
    class _Exploding:
        def topics(self, **_kw):
            raise RuntimeError("boom")

    assert gap_topics(limit=3, drive=_Exploding())      # falls through, still returns something


# -------------------- reporting -------------------- #
def test_report_and_summary_serialise():
    drive = _drive(world_model=_FakeWorldModel(0.7))
    blob = drive.report(k=3)
    assert "pressure" in blob and "gaps" in blob and "competence" in blob
    assert "epistemic drive" in drive.summary()
