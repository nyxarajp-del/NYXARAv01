"""Tests for nyxara.observe.self_report."""

from __future__ import annotations

from nyxara.observe.self_report import (ReportKind, ReportSection, SelfReporter)


def _reporter():
    r = SelfReporter()
    r.register("health", lambda: {"posture": "normal", "overall": "healthy"})
    r.register("capabilities", lambda: {"available": ["a", "b"], "locked": ["c"]})
    r.register("oversight", lambda: ["delete X?"])
    return r


def _section(report, title):
    return next((s for s in report.sections if s.title == title), None)


# -------------------- shapes -------------------- #
def test_section_to_dict():
    s = ReportSection("T", ["x"], importance=0.7)
    assert s.to_dict()["title"] == "T"


def test_report_render():
    r = SelfReporter()
    r.note("mode", "guardian")
    txt = r.full().render()
    assert "NYXARA" in txt and "Current state" in txt


def test_report_to_dict():
    r = _reporter()
    d = r.full().to_dict()
    assert d["kind"] == "full" and "sections" in d


# -------------------- logging -------------------- #
def test_log_decision_appears():
    r = SelfReporter()
    r.log_decision("did x", "because y", outcome="ok")
    sec = _section(r.full(), "Decisions and why")
    assert sec and any("did x" in ln and "because y" in ln for ln in sec.lines)


def test_decision_autonomous_tag():
    r = SelfReporter()
    r.log_decision("auto", "r", autonomous=True)
    r.log_decision("cmd", "r", autonomous=False)
    lines = _section(r.full(), "Decisions and why").lines
    assert any("autonomously" in ln for ln in lines)
    assert any("as commanded" in ln for ln in lines)


def test_note_state():
    r = SelfReporter()
    r.note("mode", "active")
    sec = _section(r.status(), "Current state")
    assert sec and any("mode: active" in ln for ln in sec.lines)


# -------------------- disclosures (Rule 6) -------------------- #
def test_failures_disclosed():
    r = SelfReporter()
    r.log_failure("task X", "it broke")
    sec = _section(r.full(), "Honest disclosures")
    assert sec and any("FAILED: task X" in ln for ln in sec.lines)


def test_uncertainties_disclosed_and_hedged():
    r = SelfReporter()
    r.log_uncertainty("the cause", confidence=0.3)
    sec = _section(r.full(), "Honest disclosures")
    assert sec and any("UNCERTAIN" in ln for ln in sec.lines)
    # hedged, not stated as fact
    assert any(("not sure" in ln or "doubt" in ln or "suspect" in ln) for ln in sec.lines)


def test_disclosures_present_when_clean():
    r = SelfReporter()
    sec = _section(r.full(), "Honest disclosures")
    assert sec and any("no failures" in ln for ln in sec.lines)


def test_disclosures_always_in_status():
    r = SelfReporter()
    r.log_failure("x", "y")
    assert _section(r.status(), "Honest disclosures") is not None


def test_disclosures_report_kind():
    r = SelfReporter()
    r.log_failure("x", "y")
    rep = r.disclosures()
    assert rep.kind is ReportKind.DISCLOSURES
    assert _section(rep, "Honest disclosures") is not None


# -------------------- providers -------------------- #
def test_health_section():
    r = _reporter()
    sec = _section(r.full(), "Health & posture")
    assert sec and any("posture: normal" in ln for ln in sec.lines)


def test_capabilities_section():
    r = _reporter()
    sec = _section(r.full(), "Capabilities")
    assert sec and any("available" in ln for ln in sec.lines)


def test_escalations_section():
    r = _reporter()
    sec = _section(r.full(), "Awaiting your decision")
    assert sec and "delete X?" in sec.lines


def test_escalations_empty_when_none():
    r = SelfReporter()
    sec = _section(r.status(), "Awaiting your decision")
    assert sec is not None and sec.lines == []


def test_broken_provider_surfaced():
    r = SelfReporter()
    r.register("health", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    sec = _section(r.full(), "Health & posture")
    assert sec and any("error" in ln.lower() for ln in sec.lines)


def test_missing_provider_skipped():
    r = SelfReporter()  # no providers
    assert _section(r.full(), "Health & posture") is None
    assert _section(r.full(), "Capabilities") is None


def test_oversight_dict_pending():
    r = SelfReporter()
    r.register("oversight", lambda: {"pending": ["approve Y?"]})
    sec = _section(r.full(), "Awaiting your decision")
    assert "approve Y?" in sec.lines


# -------------------- report kinds -------------------- #
def test_status_kind_sections():
    r = _reporter()
    r.note("k", "v")
    titles = [s.title for s in r.status().sections]
    assert "Current state" in titles and "Health & posture" in titles
    assert "Decisions and why" not in titles  # activity-only


def test_activity_kind_sections():
    r = _reporter()
    r.log_decision("x", "y")
    titles = [s.title for s in r.activity().sections]
    assert "Decisions and why" in titles and "Activity" in titles
    assert "Health & posture" not in titles


def test_activity_section_counts():
    r = SelfReporter()
    r.log_decision("a", "r", autonomous=True)
    r.log_decision("b", "r", autonomous=False)
    sec = _section(r.activity(), "Activity")
    assert any("2 decisions logged (1 autonomous)" in ln for ln in sec.lines)


def test_full_includes_everything():
    r = _reporter()
    r.note("m", "x")
    r.log_decision("d", "r")
    r.log_failure("f", "why")
    titles = {s.title for s in r.full().sections}
    assert {"Current state", "Health & posture", "Capabilities", "Decisions and why",
            "Honest disclosures", "Awaiting your decision"} <= titles


# -------------------- summary -------------------- #
def test_summary_counts():
    r = SelfReporter()
    r.log_decision("d", "r")
    r.log_failure("f", "why")
    r.log_uncertainty("u", 0.4)
    s = r.full().summary
    assert "1 decisions" in s and "1 failures" in s and "1 open uncertainties" in s


def test_summary_empty():
    assert SelfReporter().full().summary == "nothing to report"
