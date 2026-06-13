"""Tests for nyxara.growth.researcher — the autonomous research pipeline.

Runs fully offline (no tools, no network): search/read degrade to empty and the
summary falls back to a heuristic.
"""

from __future__ import annotations

from nyxara.growth.researcher import AutonomousResearcher, ResearchReport
from nyxara.knowledge.base import KnowledgeBase
from nyxara.memory.graph import KnowledgeGraph, _configure_standard_relations
from nyxara.sim.sandbox import Sandbox


# --------------------------------------------------------------------------- #
# ResearchReport
# --------------------------------------------------------------------------- #
def test_report_defaults_and_to_dict():
    rep = ResearchReport(topic="t")
    assert rep.topic == "t"
    assert rep.sources == [] and rep.key_claims == []
    assert rep.timestamp > 0
    d = rep.to_dict()
    assert d["topic"] == "t" and "elapsed_ms" in d


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def test_research_without_tools_returns_report():
    r = AutonomousResearcher(tools=None)
    rep = r.research("transformer attention mechanism")
    assert isinstance(rep, ResearchReport)
    assert rep.topic == "transformer attention mechanism"
    assert rep.sources == []          # no tools → no URLs
    assert rep.summary                 # heuristic fallback always produces something
    assert rep.elapsed_ms >= 0.0


def test_all_reports_accumulates():
    r = AutonomousResearcher()
    r.research("topic one")
    r.research("topic two")
    assert len(r.all_reports()) == 2


def test_update_kb_ingests_summary():
    kb = KnowledgeBase(name="research-test")
    r = AutonomousResearcher(knowledge=kb)
    rep = r.research("spaced repetition")
    assert rep.kb_chunks_added == 1
    assert len(kb) >= 1


# --------------------------------------------------------------------------- #
# experiment step now uses the REAL Sandbox API (regression: was sandbox.dry_run)
# --------------------------------------------------------------------------- #
def test_experiment_uses_real_sandbox():
    r = AutonomousResearcher(sandbox=Sandbox())
    result = r._experiment("topic", "a non-empty summary about topic")
    assert result == "internal validation passed"


def test_experiment_skipped_without_sandbox():
    r = AutonomousResearcher(sandbox=None)
    assert "skipped" in r._experiment("topic", "summary")


def test_research_records_experiment_result_with_sandbox():
    r = AutonomousResearcher(sandbox=Sandbox())
    rep = r.research("photosynthesis")
    # with no sources the summary is the heuristic fallback, still non-empty → passes
    assert rep.experiment_result in (
        "internal validation passed", "internal validation flagged")


def test_update_graph_with_no_claims_adds_nothing():
    graph = KnowledgeGraph()
    _configure_standard_relations(graph)
    r = AutonomousResearcher(knowledge_graph=graph)
    rep = r.research("a topic with no fetchable sources")
    # no tools → no urls/claims → no triples
    assert rep.graph_triples_added == 0
