"""Tests for nyxara.agency.tool_router — decide which tool fits a sub-task."""

from __future__ import annotations


from nyxara.agency.default_tools import build_default_tools
from nyxara.agency.tool_router import ToolChoice, ToolProfile, ToolRouter
from nyxara.agency.tools import ToolRegistry


def _router() -> ToolRouter:
    return ToolRouter(build_default_tools(ToolRegistry()))


def test_catalog_built_from_live_registry():
    r = _router()
    cat = r.catalog()
    assert "cad_model" in cat and "web_search" in cat and "run_python" in cat


def test_routes_geometry_to_cad():
    r = _router()
    c = r.choose("model a 5cm cube bracket with a 1cm hole")
    assert c is not None and c.tool == "cad_model"


def test_routes_computation_to_python():
    r = _router()
    c = r.choose("compute the stress on a 2cm steel beam")
    assert c is not None and c.tool == "run_python"


def test_routes_live_facts_to_web_search():
    r = _router()
    c = r.choose("latest news about fusion reactors today")
    assert c is not None and c.tool == "web_search"


def test_routes_fetch_to_web_fetch():
    r = _router()
    c = r.choose("fetch the webpage at https://example.com")
    assert c is not None and c.tool == "web_fetch"


def test_rank_is_ordered_best_first():
    r = _router()
    ranked = r.rank("model a cube with a hole")
    scores = [c.score for c in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0].tool == "cad_model"


def test_select_respects_min_score_and_top_k():
    r = _router()
    top = r.select("model a cube bracket", top_k=2)
    assert 1 <= len(top) <= 2
    assert all(isinstance(c, ToolChoice) for c in top)


def test_choice_carries_rationale():
    r = _router()
    c = r.choose("model a cube")
    assert c is not None and "CAD" in c.rationale


def test_empty_registry_yields_no_choice():
    r = ToolRouter(ToolRegistry())
    assert r.catalog() == []
    assert r.choose("anything") is None


def test_profiles_can_be_supplied_directly():
    profiles = [ToolProfile(name="custom", description="", cues=("widget", "gizmo"))]
    r = ToolRouter(profiles=profiles)
    c = r.choose("build a widget")
    assert c is not None and c.tool == "custom"


def test_no_intent_match_returns_none():
    r = _router()
    # gibberish that matches nothing should clear no tool over the bar
    assert r.choose("zzqq xyzzy plugh") is None
