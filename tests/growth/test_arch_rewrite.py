"""Tests for growth/arch_rewrite.py — DPO-style graph rewriting (Change set P).

Fast + standalone. Pins: cycles/god-modules are detected; a rewrite that lowers the structural score is
kept only when the behaviour gauntlet passes; a non-improving graph is left alone.
"""

from __future__ import annotations

from nyxara.growth.arch_rewrite import ArchRewriter, DependencyGraph


def test_detects_a_cycle():
    g = DependencyGraph({"a": {"b"}, "b": {"c"}, "c": {"a"}})
    cycles = g.cycles()
    assert len(cycles) >= 1
    assert g.score() >= 10.0


def test_rewrite_breaks_a_cycle_when_verified():
    g = DependencyGraph({"a": {"b"}, "b": {"c"}, "c": {"a"}})
    res = ArchRewriter().improve(g, verify=lambda before, after: True)
    assert res.applied is True
    assert res.after_score < res.before_score


def test_rewrite_rejected_when_not_behaviour_preserving():
    g = DependencyGraph({"a": {"b"}, "b": {"a"}})
    res = ArchRewriter().improve(g, verify=lambda before, after: False)
    assert res.applied is False
    assert "behaviour" in res.reason


def test_clean_graph_needs_no_rewrite():
    g = DependencyGraph({"a": {"b"}, "b": {"c"}})    # a DAG, no cycles
    res = ArchRewriter().improve(g, verify=lambda before, after: True)
    assert res.applied is False
    assert "no rewrite" in res.reason


def test_layering_violations_counted():
    layers = {"low": 0, "high": 1}
    g = DependencyGraph({"low": {"high"}})           # lower layer imports higher = violation
    assert g.layering_violations(layers) == [("low", "high")]
    assert g.score(layers=layers) >= 5.0


def test_god_module_detected_and_extracted():
    # one hub imported by / importing many
    edges = {"hub": {f"m{i}" for i in range(10)}}
    g = DependencyGraph(edges)
    assert "hub" in g.god_modules(threshold=8)
    res = ArchRewriter(god_threshold=8).improve(g, verify=lambda b, a: True)
    assert res.applied is True
    assert res.kind == "extract_module"
