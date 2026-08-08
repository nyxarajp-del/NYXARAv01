"""NYX V.01 · the dynamic neural graph — growth, decay, pruning, and hard bounds."""

from __future__ import annotations

import pytest

from nyxara.nyx.graph import DynamicNeuralGraph, concepts_in


def _graph(**kw) -> DynamicNeuralGraph:
    kw.setdefault("max_nodes", 64)
    kw.setdefault("max_edges", 256)
    kw.setdefault("decay_rate", 0.0)          # most tests want growth isolated from fade
    return DynamicNeuralGraph(**kw)


def test_concepts_in_drops_stopwords_and_keeps_order():
    got = concepts_in("The cat sat on the warm mat")
    assert got == ["cat", "sat", "warm", "mat"]


def test_concepts_in_is_fail_soft_on_junk():
    assert concepts_in("") == []
    assert concepts_in(None) == []


def test_co_activation_strengthens_an_edge():
    g = _graph()
    g.activate("cat warm")
    first = g.weight("cat", "warm")
    g.activate("cat warm")
    assert g.weight("cat", "warm") > first > 0.0


def test_repeated_pairs_beat_incidental_ones():
    """The whole point of Hebb: what fires together often should end up bound together."""
    g = _graph()
    g.activate("gravity apple falls")
    g.activate("gravity apple ground")
    assert g.weight("gravity", "apple") > g.weight("falls", "ground")


def test_new_concepts_are_born_once():
    g = _graph()
    a = g.activate("cat warm")
    b = g.activate("cat lap")
    assert set(a.born) == {"cat", "warm"}
    assert b.born == ["lap"]                   # "cat" already existed


def test_disuse_decays_and_prunes_below_threshold():
    g = _graph(decay_rate=0.5, prune_threshold=0.05)
    g.activate("cat warm")
    assert g.weight("cat", "warm") > 0.0
    for _ in range(12):
        g.activate("unrelated topic")          # ticks pass; cat~warm is never reinforced
    assert g.weight("cat", "warm") == 0.0
    assert g.stats().pruned_total > 0


def test_node_cap_is_enforced():
    g = _graph(max_nodes=8)
    for i in range(40):
        g.activate(f"concept{i} other{i}")
    assert len(g) <= 8


def test_edge_cap_is_enforced():
    g = _graph(max_nodes=64, max_edges=10)
    for i in range(20):
        g.activate(f"alpha{i} beta{i} gamma{i}")
    assert len(g.edges) <= 10


def test_rewire_budget_bounds_structural_change():
    g = _graph(rewire_budget=3)
    act = g.activate("one two three four five six seven eight")
    assert act.strengthened <= 3


def test_learn_false_reads_without_rewiring():
    g = _graph()
    g.activate("cat warm")
    before = (len(g), len(g.edges), g.weight("cat", "warm"))
    g.activate("cat warm", learn=False)
    assert (len(g), len(g.edges), g.weight("cat", "warm")) == before


def test_spreading_activation_reaches_indirect_neighbours():
    g = _graph(spread_depth=2)
    g.activate("cat warm")
    g.activate("warm sun")
    act = g.activate("cat", learn=False)
    assert "sun" in [n for n, _ in act.spread]  # cat → warm → sun


def test_neighbours_are_ranked_strongest_first():
    g = _graph()
    g.activate("cat warm")
    g.activate("cat warm")
    g.activate("cat mat")
    got = g.neighbours("cat", k=2)
    assert got[0][0] == "warm" and got[0][1] > got[1][1]


def test_gaps_surface_seen_but_unconnected_concepts():
    g = _graph()
    for _ in range(6):
        g.activate("lonely")                   # met often, never co-occurring with anything
    g.activate("cat warm mat")
    assert "lonely" in g.gaps(k=3)


def test_round_trip_preserves_shape_and_weights():
    g = _graph()
    g.activate("gravity apple falls")
    g.activate("gravity apple ground")
    data = g.to_dict()

    restored = DynamicNeuralGraph()
    assert restored.load_dict(data) is True
    assert len(restored) == len(g)
    assert len(restored.edges) == len(g.edges)
    # weights are stored rounded to 6dp to keep the sidecar small — that is the only loss
    assert restored.weight("gravity", "apple") == pytest.approx(g.weight("gravity", "apple"),
                                                               abs=1e-6)


def test_corrupt_sidecar_is_an_empty_graph_not_a_crash():
    g = _graph()
    assert g.load_dict(None) is False
    assert g.load_dict({"nodes": "not-a-list", "edges": 7}) in (True, False)
    assert len(g) == 0                          # degraded, but alive


def test_activation_never_raises_on_hostile_input():
    g = _graph()
    for junk in (None, "", "   ", "!!!  ???", "a" * 10000):
        assert g.activate(junk) is not None
