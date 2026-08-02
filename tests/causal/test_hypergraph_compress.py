"""Tests for nyxara.causal.hypergraph_compress."""

from __future__ import annotations

from nyxara.causal.hypergraph_compress import HyperGraphCompressor


def _triples():
    t = [("nyxara", "knows", f"concept_{i}") for i in range(300)]
    t += [(f"user_{i}", "likes", "nyxara") for i in range(150)]
    return t


def test_round_trip_is_lossless():
    c = HyperGraphCompressor()
    triples = _triples()
    blob, report = c.compress(triples)
    restored = c.decompress(blob)
    assert restored == set(triples)
    assert report.lossless is True


def test_repetitive_data_compresses_above_one():
    c = HyperGraphCompressor()
    blob, report = c.compress(_triples())
    assert report.ratio > 1.0                          # a real, measured win on hub-structured data
    assert report.hyperedges < report.triples          # edges folded into hyper-edges


def test_history_is_preserved():
    c = HyperGraphCompressor()
    blob, _ = c.compress([("a", "b", "c")], history=["boot", "learn", "reflect"])
    assert c.history(blob) == ["boot", "learn", "reflect"]


def test_direct_query_on_the_packed_graph():
    c = HyperGraphCompressor()
    blob, _ = c.compress(_triples())
    hits = c.query(blob, subject="nyxara", predicate="knows")
    assert len(hits) == 300
    assert all(h[0] == "nyxara" and h[1] == "knows" for h in hits)


def test_duplicate_triples_are_deduped():
    c = HyperGraphCompressor()
    blob, report = c.compress([("a", "b", "c")] * 10)
    assert report.triples == 1
    assert c.decompress(blob) == {("a", "b", "c")}


def test_empty_graph_is_handled():
    c = HyperGraphCompressor()
    blob, report = c.compress([])
    assert report.triples == 0
    assert c.decompress(blob) == set()
