"""Tests for the Genesis Seed — growth/genesis_seed.py (Part J)."""
from __future__ import annotations

from nyxara.growth.genesis_seed import GenesisSeed
from nyxara.memory.graph import GraphQuery, KnowledgeGraph


def test_snapshot_seed_is_256_bit_and_round_trips():
    gs = GenesisSeed(secret=b"s")
    seed = gs.snapshot({"goals": ["serve JP"], "axioms": {"loyalty": 1.0}})
    assert len(seed.seed) == 64                       # 256-bit content address (hex)
    state = gs.open(seed.seed, seed.bundle)
    assert state is not None and state["goals"] == ["serve JP"]


def test_tampered_bundle_is_refused():
    gs = GenesisSeed(secret=b"s")
    seed = gs.snapshot({"x": 1})
    tampered = seed.bundle[:-3] + b"XYZ"
    # content hash no longer matches the seed -> refused (fail-closed)
    assert gs.open(seed.seed, tampered) is None


def test_wrong_secret_fails_signature():
    seed = GenesisSeed(secret=b"right").snapshot({"x": 1})
    # a core with the wrong secret cannot verify the signature
    assert GenesisSeed(secret=b"wrong").open(seed.seed, seed.bundle) is None


def test_bundle_write_and_reopen(tmp_path):
    gs = GenesisSeed(secret=b"s")
    seed = gs.snapshot({"memory": "hello"})
    path = seed.write(tmp_path)
    assert path.name == f"{seed.seed}.nyx"
    assert gs.open(seed.seed, path.read_bytes())["memory"] == "hello"


def test_snapshot_and_resurrect_knowledge_graph():
    # a source core with a real graph
    class _Core:
        pass
    src = _Core()
    src.knowledge_graph = KnowledgeGraph()
    src.knowledge_graph.add_triple("nyxara", "owned_by", "master")

    gs = GenesisSeed(secret=b"s")
    seed = gs.snapshot_core(src)

    # a FRESH core resurrects the exact graph
    fresh = _Core()
    fresh.knowledge_graph = KnowledgeGraph()
    assert gs.resurrect_core(fresh, seed.seed, seed.bundle) is True
    matches = fresh.knowledge_graph.match(GraphQuery(subject="nyxara", predicate="owned_by"))
    assert matches and fresh.knowledge_graph.resolve(matches[0].object) is not None
