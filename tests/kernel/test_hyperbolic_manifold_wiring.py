"""Integration: the self-mutating hyperbolic manifold is LIVE in the core (not an orphan module).

Confirms the orchestrator builds the manifold from config, feeds it every processed turn
(_manifold_tick), reinforces active edges from the turn's real outcome (_grow → reinforce_active),
exposes the public manifold_* API, honours the config kill-switch, and persists the manifold as a
sidecar across save_state/load_state. Hermetic (TEST profile).
"""
from __future__ import annotations

from nyxara.kernel.orchestrator import Authority, Candidate, Disposition, NyxaraCore


def test_core_builds_manifold():
    core = NyxaraCore()
    assert core.hyperbolic_manifold is not None
    stats = core.manifold_stats()
    assert stats["n_nodes"] == 0 and stats["genesis_count"] == 0


def test_manifold_observe_and_nearest_via_core():
    core = NyxaraCore()
    r = core.manifold_observe("quantum gravity", concept_id="qg")
    assert r is not None and r.created
    assert core.manifold_stats()["n_nodes"] == 1
    hits = core.manifold_nearest("quantum gravity", k=1)
    assert hits and hits[0][0] == "qg"
    # distance API answers for known ids and refuses unknowns
    core.manifold_observe("sourdough bread", concept_id="bread")
    d = core.manifold_distance("qg", "bread")
    assert d is not None and d > 0.0
    assert core.manifold_distance("qg", "ghost") is None


def test_turn_feeds_manifold():
    core = NyxaraCore()
    core.process("tell me about black holes", authority=Authority.OWNER)
    assert core.manifold_stats()["n_nodes"] >= 1  # the _manifold_tick seam fired


def test_grow_reinforces_active_edges():
    core = NyxaraCore()
    mm = core.hyperbolic_manifold
    # two genesis observations: the second is born with an edge to the first (its context)
    core.manifold_observe("celestial mechanics of planetary orbits", concept_id="a")
    core.manifold_observe("a completely different topic: bread baking", concept_id="b")
    w_before = mm.edge_weight("a", "b")
    assert w_before is not None  # genesis wired the context edge
    cand = Candidate(text="ok", kind="respond", confidence=0.9, rationale="test")
    core._grow(cand, Disposition.ACT, authority=Authority.OWNER, success=True, stimulus="x")
    w_after = mm.edge_weight("a", "b")
    assert w_after is not None and w_after > w_before  # Δe = η·utility·(1 − d/d_norm) > 0


def test_disabled_by_config(monkeypatch):
    monkeypatch.setenv("NYXARA_HYPERBOLIC_MANIFOLD__ENABLED", "false")
    from nyxara.kernel.config import reload_settings
    reload_settings()
    try:
        core = NyxaraCore()
        assert core.hyperbolic_manifold is None
        assert core.manifold_stats() == {}
        assert core.manifold_observe("anything") is None
        core.process("still works without the manifold", authority=Authority.OWNER)
    finally:
        # monkeypatch only unsets env AFTER the test returns — drop it now so the
        # restored settings singleton doesn't leak "disabled" into later tests
        monkeypatch.delenv("NYXARA_HYPERBOLIC_MANIFOLD__ENABLED", raising=False)
        reload_settings()


def test_persistence_sidecar_round_trip(tmp_path):
    target = str(tmp_path / "memory.json")
    core = NyxaraCore()
    core.manifold_observe("the moon orbits the earth", concept_id="moon")
    core.manifold_observe("sourdough bread recipe", concept_id="bread")
    saved_stats = core.manifold_stats()
    assert saved_stats["n_nodes"] == 2
    core.save_state(target)
    assert (tmp_path / "hyperbolic_manifold.json").exists()

    core2 = NyxaraCore()
    core2.load_state(target)
    assert core2.manifold_stats() == saved_stats
    hits = core2.manifold_nearest("lunar orbit of the moon", k=1)
    assert hits and hits[0][0] == "moon"
