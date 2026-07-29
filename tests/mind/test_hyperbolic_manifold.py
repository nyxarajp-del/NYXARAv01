"""Tests for the self-mutating hyperbolic concept manifold (mind/hyperbolic_manifold.py).

The math must be real Poincaré-ball geometry (Möbius ops, exp/log inverses, hyperbolic — not
Euclidean — barycenters), and the mutation dynamics must be exactly as specified: node genesis
beyond τ at the weighted Fréchet mean, Hebbian edge updates Δe = η·utility·(1 − d/d_norm),
decay/prune, bounded eviction, determinism, and full persistence round-trip. No LLM anywhere.
"""

from __future__ import annotations

import math
import random

from nyxara.mind.hyperbolic_manifold import (
    BallEmbedder,
    HyperbolicConceptManifold,
    einstein_midpoint,
    exp_map,
    frechet_mean,
    log_map,
    mobius_add,
    project_to_ball,
)
from nyxara.mind.latent_geometry import poincare_distance


def _rand_ball_point(rng: random.Random, dim: int = 3, rmax: float = 0.8):
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    r = rng.uniform(0.0, rmax)
    return [r * c / n for c in v]


def test_mobius_identity_and_inverse():
    v = [0.3, -0.2, 0.1]
    zero = [0.0, 0.0, 0.0]
    assert max(abs(a - b) for a, b in zip(mobius_add(zero, v), v)) < 1e-12
    inv = mobius_add([-c for c in v], v)
    assert math.sqrt(sum(c * c for c in inv)) < 1e-9


def test_exp_log_round_trip():
    rng = random.Random(3)
    for _ in range(20):
        x = _rand_ball_point(rng)
        y = _rand_ball_point(rng)
        y2 = exp_map(x, log_map(x, y))
        assert max(abs(a - b) for a, b in zip(y, y2)) < 1e-6


def test_distance_symmetry_and_boundary_growth():
    rng = random.Random(5)
    for _ in range(10):
        u, v = _rand_ball_point(rng), _rand_ball_point(rng)
        assert abs(poincare_distance(u, v) - poincare_distance(v, u)) < 1e-12
    center = [0.0, 0.0]
    assert poincare_distance(center, [0.9, 0.0]) > poincare_distance(center, [0.1, 0.0])
    # sampled triangle inequality
    for _ in range(30):
        a, b, c = (_rand_ball_point(rng) for _ in range(3))
        assert poincare_distance(a, c) <= poincare_distance(a, b) + poincare_distance(b, c) + 1e-9


def test_einstein_midpoint_between_and_in_ball():
    u, v = [0.5, 0.1, 0.0], [-0.3, 0.4, 0.2]
    m = einstein_midpoint([u, v])
    assert math.sqrt(sum(c * c for c in m)) < 1.0
    duv = poincare_distance(u, v)
    assert abs(poincare_distance(u, m) - poincare_distance(m, v)) < 0.05 * duv
    # weighted midpoint leans toward the heavy point
    mw = einstein_midpoint([u, v], [3.0, 1.0])
    assert poincare_distance(u, mw) < poincare_distance(v, mw)


def test_frechet_mean_reduces_objective():
    rng = random.Random(9)
    pts = [_rand_ball_point(rng) for _ in range(5)]

    def objective(m):
        return sum(poincare_distance(m, p) ** 2 for p in pts)

    init = einstein_midpoint(pts)
    refined = frechet_mean(pts, iters=5)
    assert objective(refined) <= objective(init) + 1e-9


def test_project_to_ball_clips_outside_points():
    p = project_to_ball([3.0, 4.0])
    assert math.sqrt(sum(c * c for c in p)) < 1.0
    inside = project_to_ball([0.1, 0.2])
    assert inside == [0.1, 0.2]


def test_genesis_fires_beyond_tau():
    mm = HyperbolicConceptManifold(dim=8, tau=2.2, seed=11,
                                   embedder=BallEmbedder(dim=8, seed=11, alpha=1.0))
    r1 = mm.observe("the moon orbits the earth")
    assert r1.created and not math.isfinite(r1.min_dist)
    r2 = mm.observe("sourdough bread recipe")
    assert r2.created
    assert mm.stats()["n_nodes"] == 2
    # repeating a known concept assimilates, never re-creates
    r3 = mm.observe("the moon orbits the earth")
    assert not r3.created
    assert mm.stats()["n_nodes"] == 2


def test_assimilation_drifts_nearest_node():
    mm = HyperbolicConceptManifold(dim=8, tau=5.0, seed=11, place_lr=0.2)
    mm.observe("alpha beta gamma", concept_id="a")
    q = mm.embedder.embed_text("alpha beta gamma delta")
    d_before = poincare_distance(mm._nodes["a"], q)
    mm.observe("alpha beta gamma delta")  # assimilates into "a" (tau is huge)
    d_after = poincare_distance(mm._nodes["a"], q)
    assert d_after < d_before


def test_genesis_point_is_hyperbolic_barycenter():
    mm = HyperbolicConceptManifold(dim=4, tau=0.0, seed=7)
    mm.observe("first concept", concept_id="a")
    mm.observe("second concept", concept_id="b")
    r = mm.observe("third concept", concept_id="c", context_ids=["a", "b"])
    assert r.created
    p = mm._nodes["c"]
    assert math.sqrt(sum(x * x for x in p)) < 1.0
    # within the cluster: no farther from any member than the cluster's own diameter
    members = [mm._nodes["a"], mm._nodes["b"], mm.embedder.embed_text("third concept")]
    diameter = max(poincare_distance(u, v) for u in members for v in members)
    assert all(poincare_distance(p, m) <= diameter + 1e-9 for m in members)
    # genesis wired edges to its context
    assert mm.edge_weight("c", "a") is not None
    assert mm.edge_weight("c", "b") is not None


def test_hebbian_update_math():
    mm = HyperbolicConceptManifold(dim=2, tau=0.0, eta=0.1, d_norm=5.0, seed=1)
    mm._nodes["i"] = [0.1, 0.0]
    mm._nodes["j"] = [0.0, 0.1]
    d = poincare_distance(mm._nodes["i"], mm._nodes["j"])
    expected = 0.1 * 1.0 * max(0.0, 1.0 - d / 5.0)
    w = mm.reinforce("i", "j", 1.0)
    assert w is not None and abs(w - expected) < 1e-12
    # negative utility lowers the weight; never leaves [0, 1]
    w2 = mm.reinforce("i", "j", -1.0)
    assert w2 is not None and abs(w2) < 1e-12
    for _ in range(200):
        mm.reinforce("i", "j", 1.0)
    assert mm.edge_weight("i", "j") <= 1.0
    # unknown nodes / self-edges are refused
    assert mm.reinforce("i", "i", 1.0) is None
    assert mm.reinforce("i", "ghost", 1.0) is None


def test_decay_and_prune():
    mm = HyperbolicConceptManifold(dim=2, decay=0.5, edge_floor=0.05, seed=1)
    mm._nodes["i"] = [0.1, 0.0]
    mm._nodes["j"] = [0.0, 0.1]
    mm._edges[("i", "j")] = 0.2
    mm.decay_and_prune()          # 0.2 → 0.1
    assert mm.edge_weight("i", "j") is not None
    mm.decay_and_prune()          # 0.1 → 0.05 → pruned (< floor after decay)
    mm.decay_and_prune()
    assert mm.edge_weight("i", "j") is None


def test_max_nodes_eviction():
    mm = HyperbolicConceptManifold(dim=8, tau=0.0, max_nodes=16, seed=3)
    for i in range(40):
        mm.observe(f"totally distinct concept number {i} {'x' * (i % 7)}")
    assert mm.stats()["n_nodes"] <= 16


def test_determinism():
    def run():
        mm = HyperbolicConceptManifold(dim=8, tau=1.5, seed=42)
        for i in range(12):
            mm.observe(f"concept stream item {i % 5} variant {i}")
        mm.reinforce_active(0.7)
        return mm.to_dict()

    assert run() == run()


def test_persistence_round_trip():
    mm = HyperbolicConceptManifold(dim=8, tau=1.5, seed=42)
    for i in range(10):
        mm.observe(f"idea {i % 4} take {i}")
    mm.reinforce_active(1.0)
    blob = mm.to_dict()

    mm2 = HyperbolicConceptManifold(dim=8, tau=1.5, seed=42)
    mm2.load_dict(blob)
    assert mm2.stats() == mm.stats()
    assert mm2.to_dict() == blob
    # and the restored manifold behaves identically on the next observation
    r1 = mm.observe("a brand new follow-up thought")
    r2 = mm2.observe("a brand new follow-up thought")
    assert r1.to_dict() == r2.to_dict()


def test_load_dict_survives_garbage():
    mm = HyperbolicConceptManifold(dim=4, seed=1)
    mm.load_dict({"nodes": "not-a-dict", "edges": None})  # must not raise
    assert mm.stats()["n_nodes"] == 0


def test_nearest_and_distance():
    mm = HyperbolicConceptManifold(dim=8, tau=2.2, seed=11,
                                   embedder=BallEmbedder(dim=8, seed=11, alpha=1.0))
    mm.observe("the moon orbits the earth", concept_id="moon")
    mm.observe("sourdough bread recipe", concept_id="bread")
    hits = mm.nearest("lunar orbit of the moon", k=1)
    assert hits and hits[0][0] == "moon"
    d = mm.distance("moon", "bread")
    assert d is not None and d > 0.0
    assert mm.distance("moon", "ghost") is None
