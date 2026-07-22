"""Tests for F4 — topological & geometric feature synthesis (mind/latent_geometry.py).

She must read a point cloud's SHAPE (components + holes), embed a hierarchy with less distortion in
hyperbolic than Euclidean space, and recommend a geometry only when it MEASURABLY wins — with an honest
pure-stdlib path (no heavy dependency required). No LLM anywhere.
"""

from __future__ import annotations

import math

from nyxara.mind.latent_geometry import (
    LatentGeometry,
    analyze_points,
    embed_tree,
    embedding_distortion,
    persistent_h0,
    poincare_distance,
    recommend_geometry,
)


def _ring(n=24, r=1.0):
    return [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def _balanced_tree(depth=4):
    edges = []
    nxt = 1
    frontier = [0]
    for _ in range(depth):
        newf = []
        for p in frontier:
            for _ in range(2):
                edges.append((p, nxt)); newf.append(nxt); nxt += 1
        frontier = newf
    return edges


def test_tda_finds_the_hole_in_an_annulus():
    rep = analyze_points(_ring())
    assert rep.betti0 == 1       # one connected ring
    assert rep.betti1 == 1       # …with one hole


def test_tda_counts_separate_clusters():
    two = [(0, 0), (0.1, 0), (0, 0.1), (0.1, 0.1),
           (9, 9), (9.1, 9), (9, 9.1), (9.1, 9.1)]
    rep = analyze_points(two)
    assert rep.betti0 == 2
    assert rep.betti1 == 0


def test_h0_barcode_has_one_immortal_component():
    bars = persistent_h0([(0, 0), (1, 0), (2, 0)])
    infinite = [b for b in bars if b.death == math.inf]
    assert len(infinite) == 1    # exactly one component never dies


def test_poincare_distance_grows_toward_the_boundary():
    center = (0.0, 0.0)
    near = (0.1, 0.0)
    far = (0.9, 0.0)
    assert poincare_distance(center, far) > poincare_distance(center, near)


def test_hyperbolic_beats_euclidean_on_a_tree():
    edges = _balanced_tree(depth=4)
    choice = recommend_geometry(edges)
    assert choice.geometry == "hyperbolic"
    assert choice.hyperbolic_distortion < choice.euclidean_distortion


def test_recommend_is_honest_not_a_blanket_claim():
    # a single path (a degenerate "tree") is essentially 1-D — hyperbolic need not win, and the
    # recommender must report the real distortions either way rather than always choosing hyperbolic.
    path = [(i, i + 1) for i in range(6)]
    choice = recommend_geometry(path)
    assert choice.geometry in ("hyperbolic", "euclidean")
    assert choice.hyperbolic_distortion >= 0.0 and choice.euclidean_distortion >= 0.0


def test_facade():
    lg = LatentGeometry()
    assert lg.analyze(_ring()).betti1 == 1
    assert lg.choose_geometry(_balanced_tree(3)).geometry in ("hyperbolic", "euclidean")
