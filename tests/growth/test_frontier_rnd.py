"""NYXARA · tests/growth/test_frontier_rnd.py — curiosity that LEARNS BY GRADIENT DESCENT.

The frontier used to score novelty by pure distance (kNN). These prove the new Random Network
Distillation signal is a *learned* one: a predictor net trained by back-prop whose error falls where
NYXARA has been (habituation is learned, not counted) and stays high where she has not — blended with
the kNN floor, degrading to pure kNN without numpy, and persisting byte-for-byte.
"""

from __future__ import annotations

import json

import pytest

try:
    import numpy as np  # noqa: F401
    _HAS_NUMPY = True
except Exception:  # pragma: no cover - depends on install
    _HAS_NUMPY = False

needs_numpy = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")

from nyxara.growth.frontier import BehaviorDescriptor, Discovery, NoveltyArchive


def _disc(dims):
    return Discovery(descriptor=BehaviorDescriptor(dims=dims), quality=1.0, content="x")


@needs_numpy
def test_rnd_novelty_habituates_where_she_has_been():
    arc = NoveltyArchive(ndims=3, learned_novelty=True)
    assert arc._rnd is not None
    region = (0.2, 0.2, 0.2)
    fresh = (0.9, 0.9, 0.9)
    before = arc._rnd.score(region)
    for _ in range(200):
        arc.add(_disc(region))                        # repeated exposure → predictor catches up
    after_region = arc._rnd.score(region)
    after_fresh = arc._rnd.score(fresh)
    assert after_region < before                      # learned habituation (error fell by gradient)
    assert after_fresh > after_region                 # the unseen region is still novel


@needs_numpy
def test_rnd_predictor_weights_actually_move():
    arc = NoveltyArchive(ndims=3, learned_novelty=True)
    w0 = arc._rnd._pred.W[0].copy()
    for _ in range(50):
        arc.add(_disc((0.3, 0.4, 0.5)))
    assert not np.allclose(w0, arc._rnd._pred.W[0])   # real gradient steps changed the weights


def test_blend_degrades_to_pure_knn_when_disabled():
    arc = NoveltyArchive(ndims=3, learned_novelty=False)
    assert arc._rnd is None
    for _ in range(5):
        arc.add(_disc((0.1, 0.1, 0.1)))
    d = BehaviorDescriptor(dims=(0.5, 0.5, 0.5))
    assert arc.novelty(d) == arc._knn_novelty(d)      # identical to the counting floor


def test_empty_archive_is_maximally_novel():
    for learned in (True, False):
        arc = NoveltyArchive(ndims=3, learned_novelty=learned)
        assert arc.novelty(BehaviorDescriptor(dims=(0.4, 0.4, 0.4))) == 1.0


@needs_numpy
def test_rnd_roundtrips_through_json():
    arc = NoveltyArchive(ndims=3, learned_novelty=True)
    for _ in range(80):
        arc.add(_disc((0.25, 0.35, 0.45)))
    probe = (0.25, 0.35, 0.45)
    before = arc._rnd.score(probe)
    blob = json.loads(json.dumps(arc.to_dict()))       # JSON-safe end to end
    arc2 = NoveltyArchive.from_dict(blob, learned_novelty=True)
    assert arc2._rnd is not None
    assert abs(arc2._rnd.score(probe) - before) < 1e-9
