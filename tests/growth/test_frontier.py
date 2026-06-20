"""Tests for nyxara.growth.frontier — open-ended discovery (Pillar F · Edge 1)."""

from __future__ import annotations

from nyxara.growth.frontier import (
    BehaviorDescriptor,
    Discovery,
    FrontierEngine,
    NoveltyArchive,
)
from nyxara.memory.store import MemoryStore


# --------------------------------------------------------------------------- #
# descriptor
# --------------------------------------------------------------------------- #
def test_descriptor_clamps_and_buckets():
    d = BehaviorDescriptor(dims=(1.5, -0.2, 0.5))
    assert d.dims == (1.0, 0.0, 0.5)                  # clamped to [0,1]
    assert d.cell(8) == (7, 0, 4)                     # bucketed into 8 bins

def test_descriptor_distance_is_normalised():
    a = BehaviorDescriptor(dims=(0.0, 0.0, 0.0))
    b = BehaviorDescriptor(dims=(1.0, 1.0, 1.0))
    assert abs(a.distance(b) - 1.0) < 1e-9            # full diagonal -> 1.0
    assert a.distance(a) == 0.0


# --------------------------------------------------------------------------- #
# archive
# --------------------------------------------------------------------------- #
def test_add_new_niche_is_novel():
    arc = NoveltyArchive(resolution=8, ndims=3)
    r = arc.add(Discovery("q1", BehaviorDescriptor(dims=(0.1, 0.1, 0.1)), quality=0.5))
    assert r["novel"] is True and r["coverage"] == 1
    assert r["novelty"] == 1.0                        # empty archive -> maximally novel

def test_better_quality_replaces_elite_same_cell():
    arc = NoveltyArchive(resolution=8, ndims=3)
    desc = BehaviorDescriptor(dims=(0.1, 0.1, 0.1))
    arc.add(Discovery("weak", desc, quality=0.3))
    r = arc.add(Discovery("strong", desc, quality=0.9))
    assert r["novel"] is False and r["improved"] is True
    assert arc.coverage() == 1                        # same niche, not a new one
    assert arc.elites()[0].content == "strong"

def test_weaker_quality_does_not_replace():
    arc = NoveltyArchive(resolution=8, ndims=3)
    desc = BehaviorDescriptor(dims=(0.1, 0.1, 0.1))
    arc.add(Discovery("strong", desc, quality=0.9))
    r = arc.add(Discovery("weak", desc, quality=0.3))
    assert r["improved"] is False
    assert arc.elites()[0].content == "strong"

def test_novelty_drops_for_nearby_behaviour():
    arc = NoveltyArchive(resolution=8, k=3, ndims=3)
    arc.add(Discovery("a", BehaviorDescriptor(dims=(0.1, 0.1, 0.1)), quality=0.5))
    near = arc.novelty(BehaviorDescriptor(dims=(0.12, 0.1, 0.1)))
    far = arc.novelty(BehaviorDescriptor(dims=(0.9, 0.9, 0.9)))
    assert far > near                                 # the distant point is more novel

def test_sparsest_bins_points_away_from_filled_region():
    arc = NoveltyArchive(resolution=4, ndims=3)
    # fill only low corner cells on axis 0
    for _ in range(3):
        arc.add(Discovery("x", BehaviorDescriptor(dims=(0.0, 0.0, 0.0)), quality=0.5))
    bins = arc.sparsest_bins()
    assert bins[0] != 0                               # axis 0 bin 0 is crowded -> avoid it


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
def test_characterize_is_deterministic_and_domain_aware():
    eng = FrontierEngine()
    d1 = eng.characterize("What is 2 + 2?", domain="mathematics")
    d2 = eng.characterize("What is 2 + 2?", domain="mathematics")
    d3 = eng.characterize("Write a poem about the sea.", domain="language")
    assert d1.dims == d2.dims                         # deterministic
    assert d1.dims[0] != d3.dims[0]                   # different domains -> different niche axis

def test_ingest_grows_coverage_and_reports():
    eng = FrontierEngine()
    eng.ingest("What is 17 * 23?", domain="mathematics", save=False)
    eng.ingest("Reverse a linked list.", domain="code", save=False)
    rep = eng.report()
    assert rep["coverage"] == 2 and rep["total_seen"] == 2
    assert 0.0 <= rep["coverage_frac"] <= 1.0

def test_next_direction_names_an_underexplored_topic():
    eng = FrontierEngine()
    # crowd mathematics so the next direction steers elsewhere
    for i in range(6):
        eng.ingest(f"math problem {i} with numbers {i*7}+{i}", domain="mathematics", save=False)
    nd = eng.next_direction()
    assert "problems in" in nd["topic"]
    assert nd["domain"] != "mathematics"

def test_persistence_round_trips_through_memory():
    mem = MemoryStore()
    eng = FrontierEngine(memory=mem)
    eng.ingest("What is 2 + 2?", domain="mathematics")     # save=True by default
    eng.ingest("Why is the sky blue?", domain="science")
    cov = eng.report()["coverage"]

    reloaded = FrontierEngine(memory=mem)
    assert reloaded.report()["coverage"] == cov            # survived via long-term memory
    assert reloaded.archive().total_seen == 2

def test_in_process_when_no_memory():
    eng = FrontierEngine(memory=None)
    r = eng.ingest("standalone", domain="reasoning")
    assert r["coverage"] == 1                               # works without a store
    assert eng.save() is False                              # honestly reports no persistence
