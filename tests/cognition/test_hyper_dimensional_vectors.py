"""Tests for nyxara.cognition.hyper_dimensional_vectors."""

from __future__ import annotations

import pytest

from nyxara.cognition.hyper_dimensional_vectors import (
    CleanupResult,
    HyperSpace,
    Hypervector,
    ItemMemory,
    LatentSpaceMap,
    NoveltyResult,
    PatternReport,
    RandomProjector,
)


# -------------------- the algebra: blessing of dimensionality -------------------- #
def test_random_vectors_are_near_orthogonal():
    space = HyperSpace(dim=10000, seed=1)
    a, b = space.random(), space.random()
    assert abs(space.similarity(a, b)) < 0.05


def test_random_vectors_are_bipolar():
    space = HyperSpace(dim=256, seed=1)
    v = space.random()
    assert all(x in (-1.0, 1.0) for x in v.to_list())


def test_self_similarity_is_one():
    space = HyperSpace(dim=1024, seed=2)
    a = space.random()
    assert space.similarity(a, a) == pytest.approx(1.0)


# -------------------- bind / unbind (associations) -------------------- #
def test_bind_is_dissimilar_to_inputs():
    space = HyperSpace(dim=10000, seed=3)
    a, b = space.random(), space.random()
    bound = space.bind(a, b)
    assert abs(space.similarity(bound, a)) < 0.05
    assert abs(space.similarity(bound, b)) < 0.05


def test_unbind_recovers_partner():
    space = HyperSpace(dim=10000, seed=4)
    key, val = space.random(), space.random()
    bound = space.bind(key, val)
    recovered = space.unbind(bound, key)
    assert space.similarity(recovered, val) > 0.99


# -------------------- bundle (superposition / sets) -------------------- #
def test_bundle_is_similar_to_all_members():
    space = HyperSpace(dim=10000, seed=5)
    members = [space.random() for _ in range(3)]
    bundle = space.bundle(*members)
    for m in members:
        assert space.similarity(bundle, m) > 0.4


def test_empty_bundle_is_zero():
    space = HyperSpace(dim=64, seed=5)
    z = space.bundle()
    assert all(x == 0.0 for x in z.to_list())


# -------------------- permutation (order/position) -------------------- #
def test_permute_is_reversible():
    space = HyperSpace(dim=1000, seed=6)
    a = space.random()
    assert space.unpermute(space.permute(a, 5), 5) == a


def test_permutation_makes_sequences_order_sensitive():
    space = HyperSpace(dim=10000, seed=7)
    a, b = space.random(), space.random()
    ab = space.bundle(space.permute(a, 0), space.permute(b, 1))
    ba = space.bundle(space.permute(b, 0), space.permute(a, 1))
    assert space.similarity(ab, ba) < 0.2


# -------------------- item memory / cleanup -------------------- #
def test_cleanup_recovers_nearest_symbol():
    space = HyperSpace(dim=10000, seed=8)
    mem = ItemMemory(space)
    for name in ("alpha", "beta", "gamma"):
        mem.add(name)
    target = mem.get("beta")
    # add noise by bundling with an unrelated vector, beta should still dominate
    noisy = space.bundle(target, target, space.random())
    res = mem.cleanup(noisy)
    assert isinstance(res, CleanupResult)
    assert res.name == "beta"
    assert res.margin > 0


def test_cleanup_respects_exclude():
    space = HyperSpace(dim=4096, seed=9)
    mem = ItemMemory(space)
    a = mem.add("a")
    mem.add("b")
    res = mem.cleanup(a, exclude=["a"])
    assert res.name != "a"


def test_symbol_is_stable():
    space = HyperSpace(dim=512, seed=10)
    mem = ItemMemory(space)
    assert mem.symbol("x") == mem.symbol("x")


# -------------------- random projection (SimHash bridge) -------------------- #
def test_projection_preserves_angular_order():
    pytest.importorskip("numpy")
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.standard_normal(64).astype("float32")
    y = x + 0.1 * rng.standard_normal(64).astype("float32")  # close to x
    z = rng.standard_normal(64).astype("float32")            # unrelated
    proj = RandomProjector(64, 10000, seed=1)
    space = HyperSpace(dim=10000, seed=1)
    px, py, pz = proj.project(x), proj.project(y), proj.project(z)
    assert space.similarity(px, py) > space.similarity(px, pz)


def test_projection_dimension_mismatch_raises():
    proj = RandomProjector(8, 256, seed=1)
    with pytest.raises(ValueError):
        proj.project([1.0, 2.0, 3.0])


# -------------------- continuous-value (level) encoding -------------------- #
def test_numeric_encoding_is_locally_smooth():
    lm = LatentSpaceMap(dim=10000, seed=11, num_levels=64)
    near_a = lm.encode_numeric(0.50, 0.0, 1.0)
    near_b = lm.encode_numeric(0.52, 0.0, 1.0)
    far = lm.encode_numeric(0.05, 0.0, 1.0)
    s_near = lm.space.similarity(near_a, near_b)
    s_far = lm.space.similarity(near_a, far)
    assert s_near > s_far


# -------------------- relational analogy -------------------- #
def test_analogy_dollar_of_mexico_is_peso():
    lm = LatentSpaceMap(dim=10000, seed=42)
    lm.add("USA", {"name": "usa", "capital": "washington", "currency": "dollar"})
    lm.add("MEX", {"name": "mexico", "capital": "mexico_city", "currency": "peso"})
    res = lm.analogy("USA", "MEX", "dollar")
    assert res.name == "peso"


def test_analogy_capital_of_mexico():
    lm = LatentSpaceMap(dim=10000, seed=42)
    lm.add("USA", {"name": "usa", "capital": "washington", "currency": "dollar"})
    lm.add("MEX", {"name": "mexico", "capital": "mexico_city", "currency": "peso"})
    assert lm.analogy("USA", "MEX", "washington").name == "mexico_city"


# -------------------- corpus: recall, discovery, novelty -------------------- #
def _market_map():
    lm = LatentSpaceMap(dim=10000, seed=7, novelty_threshold=0.85)
    lm.add("buy_tech_1", {"action": "buy", "sector": "tech", "size": "large"})
    lm.add("buy_tech_2", {"action": "buy", "sector": "tech", "size": "small"})
    lm.add("sell_energy", {"action": "sell", "sector": "energy", "size": "large"})
    return lm


def test_nearest_recovers_similar_records():
    lm = _market_map()
    hits = lm.nearest({"action": "buy", "sector": "tech", "size": "medium"}, k=1)
    assert hits[0][0] in {"buy_tech_1", "buy_tech_2"}


def test_discover_patterns_clusters_similar_records():
    lm = _market_map()
    report = lm.discover_patterns(threshold=0.45)
    assert isinstance(report, PatternReport)
    assert any(set(cl) == {"buy_tech_1", "buy_tech_2"} for cl in report.clusters)
    assert report.n_items == 3


def test_novelty_flags_unseen_regime():
    lm = _market_map()
    nov = lm.novelty({"action": "short", "sector": "crypto", "size": "huge"})
    assert isinstance(nov, NoveltyResult)
    assert nov.is_novel


def test_novelty_does_not_flag_familiar():
    lm = _market_map()
    nov = lm.novelty({"action": "buy", "sector": "tech", "size": "large"})
    assert not nov.is_novel


def test_empty_corpus_is_always_novel():
    lm = LatentSpaceMap(dim=1024, seed=1)
    assert lm.novelty({"a": "b"}).is_novel


# -------------------- encoding dispatch + determinism -------------------- #
def test_encode_dispatch_by_type():
    lm = LatentSpaceMap(dim=2048, seed=1)
    assert isinstance(lm.encode("hello world"), Hypervector)
    assert isinstance(lm.encode({"k": "v"}), Hypervector)
    assert isinstance(lm.encode([1.0, 2.0, 3.0]), Hypervector)
    assert isinstance(lm.encode(["a", "b", "c"]), Hypervector)


def test_encoding_is_deterministic_under_seed():
    a = LatentSpaceMap(dim=4096, seed=99).encode_record({"x": "one", "y": "two"})
    b = LatentSpaceMap(dim=4096, seed=99).encode_record({"x": "one", "y": "two"})
    assert a == b


def test_top_level_exports():
    import nyxara

    assert nyxara.LatentSpaceMap is LatentSpaceMap
    assert nyxara.HyperSpace is HyperSpace


# -------------------- bounded corpus (FIFO) -------------------- #
def test_max_corpus_evicts_oldest():
    lm = LatentSpaceMap(dim=2048, seed=1, max_corpus=2)
    lm.add("a", {"x": "1"})
    lm.add("b", {"x": "2"})
    lm.add("c", {"x": "3"})
    assert len(lm) == 2
    assert lm.nearest({"x": "3"}, k=3)  # newest present
    names = {n for n, _ in lm.nearest({"x": "2"}, k=3)}
    assert "a" not in names  # oldest evicted


def test_forget_removes_item():
    lm = LatentSpaceMap(dim=1024, seed=1)
    lm.add("a", {"x": "1"})
    assert lm.forget("a")
    assert not lm.forget("a")
    assert len(lm) == 0


# -------------------- orchestrator integration -------------------- #
def test_core_wires_hyperdimensional_faculty():
    from nyxara import NyxaraCore

    core = NyxaraCore()
    assert isinstance(core.hyperdimensional, LatentSpaceMap)


def test_core_ingests_turns_into_latent_space():
    from nyxara import Authority, NyxaraCore

    core = NyxaraCore()
    core.process("market trends in technology stocks", authority=Authority.OWNER)
    core.process("energy sector and oil price behaviour", authority=Authority.OWNER)
    assert len(core.hyperdimensional) == 2
    # a repeated stimulus is recognised — not novel — once it has been seen
    seen = core.latent_novelty("market trends in technology stocks")
    assert not seen.is_novel


def test_core_latent_public_api():
    from nyxara import Authority, NyxaraCore

    core = NyxaraCore()
    core.process("buy technology shares now", authority=Authority.OWNER)
    assert core.map_latent_space("a probe sentence") is not None
    assert isinstance(core.latent_recall("technology equities", k=2), list)
    assert core.latent_patterns() is not None
