"""NYX V.01 · content-addressed memory — recall without a token window, and honest limits."""

from __future__ import annotations

from nyxara.nyx.holomem import HoloMemory


def _mem(**kw) -> HoloMemory:
    kw.setdefault("dim", 1024)                 # small but real HDC width — keeps the suite fast
    kw.setdefault("capacity", 32)
    kw.setdefault("seed", 7)
    return HoloMemory(**kw)


def test_content_addressed_recall_finds_the_right_trace():
    m = _mem()
    m.remember("a", "gravity pulls an apple toward the ground", kind="fact")
    m.remember("b", "the cat sat on the warm mat")
    rec = m.recall("apple gravity")
    assert rec.hit is not None and rec.hit.key == "a"
    assert rec.decided is True


def test_recall_is_not_recency_biased():
    """The claim that replaces the context window: age does not decide reachability."""
    m = _mem(capacity=64)
    m.remember("old", "a hummingbird hovers by beating its wings very fast")
    for i in range(40):
        m.remember(f"filler{i}", f"an unrelated remark number {i} about nothing much")
    rec = m.recall("hummingbird wings")
    assert rec.hit is not None and rec.hit.key == "old"


def test_low_confidence_recall_reports_itself_rather_than_guessing():
    m = _mem()
    m.remember("a", "gravity pulls an apple toward the ground")
    rec = m.recall("zzzz qqqq vvvv")           # nothing like anything stored
    assert rec.decided is False


def test_recall_on_empty_memory_is_empty_not_an_error():
    rec = _mem().recall("anything at all")
    assert rec.hit is None and rec.decided is False and rec.text == ""


def test_writes_in_the_same_moment_become_linked():
    m = _mem()
    m.remember("a", "the cat sat on the warm mat")
    m.remember("b", "the cat likes a warm lap")
    assert "a" in m.traces["b"].links and "b" in m.traces["a"].links


def test_links_saturate_and_stay_bounded():
    m = _mem(max_links_per_trace=2)
    for i in range(8):
        m.remember(f"k{i}", f"a sentence about topic number {i}")
    assert all(len(t.links) <= 2 for t in m.traces.values())


def test_explicit_association_is_symmetric():
    m = _mem()
    m.remember("a", "alpha content here")
    m.remember("b", "beta content here")
    m.associate("a", "b", weight=0.9)
    assert m.traces["a"].links["b"] > 0.5 and m.traces["b"].links["a"] > 0.5


def test_context_returns_the_hit_and_its_constellation():
    m = _mem()
    m.remember("a", "gravity pulls an apple toward the ground")
    m.remember("b", "an apple is a fruit that grows on a tree")
    ctx = m.context("apple gravity", k=3)
    assert ctx and ctx[0].key in {"a", "b"}
    assert len(ctx) <= 3


def test_capacity_evicts_and_reports_the_spill():
    """It forgets — on purpose. That is the honest counterpart to 'no context window'."""
    m = _mem(capacity=4)
    for i in range(12):
        m.remember(f"k{i}", f"a distinct sentence about subject {i}")
    assert len(m) <= 4
    assert m.spilled, "eviction must be surfaced so a caller can persist it elsewhere"


def test_rewriting_a_key_keeps_one_trace():
    m = _mem()
    m.remember("a", "the first version of this thought")
    m.remember("a", "the second version of this thought")
    assert len(m) == 1
    assert "second" in m.recall_key("a").text


def test_round_trip_preserves_traces_links_and_recall():
    m = _mem()
    m.remember("a", "gravity pulls an apple toward the ground", kind="fact")
    m.remember("b", "the cat sat on the warm mat")
    data = m.to_dict()

    restored = HoloMemory(dim=1024, capacity=32, seed=7)
    assert restored.load_dict(data) is True
    assert len(restored) == len(m)
    assert restored.recall_key("a").kind == "fact"
    assert restored.traces["b"].links == m.traces["b"].links
    rec = restored.recall("apple gravity")
    assert rec.hit is not None and rec.hit.key == "a"


def test_load_refuses_a_sidecar_from_a_different_space():
    """A different (dim, seed) would rebuild a field that looks fine and recalls wrongly."""
    m = _mem(dim=1024, seed=7)
    m.remember("a", "something worth keeping")
    data = m.to_dict()

    other = HoloMemory(dim=2048, capacity=32, seed=7)
    assert other.load_dict(data) is False
    assert len(other) == 0


def test_corrupt_sidecar_is_an_empty_memory_not_a_crash():
    m = _mem()
    assert m.load_dict(None) is False
    assert m.load_dict({"dim": 1024, "seed": 7, "traces": "nonsense"}) is True
    assert len(m) == 0


def test_writes_never_raise_on_hostile_input():
    m = _mem()
    for junk in (None, "", "   "):
        assert m.remember("k", junk) is None
    assert m.remember("", "text") is None
    assert m.recall(None).hit is None
