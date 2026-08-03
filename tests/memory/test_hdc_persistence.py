"""The HDC corpus and the holographic field survive a restart — by recipe, not by vector."""

from __future__ import annotations

from nyxara.cognition.hyper_dimensional_vectors import LatentSpaceMap
from nyxara.memory.holographic_field import HolographicMemoryField


def _map() -> LatentSpaceMap:
    space = LatentSpaceMap(dim=512, seed=7, max_corpus=100)
    space.add("alpha", {"colour": "red", "size": 3})
    space.add("beta", "the quick brown fox")
    space.add("gamma", ["a", "b", "c"])
    return space


def _field() -> HolographicMemoryField:
    field = HolographicMemoryField(dim=512, seed=7, capacity=50)
    field.remember("master", "Jaypal Khoja is the Master")
    field.remember("brain", "a pure NumPy transformer")
    return field


# -------------------- LatentSpaceMap -------------------- #
def test_corpus_round_trips_exactly(tmp_path):
    original = _map()
    path = tmp_path / "hdc.json"
    assert original.save(path)

    restored = LatentSpaceMap(dim=512, seed=7, max_corpus=100)
    assert restored.load(path)
    assert len(restored) == len(original)
    for name, vector in original._corpus.items():
        assert restored._corpus[name] == vector       # recomputed, and identical


def test_the_sidecar_stores_recipes_not_ten_thousand_floats(tmp_path):
    """Persisting the vectors would be megabytes, and bundled vectors are real-valued so they
    cannot be bit-packed without loss. The encoders are deterministic, so the input is enough."""
    path = tmp_path / "hdc.json"
    _map().save(path)
    assert path.stat().st_size < 4096
    assert "sources" in path.read_text(encoding="utf-8")


def test_loading_into_a_populated_map_is_still_exact(tmp_path):
    """ItemMemory draws a fresh random vector on first request, so first-request ORDER decides
    every vector. Without resetting the space, replaying recipes onto a dirty map restores a
    corpus that looks right and recalls wrongly."""
    original = _map()
    path = tmp_path / "hdc.json"
    original.save(path)

    dirty = LatentSpaceMap(dim=512, seed=7, max_corpus=100)
    dirty.add("junk", "completely unrelated noise")
    assert dirty.load(path)
    assert len(dirty) == len(original)
    for name, vector in original._corpus.items():
        assert dirty._corpus[name] == vector


def test_a_mismatched_space_is_refused(tmp_path):
    path = tmp_path / "hdc.json"
    _map().save(path)
    assert LatentSpaceMap(dim=256, seed=7).load(path) is False
    assert LatentSpaceMap(dim=512, seed=8).load(path) is False


def test_a_corrupt_sidecar_never_blocks_boot(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{ not json", encoding="utf-8")
    assert LatentSpaceMap(dim=512, seed=7).load(path) is False


def test_unreplayable_inputs_are_left_out_rather_than_persisted_wrong():
    space = LatentSpaceMap(dim=512, seed=7)
    space.add("text", "an ordinary string")
    space.add("raw", space.space.random())        # a Hypervector cannot round-trip through JSON
    assert "text" in space._sources
    assert "raw" not in space._sources


def test_forgetting_drops_the_recipe_too(tmp_path):
    space = _map()
    assert space.forget("beta")
    path = tmp_path / "hdc.json"
    space.save(path)
    restored = LatentSpaceMap(dim=512, seed=7, max_corpus=100)
    restored.load(path)
    assert "beta" not in restored._corpus


# -------------------- HolographicMemoryField -------------------- #
def test_the_field_is_rebuilt_bit_for_bit(tmp_path):
    original = _field()
    path = tmp_path / "holo.json"
    assert original.save(path)

    restored = HolographicMemoryField(dim=512, seed=7, capacity=50)
    assert restored.load(path)
    assert restored.field == original.field
    assert restored.recall("master").text == "Jaypal Khoja is the Master"


def test_the_field_vector_is_not_what_gets_stored(tmp_path):
    path = tmp_path / "holo.json"
    _field().save(path)
    assert path.stat().st_size < 2048
    assert "pairs" in path.read_text(encoding="utf-8")


def test_loading_into_a_populated_field_is_still_exact(tmp_path):
    original = _field()
    path = tmp_path / "holo.json"
    original.save(path)

    dirty = HolographicMemoryField(dim=512, seed=7, capacity=50)
    dirty.remember("junk", "unrelated noise")
    assert dirty.load(path)
    assert dirty.field == original.field
    assert dirty.recall("brain").text == "a pure NumPy transformer"


def test_a_mismatched_field_is_refused(tmp_path):
    path = tmp_path / "holo.json"
    _field().save(path)
    assert HolographicMemoryField(dim=512, seed=9).load(path) is False
    assert HolographicMemoryField(dim=256, seed=7).load(path) is False


def test_a_corrupt_field_sidecar_is_survivable(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("nonsense", encoding="utf-8")
    assert HolographicMemoryField(dim=512, seed=7).load(path) is False
