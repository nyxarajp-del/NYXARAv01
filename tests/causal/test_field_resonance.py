"""Tests for nyxara.causal.field_resonance."""

from __future__ import annotations

from nyxara.causal.field_resonance import ResonanceField, embed_field


def test_embedding_is_stable_and_unit_norm():
    a = embed_field("gravity attracts mass", dim=256)
    b = embed_field("gravity attracts mass", dim=256)
    assert a == b                                   # deterministic
    assert len(a) == 256
    norm = sum(x * x for x in a) ** 0.5
    assert abs(norm - 1.0) < 1e-9 or norm == 0.0    # unit norm


def test_resonance_prefers_the_matching_concept():
    field = ResonanceField(dim=512, resonance_floor=0.05)
    field.imprint("gravity", "gravity mass weight falling objects attract force")
    field.imprint("python", "python code function class module programming language")
    best = field.best("why do heavy objects fall — gravity and weight")
    assert best is not None
    assert best.label == "gravity"
    assert -1.0 <= best.score <= 1.0


def test_self_resonance_is_maximal():
    field = ResonanceField(dim=512)
    text = "sovereign cognitive architecture owned by the Master"
    field.imprint("identity", text)
    hit = field.best(text)
    assert hit is not None and hit.label == "identity"
    assert hit.score > 0.99                          # a field resonates maximally with itself


def test_gap_detection_triggers_when_nothing_resonates():
    field = ResonanceField(dim=512, resonance_floor=0.3)
    field.imprint("weather", "rain sun cloud storm temperature forecast")
    assert field.is_gap("zzyzx qwibble frobnicate quuxatron") is True
    assert field.is_gap("what is the rain forecast temperature") is False


def test_capacity_evicts_oldest():
    field = ResonanceField(dim=64, capacity=3)
    for i in range(5):
        field.imprint(f"c{i}", f"concept number {i}")
    assert len(field) == 3
    assert "c0" not in field.labels() and "c4" in field.labels()


def test_reinforce_raises_weight_and_ranking():
    field = ResonanceField(dim=128, resonance_floor=0.0)
    field.imprint("a", "shared token alpha beta")
    field.imprint("b", "shared token alpha beta")
    field.reinforce("b", by=5.0)
    hits = field.resonate("shared token alpha beta", top=2)
    assert hits[0].label == "b"                      # reinforced field ranks first
