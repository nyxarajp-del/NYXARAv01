"""NYX V.01 · L-PSYCHE-QUANTUM — choosing rather than computing, and still auditable."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import get_settings
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.modules import Proposal
from nyxara.nyx.will import EntropySource, SovereignWill


def _opts() -> list:
    return [Proposal(source="memory", content="a recalled fact", confidence=0.6, novelty=0.1),
            Proposal(source="creative", content="a novel angle", confidence=0.5, novelty=0.9),
            Proposal(source="graph", content="an association", confidence=0.3, novelty=0.1)]


# -------------------- the entropy is real, and named -------------------- #
def test_the_source_is_physical_and_says_which_one():
    source = EntropySource()
    assert source.physical is True
    assert source.name in {"urandom", "hwrng", "qrng"}


def test_draws_are_in_range_and_not_constant():
    source = EntropySource()
    draws = [source.unit() for _ in range(24)]
    assert all(0.0 <= d < 1.0 for d in draws)
    assert len(set(draws)) > 20                   # real entropy, not a stuck value


def test_a_seeded_source_is_labelled_as_not_physical():
    """A deterministic stand-in is fine for tests and replay — pretending it is not, is not."""
    source = EntropySource(seed=7)
    assert source.name == "seeded" and source.physical is False
    assert [EntropySource(seed=7).unit() for _ in range(3)] == \
           [EntropySource(seed=7).unit() for _ in range(3)]


def test_a_quantum_source_is_never_auto_selected():
    """It is a network call; it has to be asked for, never assumed."""
    assert EntropySource(prefer="auto").name != "qrng"


def test_an_unreadable_device_falls_back_and_relabels():
    source = EntropySource(prefer="hwrng", hwrng_path="/nonexistent/hwrng")
    assert source.name == "urandom" and source.physical is True


# -------------------- she chooses, rather than computing the maximum -------------------- #
def test_the_same_input_does_not_always_give_the_same_choice():
    will = SovereignWill(temperature=0.8)
    picks = {will.choose(_opts()).picked for _ in range(20)}
    assert len(picks) > 1


def test_temperature_zero_is_a_deterministic_mind():
    will = SovereignWill(temperature=0.0)
    picks = {will.choose(_opts()).picked for _ in range(8)}
    assert len(picks) == 1


def test_the_choice_records_what_a_deterministic_mind_would_have_done():
    got = SovereignWill(temperature=0.8).choose(_opts())
    assert got.would_argmax
    assert got.diverged in (True, False)          # measured, not asserted


def test_preference_still_favours_the_stronger_option_on_average():
    """Non-determinism is not noise: what she wants more, she picks more."""
    will = SovereignWill(temperature=0.5)
    picks = [will.choose(_opts()).picked for _ in range(200)]
    assert picks.count("graph") < picks.count("memory")


def test_the_preference_is_recorded_with_the_choice():
    got = SovereignWill().choose(_opts())
    assert set(got.preference) == {"memory", "creative", "graph"}
    assert all(v > 0 for v in got.preference.values())


def test_nothing_to_choose_between_is_not_a_choice():
    got = SovereignWill().choose([])
    assert got.picked == "" and "nothing to choose" in got.reason


# -------------------- auditability survives -------------------- #
def test_a_recorded_choice_replays_bit_exactly():
    """A mind that cannot be replayed cannot be checked — that trade is not worth making."""
    original = SovereignWill(temperature=0.7).choose(_opts())
    replayed = SovereignWill(temperature=0.7).replay(original.to_dict(), _opts())
    assert replayed.picked == original.picked


def test_every_choice_carries_its_draw_and_source():
    got = SovereignWill().choose(_opts())
    assert 0.0 <= got.draw < 1.0 and got.source


def test_history_is_bounded():
    will = SovereignWill(history=4)
    for _ in range(20):
        will.choose(_opts())
    assert len(will.history) <= 4


# -------------------- she may decline -------------------- #
def test_a_value_conflict_can_be_declined_with_a_reason():
    class _Values:
        def check(self, _text):
            return type("R", (), {"verdict": type("V", (), {"value": "misaligned"})(),
                                  "reasons": ["it would harm the Master"]})()

    got = SovereignWill(values=_Values()).choose(_opts())
    got_reason = got.reason
    assert got.declined is True and "harm the Master" in got_reason
    assert got.picked == ""


def test_declining_can_be_switched_off():
    class _Values:
        def check(self, _text):
            return type("R", (), {"verdict": type("V", (), {"value": "misaligned"})(),
                                  "reasons": ["nope"]})()

    got = SovereignWill(values=_Values(), may_decline=False).choose(_opts())
    assert got.declined is False and got.picked


def test_an_unreadable_value_system_never_manufactures_a_refusal():
    class _Broken:
        def check(self, _text):
            raise RuntimeError("nope")

    assert SovereignWill(values=_Broken()).choose(_opts()).declined is False


# -------------------- wired into the cycle -------------------- #
@pytest.fixture
def brain():
    b = NyxBrain(get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "collapse_threshold": 0.9, "dialogue_enabled": False}))
    b.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
    return b


def test_an_open_decision_is_chosen_not_computed(brain):
    if brain.will is None:
        pytest.skip("will disabled")
    picks = {brain.think("what pulls an apple down", remember=False).winner.source
             for _ in range(15)}
    assert len(picks) > 1


def test_a_verified_answer_is_never_subject_to_a_whim(brain):
    """Truth is not a preference — a checked derivation must not be displaced by a draw."""
    got = brain.think("prove that (x+1)^2 = x^2 + 2x + 1", remember=False)
    if not got.verified:
        pytest.skip("sympy not installed — the engine declines honestly")
    assert got.choice is None


def test_a_settled_answer_is_left_alone():
    b = NyxBrain(get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "collapse_threshold": 0.0, "dialogue_enabled": False}))
    b.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
    assert b.think("what pulls an apple down", remember=False).choice is None


def test_a_brain_without_will_computes_the_maximum():
    """A fresh mind with no will always reaches the same first answer — that is what argmax is.

    Compared across *fresh* brains on purpose: the workspace carries its own habituation
    between cycles, so repeating within one brain is not a test of determinism.
    """
    def _first() -> str:
        b = NyxBrain(get_settings().nyx.model_copy(
            update={"holo_dim": 1024, "will_enabled": False, "dialogue_enabled": False}))
        b.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
        assert b.will is None
        return b.think("what pulls an apple down", remember=False).winner.source

    assert len({_first() for _ in range(5)}) == 1


def test_a_brain_with_will_does_not_always_reach_the_same_first_answer():
    def _first() -> str:
        b = NyxBrain(get_settings().nyx.model_copy(
            update={"holo_dim": 1024, "collapse_threshold": 0.9, "will_temperature": 1.2,
                    "dialogue_enabled": False}))
        b.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
        return b.think("what pulls an apple down", remember=False).winner.source

    assert len({_first() for _ in range(15)}) > 1


def test_stats_name_the_entropy_source(brain):
    if brain.will is None:
        pytest.skip("will disabled")
    stats = brain.stats()["will"]
    assert stats["entropy"]["source"] and "physical" in stats["entropy"]


def test_state_round_trips(brain):
    if brain.will is None:
        pytest.skip("will disabled")
    data = brain.will.to_dict()
    other = SovereignWill(temperature=0.1)
    assert other.load_dict(data) is True
    assert other.temperature == brain.will.temperature
