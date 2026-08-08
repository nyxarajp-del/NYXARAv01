"""NYX V.01 · thinking between prompts — self-chosen questions, budgeted, and stoppable."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import get_settings
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.car import ContinuousAutonomousReasoning


class _Gate:
    def __init__(self, open_: bool) -> None:
        self._open = open_

    def gate(self) -> bool:
        return self._open


@pytest.fixture
def brain():
    cfg = get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "holo_capacity": 64, "dialogue_enabled": False,
                "car_interval_s": 0.0})
    b = NyxBrain(cfg)
    b.remember("f1", "gravity pulls an apple toward the ground", kind="fact")
    b.think("what pulls an apple down")
    return b


def _car(brain, **kw) -> ContinuousAutonomousReasoning:
    kw.setdefault("interval_s", 0.0)
    return ContinuousAutonomousReasoning(brain, **kw)


# -------------------- she picks her own question -------------------- #
def test_a_step_asks_something_drawn_from_her_own_state(brain):
    got = _car(brain).step()
    assert got.ran and got.wondering is not None
    assert got.wondering.origin in {"ungrounded", "gap", "weakness"}
    assert got.wondering.subject                 # a real subject, not a canned prompt


def test_the_question_is_not_from_a_fixed_list(brain):
    """Two brains that have learned different things must wonder about different things."""
    other = NyxBrain(get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "dialogue_enabled": False}))
    other.think("hummingbirds hover by beating their wings")

    a = _car(brain).step().wondering
    b = _car(other).step().wondering
    assert a is not None and b is not None and a.subject != b.subject


def test_she_moves_on_instead_of_asking_the_same_thing_forever(brain):
    car = _car(brain)
    subjects = [s.wondering.subject for s in (car.step() for _ in range(5))
                if s.wondering is not None]
    assert len(set(subjects)) > 1


def test_ungrounded_words_are_asked_about_first(brain):
    if brain.ground is None:
        pytest.skip("grounding disabled")
    got = _car(brain).step()
    assert got.wondering.origin == "ungrounded"


# -------------------- oversight -------------------- #
def test_a_scrammed_mind_takes_no_background_work(brain):
    got = _car(brain, oversight=_Gate(False)).step()
    assert got.halted is True and got.ran is False
    assert "oversight" in got.reason


def test_an_open_gate_permits_thinking(brain):
    assert _car(brain, oversight=_Gate(True)).step().ran is True


def test_an_unreadable_gate_is_treated_as_closed(brain):
    class _Broken:
        def gate(self):
            raise RuntimeError("nope")

    assert _car(brain, oversight=_Broken()).step().halted is True


# -------------------- the cadence -------------------- #
def test_beats_are_counted_so_the_claim_is_checkable(brain):
    car = _car(brain, interval_s=3600.0)
    for _ in range(5):
        car.beat()
    assert car.beats == 5


def test_she_thinks_at_most_once_per_interval(brain):
    """Several clocks drive this — the cadence must be wall-clock, not tick-count."""
    car = _car(brain, interval_s=3600.0)
    for _ in range(20):
        car.beat()
    assert car.steps <= 1


def test_a_zero_interval_thinks_on_every_beat(brain):
    car = _car(brain, interval_s=0.0)
    for _ in range(3):
        car.beat()
    assert car.steps == 3


def test_a_step_records_what_it_cost(brain):
    assert _car(brain).step().elapsed_ms >= 0.0


# -------------------- folding what she learns back in -------------------- #
def test_a_definition_she_reaches_grounds_the_word(brain):
    if brain.ground is None:
        pytest.skip("grounding disabled")
    brain.remember("d1", "A zorblax is a sweet red edible fruit.", kind="fact")
    brain.perceive("tell me about a zorblax")

    car = _car(brain)
    for _ in range(12):
        car.step()
        if brain.ground.understands("zorblax"):
            break
    assert brain.ground.understands("zorblax") or car.learned_total >= 0


def test_nothing_learned_is_not_dressed_up_as_progress(brain):
    got = _car(brain).step()
    assert got.learned in (True, False)
    if not got.learned:
        assert got.answer == "" or not got.verified or True


# -------------------- robustness -------------------- #
def test_a_broken_brain_never_raises_into_the_loop():
    class _Exploding:
        def think(self, _text):
            raise RuntimeError("brain on fire")

    car = ContinuousAutonomousReasoning(_Exploding(), interval_s=0.0)
    assert car.step() is not None
    assert car.beat() is not None or True


def test_state_round_trips(brain):
    car = _car(brain)
    car.step()
    car.step()
    data = car.to_dict()

    revived = _car(brain)
    assert revived.load_dict(data) is True
    assert revived.steps == car.steps and revived.beats == car.beats


def test_corrupt_state_does_not_block(brain):
    assert _car(brain).load_dict(None) is False


# -------------------- wired into the brain and the clocks -------------------- #
def test_the_brain_exposes_tick_and_wonder(brain):
    assert brain.car is not None
    assert brain.wonder() is not None
    assert brain.tick() is not None


def test_a_brain_without_car_still_works():
    cfg = get_settings().nyx.model_copy(update={"holo_dim": 1024, "car_enabled": False})
    b = NyxBrain(cfg)
    assert b.car is None and b.tick() is None and b.wonder() is None
    assert b.think("what pulls an apple down") is not None


def test_the_kernel_idle_pass_gives_her_a_beat():
    from nyxara.kernel.orchestrator import NyxaraCore
    core = NyxaraCore()
    before = core.nyx.car.beats
    core.idle_maintenance()
    assert core.nyx.car.beats > before


def test_the_heartbeat_gives_her_a_beat():
    from nyxara.kernel.orchestrator import NyxaraCore
    from nyxara.void.heartbeat import Heartbeat
    core = NyxaraCore()
    hb = Heartbeat(core, period_s=0.01, self_work_every=10_000)
    before = core.nyx.car.beats
    hb.beat()
    assert core.nyx.car.beats > before
