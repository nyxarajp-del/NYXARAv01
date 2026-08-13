"""The continuous beat: growth between turns, and a mind that stops when told to."""

from __future__ import annotations

from nyxara.njp.brain import NJPBrain
from nyxara.njp.fabric import Fabric
from nyxara.njp.pulse import PulseEngine


class _Brain:
    """Just enough of a brain for the pulse to drive."""

    def __init__(self):
        self.fabric = Fabric(seed=4, manifold_dim=512)
        self.ledger = None
        self.evolver = None


class _Scram:
    def gate(self):
        return False


def _lived(brain) -> object:
    brain.fabric.stimulate([1, 2, 3, 4])
    return brain.fabric.settle()


def test_a_beat_expands_what_was_lived_since_the_last_one():
    b = _Brain()
    p = PulseEngine(b, every_s=0.0)
    p.submit(_lived(b), outcome=1.0)
    before = b.fabric.n_synapses
    rep = p.beat(force=True)
    assert rep.expanded == 1
    assert b.fabric.n_synapses > before
    assert rep.grown > 0


def test_a_scrammed_mind_does_not_grow():
    """Autonomy is not extra permission."""
    b = _Brain()
    p = PulseEngine(b, every_s=0.0)
    p.submit(_lived(b), outcome=1.0)
    before = b.fabric.n_synapses
    rep = p.beat(oversight=_Scram(), force=True)
    assert rep.blocked
    assert b.fabric.n_synapses == before
    assert p.blocked_beats == 1


def test_the_queue_is_bounded_and_says_when_it_dropped():
    """If the host stops beating, experience must not accumulate without limit."""
    b = _Brain()
    p = PulseEngine(b, queue_capacity=16)
    for _ in range(40):
        p.submit(_lived(b), outcome=1.0)
    assert len(p.queue) <= 16
    assert p.dropped > 0


def test_it_rate_limits_itself_so_any_caller_cadence_is_safe():
    b = _Brain()
    p = PulseEngine(b, every_s=3600.0)
    p.beat(force=True)
    beats = p.beats
    for _ in range(10):
        p.beat()                       # not forced: the cadence must hold them off
    assert p.beats == beats


def test_the_brain_beats_without_starting_a_thread():
    """The kernel's clock drives it; a thread mutating the brain under a turn is a race."""
    import threading
    before = threading.active_count()
    brain = NJPBrain()
    brain.think("gravity pulls the apple down")
    brain.tick()
    assert threading.active_count() == before
