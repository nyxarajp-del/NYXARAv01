"""Real backpropagation over the fabric, and the check that proves it is real.

Every update in the fabric is local: a synapse changes because its two endpoints fired, and
nothing tells it what the network as a whole was trying to achieve. This is the other kind of
learning — and "real backprop" is easy to assert and easy to get subtly wrong, so it is checked
against finite differences rather than trusted.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from nyxara.growth import genesis_numpy as G  # noqa: E402
from nyxara.njp.learn import Readout, available, gradcheck  # noqa: E402


@pytest.fixture()
def head() -> Readout:
    return Readout(width=16, hidden=8, lr=0.05, seed=1)


def _batch():
    return [([1, 2, 3], [4, 5]), ([4, 5], [6]), ([6], [1, 2])]


# ---- the engine is reachable ------------------------------------------------ #
def test_the_autodiff_engine_is_exported():
    """It existed and was invisible: __all__ named only the model, so every caller that wanted
    gradients had to write its own engine."""
    for name in ("Tensor", "backward", "matmul", "sigmoid", "relu", "layernorm"):
        assert name in G.__all__, f"{name} is not exported"
        assert hasattr(G, name)


def test_gradients_are_available_here():
    assert available()


# ---- THE test: are these really gradients? ---------------------------------- #
def test_every_parameter_gradient_matches_finite_differences(head: Readout):
    """The check that decides whether "real backprop" is true.

    A sign error or a missing term in one backward closure still trains, just worse, and nothing
    else in a training loop would ever notice.
    """
    x, y = head._batch(_batch())

    def loss_of(_p):
        return head._loss(head._forward(G.Tensor(x)), y)

    for name, param in (("W1", head.W1), ("b1", head.b1),
                        ("W2", head.W2), ("b2", head.b2)):
        passed, error = gradcheck(loss_of, param)
        assert passed, f"{name} gradient disagrees with finite differences by {error:.3e}"


def test_gradcheck_catches_a_wrong_gradient():
    """The check has to be able to fail, or it proves nothing."""
    a = G.Tensor(np.array([[0.3, -0.7]]), requires_grad=True)

    def broken(p):
        out = G.Tensor(np.sum(p.data ** 2), _prev=(p,))
        # Deliberately wrong: the true derivative of sum(x^2) is 2x, not x.
        out._backward = lambda: G._acc(p, p.data * float(out.grad))
        return out

    passed, error = gradcheck(broken, a)
    assert not passed and error > 1e-6


def test_a_correct_hand_written_gradient_passes():
    a = G.Tensor(np.array([[0.3, -0.7]]), requires_grad=True)

    def correct(p):
        out = G.Tensor(np.sum(p.data ** 2), _prev=(p,))
        out._backward = lambda: G._acc(p, 2.0 * p.data * float(out.grad))
        return out

    passed, _error = gradcheck(correct, a)
    assert passed


# ---- it actually learns ------------------------------------------------------ #
def test_training_reduces_the_loss(head: Readout):
    batch = _batch()
    before = head.loss_on(batch)
    for _ in range(100):
        head.train_step(batch)
    assert head.loss_on(batch) < before * 0.1, "the head barely learned anything"


def test_it_learns_the_actual_mapping():
    head = Readout(width=32, hidden=16, lr=0.05, seed=7)
    batch = [([1, 2, 3], [4, 5]), ([8], [9])]
    for _ in range(200):
        head.train_step(batch)
    predicted = {c for c, _p in head.predict([1, 2, 3], k=2)}
    assert {4, 5} & predicted, f"expected 4 or 5, got {predicted}"


def test_a_step_reports_whether_it_helped(head: Readout):
    step = head.train_step(_batch())
    assert step.samples == 3
    assert step.grad_norm > 0.0
    assert step.improved


def test_gradients_are_cleared_between_steps(head: Readout):
    """Accumulating across steps would silently scale the learning rate by the step count."""
    head.train_step(_batch())
    first = head.last.grad_norm
    head.train_step(_batch())
    assert head.last.grad_norm < first * 5, "gradients look accumulated rather than reset"


def test_an_empty_batch_changes_nothing(head: Readout):
    before = [p.data.copy() for p in head._params]
    step = head.train_step([])
    assert step.samples == 0
    for p, was in zip(head._params, before):
        assert np.allclose(p.data, was)


def test_saturated_outputs_do_not_produce_nan(head: Readout):
    """A sigmoid saturating to exactly 0 or 1 makes the loss inf and every gradient nan — which
    does not raise, it silently destroys every parameter on the next step."""
    for _ in range(400):
        head.train_step(_batch())
    for p in head._params:
        assert np.all(np.isfinite(p.data)), "a parameter went nan/inf"
    assert np.isfinite(head.loss_on(_batch()))


# ---- shape and honesty --------------------------------------------------------- #
def test_the_head_has_real_parameters(head: Readout):
    assert head.stats()["parameters"] == 16 * 8 + 8 + 8 * 16 + 16


def test_slots_are_stable_across_calls(head: Readout):
    """What was learned has to stay addressable as the fabric grows new cells."""
    assert head.slot(12345) == head.slot(12345)


def test_weights_survive_a_reload(head: Readout):
    for _ in range(20):
        head.train_step(_batch())
    fresh = Readout(width=16, hidden=8, seed=99)
    fresh.load_dict(head.to_dict())
    assert np.allclose(fresh.W1.data, head.W1.data)
    assert fresh.loss_on(_batch()) == pytest.approx(head.loss_on(_batch()))


def test_weights_survive_a_reload_into_a_head_of_the_default_width():
    """The reload above restores into a head built at the *same* width, which is the easy case.

    A real resume never is. The head doubles its own width as the fabric grows, so the state on
    disk is 4096 or 8192 wide and the brain that loads it was constructed at 512. `to_dict` wrote
    `width` and `load_dict` never read it, so every saved tensor failed the shape guard and was
    **silently skipped** — the whole trained head discarded, with nothing raised and nothing
    logged. The slot map was restored regardless, and the readout then never took another
    gradient step for the life of the process: a resumed run learned facts and concepts while its
    only gradient learner sat dead at a frozen step count.
    """
    grown = Readout(width=16, hidden=8, seed=3, max_width=64)
    while grown.width <= 16:
        grown._grow()
    for _ in range(10):
        grown.train_step(_batch())
    assert grown.width > 16 and grown.growths > 0

    fresh = Readout(width=16, hidden=8, seed=99)     # the default-width brain a resume builds
    fresh.load_dict(grown.to_dict())

    assert fresh.width == grown.width, "saved width was ignored, so the head was rebuilt wrong"
    assert fresh.growths == grown.growths
    for restored, saved in zip(fresh._params, grown._params):
        assert np.allclose(restored.data, saved.data), "trained weights were silently discarded"


def test_a_reloaded_head_keeps_training(head: Readout):
    """The symptom that made the silent discard expensive: it stopped learning, permanently."""
    for _ in range(5):
        head.train_step(_batch())
    fresh = Readout(width=16, hidden=8, seed=99)
    fresh.load_dict(head.to_dict())
    before = fresh.steps
    for _ in range(5):
        fresh.train_step(_batch())
    assert fresh.steps > before, "a reloaded head took no further gradient steps"


def test_a_mismatched_state_is_rejected_whole_rather_than_half_applied():
    """Two of four tensors restored is not a partially trained head. It is a broken one."""
    head = Readout(width=16, hidden=8, seed=3)
    for _ in range(5):
        head.train_step(_batch())
    saved = head.to_dict()
    saved["params"][2] = [[0.0]]                     # corrupt one tensor's shape

    fresh = Readout(width=16, hidden=8, seed=99)
    original = [p.data.copy() for p in fresh._params]
    fresh.load_dict(saved)
    for now, then in zip(fresh._params, original):
        assert np.allclose(now.data, then), "a rejected state still wrote some of its tensors"


def test_growing_recounts_collisions_in_the_table_that_now_exists():
    """The one number that says whether growth worked was the one number growth did not update.

    ``_grow`` rehashes every cell into a table twice the size — that is the whole point — but it
    never reset ``_collisions``, so a head that had just doubled its width to stop colliding went
    on reporting the smaller table's count, and ``collision_rate`` described a table that no
    longer existed.
    """
    # Pinned at its starting width so it cannot grow away from the crowding: `slot` doubles the
    # table on its own once the load factor is passed, which is exactly the behaviour under test.
    head = Readout(width=16, hidden=8, seed=3, max_width=16)
    for cell in range(200):
        head.slot(cell)
    crowded = head.stats()["slot_collisions"]
    assert crowded > 0, "the table did not actually collide; test inconclusive"

    head.max_width = 256                          # now let it grow, and recount
    assert head._grow()
    assert head.stats()["slot_collisions"] < crowded, "collisions survived a doubling untouched"
    assert all(0 <= s < head.width for s in head._slots.values())


def test_a_restored_slot_map_never_points_outside_the_table():
    """An index past the end of the row is what turned this into silent, permanent breakage."""
    head = Readout(width=16, hidden=8, seed=3)
    for _ in range(5):
        head.train_step(_batch())
    saved = head.to_dict()
    saved["slots"]["999999"] = 10_000               # a slot from a much wider head

    fresh = Readout(width=16, hidden=8, seed=99)
    fresh.load_dict(saved)
    assert all(0 <= s < fresh.width for s in fresh._slots.values())


# ---- wired: the dream actually trains ------------------------------------------- #
def _lived() -> "NJPBrain":  # noqa: F821
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    for line in ("gravity pulls the apple down", "my name is Jay", "I live in Mumbai",
                 "Ravi lives in Delhi", "Ravi works at Infosys"):
        brain.think(line)
    return brain


def test_the_brain_builds_a_readout():
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    assert brain.readout is not None
    assert brain.readout.stats()["parameters"] > 0


def test_the_dream_replays_real_episodes():
    brain = _lived()
    report = brain.pulse.beat(force=True)
    assert report.replayed > 0, "the dream replayed nothing"


def test_the_dream_actually_reduces_loss():
    """It used to touch timestamps and report dreamed=True while changing nothing that could ever
    affect an answer."""
    brain = _lived()
    report = brain.pulse.beat(force=True)
    assert report.dream_loss_before > 0.0
    assert report.dream_loss_after < report.dream_loss_before


def test_the_dream_trains_the_readout():
    brain = _lived()
    before = brain.readout.stats()["steps"]
    brain.pulse.beat(force=True)
    assert brain.readout.stats()["steps"] > before


def test_both_experiential_levels_are_replayed():
    """Autobiographical entries are episodes too. They are protected from forgetting, which is not
    a reason to exclude them from replay — a session mostly about the Master replayed nothing."""
    from nyxara.njp.levels import Level

    brain = _lived()
    assert brain.levels.at(Level.AUTOBIOGRAPHICAL), "fixture stored nothing autobiographical"
    assert brain.pulse.beat(force=True).replayed >= len(
        brain.levels.at(Level.AUTOBIOGRAPHICAL))


def test_the_dream_does_not_mutate_the_live_fabric():
    """Re-settling during an offline pass would change the very thing it is learning about."""
    brain = _lived()
    before = (brain.fabric.n_cells, brain.fabric.n_synapses, brain.fabric.tick)
    brain.pulse._dream(type(brain.pulse.last)())
    assert (brain.fabric.n_cells, brain.fabric.n_synapses, brain.fabric.tick) == before


def test_an_organ_that_is_off_is_absent():
    from nyxara.njp.brain import NJPBrain

    class _Off:
        readout_enabled = False

    brain = NJPBrain(_Off())
    assert brain.readout is None
    assert "readout" not in brain.stats()
