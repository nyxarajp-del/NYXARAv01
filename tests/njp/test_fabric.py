"""The Fluid Neural Automata: the local law, and that it actually settles."""

from __future__ import annotations

from nyxara.njp.fabric import Fabric


def _chain(f: Fabric, n: int = 8, w: float = 0.6) -> None:
    for i in range(1, n):
        f.connect(i, i + 1, w)


def test_same_seed_same_fabric():
    """Determinism is what makes 'it grew' checkable rather than an anecdote."""
    runs = []
    for _ in range(2):
        f = Fabric(seed=7, manifold_dim=512)
        _chain(f)
        f.stimulate([1, 2])
        res = f.settle()
        runs.append(res.trace)
    assert runs[0] == runs[1]


def test_settle_terminates_and_reports_which_way():
    """A settle either goes quiet on its own or hits the cap — and says which."""
    f = Fabric(seed=7, manifold_dim=512, max_settle_steps=6)
    _chain(f)
    f.stimulate([1])
    res = f.settle()
    assert res.steps <= 6
    assert isinstance(res.quiescent, bool)


def test_refractory_cell_cannot_fire_twice_running():
    """The refractory period is part of the law, not decoration."""
    f = Fabric(seed=7, manifold_dim=512, refractory=3)
    f.connect(1, 2, 5.0)
    f.connect(2, 1, 5.0)          # a loop that would oscillate every tick without refractoriness
    f.stimulate([1])
    res = f.settle(max_steps=6)
    for step in res.trace[1:]:
        assert len(step) <= 1


def test_stimulated_cells_are_step_zero():
    """External drive IS firing. Without this a cold fabric could never grow at all."""
    f = Fabric(seed=7, manifold_dim=512)
    f.stimulate([10, 11, 12])
    res = f.settle()
    assert res.trace and res.trace[0] == (10, 11, 12)
    assert res.seed == (10, 11, 12)


def test_no_self_synapse():
    """A cell exciting itself never settles."""
    f = Fabric(seed=7, manifold_dim=512)
    assert f.connect(5, 5, 1.0) is False
    assert f.n_synapses == 0
