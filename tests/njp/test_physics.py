"""Real membrane physics, and honesty about the hardware it runs on.

The dynamics used to be `state * leak`, with `leak` a dimensionless per-step constant and the
refractory period counted in integer ticks. That resembles a leaky integrator; it is not one.
"""

from __future__ import annotations

import math

import pytest

from nyxara.njp.backend import NumpyBackend, detect, probe
from nyxara.njp.fabric import Fabric


# ---- the membrane equation --------------------------------------------------- #
def test_decay_is_the_analytic_solution():
    """dV/dt = -V/tau_m, solved exactly over dt for constant input: V <- V*exp(-dt/tau_m)."""
    fabric = Fabric(dt=0.001, tau_m=0.02)
    assert fabric.decay == pytest.approx(math.exp(-0.001 / 0.02), abs=1e-12)


def test_the_trajectory_does_not_depend_on_the_step_size():
    """The test that separates real physics from a renamed constant: halving dt and doubling the
    steps must reach the same membrane potential."""
    def after_20ms(dt: float) -> float:
        fabric = Fabric(dt=dt, tau_m=0.02)
        v = 1.0
        for _ in range(int(round(0.020 / dt))):
            v *= fabric.decay
        return v

    coarse, fine = after_20ms(0.001), after_20ms(0.0005)
    assert coarse == pytest.approx(fine, abs=1e-9)
    assert coarse == pytest.approx(math.exp(-1.0), abs=1e-9)


def test_a_shorter_time_constant_decays_faster():
    assert Fabric(tau_m=0.01).decay < Fabric(tau_m=0.02).decay


def test_time_is_measured_in_seconds():
    fabric = Fabric(dt=0.001)
    assert fabric.now == 0.0
    fabric.connect(1, 2, 0.5)
    fabric.stimulate([1])
    fabric.settle()
    assert fabric.now > 0.0
    assert fabric.tau_ref == pytest.approx(0.002)


def test_an_explicit_leak_is_honoured_exactly():
    """Keying the sentinel off the default *value* made an explicit leak=0.85 silently become
    tau_m's decay instead — equal only by coincidence, and never for any other value."""
    for leak in (0.85, 0.5, 0.99):
        assert Fabric(leak=leak).decay == pytest.approx(leak, abs=1e-9)


def test_the_default_uses_real_time_constants_not_a_bare_multiplier():
    fabric = Fabric()
    assert fabric.tau_m == pytest.approx(0.02)
    assert fabric.decay == pytest.approx(math.exp(-fabric.dt / fabric.tau_m))


# ---- the backend ------------------------------------------------------------- #
def test_the_live_backend_is_named():
    fabric = Fabric()
    assert fabric.stats()["backend"]["backend"] == fabric.backend.name


def test_no_neuromorphic_hardware_is_reported_as_absent():
    """A backend layer that quietly fell back while still describing itself as neuromorphic would
    be worse than no layer at all: an unverifiable hardware claim on every call."""
    found = probe()
    backend = detect()
    if not found["available"]:
        assert isinstance(backend, NumpyBackend)
        assert backend.name == "numpy"
        assert backend.stats()["hardware"] is False


def test_the_probe_names_what_it_looked_for():
    found = probe()
    assert "neuromorphic" in found and "numpy" in found
    assert isinstance(found["available"], bool)


def test_stats_report_the_real_physics():
    physics = Fabric(dt=0.002, tau_m=0.05, tau_ref=0.004).stats()["physics"]
    assert physics["dt_s"] == 0.002
    assert physics["tau_m_s"] == 0.05
    assert physics["tau_ref_s"] == 0.004


# ---- the compiled path is still the same automaton ----------------------------- #
def test_the_compiled_form_is_rebuilt_only_when_structure_changes():
    fabric = Fabric()
    fabric.connect(1, 2, 1.5)
    fabric.stimulate([1])
    fabric.settle()
    compiled = fabric.compilations
    for _ in range(5):
        fabric.stimulate([1])
        fabric.settle()
    assert fabric.compilations == compiled, "recompiling with no structural change"


def test_growth_invalidates_the_compiled_form():
    fabric = Fabric()
    fabric.connect(1, 2, 1.5)
    fabric.stimulate([1])
    fabric.settle()
    compiled = fabric.compilations
    fabric.connect(2, 3, 1.5)          # a structural edit
    fabric.stimulate([1])
    fabric.settle()
    assert fabric.compilations > compiled, "a new synapse did not invalidate the compiled form"


def test_a_signal_still_propagates():
    """Correctness before speed: the automaton must still be the automaton."""
    fabric = Fabric(threshold=1.0)
    fabric.connect(1, 2, 1.5)
    fabric.connect(2, 3, 1.5)
    fabric.stimulate([1])
    result = fabric.settle()
    assert 2 in result.fired and 3 in result.fired


def test_the_refractory_period_actually_blocks():
    fabric = Fabric(threshold=1.0, tau_ref=1.0)   # a full second: nothing may fire twice
    fabric.connect(1, 2, 5.0)
    fabric.stimulate([1])
    fabric.settle()
    hits = fabric.cells[2].hits
    fabric.stimulate([1])
    fabric.settle()
    assert fabric.cells[2].hits == hits, "a refractory cell fired again"


def test_cells_are_synced_after_a_settle():
    """The vectorised state is authoritative during a settle; everything that reads a Cell runs
    after one, so the sync happens at that boundary."""
    fabric = Fabric(threshold=1.0)
    fabric.connect(1, 2, 1.5)
    fabric.stimulate([1])
    fabric.settle()
    assert fabric.cells[2].hits > 0
    assert fabric.cells[2].last_fired > 0.0


def test_determinism_survives_vectorisation():
    def run() -> tuple:
        fabric = Fabric(seed=7, threshold=1.0)
        for a, b in ((1, 2), (2, 3), (1, 3), (3, 4)):
            fabric.connect(a, b, 1.2)
        fabric.stimulate([1])
        return tuple(fabric.settle().trace)

    assert run() == run()


# ---- the candidate pool ---------------------------------------------------------- #
def test_a_small_fabric_decodes_against_everything():
    fabric = Fabric()
    for i in range(1, 20):
        fabric.connect(i, i + 1, 0.5)
    assert len(fabric._candidate_pool()) == fabric.n_cells


def test_a_large_fabric_bounds_the_candidate_pool():
    """Scoring cells that have never fired does not add a possibility, it adds noise that dilutes
    the margin — the very number that decides whether a prediction is trusted."""
    fabric = Fabric(candidate_cap=64)
    for i in range(1, 500):
        fabric.connect(i, i + 1, 0.5)
    assert len(fabric._candidate_pool()) <= 64
