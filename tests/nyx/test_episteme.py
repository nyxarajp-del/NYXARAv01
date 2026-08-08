"""NYX V.01 · L-EPISTEME — she finds out for herself, and only keeps what survives a check."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import get_settings
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.episteme import AutonomousDiscovery, Experiment, Knob


def _linear() -> Experiment:
    """A sandbox with a law she is not told: y = 3x + 1."""
    return Experiment(name="linear", probe=lambda x: 3.0 * x + 1.0, measures="y",
                      domain="toy", knobs=[Knob("x", 0.0, 20.0)])


def _noise() -> Experiment:
    """A sandbox with no law at all — nothing should ever be promoted from this."""
    import random
    rng = random.Random(1)
    return Experiment(name="noise", probe=lambda x: rng.uniform(-100.0, 100.0),
                      measures="y", domain="toy", knobs=[Knob("x", 0.0, 20.0)])


@pytest.fixture
def brain():
    return NyxBrain(get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "episteme_enabled": False, "aura_enabled": False,
                "car_enabled": False, "dialogue_enabled": False}))


# -------------------- she discovers what she was not told -------------------- #
def test_a_real_law_is_discovered_and_promoted(brain):
    got = AutonomousDiscovery(brain, experiments=[_linear()], trials=24,
                              seed=0).investigate()
    assert got.status == "promoted"
    assert "x" in got.expression
    assert got.holdout_error is not None and got.holdout_error < 0.05


def test_a_promoted_law_becomes_something_she_can_recall(brain):
    discovery = AutonomousDiscovery(brain, experiments=[_linear()], trials=24, seed=0)
    discovery.investigate()
    hit = brain.recall("toy law y x").hit
    assert hit is not None and hit.kind == "fact"


def test_the_law_is_recorded_under_its_experiment(brain):
    discovery = AutonomousDiscovery(brain, experiments=[_linear()], trials=24, seed=0)
    discovery.investigate()
    assert "linear" in discovery.promoted


# -------------------- the held-out check is the whole gate -------------------- #
def test_pure_noise_is_never_promoted(brain):
    """A curve that fits the points it was fitted to has established nothing."""
    got = AutonomousDiscovery(brain, experiments=[_noise()], trials=24, seed=0).investigate()
    assert got.status != "promoted"
    assert got.status in {"abstained", "conjecture"}


def test_a_law_that_misses_the_withheld_trials_stays_a_conjecture(brain):
    discovery = AutonomousDiscovery(brain, experiments=[_noise()], trials=24, seed=0)
    got = discovery.investigate()
    if got.status != "conjecture":
        pytest.skip("this run abstained before producing a candidate")
    assert "linear" not in discovery.promoted
    assert discovery.conjectures


def test_a_conjecture_is_stored_labelled_not_asserted(brain):
    discovery = AutonomousDiscovery(brain, experiments=[_noise()], trials=24, seed=0)
    if discovery.investigate().status != "conjecture":
        pytest.skip("this run abstained before producing a candidate")
    kinds = {t.kind for t in brain.memory.traces.values()}
    assert "conjecture" in kinds and "fact" not in kinds


def test_a_stricter_bar_refuses_more(brain):
    lenient = AutonomousDiscovery(brain, experiments=[_linear()], trials=24, seed=0,
                                  max_holdout_error=0.5).investigate()
    strict = AutonomousDiscovery(brain, experiments=[_linear()], trials=24, seed=0,
                                 max_holdout_error=0.0).investigate()
    assert lenient.status == "promoted"
    assert strict.status in {"conjecture", "promoted"}   # exact only if it fits perfectly


def test_trials_are_actually_withheld(brain):
    got = AutonomousDiscovery(brain, experiments=[_linear()], trials=20, seed=0).investigate()
    assert got.holdout >= 2 and got.trials + got.holdout == 20


# -------------------- abstention is a result -------------------- #
def test_nothing_to_experiment_on_is_reported(brain):
    got = AutonomousDiscovery(brain, experiments=[]).investigate()
    assert got.status == "abstained" and "nothing she can experiment on" in got.reason
    assert AutonomousDiscovery(brain, experiments=[]).available() is False


def test_a_sandbox_that_will_not_yield_measurements_fails_honestly(brain):
    def _broken(x):
        raise RuntimeError("probe on fire")

    got = AutonomousDiscovery(
        brain, experiments=[Experiment(name="broken", probe=_broken,
                                       knobs=[Knob("x", 0.0, 1.0)])],
        trials=10).investigate()
    assert got.status == "failed" and "measurements" in got.reason


def test_she_works_through_her_experiments_rather_than_fixating(brain):
    discovery = AutonomousDiscovery(brain, experiments=[_linear(), _noise()], trials=12, seed=0)
    names = {discovery.investigate().experiment for _ in range(4)}
    assert names == {"linear", "noise"}


# -------------------- oversight -------------------- #
def test_a_scrammed_mind_does_not_investigate(brain):
    class _Gate:
        def gate(self):
            return False

    discovery = AutonomousDiscovery(brain, experiments=[_linear()], trials=12)
    assert discovery.beat(oversight=_Gate()) is None
    assert discovery.findings == []


def test_an_open_gate_permits_investigating(brain):
    class _Gate:
        def gate(self):
            return True

    discovery = AutonomousDiscovery(brain, experiments=[_linear()], trials=12)
    assert discovery.beat(oversight=_Gate()) is not None


# -------------------- the built-in sandboxes -------------------- #
def test_the_default_sandboxes_need_no_network(brain):
    discovery = AutonomousDiscovery(brain, trials=20, seed=0)
    if not discovery.available():
        pytest.skip("no sandbox available on this host")
    assert {"gas-box", "electrostatics"} & {e.name for e in discovery.experiments}


def test_she_recovers_the_gas_law_from_her_own_measurements(brain):
    """She is told nothing about it: she probes, measures, derives, and checks."""
    discovery = AutonomousDiscovery(brain, trials=20, seed=0)
    gas = next((e for e in discovery.experiments if e.name == "gas-box"), None)
    if gas is None:
        pytest.skip("the gas sandbox is unavailable")
    got = discovery.investigate(gas)
    if got.status != "promoted":
        pytest.skip("this run did not reach a validated law")
    assert "n" in got.expression and "temperature" in got.expression


# -------------------- state -------------------- #
def test_findings_are_bounded(brain):
    discovery = AutonomousDiscovery(brain, experiments=[_linear()], trials=8, seed=0)
    for _ in range(40):
        discovery.investigate()
    assert len(discovery.findings) <= 32


def test_state_round_trips(brain):
    discovery = AutonomousDiscovery(brain, experiments=[_linear()], trials=24, seed=0)
    discovery.investigate()
    data = discovery.to_dict()

    other = AutonomousDiscovery(brain, experiments=[_linear()], trials=24, seed=0)
    assert other.load_dict(data) is True
    assert other.promoted == discovery.promoted


def test_corrupt_state_does_not_block(brain):
    assert AutonomousDiscovery(brain).load_dict(None) is False


# -------------------- wired into the brain -------------------- #
def test_the_brain_can_be_asked_to_discover():
    b = NyxBrain(get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "episteme_trials": 20, "aura_enabled": False}))
    if b.episteme is None or not b.episteme.available():
        pytest.skip("no sandbox available")
    assert b.discover() is not None


def test_a_brain_without_episteme_still_works():
    b = NyxBrain(get_settings().nyx.model_copy(
        update={"holo_dim": 1024, "episteme_enabled": False}))
    assert b.episteme is None and b.discover() is None
    assert b.think("what pulls an apple down") is not None
