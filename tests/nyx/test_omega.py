"""NYX V.02 · L-OMEGA — she tunes the mind she thinks with, and can always put it back.

L-OMNI rewrites her code; `author` writes new code. Neither touches the constants her graph and
memory actually run on. Most of this file is about the restraints that make that safe: the
safety core is not a knob, anything that does not beat its own baseline is rolled back, and
below a sample floor she does not evolve at all.
"""

from __future__ import annotations

import pytest

from nyxara.kernel.config import NyxConfig
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.graph import DynamicNeuralGraph
from nyxara.nyx.holomem import HoloMemory
from nyxara.nyx.omega import PROTECTED, SelfEvolutionKernel, Step


def _brain(**kw) -> NyxBrain:
    kw.setdefault("omega_every_s", 0.0)
    kw.setdefault("omega_min_samples", 1)
    return NyxBrain(NyxConfig(**kw))


def _warm(brain: NyxBrain) -> NyxBrain:
    """A brain with structure, so the graph-health probe has something to measure."""
    for line in ("gravity pulls the apple down", "the apple falls to the ground",
                 "a heavier flywheel smooths an engine", "the engine turns the flywheel"):
        brain.perceive(line)
    return brain


def _kernel(**kw) -> SelfEvolutionKernel:
    brain = _brain()
    kw.setdefault("every_s", 0.0)
    kw.setdefault("min_samples", 1)
    return SelfEvolutionKernel(brain, **kw)


# --------------------------------------------------------------------------- #
# The knob seams — a whitelist, clamped, and nothing else
# --------------------------------------------------------------------------- #
def test_the_graph_exposes_its_tunables_and_only_those():
    graph = DynamicNeuralGraph()
    knobs = graph.knobs()
    assert set(knobs) == set(DynamicNeuralGraph.KNOBS)
    assert "max_nodes" not in knobs        # capacity is a resource decision, not a knob


def test_an_unknown_knob_does_nothing_rather_than_growing_an_attribute():
    graph = DynamicNeuralGraph()
    assert graph.apply_knobs({"not_a_knob": 1.0}) == {}
    assert not hasattr(graph, "not_a_knob")


def test_a_knob_is_clamped_to_its_declared_range():
    graph = DynamicNeuralGraph()
    applied = graph.apply_knobs({"hebbian_rate": 99.0})
    assert applied["hebbian_rate"] == DynamicNeuralGraph.KNOBS["hebbian_rate"][1]


def test_integer_knobs_stay_integers():
    graph = DynamicNeuralGraph()
    graph.apply_knobs({"spread_depth": 3.7})
    assert graph.spread_depth == 3 and isinstance(graph.spread_depth, int)


def test_memory_keeps_the_field_threshold_in_step():
    """A knob that looks tuned while recall behaves exactly as before is not tuned."""
    memory = HoloMemory(dim=512, capacity=32)
    memory.apply_knobs({"recall_threshold": 0.33})
    assert memory.recall_threshold == pytest.approx(0.33)
    assert memory.field.recall_threshold == pytest.approx(0.33)


# --------------------------------------------------------------------------- #
# Fitness is measured, and a thin record stops her
# --------------------------------------------------------------------------- #
def test_a_thin_record_stops_her_evolving_at_all():
    kernel = SelfEvolutionKernel(_brain(), every_s=0.0, min_samples=50)
    step = kernel.evolve()
    assert step.status == "skipped"
    assert "fitting noise" in step.reason


def test_the_cadence_is_real():
    kernel = SelfEvolutionKernel(_brain(), every_s=3600.0, min_samples=1)
    kernel.evolve(force=True)                 # sets the clock
    step = kernel.evolve()
    assert step.status == "skipped" and "not due" in step.reason
    assert "no baseline left" in step.reason


def test_fitness_reports_its_sample_count():
    fit = _kernel().fitness()
    assert fit.samples >= 0 and fit.why
    assert fit.enough is (fit.samples >= 1)


def test_an_empty_graph_earns_no_free_health_bonus():
    """Otherwise every knob wins on a cold boot, before she has learned anything."""
    kernel = _kernel()
    assert kernel._score_after(kernel.brain.graph, "hebbian_rate") == \
        pytest.approx(kernel.fitness().score)


# --------------------------------------------------------------------------- #
# The safety core is not a knob
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(PROTECTED))
def test_every_protected_name_is_refused(name):
    assert SelfEvolutionKernel._protected(name)
    assert SelfEvolutionKernel._protected(f"nyx_{name}_rate")


def test_no_tunable_knob_is_anywhere_near_the_safety_core():
    for cls in (DynamicNeuralGraph, HoloMemory):
        for knob in cls.KNOBS:
            assert not SelfEvolutionKernel._protected(knob), knob


def test_a_protected_knob_is_refused_and_recorded(monkeypatch):
    kernel = _kernel()
    monkeypatch.setattr(kernel, "_pick",
                        lambda targets: ("graph", kernel.brain.graph, "loyalty_rate"))
    step = kernel.evolve(force=True)
    assert step.status == "refused"
    assert "never becomes one" in step.reason
    assert kernel.refused == 1


# --------------------------------------------------------------------------- #
# The gauntlet: nothing that fails to beat its baseline survives
# --------------------------------------------------------------------------- #
def test_a_change_that_does_not_beat_its_baseline_is_rolled_back(monkeypatch):
    kernel = _kernel()
    before = kernel.knobs()
    monkeypatch.setattr(kernel, "_score_after", lambda obj, knob: 0.0)
    monkeypatch.setattr(kernel, "fitness",
                        lambda: type("F", (), {"score": 0.9, "enough": True, "samples": 99,
                                               "why": "", "to_dict": lambda s: {}})())
    step = kernel.evolve(force=True)
    assert step.status == "rolled_back"
    assert kernel.knobs() == before            # exactly back, not approximately
    assert step.now == step.was


def test_a_change_that_beats_its_baseline_is_kept():
    kernel = _kernel()
    _warm(kernel.brain)
    before = kernel.knobs()
    step = kernel.evolve(force=True)
    assert step.status == "promoted"
    assert kernel.knobs() != before


def test_without_a_gauntlet_nothing_is_applied(monkeypatch):
    kernel = _kernel()
    before = kernel.knobs()
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kw):
        if name == "nyxara.nyx5.autopoiesis":
            raise ImportError("no gauntlet")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    step = kernel.evolve(force=True)
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert step.status == "rolled_back" and "unavailable" in step.reason
    assert kernel.knobs() == before


# --------------------------------------------------------------------------- #
# Reversibility — the point of the layer
# --------------------------------------------------------------------------- #
def test_a_checkpoint_restores_every_knob_exactly():
    kernel = _kernel()
    _warm(kernel.brain)
    kernel.checkpoint(note="start")
    original = kernel.knobs()
    for _ in range(5):
        kernel.evolve(force=True)
    assert kernel.knobs() != original
    assert kernel.rollback()
    assert kernel.knobs() == original


def test_rollback_without_a_point_is_a_no_op_not_a_crash():
    assert _kernel().rollback() is False


def test_a_deliberately_broken_knob_can_be_walked_back():
    kernel = _kernel()
    kernel.checkpoint(note="good")
    kernel.brain.graph.apply_knobs({"prune_threshold": 0.2, "decay_rate": 0.2})
    assert kernel.brain.graph.prune_threshold == pytest.approx(0.2)
    kernel.rollback()
    assert kernel.brain.graph.prune_threshold == pytest.approx(0.02)


def test_every_step_is_in_the_ledger():
    kernel = _kernel()
    for _ in range(3):
        kernel.evolve(force=True)
    assert len(kernel.ledger) == 3
    assert all(s.status in ("promoted", "rolled_back", "refused", "skipped")
               for s in kernel.ledger)


# --------------------------------------------------------------------------- #
# Redesign, when knobs stop helping
# --------------------------------------------------------------------------- #
def test_a_stall_stops_her_turning_knobs_and_asks_for_a_new_rule(monkeypatch):
    kernel = _kernel(stall_after=1, rule_population=2, rule_generations=1, rule_budget_s=0.6)
    kernel._stalls = 5
    step = kernel.evolve(force=True)
    assert step.target == "rule" and step.knob in ("learning-rule", "")
    assert "stalled" in step.reason


def test_rule_synthesis_can_be_switched_off():
    kernel = _kernel(stall_after=1, rule_synth_on_stall=False)
    kernel._stalls = 5
    step = kernel.evolve(force=True)
    assert step.target in ("graph", "memory")     # she keeps turning knobs instead


# --------------------------------------------------------------------------- #
# Corrigibility
# --------------------------------------------------------------------------- #
class _Scrammed:
    def scrammed(self) -> bool:
        return True


def test_scram_stops_her_reshaping_herself():
    kernel = _kernel()
    before = kernel.knobs()
    step = kernel.evolve(oversight=_Scrammed(), force=True)
    assert step.status == "skipped" and "scrammed" in step.reason
    assert kernel.knobs() == before


def test_an_unreadable_oversight_is_treated_as_stop():
    class _Broken:
        @property
        def paused(self):
            raise RuntimeError("unreadable")
    assert SelfEvolutionKernel._stopped(_Broken()) is True


# --------------------------------------------------------------------------- #
# Through the brain, and across a restart
# --------------------------------------------------------------------------- #
def test_evolve_is_reachable_on_the_brain():
    step = _brain().evolve(force=True)
    assert step is not None and step.status in ("promoted", "rolled_back", "refused")


def test_tuned_knobs_survive_a_restart():
    brain = _warm(_brain())
    for _ in range(3):
        brain.evolve(force=True)
    tuned = brain.omega.knobs()
    reborn = _brain()
    reborn.load_dict(brain.to_dict())
    assert reborn.omega.knobs() == tuned


def test_disabling_the_layer_leaves_v01_behaviour():
    brain = NyxBrain(NyxConfig(omega_enabled=False))
    assert brain.omega is None and brain.evolve(force=True) is None


# --------------------------------------------------------------------------- #
# Honesty and fail-soft
# --------------------------------------------------------------------------- #
def test_stats_say_what_she_tunes_and_what_she_never_will():
    note = _kernel().stats()["note"]
    assert "knobs, not her constitution" in note
    assert "not 'better" in note.lower()
    assert "rolled back" in note and "sample floor" in note


def test_the_bandit_records_what_it_tried():
    kernel = _kernel()
    _warm(kernel.brain)
    for _ in range(4):
        kernel.evolve(force=True)
    bandit = kernel.stats()["bandit"]
    assert bandit and all("pulls" in v for v in bandit.values())


def test_everything_is_fail_soft_on_junk():
    kernel = _kernel()
    assert kernel.load_dict(None) is False
    assert kernel.load_dict({"ledger": ["junk"]}) is True
    assert Step().to_dict()["status"] == ""


def test_a_brain_with_nothing_tunable_says_so():
    kernel = SelfEvolutionKernel(object(), every_s=0.0, min_samples=1)
    step = kernel.evolve(force=True)
    assert step.status == "skipped" and "nothing tunable" in step.reason
