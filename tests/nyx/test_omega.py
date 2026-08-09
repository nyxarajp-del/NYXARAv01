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


#: Distinct enough that the traces they produce do not all collide on recall. Near-identical
#: turns retrieve each other and the replay probe reads flat at every threshold, which measures
#: the corpus rather than the knob.
_LINES = ("gravity pulls the apple toward the ground",
          "a heavier flywheel smooths the running of an engine",
          "friction converts motion into heat inside a bearing",
          "the governor limits how much steam reaches the valve",
          "lubrication reduces wear between a piston and its cylinder")


def _resolved(brain: NyxBrain, n: int = 12) -> NyxBrain:
    """Give her real resolved assessments and real stored traces, both honestly.

    ``force`` waives the cadence and deliberately **not** the sample floor, so a test that wants
    her to actually evolve has to give her something to have measured — which is the whole point
    of the floor, and is exactly what the old ``force=True`` shortcut was quietly skipping.
    """
    for index in range(max(1, n)):
        thought = brain.think(f"{_LINES[index % len(_LINES)]} — case {index}")
        brain.resolve(thought, correct=1.0)
    return brain


def _kernel(**kw) -> SelfEvolutionKernel:
    brain = _resolved(_brain())
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
    kernel = SelfEvolutionKernel(_brain(), every_s=0.0, min_samples=1)
    assert kernel.brain.graph.stats().edges == 0
    assert kernel._score_after(kernel.brain.graph, "hebbian_rate") == \
        pytest.approx(kernel.fitness().score)


def test_an_uninformative_graph_earns_no_free_health_bonus_either():
    """"Empty" was never the real threshold — *uninformative* is.

    A three-edge graph's mean degree is an artefact of two turns, not a property of her
    topology, and scoring it handed every knob most of the free win that guarding only the
    empty case was supposed to prevent.
    """
    kernel = SelfEvolutionKernel(_brain(), every_s=0.0, min_samples=1,
                                 min_edges_for_health=24, replay_k=0)
    kernel.brain.perceive("the engine turns the flywheel")
    edges = kernel.brain.graph.stats().edges
    assert 0 < edges < kernel.min_edges_for_health
    assert kernel._score_after(kernel.brain.graph, "hebbian_rate") == \
        pytest.approx(kernel.fitness().score)


def test_the_health_probe_still_reads_a_graph_that_has_enough_structure():
    """The guard abstains while the graph says nothing; it does not switch the probe off."""
    kernel = SelfEvolutionKernel(_brain(), every_s=0.0, min_samples=1,
                                 min_edges_for_health=4, replay_k=0)
    _warm(kernel.brain)
    assert kernel.brain.graph.stats().edges >= 4
    assert kernel._score_after(kernel.brain.graph, "hebbian_rate") != \
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
    """A real hill-climb, on the one knob the one-step gauntlet can actually see.

    ``recall_threshold`` set too high loses retrieval she could otherwise make; lowering it
    raises the replay score immediately, so the gauntlet has something real to promote on.
    """
    kernel = _kernel(stall_after=99)
    kernel.brain.memory.apply_knobs({"recall_threshold": 0.3})   # deliberately too high
    before = kernel.knobs()
    for _ in range(80):
        if kernel.evolve(force=True).status == "promoted":
            break
    assert kernel.promoted >= 1
    assert kernel.knobs() != before
    # …and it is an actual improvement, not merely a change that was allowed through.
    assert kernel.brain.memory.knobs()["recall_threshold"] < 0.3


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
    kernel = _kernel(stall_after=99)
    kernel.brain.memory.apply_knobs({"recall_threshold": 0.3})
    kernel.checkpoint(note="start")
    original = kernel.knobs()
    for _ in range(80):
        if kernel.evolve(force=True).status == "promoted":
            break
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
    step = _resolved(_brain()).evolve(force=True)
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
    step = kernel.evolve(force=True, ignore_sample_floor=True)
    assert step.status == "skipped" and "nothing tunable" in step.reason


# --------------------------------------------------------------------------- #
# The three guarantees the docs make — each one was measurably not true
# --------------------------------------------------------------------------- #
def test_force_waives_the_cadence_and_not_the_sample_floor():
    """The floor is an epistemic guard; the cadence is a schedule. Only one is overrulable.

    Both ``/nyx omega evolve`` and ``POST /v1/nyx/evolve`` pass ``force=True``, so a ``force``
    that also waived the floor meant "she does not evolve below the floor" was false on every
    path anyone actually uses.
    """
    kernel = SelfEvolutionKernel(_brain(), every_s=3600.0, min_samples=12)
    assert not kernel.fitness().enough
    kernel.evolve()                              # consumes the first slot

    step = kernel.evolve()                       # blocked by the cadence
    assert step.status == "skipped" and "not due" in step.reason

    step = kernel.evolve(force=True)             # cadence waived, floor still holds
    assert step.status == "skipped" and "fitting noise" in step.reason
    assert not step.knob                         # nothing was touched


def test_waiving_the_floor_needs_its_own_name_and_marks_the_step_unmeasured():
    kernel = SelfEvolutionKernel(_brain(), every_s=0.0, min_samples=12)
    step = kernel.evolve(force=True, ignore_sample_floor=True)
    assert step.knob and step.status in ("promoted", "rolled_back")
    assert step.measured is False
    assert "sample floor was waived" in step.reason
    assert "UNMEASURED" in step.render()


def test_a_step_taken_above_the_floor_is_marked_measured():
    kernel = _kernel()
    assert kernel.fitness().enough
    assert kernel.evolve(force=True).measured is True


def test_the_baseline_is_read_on_the_same_scale_the_gauntlet_scores_on():
    """Baseline was raw fitness while the score included the probes.

    Measured before the fix: every knob was promoted, all with an identical score, because the
    gap was the probe's and not the knob's — so "roll back anything that did not beat its own
    baseline" never actually tested a knob.
    """
    kernel = _kernel(stall_after=99)
    for _ in range(6):
        step = kernel.evolve(force=True)
        if step.knob and step.status in ("promoted", "rolled_back"):
            assert step.baseline == pytest.approx(
                kernel._score_after(kernel.targets()[step.target], step.knob))


def test_a_knob_the_gauntlet_cannot_see_is_not_reported_as_having_failed():
    """"Did not beat baseline" reads as evidence against the knob. None was gathered."""
    kernel = _kernel(stall_after=99)
    inert = [kernel.evolve(force=True) for _ in range(12)]
    unmoved = [s for s in inert if s.knob and s.scored == pytest.approx(s.baseline)]
    assert unmoved
    for step in unmoved:
        assert "moved nothing the one-step gauntlet can measure" in step.reason
    assert any(v["inert"] for v in kernel.stats()["bandit"].values())


def test_an_unavailable_gauntlet_still_says_it_was_unavailable(monkeypatch):
    """A refusal to run must not be mistaken for a change that measured nothing."""
    kernel = _kernel()
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kw):
        if name == "nyxara.nyx5.autopoiesis":
            raise ImportError("no gauntlet")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    step = kernel.evolve(force=True)
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert "unavailable" in step.reason


# --------------------------------------------------------------------------- #
# The replay probe — the only part of the score a knob can move in one step
# --------------------------------------------------------------------------- #
def test_the_replay_probe_responds_to_the_knob_that_governs_retrieval():
    kernel = _kernel()
    kernel.brain.memory.apply_knobs({"recall_threshold": 0.02})
    generous = kernel._replay()
    kernel.brain.memory.apply_knobs({"recall_threshold": 0.6})
    strict = kernel._replay()
    assert generous is not None and strict is not None
    assert strict < generous            # a threshold nothing clears recalls nothing


def test_the_replay_probe_asks_with_the_cue_not_the_stored_text():
    """A trace's own text matches itself at ~1.0, so no threshold ever binds and the probe
    reads identically at every setting — it would measure nothing at all."""
    kernel = _kernel()
    traces = list(kernel.brain.memory.traces.values())
    assert any(t.cue for t in traces)
    by_text = [kernel.brain.memory.recall(t.text, k=1) for t in traces if t.cue]
    assert all(g.score > 0.9 for g in by_text if g.hit is not None)


def test_the_replay_probe_abstains_rather_than_scoring_zero_on_an_empty_store():
    """Nothing stored is not the same as nothing recalled."""
    kernel = SelfEvolutionKernel(_brain(), every_s=0.0, min_samples=1)
    assert kernel._replay() is None


def test_the_replay_probe_can_be_switched_off():
    kernel = SelfEvolutionKernel(_resolved(_brain()), every_s=0.0, min_samples=1, replay_k=0)
    assert kernel._replay() is None
