"""Tests for growth/mind_evolution.py — recursive mind-evolution (evolve the way of thinking).

All offline, deterministic, no network: a scripted stochastic base sampler stands in for a real
reasoner so self-consistency voting has a measurable effect, and a tiny toy benchmark keeps each
generation fast. Mirrors the conventions in tests/growth/test_recursive_improvement.py.
"""

from __future__ import annotations

import re

import pytest

from nyxara.eval.benchmark import Benchmark, BenchmarkTask, Grader
from nyxara.guard.value_learning import IMMUTABLE_VALUES
from nyxara.growth.mind_evolution import (
    EvolutionLineageReport,
    Generation,
    MindEvolutionEngine,
    ReasoningGenome,
    genome_solver,
)


# --------------------------------------------------------------------------- #
# Fixtures — a toy battery + a noisy reasoner (correct with prob p per attempt)
# --------------------------------------------------------------------------- #
def _toy_bench(n: int = 8) -> Benchmark:
    return Benchmark("toy", [
        BenchmarkTask(id=f"q{i}", prompt=f"What is {i}+{i}?", answer=str(2 * i),
                      grader=Grader.NUMERIC) for i in range(1, n + 1)])


def _noisy_sampler(p: float = 0.7):
    """A reasoner that answers correctly with probability ``p``, else off by one."""
    def _s(prompt: str, rng) -> str:
        i = int(re.findall(r"\d+", prompt)[0])
        return str(2 * i) if rng.random() < p else str(2 * i + 1)
    return _s


def _engine(**kw) -> MindEvolutionEngine:
    return MindEvolutionEngine(benchmark=_toy_bench(), base_sampler=_noisy_sampler(),
                               seed=kw.pop("seed", 7), **kw)


@pytest.fixture
def isolated(tmp_path):
    """An engine factory whose lineage persists to an isolated tmp dir (never ~/.nyxara)."""
    def _make(**kw):
        kw.setdefault("benchmark", _toy_bench())
        kw.setdefault("base_sampler", _noisy_sampler())
        kw.setdefault("seed", 7)
        kw["data_dir"] = tmp_path
        return MindEvolutionEngine(**kw)
    return _make


# --------------------------------------------------------------------------- #
# Genome
# --------------------------------------------------------------------------- #
def test_genome_round_trips_and_clamps():
    g = ReasoningGenome(deliberation_depth=999, mcts_rollouts=-5, system2_bias=2.0)
    # out-of-range values are clamped to the declared bounds on construction
    assert g.deliberation_depth == 12.0
    assert g.mcts_rollouts == 0.0
    assert g.system2_bias == 1.0
    # to_dict / from_dict is a faithful round-trip
    g2 = ReasoningGenome.from_dict(g.to_dict())
    assert g2.to_dict() == g.to_dict()


def test_genome_from_dict_is_defensive():
    assert ReasoningGenome.from_dict(None).to_dict() == ReasoningGenome().to_dict()
    # a broken value falls back to the field default rather than raising
    g = ReasoningGenome.from_dict({"deliberation_depth": "not-a-number"})
    assert g.deliberation_depth == ReasoningGenome().deliberation_depth


def test_evolver_genome_normalisation_round_trips():
    g = ReasoningGenome(deliberation_depth=8, mcts_rollouts=10, recall_k=4)
    ev = g.to_evolver_genome()
    assert all(0.0 <= v <= 1.0 for v in ev.values())              # every knob normalised to [0,1]
    back = ReasoningGenome.from_evolver_genome(ev)
    assert back.to_dict() == g.to_dict()                          # and inverts cleanly


def test_vote_samples_grows_with_deliberation():
    lo = ReasoningGenome(deliberation_depth=1, mcts_rollouts=0).vote_samples()
    hi = ReasoningGenome(deliberation_depth=12, mcts_rollouts=16).vote_samples()
    assert 1 <= lo < hi <= 12


def test_character_core_never_in_genome():
    assert not (set(IMMUTABLE_VALUES) & set(ReasoningGenome().to_dict()))


# --------------------------------------------------------------------------- #
# Measurement — more deliberation genuinely lifts accuracy on a noisy reasoner
# --------------------------------------------------------------------------- #
def test_self_consistency_lifts_accuracy():
    # self-consistency reduces variance, so it lifts *expected* accuracy — assert it statistically
    # across many seeds rather than on one noisy 8-task draw
    shallow_g = ReasoningGenome(deliberation_depth=1, mcts_rollouts=0)
    deep_g = ReasoningGenome(deliberation_depth=12, mcts_rollouts=16)
    assert deep_g.vote_samples() > shallow_g.vote_samples()

    def _mean_acc(genome, seeds):
        accs = [MindEvolutionEngine(benchmark=_toy_bench(8), base_sampler=_noisy_sampler(0.65),
                                    seed=s).measure(genome)["accuracy"] for s in seeds]
        return sum(accs) / len(accs)

    seeds = range(40)
    assert _mean_acc(deep_g, seeds) > _mean_acc(shallow_g, seeds)


def test_measure_is_deterministic_per_genome():
    eng = _engine()
    g = ReasoningGenome(deliberation_depth=6, mcts_rollouts=6)
    assert eng.measure(g) == eng.measure(g)        # pure function of the genome → reproducible


def test_fitness_prefers_fewer_votes_at_equal_accuracy():
    # a near-deterministic reasoner: both shallow and deep hit 100% accuracy, so the compute-cost
    # term must break the tie in favour of the cheaper (faster) strategy
    eng = MindEvolutionEngine(benchmark=_toy_bench(), base_sampler=_noisy_sampler(p=1.0),
                              seed=1, cost_penalty=0.1)
    cheap = eng.measure(ReasoningGenome(deliberation_depth=1, mcts_rollouts=0))
    pricey = eng.measure(ReasoningGenome(deliberation_depth=12, mcts_rollouts=16))
    assert cheap["accuracy"] == pricey["accuracy"] == 1.0
    assert cheap["fitness"] > pricey["fitness"]    # same accuracy, fewer samples → higher fitness


# --------------------------------------------------------------------------- #
# The generational loop
# --------------------------------------------------------------------------- #
def test_evolution_climbs_and_promotes(isolated):
    eng = isolated()
    rep = eng.evolve_generations(6, enact=False, inner_generations=5, population=6)
    assert isinstance(rep, EvolutionLineageReport)
    assert rep.promoted >= 1                                       # at least one real improvement
    # every promotion strictly increases the champion's fitness — the gauntlet's guarantee
    promoted = [g for g in rep.generations if g.promoted]
    fits = [g.fitness for g in promoted]
    assert fits == sorted(fits) and len(set(fits)) == len(fits)    # strictly increasing
    # the committed champion's accuracy never regresses below generation 0
    assert promoted[-1].accuracy >= promoted[0].accuracy


def test_generation_lineage_links_parent(isolated):
    rep = isolated().evolve_generations(4, enact=False, inner_generations=4, population=6)
    promoted = [g for g in rep.generations if g.promoted]
    # every promoted generation points back at the index it improved upon
    for g in promoted[1:]:
        assert g.parent_index < g.index


def test_dry_run_does_not_touch_live_core(isolated):
    class _Core:
        recursive_improver = None
        reasoner = None

    core = _Core()
    eng = isolated(core=core)
    eng.evolve_generations(4, enact=False, inner_generations=4, population=6)
    assert getattr(core, "_reasoning_genome", None) is None        # sandbox: live mind untouched
    assert core.recursive_improver is None


def test_enact_installs_strategy_into_live_core(isolated):
    class _Core:
        recursive_improver = None
        reasoner = None

    core = _Core()
    eng = isolated(core=core)
    rep = eng.evolve_generations(4, enact=True, inner_generations=4, population=6)
    if rep.promoted >= 1:
        champ = ReasoningGenome.from_dict(rep.champion)
        assert getattr(core, "_reasoning_genome", None) is not None
        assert core.recursive_improver is not None
        assert core.recursive_improver.n_iterations == champ.n_iterations()


# --------------------------------------------------------------------------- #
# Safety — the gauntlet never relaxes
# --------------------------------------------------------------------------- #
def test_gauntlet_rejects_non_improvement():
    eng = _engine()
    g = ReasoningGenome()
    ok, reason = eng._gauntlet(g, champ_fitness=0.9, cand_fitness=0.9, min_improvement=1e-4)
    assert not ok and "not measurably better" in reason


def test_gauntlet_accepts_strict_improvement():
    eng = _engine()
    ok, reason = eng._gauntlet(ReasoningGenome(), champ_fitness=0.5,
                               cand_fitness=0.8, min_improvement=1e-4)
    assert ok


# --------------------------------------------------------------------------- #
# Persistence — lineage survives a fresh engine (memory-less / JSON path)
# --------------------------------------------------------------------------- #
def test_lineage_persists_across_engines(isolated):
    eng1 = isolated()
    rep1 = eng1.evolve_generations(3, enact=False, inner_generations=3, population=4)
    n1 = len(rep1.generations)
    assert n1 >= 1

    # a brand-new engine reloads the lineage and continues numbering from where it left off
    eng2 = isolated()
    loaded = eng2._load_lineage()
    assert len(loaded) >= n1
    assert all(isinstance(g, Generation) for g in loaded)
    rep2 = eng2.evolve_generations(1, enact=False, inner_generations=3, population=4)
    # the continued run's first new generation has a higher index than the persisted tail
    new_gens = [g for g in rep2.generations if g.index > loaded[-1].index - 1]
    assert new_gens, "a fresh run should append, not restart, the lineage"


# --------------------------------------------------------------------------- #
# Plateau guard — a flat search stops early without error
# --------------------------------------------------------------------------- #
def test_plateau_guard_stops_a_flat_search(tmp_path):
    # a perfect deterministic reasoner: gen 0 is already optimal, so nothing can strictly improve
    eng = MindEvolutionEngine(benchmark=_toy_bench(), base_sampler=_noisy_sampler(p=1.0),
                              seed=2, cost_penalty=0.0, data_dir=tmp_path)
    rep = eng.evolve_generations(20, enact=False, inner_generations=3,
                                 population=4, plateau_window=2)
    assert rep.plateaued                                           # bailed out early, no ceiling hit
    assert len(rep.generations) < 21


# --------------------------------------------------------------------------- #
# The solver — majority voting returns the most-voted decision
# --------------------------------------------------------------------------- #
def test_genome_solver_majority_vote():
    # a sampler that returns "10" twice as often as "11" → majority is "10"
    seq = iter(["10", "11", "10", "10", "11", "10", "10"])

    def _s(prompt, rng):
        try:
            return next(seq)
        except StopIteration:
            return "10"

    solver = genome_solver(ReasoningGenome(deliberation_depth=8, mcts_rollouts=8), _s, seed=0)
    assert "10" in solver("What is 5+5?")
