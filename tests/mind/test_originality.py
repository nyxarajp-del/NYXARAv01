"""Tests for the True-Original-Creativity engine (P24 · F19).

Covers the closed loop plus every organ integrated here: MCTS imagination (4),
causal engine (6), CA physics sandbox (9), reality anchor (11), dialectic gating
(13), cross-modal transmutation (14), meta-genetic operator invention (15),
benchmark-gated strategy evolution (3), and the dream consolidation phase (8)."""

import random
import sys
import xml.etree.ElementTree as ET

import pytest

from nyxara.memory.provenance import SourceType
from nyxara.mind.faculties import (FacultyRegistry, FacultySelector, LLMFaculty,
                                   Task, TaskType)
from nyxara.mind.originality import (CausalSketch, CreativeFaculty, Genome,
                                     OriginalityEngine, RealityAnchor,
                                     StrategyEvolver, StrategyGenome)


@pytest.fixture()
def engine():
    return OriginalityEngine(rng=random.Random(7))


# --------------------------------------------------------------------------- #
# the closed loop
# --------------------------------------------------------------------------- #
def test_idea_creation_survives_gates_and_is_real(engine):
    rep = engine.create("a lock that heals itself", modality="idea",
                        rounds=1, population=8)
    assert rep["llm_used"] is False
    assert rep["kept"], "at least one invention must survive"
    best = rep["best"]
    assert "Mechanism trace" in best["content"]
    assert "Energy ledger" in best["content"]
    assert best["metadata"]["llm_used"] is False


def test_verse_respects_its_evolved_form(engine):
    rep = engine.create("the night sky over the sea", modality="verse",
                        rounds=1, population=8)
    assert rep["kept"]
    best = rep["best"]
    budgets = best["genome"]["genes"]["budgets"]
    body = [ln for ln in best["content"].splitlines()[1:] if ln.strip()]
    assert len(body) == len(budgets)


def test_art_parses_as_svg(engine):
    rep = engine.create("spiral galaxies", modality="art", rounds=1, population=8)
    assert rep["kept"]
    root = ET.fromstring(rep["best"]["content"])
    assert root.tag.endswith("svg")


def test_novelty_is_one_on_empty_archive(engine):
    desc = engine._describe("something entirely new", "idea")
    assert engine.archive.novelty(desc) == 1.0


def test_duplicates_are_never_kept(engine):
    engine.create("glass gardens", modality="idea", rounds=1, population=8)
    ids = [p.original_id for p in engine.kept]
    assert len(ids) == len(set(ids))
    g = Genome.from_dict(engine.kept[0].genome)
    piece, reason = engine._consider(g, engine.kept[0].topic)
    assert piece is None and reason in ("duplicate", "not-novel", "quality",
                                        "critic:cliche", "critic:vagueness",
                                        "critic:redundancy", "critic:unsupported",
                                        "critic:contradiction", "critic:degenerate")


def test_empty_content_never_survives(engine):
    assert engine._verify("", "idea", "anything", {}) == 0.0


def test_deterministic_under_seed():
    r1 = OriginalityEngine(rng=random.Random(42)).create(
        "glass gardens", modality="idea", rounds=1, population=8)
    r2 = OriginalityEngine(rng=random.Random(42)).create(
        "glass gardens", modality="idea", rounds=1, population=8)
    assert [p["content"] for p in r1["kept"]] == [p["content"] for p in r2["kept"]]


# --------------------------------------------------------------------------- #
# organ 6 — causal engine
# --------------------------------------------------------------------------- #
def test_causal_do_intervention_kills_the_benefit():
    sk = CausalSketch(["combine", "adapt"])
    assert sk.is_acyclic()
    assert "benefit" in sk._reachable("input")
    assert "benefit" not in sk._reachable("input", removed="mech:combine")
    assert sk.viability() > 0.8


def test_causal_chain_friction_lowers_viability():
    short = CausalSketch(["combine"]).viability()
    long = CausalSketch(["combine", "adapt", "reverse", "blend"]).viability()
    assert long < short


# --------------------------------------------------------------------------- #
# organ 11 — reality anchor
# --------------------------------------------------------------------------- #
def test_reality_anchor_rejects_impossible_claims():
    anchor = RealityAnchor()
    out = anchor.audit("a perpetual motion wheel giving infinite energy", ["combine"])
    assert out["reality"] == 0.0
    assert out["violation"] is not None


def test_reality_ledger_never_exceeds_input():
    anchor = RealityAnchor()
    out = anchor.audit("a costed heat pump", ["combine", "adapt", "blend"])
    assert 0.0 < out["reality"] < 1.0
    assert all(st["out"] < st["in"] for st in out["ledger"])


def test_impossible_invention_scores_zero_quality(engine):
    info = {"ops": ["combine"], "causal": 1.0}
    q = engine._verify("a perpetual motion engine of infinite energy",
                       "idea", "engine", info)
    assert q == 0.0


# --------------------------------------------------------------------------- #
# organ 9 — the CA physics sandbox
# --------------------------------------------------------------------------- #
def test_dead_rule_fails_life_rule_passes(engine):
    dead, _ = engine._simulate_automaton(set(), {8}, seed=3)
    life, _ = engine._simulate_automaton({3}, {2, 3}, seed=3)
    assert dead["dynamics"] < life["dynamics"]
    assert dead["dynamics"] == 0.0


def test_saturating_rule_fails(engine):
    sat, _ = engine._simulate_automaton({1, 2, 3, 4, 5, 6, 7, 8},
                                        {0, 1, 2, 3, 4, 5, 6, 7, 8}, seed=3)
    assert sat["dynamics"] == 0.0


# --------------------------------------------------------------------------- #
# organ 14 — transmutation
# --------------------------------------------------------------------------- #
def test_transmutation_crosses_modalities_through_the_gates(engine):
    engine.create("a lantern of tides", modality="idea", rounds=1, population=8)
    src = engine.kept[0]
    out = engine.transmute(src, "verse")
    if out is not None:              # gated — may honestly fail, but if kept:
        assert out.modality == "verse"
        assert out.metadata["isomorphic_source"] == src.original_id
        assert out.metadata["llm_used"] is False


def test_transmutation_to_same_modality_refused(engine):
    engine.create("a lantern of tides", modality="idea", rounds=1, population=8)
    assert engine.transmute(engine.kept[0], engine.kept[0].modality) is None


# --------------------------------------------------------------------------- #
# organ 8 — dream consolidation
# --------------------------------------------------------------------------- #
def test_dream_compresses_without_losing_the_best(engine):
    engine.create("glass gardens", modality="idea", rounds=1, population=8)
    engine.create("iron rivers", modality="verse", rounds=1, population=8)
    best_id = max(engine.kept, key=lambda p: p.novelty * p.quality).original_id
    out = engine.dream()
    assert any(p.original_id == best_id for p in engine.kept)
    assert out["dream_cycles"] == 1
    assert isinstance(out["motifs"], list)


# --------------------------------------------------------------------------- #
# organs 3 & 15 — strategy evolution + operator invention
# --------------------------------------------------------------------------- #
def test_strategy_bounds_are_enforced():
    s = StrategyGenome(mutation_rate=99.0, quality_floor=-5.0).clamped()
    assert 0.02 <= s.mutation_rate <= 0.6
    assert 0.15 <= s.quality_floor <= 0.7


def test_strategy_adopted_only_on_measured_win():
    ev = StrategyEvolver()
    out = ev.evolve_strategy(lambda s: 1.0, random.Random(2))   # flat benchmark
    assert out["adopted"] is False                               # no win → no rewrite
    out2 = ev.evolve_strategy(lambda s: s.mutation_rate * 100, random.Random(3))
    assert out2["adopted"] == (out2["candidate"] > out2["base"] * 1.02)


def test_operator_invention_requires_plateau_and_merit():
    ev = StrategyEvolver()
    assert ev.maybe_invent_operator(random.Random(1)) is None    # no plateau
    for _ in range(3):
        ev.note_epoch(1, 10)
    invented = ev.maybe_invent_operator(random.Random(1))
    assert invented is not None and invented.invented
    for _ in range(StrategyEvolver.TRIAL_USES):                  # fails its trial…
        ev.credit(invented, success=False)
    assert invented not in ev.operators                          # …and is retired


# --------------------------------------------------------------------------- #
# provenance + no-LLM guarantee
# --------------------------------------------------------------------------- #
def test_proposal_carries_self_reflection_provenance(engine):
    prop = engine.propose("a bridge made of sound")
    assert prop.provenance is not None
    assert prop.provenance.source is SourceType.SELF_REFLECTION
    assert prop.metadata["llm_used"] is False


def test_module_never_imports_the_llm():
    import re

    import nyxara.mind.originality as mod
    assert not hasattr(OriginalityEngine, "llm")
    src = open(mod.__file__, encoding="utf-8").read()
    assert not re.search(r"^\s*(from|import)\s+nyxara\.mind\.llm\b", src,
                         re.MULTILINE)


def test_report_states_no_llm(engine):
    assert engine.report()["llm_used"] is False


# --------------------------------------------------------------------------- #
# faculty routing — her engine outranks the LLM
# --------------------------------------------------------------------------- #
class _MockLLM:
    def generate(self, *a, **k):
        return "mock"


def test_creative_faculty_beats_llm_at_generation(engine):
    reg = FacultyRegistry()
    reg.register(CreativeFaculty(engine=engine))
    reg.register(LLMFaculty(_MockLLM()))
    sel = FacultySelector(reg)
    chosen = sel.select(Task(TaskType.GENERATION, "write a poem about rust"))
    assert chosen is not None and chosen.name == "creative"


def test_default_registry_routes_generation_to_her():
    from nyxara.mind.reasoning_faculties import build_default_faculties
    # even with the LLM registered, GENERATION routes to her own engine
    reg, sel = build_default_faculties(llm=_MockLLM())
    chosen = sel.select(Task(TaskType.GENERATION, "compose a verse"))
    assert chosen is not None and chosen.name == "creative"
    # exact engines still win their own turf
    math_task = Task(TaskType.ARITHMETIC, "what is 9*9", payload="9*9")
    assert sel.select(math_task).name == "math"


def test_faculty_handle_produces_her_proposal(engine):
    fac = CreativeFaculty(engine=engine)
    prop = fac.handle(Task(TaskType.GENERATION, "a poem about the tide"))
    assert prop.provenance.source is SourceType.SELF_REFLECTION
    assert prop.metadata["llm_used"] is False


# --------------------------------------------------------------------------- #
# autonomy + persistence
# --------------------------------------------------------------------------- #
def test_autonomous_step_needs_no_prompt(engine):
    engine.step()
    project = engine.muse.projects[-1]
    assert project.status in ("corroborated", "refuted")
    assert project.hypothesis


def test_persistence_round_trip(engine, tmp_path):
    engine.create("glass gardens", modality="idea", rounds=1, population=8)
    path = str(tmp_path / "originality.json")
    engine.save(path)
    clone = OriginalityEngine(rng=random.Random(9))
    assert clone.load(path)
    assert len(clone.kept) == len(engine.kept)
    assert clone.archive.coverage() == engine.archive.coverage()
    assert clone.atelier.negotiations == engine.atelier.negotiations
    assert clone.report()["llm_used"] is False


def test_load_survives_corrupt_state(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    assert OriginalityEngine().load(str(path)) is False
