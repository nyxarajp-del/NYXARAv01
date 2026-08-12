"""Layers 1–17 and the stack. Each test targets the contract that layer would fail silently on.

Three of these exist because the layer in question shipped inert during development and looked
fine: :func:`test_world_model_decay_gate_actually_learns` (the gate's gradient was identically
zero), :func:`test_contradiction_finds_disconfirming_evidence` (it searched the wrong store and
found nothing forever), and :func:`test_stack_exposes_all_eighteen_layers` (Layer 16's attribute
name did not match its registry key, so it vanished from ``layers()``).
"""

from __future__ import annotations

import numpy as np
import pytest

from nyxara.nyx001.layers import (
    CognitiveCompression,
    CognitiveStack,
    ContradictionEngine,
    CounterfactualEngine,
    CuriosityEngine,
    DynamicAttention,
    EpisodicMemory,
    MetaLearner,
    Observation,
    Perception,
    Planner,
    ReasoningEngine,
    ResourceAwareCognition,
    SelfCritique,
    SelfImprovement,
    SelfModel,
    SemanticMemory,
    Substrate,
    WorldModel,
)
from nyxara.nyx001.layers.l05_attention import Candidate


@pytest.fixture
def sub():
    return Substrate(seed=13, rung="toy")


class _Pinned:
    """A substrate pinned to a small width, so learning tests stay fast but real."""

    def __init__(self, width=24, seed=5):
        self._s = Substrate(seed=seed, rung="toy")
        self.width = width

    def stream(self, name):
        return self._s.stream(name)

    def note_params(self, *t):
        return 0


def _rotation(width, theta=0.7, decay=0.99):
    R = np.eye(width)
    for i in range(0, width - 1, 2):
        R[i, i] = R[i + 1, i + 1] = np.cos(theta)
        R[i, i + 1] = -np.sin(theta)
        R[i + 1, i] = np.sin(theta)

    def step(s):
        nxt = decay * (R @ s)
        nxt[0] += 0.15 * np.tanh(3.0 * s[1])
        return nxt
    return step


# --------------------------------------------------------------------------- #
# L0 / L1
# --------------------------------------------------------------------------- #
def test_birth_certificate_identifies_a_system(sub):
    other = Substrate(seed=13, rung="toy")
    assert sub.birth_certificate().same_system_as(other.birth_certificate())
    assert not sub.birth_certificate().same_system_as(
        Substrate(seed=99, rung="toy").birth_certificate())


def test_streams_are_private_per_layer(sub):
    a = sub.stream("alpha").normal(0, 1, 8)
    b = sub.stream("beta").normal(0, 1, 8)
    assert not np.allclose(a, b)
    assert np.allclose(a, sub.stream("alpha").normal(0, 1, 8)), "a stream must be stable"


def test_perception_refuses_to_name_a_one_off(sub):
    """Naming a single occurrence manufactures structure every layer above trusts."""
    p = Perception(sub, min_support=2)
    once = p.perceive(Observation(payload="unrepeated singular phrase"))
    assert once.entities == [], "named something seen exactly once"
    for _ in range(3):
        again = p.perceive(Observation(payload="repeated repeated phrase"))
    assert again.entities, "never named anything despite repetition"


def test_perception_novelty_is_higher_for_the_unfamiliar(sub):
    """The very first percept has novelty 0 by construction (the trace starts *at* it), so the
    meaningful comparison is a settled input against a genuinely different one."""
    p = Perception(sub)
    for _ in range(10):
        p.perceive(Observation(payload="the same sentence again"))
    familiar = p.perceive(Observation(payload="the same sentence again")).novelty
    strange = p.perceive(Observation(payload="entirely unrelated vocabulary elsewhere")).novelty
    assert strange > familiar, f"novel input scored no higher: {strange} vs {familiar}"


# --------------------------------------------------------------------------- #
# L2 — the central organ
# --------------------------------------------------------------------------- #
@pytest.mark.timeout(900)
def test_world_model_learns_and_beats_the_copy_baseline():
    """A residual model that learned nothing predicts no change. It must do better than that."""
    wm = WorldModel(_Pinned(width=24), lr=5e-3)
    system = _rotation(24)
    rng = np.random.default_rng(0)
    s = rng.normal(0, 0.5, 24)
    for i in range(6000):
        if i % 25 == 0:
            s = rng.normal(0, 0.5, 24)
            wm.reset_state()
        wm.predict(s)
        nxt = system(s)
        wm.observe(nxt)
        s = nxt
    assert wm.is_learning() is True
    curve = wm.learning_curve()
    assert sum(curve[-2:]) < sum(curve[:2]), f"error did not fall: {curve}"

    s = rng.normal(0, 0.5, 24)
    wm.reset_state()
    m, c = [], []
    for _ in range(200):
        p = wm.predict(s)
        nxt = system(s)
        m.append(float(np.mean((p.state - nxt) ** 2)))
        c.append(float(np.mean((s - nxt) ** 2)))
        s = nxt
    assert np.mean(m) < np.mean(c), "no better than copying its own input"


def test_world_model_decay_gate_actually_learns():
    """The gate was once passed detached: gradient identically zero, recurrence decorative."""
    wm = WorldModel(_Pinned(width=16), lr=5e-3)
    before = wm.a_raw.data.copy()
    system = _rotation(16)
    rng = np.random.default_rng(1)
    s = rng.normal(0, 0.5, 16)
    for _ in range(60):
        wm.predict(s)
        s2 = system(s)
        wm.observe(s2)
        s = s2
    assert not np.allclose(before, wm.a_raw.data), "the decay gate never moved — gradient is zero"
    gate = 1.0 / (1.0 + np.exp(-wm.a_raw.data))
    assert float(gate.min()) > 0.0 and float(gate.max()) < 1.0, "gate escaped (0,1)"


def test_world_model_reports_none_before_it_can_say(sub):
    wm = WorldModel(_Pinned(width=16))
    assert wm.mean_error() is None
    assert wm.is_learning() is None
    assert wm.uncertainty() == 1.0


def test_rollout_confidence_decays_with_horizon():
    wm = WorldModel(_Pinned(width=16))
    rolls = wm.rollout(np.ones(16), horizon=5)
    assert len(rolls) == 5
    assert rolls[-1].uncertainty >= rolls[0].uncertainty


# --------------------------------------------------------------------------- #
# L3 / L4 / L14
# --------------------------------------------------------------------------- #
def test_replay_damping_stops_one_episode_monopolising(sub):
    m = EpisodicMemory(sub, capacity=64)
    shock = m.remember("shocking", context=np.ones(24), prediction_error=10.0)
    first = shock.priority
    shock.replays = 8
    assert shock.priority < first, "priority does not decay with replays"


def test_relevance_is_credited_backwards_not_at_write_time(sub):
    m = EpisodicMemory(sub, capacity=64)
    v = np.ones(24)
    ep = m.remember("something", context=v)
    assert ep.future_relevance == 0.0, "relevance must not be assigned when written"
    assert m.credit(v) >= 1
    assert ep.future_relevance > 0.0, "credit() did not reach the episode"


def test_memory_reports_what_it_forgot(sub):
    m = EpisodicMemory(sub, capacity=4)
    for i in range(12):
        m.remember(f"ep{i}", context=np.random.default_rng(i).normal(0, 1, 24),
                   prediction_error=float(i))
    assert m.evicted > 0, "a bounded store that never reports eviction is claiming infinity"
    assert len(m) <= 4


def test_concepts_must_be_earned(sub):
    sem = SemanticMemory(sub, min_support=3)
    from nyxara.nyx001.layers.types import Experience
    v = np.ones(24)
    c = sem.absorb(Experience(key="a", context=v))
    assert c.is_thin, "a cluster of one is not a concept"
    assert sem.concepts() == [], "thin concepts must not be reported as earned"


def test_compression_rejects_a_lossy_result(sub):
    """"Without destroying information" is checked here, not asserted."""
    comp = CognitiveCompression(sub, fidelity_floor=0.999, min_episodes=4, max_codes=1)
    from nyxara.nyx001.layers.types import Experience
    rng = np.random.default_rng(0)
    eps = [Experience(key=f"e{i}", context=rng.normal(0, 1, 32)) for i in range(16)]
    out = comp.compress("c0", eps)
    assert out.fidelity < 0.999
    assert not out.accepted, "accepted a compression that failed its own fidelity floor"
    assert out.bits_saved == 0, "claimed savings for a rejected compression"


def test_compression_accepts_and_reconstructs_real_structure(sub):
    comp = CognitiveCompression(sub, fidelity_floor=0.8, min_episodes=4, max_codes=3)
    from nyxara.nyx001.layers.types import Experience
    rng = np.random.default_rng(0)
    basis = rng.normal(0, 1, (2, 32))
    eps = [Experience(key=f"e{i}", context=(rng.normal(0, 1, 2) @ basis)) for i in range(20)]
    out = comp.compress("c0", eps)
    assert out.accepted and out.fidelity > 0.8
    code = comp.encode("c0", eps[0].context)
    assert code is not None and comp.decode("c0", code) is not None


# --------------------------------------------------------------------------- #
# L5 — nine factors
# --------------------------------------------------------------------------- #
def test_attention_uses_more_than_query_key_match(sub):
    att = DynamicAttention(sub)
    v = np.ones(8)
    plain = Candidate(key="plain", vec=v)
    urgent = Candidate(key="urgent", vec=v, prediction_error=1.0, goal_alignment=1.0)
    alloc = att.allocate(v, [plain, urgent])
    assert alloc.winner.key == "urgent", "identical q·k, yet the extra factors changed nothing"
    assert set(alloc.factors) >= {"match", "G", "U", "M", "E", "T", "R"}


def test_starved_budget_sharpens_attention(sub):
    att = DynamicAttention(sub)
    cands = [Candidate(key=f"c{i}", vec=np.eye(4)[i % 4], prediction_error=i * 0.2)
             for i in range(4)]
    ample = att.allocate(np.ones(4), cands, resource_budget=1.0)
    starved = att.allocate(np.ones(4), cands, resource_budget=0.0)
    assert starved.temperature < ample.temperature
    assert starved.entropy <= ample.entropy + 1e-9, "a starved budget must commit, not spread"


# --------------------------------------------------------------------------- #
# L6 / L10
# --------------------------------------------------------------------------- #
def test_untestable_hypothesis_is_untested_not_refuted(sub):
    r = ReasoningEngine(sub)
    r.propose("a precedes b")
    r.evaluate("a precedes b", lambda: None)
    h = r.get("a precedes b")
    assert h.tested == 0, "a claim that could not be tested must not be scored"
    assert h.confidence == 0.5


def test_verifiable_outranks_a_more_confident_guess(sub):
    r = ReasoningEngine(sub)
    r.propose("checked", verifiable=True)
    guess = r.propose("guess")
    guess.confidence = 0.99
    assert r.best(k=1)[0].claim == "checked", "a louder guess beat a checked derivation"


def test_contradiction_finds_disconfirming_evidence(sub):
    """It once searched episode text for concept names: 732 attacks, zero findings, forever."""
    r = ReasoningEngine(sub)
    c = ContradictionEngine(sub)

    class Sem:
        _succession = {"c0\x00c1": 2.0, "c0\x00c2": 9.0, "c3\x00c4": 5.0}

    false_claim = r.propose("c0 precedes c1")
    attack = c.attack(false_claim, memory=None, reasoning=r, semantic=Sem())
    assert attack.found, "failed to disconfirm a claim the evidence contradicts"
    assert r.get("c0 precedes c1").confidence < 0.5

    true_claim = r.propose("c3 precedes c4")
    assert not c.attack(true_claim, None, r, Sem()).found, "manufactured a contradiction"


def test_shake_lowers_confidence_that_was_never_earned(sub):
    r = ReasoningEngine(sub)
    c = ContradictionEngine(sub)
    h = r.propose("confident but untested")
    h.confidence = 0.95
    assert c.shake([h])
    assert h.confidence < 0.95, "a 0.95 belief on zero tests was left alone"


# --------------------------------------------------------------------------- #
# L7 / L8
# --------------------------------------------------------------------------- #
def test_counterfactual_refuses_to_claim_from_an_unsure_model(sub):
    wm = WorldModel(_Pinned(width=16))
    cf = CounterfactualEngine(sub, wm, trust_ceiling=0.01)
    out = cf.what_if_not(np.ones(16), ["a", "b"], remove=0)
    assert not out.trustworthy, "claimed a causal attribution from an untrained model"
    assert "cannot say" in cf.explain(out)


def test_plan_is_infeasible_when_the_model_cannot_see(sub):
    wm = WorldModel(_Pinned(width=16))
    p = Planner(sub, wm, trust_ceiling=0.0)
    plan = p.plan("do something", np.ones(16), action_space=["a", "b"])
    assert not plan.feasible
    assert "not trustworthy" in p.explain(plan) or "no plan" in p.explain(plan)


def test_planner_without_an_action_space_returns_no_plan(sub):
    p = Planner(sub, WorldModel(_Pinned(width=16)))
    assert not p.plan("goal", np.ones(16), action_space=[]).feasible


# --------------------------------------------------------------------------- #
# L9 / L11 / L12 / L13 / L15 / L17
# --------------------------------------------------------------------------- #
def test_curiosity_abandons_a_region_with_no_progress(sub):
    """The noisy-TV trap: a region that stays wrong forever must be dropped, not chased."""
    cur = CuriosityEngine(sub)
    for _ in range(10):
        cur.observe("noise", 0.9)
    assert cur.stale(), "kept chasing a region with high error and zero progress"
    assert all(r.key != "noise" for r in cur.frontier()), "stale region is still on the frontier"


def test_curiosity_frontier_prefers_falling_error(sub):
    cur = CuriosityEngine(sub)
    for e in (0.9, 0.7, 0.5, 0.3, 0.1):
        cur.observe("learnable", e)
    for _ in range(5):
        cur.observe("flat", 0.5)
    top = cur.frontier(k=1)
    assert top and top[0].key == "learnable"


def test_critique_discards_a_repair_that_does_not_measurably_help(sub):
    sc = SelfCritique(sub)
    out = sc.run("a perfectly adequate answer with content", confidence=0.5,
                 calibration=0.5, support=["evidence"])
    assert out.clean, "found a defect in an answer with no checkable defect"
    assert out.final == out.original


def test_critique_lowers_overconfidence_it_can_measure(sub):
    sc = SelfCritique(sub)
    out = sc.run("an answer with real content here", confidence=0.95,
                 calibration=0.3, support=["e"])
    assert any(f.kind == "overconfidence" for f in out.findings)
    assert out.improved and out.final != out.original


def test_self_model_reports_none_before_it_can_know(sub):
    sm = SelfModel(sub, min_samples=20)
    assert sm.calibration() is None
    assert sm.competence() is None
    for _ in range(5):
        sm.record(0.9, 1.0)
    assert sm.calibration() is None, "reported a calibration from 5 samples"
    assert sm.report().is_thin


def test_self_model_detects_overconfidence(sub):
    sm = SelfModel(sub, min_samples=10)
    for _ in range(40):
        sm.record(0.9, 0.1)          # says 0.9, is right 10% of the time
    assert sm.confidence_bias() > 0.5
    assert sm.adjusted(0.9) < 0.9
    assert "overconfident" in sm.report().summary()


def test_meta_learning_stays_inside_its_bounds(sub):
    ml = MetaLearner(sub)
    from nyxara.nyx001.layers.l13_meta_learning import _BOUNDS
    for _ in range(30):
        s = ml.mutate()
        if s is None:
            continue
        for k, v in s.params.items():
            lo, hi = _BOUNDS[k]
            assert lo <= v <= hi, f"{k}={v} escaped [{lo}, {hi}]"


def test_meta_learning_will_not_rank_before_it_has_evidence(sub):
    ml = MetaLearner(sub)
    assert ml.best() is None
    assert not ml.confident()
    assert ml.improved() is None


def test_resource_allocates_more_to_harder_tasks(sub):
    r = ResourceAwareCognition(sub)
    easy = r.allocate(r.estimate(novelty=0.0, uncertainty=0.0, recall_confidence=1.0))
    hard = r.allocate(r.estimate(novelty=1.0, uncertainty=1.0, recall_confidence=0.0,
                                 concept_covered=False))
    assert hard.seconds > easy.seconds and hard.steps > easy.steps


def test_self_improvement_is_fail_closed(sub):
    """Every path that is not an explicit pass must refuse, with a reason."""
    si = SelfImprovement(sub, min_improvement=0.5)
    layers = {"semantic": SemanticMemory(sub)}
    outs = si.cycle(layers, task=lambda layer: 0.0)     # no improvement possible
    assert outs, "proposed nothing at all"
    assert not any(o.rolled_out for o in outs), "rolled out a change with zero improvement"
    assert all(o.reason for o in outs), "a rejection with no reason"


def test_self_improvement_cannot_widen_its_own_bounds(sub):
    """It may tune a compute ceiling; it may never tune the gate that judges its own proposals."""
    from nyxara.nyx001.layers.l17_self_improve import _MUTABLE
    si = SelfImprovement(sub)
    gate_attrs = {"min_improvement", "max_rollouts_per_cycle", "enabled", "lineage_cap"}
    for key in _MUTABLE:
        layer, attr = key.split(".", 1)
        assert layer != "self_improvement", f"{key} targets the gate layer itself"
        assert attr not in gate_attrs, f"{key} lets it move its own acceptance gate"
    for attr in gate_attrs:
        assert hasattr(si, attr), f"{attr} is not actually a gate attribute any more"


def test_self_improvement_can_be_reverted(sub):
    si = SelfImprovement(sub, min_improvement=-1e9)     # accept anything, to exercise the rollback
    sem = SemanticMemory(sub)
    original = sem.radius
    outs = si.cycle({"semantic": sem}, task=lambda layer: 1.0)
    rolled = [o for o in outs if o.rolled_out]
    if rolled:
        assert si.revert({"semantic": sem}) >= 1
        assert abs(sem.radius - original) < 1e-9 or sem.radius == original


# --------------------------------------------------------------------------- #
# The stack
# --------------------------------------------------------------------------- #
class _Cfg:
    seed = 21
    scale = "toy"
    lr = 5e-3
    memory_capacity = 256


def test_stack_exposes_all_eighteen_layers():
    """Layer 16's attribute name once did not match its registry key, so it vanished silently."""
    stack = CognitiveStack(_Cfg())
    names = set(stack.layers()) | {"substrate"}
    assert len(names) == 18, f"expected 18 layers, found {len(names)}: {sorted(names)}"
    assert "active_learning" in names, "Layer 16 is missing from layers()"


def test_stack_cycle_runs_every_layer_without_failure():
    stack = CognitiveStack(_Cfg())
    result = stack.cycle("alpha beta gamma")
    assert result.failed == [], f"layers raised: {result.failed}"
    assert result.ran, "no layer ran at all"
    assert result.cognitive is not None


def test_stack_cognitive_state_comes_from_real_measurements():
    stack = CognitiveStack(_Cfg())
    for _ in range(12):
        r = stack.cycle("alpha beta gamma")
    st = r.cognitive
    assert 0.0 <= st.error <= 1.0 and 0.0 <= st.uncertainty <= 1.0
    # Read during the cycle, before this turn's own episode is stored — so it trails the final
    # occupancy by at most the handful of episodes the last cycle wrote.
    assert st.memory_pressure == pytest.approx(stack.episodic.pressure(), abs=0.02)
    # Likewise read before this turn's own gradient step, so it trails by one update.
    assert st.uncertainty == pytest.approx(stack.world_model.uncertainty(), rel=0.25)


def test_a_disabled_layer_is_absent_not_zero():
    class Off(_Cfg):
        l09_enabled = False
    stack = CognitiveStack(Off())
    assert stack.curiosity is None
    assert "curiosity" not in stack.layers()
    assert "curiosity" not in stack.stats()


def test_stack_contains_a_failing_layer():
    """One layer raising must not kill the cycle — that is what makes this modular."""
    stack = CognitiveStack(_Cfg())

    def boom(*a, **k):
        raise RuntimeError("deliberate")
    stack.semantic.absorb = boom
    result = stack.cycle("alpha beta gamma")
    assert "semantic" in result.failed
    assert result.ran, "one failing layer stopped the whole cycle"


def test_stack_tests_the_hypotheses_it_forms():
    """L6 once accumulated claims nothing ever evaluated: tested stayed 0 forever."""
    stack = CognitiveStack(_Cfg())
    for tok in ["alpha one", "beta two", "gamma three"] * 20:
        stack.cycle(tok)
    assert stack.reasoning.stats()["held"] > 0
    assert stack.reasoning.stats()["tested"] > 0, "formed hypotheses but never tested any"


def test_stack_round_trips_through_a_sidecar():
    stack = CognitiveStack(_Cfg())
    for _ in range(6):
        stack.cycle("alpha beta")
    fresh = CognitiveStack(_Cfg())
    fresh.load_dict(stack.to_dict())
    assert fresh.cycles == stack.cycles


def test_stack_reports_none_for_learning_before_it_can_say():
    assert CognitiveStack(_Cfg()).is_learning() is None
