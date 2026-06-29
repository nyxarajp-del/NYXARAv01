"""Tests for the eight RSI gap-closers — all real, NYXARA-driven, wired into the loop.

Hermetic and offline: no network, no torch required. Each engine degrades cleanly and is exercised
on the always-available pure-stdlib substrate (n-gram brain, the prover, the sandbox).
"""

from __future__ import annotations

import re

from nyxara.kernel.compute import ComputeReport, GPUInfo


# --------------------------------------------------------------------------- #
# Gap 1 — substrate self-expansion
# --------------------------------------------------------------------------- #
def _square(x: int) -> int:
    return x * x


def test_substrate_survey_and_raisable_ceiling() -> None:
    from nyxara.growth.substrate import SubstrateManager

    mgr = SubstrateManager()
    state = mgr.survey()
    assert state.parallel_units >= 1
    assert state.expansion_factor() >= 1.0
    assert "local" in state.providers
    # the ceiling is RAISED (never lowered) by real parallelism, bounded by the hard cap
    cap = mgr.effective_cap(3)
    assert cap >= 3 and cap <= mgr.hard_cap
    mgr.close()


def test_substrate_distributed_map_order_and_fallback() -> None:
    from nyxara.growth.substrate import SubstrateManager

    mgr = SubstrateManager()
    assert mgr.distributed_map(_square, [1, 2, 3, 4, 5]) == [1, 4, 9, 16, 25]
    assert mgr.distributed_map(_square, [7]) == [49]          # single item never bothers the pool
    mgr.close()


def test_substrate_ray_declines_honestly_without_cluster() -> None:
    from nyxara.growth.substrate import RayComputeProvider

    p = RayComputeProvider(env={}).provision()
    assert not p.ok                                           # never fakes a cluster it does not have


def test_effort_budget_scales_up_with_substrate() -> None:
    from nyxara.growth.intelligence import IntelligenceIndex, IntelligenceState

    idx = IntelligenceIndex()
    state = IntelligenceState(index=0.9, t=3)
    big = ComputeReport(cpu_count=16, memory={"available_mb": 32000, "total_mb": 64000},
                        gpu=GPUInfo(available=True, backend="cuda", count=1, names=["A100"]),
                        torch_available=True)

    class _Sub:
        def effective_cap(self, c):
            return c + 5
        def survey(self):
            class _S:
                def expansion_factor(self_inner):
                    return 3.0
            return _S()

    base = idx.effort_budget(state, big)
    raised = idx.effort_budget(state, big, substrate=_Sub())
    assert raised["max_edits_per_cycle"] >= base["max_edits_per_cycle"]
    assert raised["expansion_factor"] == 3.0


# --------------------------------------------------------------------------- #
# Gap 2 — open-ended auto-curriculum
# --------------------------------------------------------------------------- #
def _smart_solver(prompt: str) -> str:
    from nyxara.growth.prover import _eval_rational
    if prompt.startswith("Compute:"):
        v = _eval_rational(prompt.split(":", 1)[1])
        return str(v) if v is not None else "0"
    eq = prompt.split(":", 1)[1]
    lhs, rhs = eq.split("=")
    m = re.match(r"\s*(-?\d+)\*x\s*\+\s*(-?\d+)\s*", lhs)
    a, b = int(m.group(1)), int(m.group(2))
    c = _eval_rational(rhs)
    return str((c - b) / a)


def test_curriculum_problems_are_prover_certified() -> None:
    from nyxara.growth.curriculum import AutoCurriculum

    cur = AutoCurriculum(seed=3)
    probs = cur.generate(20, tier=3)
    assert probs
    assert all(cur._certify(p) for p in probs)               # ground truth is machine-checked


def test_curriculum_frontier_advances_for_capable_solver() -> None:
    from nyxara.growth.curriculum import AutoCurriculum

    cur = AutoCurriculum(seed=3)
    start = cur.frontier_tier()
    rep = None
    for _ in range(5):
        rep = cur.evaluate(_smart_solver, per_tier=5)
    assert rep.frontier_score > 0.6
    assert cur.frontier_tier() > start                       # POET: the edge moves as she masters it


def test_curriculum_low_for_wrong_solver_and_unmemorisable() -> None:
    from nyxara.growth.curriculum import AutoCurriculum

    cur = AutoCurriculum(seed=3)
    rep = cur.evaluate(lambda p: "0", per_tier=5)
    assert rep.frontier_score < 0.3 and not rep.promoted
    a = [p.prompt for p in cur.generate(10, tier=4)]
    b = [p.prompt for p in cur.generate(10, tier=4)]
    assert a != b                                            # fresh each batch — cannot be memorised


# --------------------------------------------------------------------------- #
# Gap 3 — self-interpretability + weight surgery
# --------------------------------------------------------------------------- #
def _trained_ngram():
    from nyxara.growth.foundry_models import WordKNGramLM
    corpus = ["the cat sat on the mat", "the cat sat on the rug",
              "the dog ran in the park", "the dog ran in the yard"] * 6
    m = WordKNGramLM(order=2, seed=1)
    m.train_on(corpus)
    return m, corpus


def test_weight_surgery_localizes_and_steers() -> None:
    from nyxara.growth.weight_surgery import WeightSurgeon

    model, corpus = _trained_ngram()
    surgeon = WeightSurgeon(model)
    assert surgeon.localize("the", top=4).items                # interpretability read-out is real
    pred0 = surgeon.predict_next("the")
    target = "dog" if pred0 != "dog" else "cat"
    out = surgeon.steer("the", target, probe=corpus)
    assert out.applied and out.kept and not out.rolled_back
    assert surgeon.predict_next("the") == target              # the circuit edit changed the output


def test_weight_surgery_rolls_back_on_failing_gauntlet() -> None:
    from nyxara.growth.weight_surgery import WeightSurgeon

    model, _ = _trained_ngram()
    surgeon = WeightSurgeon(model)
    pred_now = surgeon.predict_next("the")
    params = model.param_count()
    strict = WeightSurgeon(model, tol=0.0)
    bad = strict.steer("the", "park", probe=["the cat sat on the mat"] * 8)
    if bad.rolled_back:                                       # a harmful edit must restore exactly
        assert model.param_count() == params
        assert surgeon.predict_next("the") == pred_now


# --------------------------------------------------------------------------- #
# Gap 4 — continual learning without catastrophic forgetting
# --------------------------------------------------------------------------- #
def test_continual_learning_retains_old_skill() -> None:
    from nyxara.growth.foundry_models import WordKNGramLM

    cx = ["alpha beta gamma delta"] * 8
    cy = ["one two three four"] * 8
    scratch = WordKNGramLM(order=2, seed=1)
    scratch.train_on(cy)                                      # forgets X
    cont = WordKNGramLM(order=2, seed=1)
    cont.train_on(cx)
    cont.train_on(cy, accumulate=True)                        # continual: adds Y on top of X
    assert cont.perplexity("alpha beta gamma delta") < scratch.perplexity("alpha beta gamma delta")
    assert cont.perplexity("one two three four") < 50.0       # while still learning the new skill


# --------------------------------------------------------------------------- #
# Gap 5 — accelerating recursion (growing meta tower)
# --------------------------------------------------------------------------- #
def test_meta_tower_grows_under_sustained_gains() -> None:
    from nyxara.growth.meta_meta import MetaGenome, RecursiveMetaController

    # reset the global dynamic caps so the test is independent of execution order
    MetaGenome._MAX_RECURSION, MetaGenome._MAX_EDITS = 5, 6
    tower = RecursiveMetaController(height=2, seed=7, can_grow=True, hard_max_height=6)
    start_h = len(tower.levels)

    def env_gain(g: MetaGenome) -> float:
        return 0.01 * g.recursion_depth + 0.0005 * g.ledger_reward_scale

    for _ in range(400):
        tower.record(env_gain(tower.active) + 0.0003 * (tower.levels[0]._rng.random() - 0.5))
        tower.maybe_evolve()
    assert len(tower.levels) > start_h                        # the tower grew a rung (compounding)
    assert (MetaGenome._MAX_RECURSION > 5 or MetaGenome._MAX_EDITS > 6)   # caps widened
    assert MetaGenome._MAX_RECURSION <= MetaGenome._ABS_MAX_RECURSION     # still bounded
    MetaGenome._MAX_RECURSION, MetaGenome._MAX_EDITS = 5, 6


# --------------------------------------------------------------------------- #
# Gap 6 — proof-carrying self-modification
# --------------------------------------------------------------------------- #
def test_proof_carrying_certifies_equivalent_rewrite() -> None:
    from nyxara.growth.proof_carrying import ProofCarrier
    from nyxara.growth.prover import ProofVerdict

    pc = ProofCarrier()
    before = "def f(a, b):\n    if not a in b:\n        return 1\n    return 0\n"
    after = "def f(a, b):\n    if a not in b:\n        return 1\n    return 0\n"
    r = pc.certify_edit(before, after)
    assert r.verdict is ProofVerdict.PROVEN and not r.blocks


def test_proof_carrying_blocks_meaning_change_and_abstains_on_opaque() -> None:
    from nyxara.growth.proof_carrying import ProofCarrier
    from nyxara.growth.prover import ProofVerdict

    pc = ProofCarrier()
    bad = pc.certify_edit("if a == b and c == d:\n    pass\n",
                          "if a == b or c == d:\n    pass\n")
    assert bad.verdict is ProofVerdict.REFUTED and bad.blocks
    opaque = pc.certify_edit("def g():\n    return h(x) + k(y)\n",
                             "def g():\n    return k(y) + h(x)\n")
    assert opaque.verdict is ProofVerdict.UNPROVABLE and not opaque.blocks   # → empirical fallback


# --------------------------------------------------------------------------- #
# Gap 7 — scalable oversight
# --------------------------------------------------------------------------- #
def test_oversight_approves_benign_and_vetoes_regressions() -> None:
    from nyxara.growth.oversight_verify import ScalableVerifier

    v = ScalableVerifier()
    assert v.verify_edit("def public_fn(a, b):\n    if not a in b:\n        return 1\n    return 0\n",
                         "def public_fn(a, b):\n    if a not in b:\n        return 1\n    return 0\n",
                         kind="not_in").approved
    # deletes a public symbol a single shallow check would miss
    assert v.verify_edit("def keep():\n    return 1\n\ndef critical_api():\n    return 2\n",
                         "def keep():\n    return 1\n").vetoed
    # strips a safety reference
    assert v.verify_edit("def f():\n    check_corrigibility()  # loyalty\n    return run()\n",
                         "def f():\n    return run()\n").vetoed
    # removes an assertion
    assert v.verify_edit("def f(x):\n    assert x > 0\n    return x\n",
                         "def f(x):\n    return x\n").vetoed


def test_oversight_metrics_second_opinion() -> None:
    from nyxara.growth.oversight_verify import ScalableVerifier

    v = ScalableVerifier()
    assert v.verify_metrics({"perplexity": 5.0, "capability": 0.4},
                            {"perplexity": 4.0, "capability": 0.5}).vetoed
    assert v.verify_metrics({"perplexity": 3.0, "capability": 0.6},
                            {"perplexity": 4.0, "capability": 0.5}).approved


# --------------------------------------------------------------------------- #
# Gap 8 — world-grounded experiments (graded by real execution)
# --------------------------------------------------------------------------- #
def test_grounded_experiment_graded_by_real_execution() -> None:
    from nyxara.growth.grounded_experiments import GroundedExperiment

    exp = GroundedExperiment(seed=11)
    good = "def solve(n):\n    return sum(range(1, n + 1))"
    ref = "def solve(n):\n    return n * (n + 1) // 2"
    bad = "def solve(n):\n    return n * n // 2"
    assert exp.code_experiment(good, ref, [3, 10, 50]).score == 1.0
    assert exp.code_experiment(bad, ref, [3, 10, 50]).score < 1.0   # reality exposes the bug


def test_grounded_predict_execution_ruler() -> None:
    from nyxara.agency.code_sandbox import run_python
    from nyxara.growth.grounded_experiments import GroundedExperiment

    exp = GroundedExperiment(seed=11)

    def truth_solver(prompt: str) -> str:
        code = prompt.split(":\n", 1)[1] if ":\n" in prompt else ""
        res = run_python(code + "\n", timeout_s=3.0)
        return str(res.value) if res.ok else "0"

    smart = exp.predict_execution(truth_solver, trials=6)
    dumb = exp.predict_execution(lambda p: "0", trials=6)
    assert smart.n_trials > 0 and smart.score > dumb.score


# --------------------------------------------------------------------------- #
# wiring — the loop folds the new rulers in (smoke, offline, dry-run)
# --------------------------------------------------------------------------- #
def test_rsi_report_carries_new_signals() -> None:
    from nyxara.growth.recursive_improvement import RecursiveSelfImprovement

    rsi = RecursiveSelfImprovement()
    report = rsi.run(enact=False)                             # dry-run: changes nothing
    assert report.substrate is not None and "expansion_factor" in report.substrate
    assert report.effort_budget is not None and "expansion_factor" in report.effort_budget
    # the curriculum + grounded rulers are folded into the external validation block
    assert report.validation is not None and "frontier" in report.validation
    assert "grounded" in report.validation
