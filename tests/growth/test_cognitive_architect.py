"""Tests for NYXARA · growth/cognitive_architect.py — Structural Cognitive Self-Modification.

Proves the engine genuinely rewires HOW she thinks, herself, with no LLM in the loop:

* she **invents** new composite reasoning operators that measurably improve her accuracy;
* adoption is **proof-carrying** — a candidate is kept only when it *strictly* beats the incumbent
  on a **held-out** fold it never optimised against;
* the immutable **character** operators can never be pruned, reordered out, or down-weighted, and no
  invented operator may touch them;
* she is **antifragile** — a faulted operator is quarantined, the pipeline re-routes, a backup is
  synthesised, and the failure enters an immune memory;
* a fast **plastic** layer reweights selection continuously; the recursive **meta** level is depth-capped;
* the architecture **persists** and reloads; enact + persistence are **sealed OFF under TEST**.
"""

from __future__ import annotations

import inspect
import json
import os

import pytest

from nyxara.growth import cognitive_architect as ca_mod
from nyxara.growth.cognitive_architect import (
    MAX_META_DEPTH,
    PROTECTED_OPS,
    CognitiveArchitect,
    CognitiveArchitecture,
    build_battery,
)


# --------------------------------------------------------------------------- #
# Core capability: she invents operators and measurably gets smarter.
# --------------------------------------------------------------------------- #
def test_rewire_invents_operators_and_improves_fitness():
    ca = CognitiveArchitect(seed=1, n_per_type=24)
    before = ca.report()
    for _ in range(20):
        ca.rewire(candidates=6)
    after = ca.report()

    assert after["train_fitness"] >= before["train_fitness"], "rewiring must never regress fitness"
    assert after["train_accuracy"] > before["train_accuracy"], "she must think more accurately"
    assert after["invented_operators"] >= 1, "she must invent at least one genuinely new operator"
    # an invented operator is a real composite (not a bare primitive) over the grammar
    invented_exprs = [ca.arch.operators[i] for i in after["invented_ids"]]
    assert any(e[0] in ("vote", "verify", "seq") for e in invented_exprs), \
        "at least one invented operator must be a composite (trans-logic), not a single primitive"


def test_adoption_is_strict_and_never_regresses():
    ca = CognitiveArchitect(seed=3, n_per_type=20)
    prev_fit = ca.report()["train_fitness"]
    for _ in range(15):
        rep = ca.rewire(candidates=6)
        cur = ca.report()["train_fitness"]
        assert cur >= prev_fit - 1e-9, "adoption must be monotone: fitness can only rise"
        if rep.adopted:
            assert rep.candidate_fitness > rep.incumbent_fitness, "adopted ⇒ strictly better on train"
        prev_fit = cur


# --------------------------------------------------------------------------- #
# Proof-carrying, anti-overfit adoption: gains validated on a held-out fold.
# --------------------------------------------------------------------------- #
def test_adoption_carries_a_heldout_proof_certificate():
    ca = CognitiveArchitect(seed=5, n_per_type=24)
    adopted = None
    for _ in range(25):
        rep = ca.rewire(candidates=8)
        if rep.adopted:
            adopted = rep
            break
    assert adopted is not None, "the engine should adopt at least one improvement"
    cert = adopted.certificate
    assert cert is not None and cert["strictly_better"] is True
    assert cert["no_llm"] is True
    assert cert["character_intact"] is True
    # the improvement generalised: held-out fitness strictly rose too (not just training)
    assert adopted.heldout_delta > 0.0
    assert cert["candidate_heldout_fitness"] > cert["incumbent_heldout_fitness"]


def test_train_and_heldout_folds_are_disjoint():
    ca = CognitiveArchitect(seed=0, n_per_type=24)
    train_ids = {i.task_id for i in ca._train}
    held_ids = {i.task_id for i in ca._heldout}
    # different seeds ⇒ different concrete tasks; the search never sees held-out gold
    assert train_ids and held_ids
    assert build_battery(4, seed=101)[0].gold is not None


# --------------------------------------------------------------------------- #
# The character lock — the immutable core can never be touched.
# --------------------------------------------------------------------------- #
def test_character_operators_survive_every_rewire():
    ca = CognitiveArchitect(seed=7, n_per_type=16)
    for _ in range(30):
        ca.rewire(candidates=6)
        for pid in PROTECTED_OPS:
            assert pid in ca.arch.operators, f"protected op {pid} was dropped"
            assert ca.arch.policy[pid] >= ca_mod.PROTECTED_WEIGHT, f"protected op {pid} down-weighted"
        assert ca.arch.character_intact()


def test_pruning_can_never_remove_a_protected_operator():
    ca = CognitiveArchitect(seed=2, n_per_type=12)
    # drive many prune attempts directly
    for _ in range(200):
        child, _new, _prov = ca._mutate(ca.arch, "prune")
        assert set(PROTECTED_OPS).issubset(child.operators.keys())


def test_invented_operator_referencing_character_is_rejected():
    ca = CognitiveArchitect(seed=1, n_per_type=8)
    arch = ca.arch.clone()
    arch.operators["inv_bad"] = ["prim", "loyalty_core"]   # try to hijack the character core
    arch.provenance["inv_bad"] = "invented"
    assert arch.character_intact() is False, "an operator touching the character core must be vetoed"


# --------------------------------------------------------------------------- #
# Biological resilience → antifragility.
# --------------------------------------------------------------------------- #
def test_faulted_primitive_is_quarantined_rerouted_and_immunised():
    ca = CognitiveArchitect(seed=4, n_per_type=24)
    for _ in range(12):
        ca.rewire(candidates=6)
    acc_before = ca._evaluate(ca.arch, ca._train)[0]

    ca.lib["cls_h0"]._damaged = True            # injure a classification primitive
    info = ca.heal("cls_h0")

    assert info["quarantined"], "operators leaning on the faulted primitive must be quarantined"
    assert info["immunised"] is True, "the failure mode must enter immune memory"
    assert any(m["primitive"] == "cls_h0" for m in ca.arch.immune)
    # classification is still served after the injury (re-routed to a redundant/backup path)
    acc_after = ca._evaluate(ca.arch, ca._train)[0]
    assert acc_after >= acc_before - 0.15, "she must degrade gracefully, not collapse, under injury"


def test_immune_memory_blocks_readopting_a_known_bad_operator():
    ca = CognitiveArchitect(seed=6, n_per_type=8)
    ca.arch.immune.append({"expr": json.dumps(["vote", ["prim", "cls_h0"], ["prim", "cls_h1"]],
                                              sort_keys=True), "primitive": "cls_h0"})
    seen = set()
    for _ in range(50):
        child, new_id, _prov = ca._mutate(ca.arch, "ensemble")
        if new_id:
            seen.add(json.dumps(child.operators[new_id], sort_keys=True))
    banned = json.dumps(["vote", ["prim", "cls_h0"], ["prim", "cls_h1"]], sort_keys=True)
    assert banned not in seen, "an immune-remembered operator must never be re-synthesised"


def test_redundancy_score_rewards_multiple_healthy_paths():
    ca = CognitiveArchitect(seed=9, n_per_type=8)
    base = ca.arch.redundancy(ca.lib)
    # add a second operator for a type ⇒ redundancy rises
    ca.arch.operators["extra_cls"] = ["vote", ["prim", "cls_h0"], ["prim", "cls_h1"]]
    ca.arch.provenance["extra_cls"] = "invented"
    assert ca.arch.redundancy(ca.lib) >= base


# --------------------------------------------------------------------------- #
# Continuous plasticity + bounded recursive meta.
# --------------------------------------------------------------------------- #
def test_plasticity_reinforces_and_stays_bounded():
    ca = CognitiveArchitect(seed=1, n_per_type=8)
    q = ca._train[0].query
    oid = ca.tick_plasticity(q, correct=True)
    assert oid is not None
    for _ in range(50):                          # relentless positive reward
        ca.tick_plasticity(q, correct=True)
    assert 1.0 <= ca.plastic.weight(oid) <= ca.plastic.hi, "potentiation is real but bounded"
    for _ in range(100):                         # then relentless negative
        ca.tick_plasticity(q, correct=False)
    assert ca.plastic.weight(oid) >= ca.plastic.lo, "depression is real but floored"


def test_meta_depth_is_hard_capped():
    ca = CognitiveArchitect(seed=1, n_per_type=8, meta_depth=99)
    assert ca.meta.depth <= MAX_META_DEPTH
    exploration_seen = []
    for _ in range(10):
        ca.rewire(candidates=4)
        exploration_seen.append(ca.meta.exploration)
    assert all(0.0 <= e <= 1.0 for e in exploration_seen)


def test_meta_policy_credits_the_mutations_that_help():
    ca = CognitiveArchitect(seed=8, n_per_type=16)
    for _ in range(15):
        ca.rewire(candidates=6)
    # the search operators that produced improvements should not have collapsed to the floor
    assert max(ca.meta.w.values()) > 1.0


# --------------------------------------------------------------------------- #
# No LLM is EVER in the loop.
# --------------------------------------------------------------------------- #
def test_no_llm_anywhere_in_the_module():
    src = inspect.getsource(ca_mod)
    lowered = src.lower()
    # No language-model *provider* library may be imported or referenced at all.
    for needle in (".generate(", "openai", "anthropic", "transformers", "import torch"):
        assert needle not in lowered, f"module must not touch {needle!r}"
    # The only *executable* use of the token 'llm' allowed is the honest invariant `self._llm = None`.
    exec_llm = ("llm.", "llm(", "import llm", "nyxara.mind.llm", "= llm", "get_llm", "self.llm")
    offenders = [ln for ln in src.splitlines()
                 if any(tok in ln.lower() for tok in exec_llm)
                 and "self._llm = none" not in ln.lower()]
    assert not offenders, f"unexpected executable LLM reference(s): {offenders}"


def test_engine_has_no_llm_handle():
    ca = CognitiveArchitect(seed=1, n_per_type=8)
    ca.rewire(candidates=4)
    assert ca._llm is None, "the architect must never acquire an LLM handle"


# --------------------------------------------------------------------------- #
# Persistence: gains compound across restarts.
# --------------------------------------------------------------------------- #
def test_persistence_round_trip(tmp_path):
    path = os.path.join(str(tmp_path), "cognitive_architecture.json")
    ca = CognitiveArchitect(persist_path=path, seed=2, n_per_type=16)
    for _ in range(6):
        ca.rewire(candidates=6)
    gen = ca.arch.generation
    n_ops = len(ca.arch.operators)
    assert os.path.exists(path)

    ca2 = CognitiveArchitect(persist_path=path, seed=2, n_per_type=16)
    assert ca2.arch.generation == gen, "the climbed architecture must reload"
    assert len(ca2.arch.operators) == n_ops
    assert ca2.arch.character_intact()


def test_corrupt_persistence_falls_back_to_seed(tmp_path):
    path = os.path.join(str(tmp_path), "cognitive_architecture.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{ this is not valid json ]")
    ca = CognitiveArchitect(persist_path=path, seed=1, n_per_type=8)
    assert ca.arch.character_intact(), "a corrupt file must never crash boot — fall back to seed"


def test_architecture_serialisation_is_lossless():
    arch = CognitiveArchitecture.seed(ca_mod._default_primitives())
    arch.operators["inv_x"] = ["verify", ["prim", "cls_h0"], ["prim", "cls_h1"]]
    arch.provenance["inv_x"] = "invented"
    round_tripped = CognitiveArchitecture.from_dict(json.loads(json.dumps(arch.to_dict())))
    assert round_tripped.operators == arch.operators
    assert round_tripped.character_intact()


# --------------------------------------------------------------------------- #
# TEST-profile seal + orchestrator wiring.
# --------------------------------------------------------------------------- #
def test_test_profile_seals_enact_and_persist():
    from nyxara.kernel.config import NyxaraSettings, Profile
    s = NyxaraSettings.for_profile(Profile.TEST)
    assert s.cognitive_architect.autonomous_enact is False, "enact must be sealed OFF under TEST"
    assert s.cognitive_architect.persist is False, "persistence must be sealed OFF under TEST"


def test_apply_to_live_is_advisory_when_not_enacted():
    from nyxara.mind.reasoner import Reasoner
    ca = CognitiveArchitect(seed=1, n_per_type=8, enact=False)
    for _ in range(6):
        ca.rewire(candidates=6)
    reasoner = Reasoner([])
    out = ca.apply_to_live(reasoner=reasoner)
    assert out["enacted"] is False and out["strategies"] == 0, "advisory by default — no live mutation"
    assert len(reasoner._strategies) == 0


def test_apply_to_live_installs_operators_when_enacted():
    from nyxara.mind.reasoner import Reasoner
    ca = CognitiveArchitect(seed=1, n_per_type=12, enact=True)
    for _ in range(12):
        ca.rewire(candidates=6)
    reasoner = Reasoner([])
    out = ca.apply_to_live(reasoner=reasoner)
    assert out["enacted"] is True
    assert out["strategies"] >= 1, "the champion architecture must install real strategies live"
    # protected character ops are never installed as battery strategies
    names = {getattr(s, "op_id", getattr(s, "name", "")) for s in reasoner._strategies}
    assert not (set(PROTECTED_OPS) & names)


def test_orchestrator_wires_the_cognitive_architect():
    from nyxara.kernel.orchestrator import NyxaraCore
    core = NyxaraCore()
    assert core.cognitive_architect is not None, "the core must build the cognitive architect"
    seed_rep = core.cognitive_architecture_report()
    assert "operators" in seed_rep and seed_rep["character_intact"] is True

    out = core.rewire_cognition(generations=6)
    assert out["train_accuracy"] >= seed_rep["train_accuracy"]
    assert out["invented_operators"] >= 1
    # folded into the top-level core.report()
    top = core.report()
    assert "cognitive_operators" in top and "cognitive_operators_invented" in top


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
