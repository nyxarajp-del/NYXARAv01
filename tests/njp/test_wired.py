"""NJP is wired into the running system — and the three old brains are genuinely gone."""

from __future__ import annotations

import importlib

import pytest

from nyxara.kernel.config import get_settings
from nyxara.kernel.orchestrator import NyxaraCore


@pytest.fixture(scope="module")
def core() -> NyxaraCore:
    return NyxaraCore()


@pytest.mark.parametrize("module", ["nyxara.nyx", "nyxara.nyx5", "nyxara.nyx001"])
def test_the_old_brains_are_gone(module: str):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_the_old_brain_handles_are_gone(core: NyxaraCore):
    for attr in ("nyx", "nyx5", "nyx001"):
        assert not hasattr(core, attr), f"core.{attr} should no longer exist"


def test_njp_is_built_and_is_the_reason_seat(core: NyxaraCore):
    assert core.njp is not None
    assert type(core.reasoner).__name__ == "NJPReasoner"


def test_the_settings_carry_njp_and_not_the_old_sections():
    settings = get_settings()
    assert settings.njp.enabled is True
    for gone in ("nyx", "nyx5", "nyx001"):
        assert not hasattr(settings, gone)


def test_self_editing_is_sealed_off_under_test():
    """A hermetic suite must never edit the live source tree."""
    assert get_settings().njp.evolve_enabled is False


def test_a_turn_through_the_kernel_grows_the_fabric(core: NyxaraCore):
    """The claim, end to end through the real orchestrator."""
    before = core.njp.fabric.n_synapses
    core.njp_think("gravity pulls the apple toward the ground")
    assert core.njp.fabric.n_synapses > before


def test_the_kernel_exposes_the_growth_surface(core: NyxaraCore):
    assert "fabric" in core.njp_stats()
    assert core.njp_expand() is not None
    assert isinstance(core.njp_ledger(), dict)
    assert isinstance(core.njp_tick(), dict)


def test_the_sidecar_round_trips_through_the_kernel(core: NyxaraCore, tmp_path):
    """njp.json is why tomorrow's brain is today's brain."""
    target = str(tmp_path / "memory.json")
    core.njp_think("the moon orbits the earth")
    grown = core.njp.fabric.n_synapses
    core._save_njp(target)

    fresh = NyxaraCore()
    assert fresh.njp.fabric.n_synapses != grown or grown == 0
    fresh._load_njp(target)
    assert fresh.njp.fabric.n_synapses == grown


def test_the_server_exposes_njp_and_no_old_routes():
    from nyxara.server.app import create_app
    paths = {r.path for r in create_app().routes if hasattr(r, "path")}
    assert "/v1/njp/status" in paths
    assert "/v1/njp/fabric" in paths
    assert not any(p.startswith("/v1/nyx/") or p.startswith("/v1/nyx001/") for p in paths)


def test_the_prove_route_refuses_what_is_not_a_formula():
    """The refusal is the feature: a proof runs on formally expressible claims and says so
    plainly about everything else, rather than returning a confident verdict on prose."""
    from fastapi.testclient import TestClient
    from nyxara.server.app import create_app
    client = TestClient(create_app())
    got = client.post("/v1/njp/prove", json={"stimulus": "gravity pulls the apple down"}).json()
    assert str(got.get("status", "")).lower() not in ("proved", "refuted")


def test_the_intent_route_reads_a_hinglish_command_with_its_ordering():
    from fastapi.testclient import TestClient
    from nyxara.server.app import create_app
    client = TestClient(create_app())
    got = client.post("/v1/njp/intent",
                      json={"stimulus": "mera code fix kar do lekin pehle test chala"}).json()
    assert got.get("kind") == "command"
    assert got.get("ordering")


def test_the_truth_route_reports_every_source():
    from fastapi.testclient import TestClient
    from nyxara.server.app import create_app
    client = TestClient(create_app())
    got = client.post("/v1/njp/truth", json={"stimulus": "an apple falls down"}).json()
    assert "verdict" in got and "evidence" in got
    assert got["verdict"] in ("established", "supported", "conjecture", "refuted", "abstained")


def test_the_brain_survives_concurrent_turns():
    """The kernel runs hypothesis framings in parallel against this one brain object.

    NJPReasoner.__call__ takes **kwargs, so `_reasoner_parallelizable` returns True and several
    threads call think() on the same instance at once. A fabric is one organism: without
    serialising, two turns rewire the adjacency dicts simultaneously — which corrupts them, and
    iterating one while another mutates raises outright.
    """
    import threading

    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    errors = []

    def worker(n: int) -> None:
        try:
            for k in range(20):
                brain.think(f"thread {n} turn {k} gravity apple moon earth")
        except Exception as exc:  # noqa: BLE001 — the point of the test is that none escape
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:3]
    assert brain.turns == 160
    # every forward edge must have its matching reverse index, or a prune can resurrect a synapse
    orphans = [(pre, post) for pre, targets in brain.fabric.out.items()
               for post in targets if pre not in brain.fabric.inn.get(post, set())]
    assert not orphans


def test_a_disabled_foundry_is_wired_but_never_trained():
    """Wiring and training are separate decisions.

    AutoForge called foundry.self_improve() unconditionally, so idle maintenance ran real NumPy
    training even with foundry.enabled=false — which is how a hermetic run ended up training a
    network until pytest-timeout killed the whole process. The forge must still be *wired* (its
    wiring is asserted elsewhere and is worth inspecting), but a disabled one must not train.
    """
    from nyxara.growth.autoforge import AutoForge

    class _Foundry:
        """Fails loudly if it is ever asked to train."""

        def self_improve(self, **_kw):
            raise AssertionError("a disabled forge must never train")

    class _Flywheel:
        def count(self):
            return 10_000

        def __len__(self):
            return 10_000

    disabled = AutoForge(foundry=_Foundry(), flywheel=_Flywheel(),
                         min_examples=1, enabled=False)
    result = disabled.run_cycle()
    assert result.trained is False
    assert "disabled" in result.reason        # reported as a skip, not silently a quiet no-op


def test_an_injected_foundry_is_never_second_guessed_by_ambient_config():
    """A caller that hands in its own foundry has already made the decision.

    The first attempt at the gate read global settings inside run_cycle, which vetoed foundries
    that tests inject deliberately and broke two of autoforge's own tests. `enabled` defaults to
    True for exactly this reason.
    """
    from nyxara.growth.autoforge import AutoForge

    forge = AutoForge(foundry=object(), flywheel=None, min_examples=1)
    assert forge.enabled is True


# --------------------------------------------------------------------------- #
# NJP V.02 — the brain is no longer an island
# --------------------------------------------------------------------------- #
def test_the_grounder_sees_the_kernels_knowledge_graph(core: NyxaraCore):
    """Without this everything she grounds lands in a private store the system cannot see."""
    assert core.njp.grounder is not None
    assert core.njp.grounder.graph is core.knowledge_graph


def test_the_concept_ladder_is_built(core: NyxaraCore):
    assert core.njp.ladder is not None


def test_a_grounded_fact_reaches_the_shared_graph(core: NyxaraCore):
    before = len(core.knowledge_graph)
    core.njp_think("Ravi works at Infosys")
    assert len(core.knowledge_graph) > before


def test_the_brain_answers_through_the_kernel(core: NyxaraCore):
    core.njp_think("my name is Jay")
    assert "Jay" in (core.njp_think("what is my name").answer or "")


def test_the_new_organs_have_config_gates():
    settings = get_settings()
    assert settings.njp.grounding_enabled is True
    assert settings.njp.concepts_enabled is True


def test_an_organ_that_is_off_is_absent_not_zeroed():
    """The package's standing rule: a disabled organ must not report fabricated zeros."""
    from nyxara.njp.brain import NJPBrain

    class _Off:
        grounding_enabled = False
        concepts_enabled = False

    brain = NJPBrain(_Off())
    assert brain.grounder is None and brain.ladder is None
    assert "grounding" not in brain.stats()


# --------------------------------------------------------------------------- #
# Cadences that had no knob
# --------------------------------------------------------------------------- #
def test_every_turn_cadence_is_reachable_from_config():
    """Three of the six took the constructor's defaults and could not be reached.

    `attack_every` is the one that mattered: a 1,352-pair study run built 1,750 beliefs and, at
    one attack per twenty turns with `limit=1`, challenged roughly 67 of them. The adversary was
    not missing — it was rationed, and there was no knob to un-ration it.
    """
    from nyxara.njp.brain import NJPBrain

    class Cfg:
        njp = None
        attack_every_turns = 3
        assess_every_turns = 5
        ledger_every_turns = 7

    tuned = NJPBrain(Cfg())
    assert (tuned.loop.attack_every, tuned.loop.assess_every, tuned.loop.ledger_every) == (3, 5, 7)
    default = NJPBrain()
    assert (default.loop.attack_every, default.loop.assess_every,
            default.loop.ledger_every) == (20, 24, 32)


def test_the_pulse_queue_has_no_feeder_and_says_so():
    """`submit` claimed the brain called it every think. It has no callers, and must not.

    `NJPBrain._expand` grows the fabric synchronously inside the turn so that "after every
    conversation" means the synapse count differs when the turn returns; its own comment records
    that routing turns through the pulse as well would double-count every one. So the queue is
    for out-of-turn experience, nothing currently produces any, and the expand counters are
    structurally zero rather than broken.
    """
    from nyxara.njp.pulse import PulseEngine

    assert "out-of-turn" in (PulseEngine.submit.__doc__ or "")
    assert "must not gain that one" in (PulseEngine.submit.__doc__ or "")


def test_the_always_on_daemon_beats_njps_pulse():
    """`brain.tick()` was reachable only from an HTTP poke, and this file never mentioned njp.

    The cost was not the pulse. It was everything downstream: `evolve.beat()` runs only from the
    pulse, `forge.MetamorphicCompiler` only from `evolve.accelerate()`, and
    `autopoiesis.AutopoieticRewriter` only from the forge — three organs, written and tested,
    behind a door nothing opened on its own.
    """
    from nyxara.kernel.autonomic import AutonomicLoop
    from nyxara.kernel.orchestrator import NyxaraCore

    core = NyxaraCore()
    assert core.njp is not None, "no njp brain; this test cannot say anything"
    before = core.njp.pulse.beats
    loop = AutonomicLoop(core=core)
    loop.run_for(3)
    assert loop.njp_pulses == 3
    assert core.njp.pulse.beats > before, "the daemon ran and the pulse never beat"


def test_the_pulse_hook_survives_a_core_without_njp():
    """Fail-soft like every other hook in that loop: a pulse is a capability, never fatal."""
    from nyxara.kernel.autonomic import AutonomicLoop

    class Bare:
        oversight = None

    loop = AutonomicLoop(core=Bare())
    loop._maybe_njp_pulse()
    assert loop.njp_pulses == 0


# --------------------------------------------------------------------------- #
# NJP V.06 — the cortex is CONNECTED, not merely built
# --------------------------------------------------------------------------- #
def test_the_cortex_organs_are_built_and_reachable(core: NyxaraCore):
    """The defect this guards against has happened in this repo before: organs that were built,
    reachable, and connected to nothing. A cortex nothing calls is exactly the arrangement that
    produced weights with no capability behind them."""
    assert core.njp.cortex is not None
    assert core.njp.router is not None
    assert core.njp.epistemic is not None
    for organ in ("cortex", "router", "epistemic"):
        assert organ in core.njp.stats(), f"{organ} must report itself"


def test_a_turn_actually_consults_the_router():
    """Not "the router exists" — that the turn path reaches it."""
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    thought = brain.think("why does aag cause garmi?")
    assert thought.routing is not None, "think() must reach the router"
    assert thought.routing.cortex_permitted is True
    assert brain.router.routed >= 1


def test_a_greeting_still_cannot_reach_the_cortex_through_the_turn_path():
    """The hole `relevance.py` closed must not reopen behind a bigger model."""
    from nyxara.njp.brain import NJPBrain
    from nyxara.njp.router import Seat

    thought = NJPBrain().think("hi NYXARA")
    assert thought.routing is not None
    assert thought.routing.cortex_permitted is False
    assert thought.routing.seats == (Seat.NJP,)
    assert thought.cortex is None, "an unconsulted cortex reports nothing, not an empty pass"


def test_an_unreachable_cortex_costs_the_turn_nothing():
    """No weights on this machine — and the turn must be indistinguishable from before."""
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    assert brain.cortex.available() is False
    thought = brain.think("aag se garmi hoti hai")
    assert thought.cortex is None and thought.arbitration is None
    assert thought.growth is not None, "the rest of the turn ran exactly as it always did"


def test_a_cortex_that_raises_never_takes_the_turn_down():
    from nyxara.njp.brain import NJPBrain

    class _Hostile:
        def available(self):
            raise RuntimeError("the runtime fell over mid-turn")

    brain = NJPBrain()
    brain.cortex = _Hostile()
    thought = brain.think("why does aag cause garmi?")
    assert thought.growth is not None
    assert thought.cortex is None


def test_the_cortex_organs_have_config_gates():
    from nyxara.njp.brain import NJPBrain

    class _Off:
        cortex_enabled = False
        router_enabled = False
        epistemic_enabled = False

    brain = NJPBrain(_Off())
    assert brain.cortex is None and brain.router is None and brain.epistemic is None
    stats = brain.stats()
    assert "cortex" not in stats and "router" not in stats and "epistemic" not in stats


def test_a_turn_with_no_router_does_not_consult_the_cortex_either():
    """The router is the gate. Without it, nothing downstream may run on its own initiative."""
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    brain.router = None
    thought = brain.think("why does aag cause garmi?")
    assert thought.routing is None and thought.cortex is None


def test_a_statement_runs_the_world_model_generator_and_a_question_runs_hypotheses():
    """Both halves of the cortex have to be reachable from the turn path.

    A question asks for rivals; a statement is where relations and causal claims actually live, and
    a statement permits no reasoning — so gating both halves on the same permission would have left
    the World Model Generator built and unreachable, which is the defect this repo has shipped
    before.
    """
    import json

    from nyxara.njp.brain import NJPBrain
    from nyxara.njp.cortex import Cortex
    from nyxara.njp.grounding import Provenance

    class _Reply:
        def __init__(self, text):
            self.text, self.model = text, "fake-cortex"

        def parse_json(self):
            return json.loads(self.text)

    class _FakeLLM:
        def __init__(self, *replies):
            self._replies = list(replies)
            self.asked = []
            self._providers = {}

        def provider_status(self):
            return {"qwythos": True}

        def complete_with(self, name, req):
            self.asked.append(req.messages[-1].content if req.messages else "")
            return _Reply(self._replies.pop(0) if self._replies else "{}")

    relations = json.dumps({"relations": [
        {"subject": "aag", "predicate": "causes", "object": "garmi", "confidence": 0.9}]})
    brain = NJPBrain()
    brain.cortex = Cortex(llm=_FakeLLM(relations))
    stated = brain.think("aag se garmi hoti hai")
    assert stated.routing.extraction_permitted is True
    assert stated.routing.cortex_permitted is False
    assert stated.cortex is not None and len(stated.cortex.triples) == 1
    assert "Extract only what the text supports" in brain.cortex._llm.asked[0]

    # and what it extracted went in as a proposal, beside the Master's own observed statement
    provenances = {t.provenance for bucket in brain.grounder.facts.values() for t in bucket}
    assert Provenance.PROPOSED in provenances
    assert Provenance.OBSERVED in provenances, "the Master's own words are not a hypothesis"

    hypotheses = json.dumps({"hypotheses": [
        {"claim": "a", "predicts": ["x"], "confidence": 0.4},
        {"claim": "b", "predicts": ["y"], "confidence": 0.3},
        {"claim": "c", "predicts": ["z"], "confidence": 0.2}]})
    asking = NJPBrain()
    asking.cortex = Cortex(llm=_FakeLLM(hypotheses))
    asked = asking.think("why does aag cause garmi?")
    assert asked.routing.cortex_permitted is True
    assert asked.cortex is not None and len(asked.cortex.hypotheses) == 3
    assert "competing" in asking.cortex._llm.asked[0].lower() or \
           "compete" in asking.cortex._llm.asked[0].lower()


def test_a_greeting_runs_neither_half_of_the_cortex():
    from nyxara.njp.brain import NJPBrain

    thought = NJPBrain().think("hi NYXARA")
    assert thought.routing.cortex_permitted is False
    assert thought.routing.extraction_permitted is False


def test_the_router_uses_the_brains_own_bandit_not_a_private_one():
    """It was reading a ``meta_learner`` attribute the self-model does not have, so the bandit
    branch was unreachable and the seat order silently fell through to the posteriors every time.

    One learner, shared with ``metareason``: two private bandits would split the evidence about
    what actually works across two records that never see each other.
    """
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    assert brain.router.learner is brain.meta
    assert brain.metareason.meta_learner is brain.meta

    routing = brain.router.route("why does aag cause garmi?")
    assert routing.basis == "bandit"
    # arms are namespaced, so the seat arm cannot collide with the strategy arms
    assert "seat:causal" in brain.meta.strategies
    assert not any(a.startswith("seat:") and a.startswith("strategy:")
                   for a in brain.meta.strategies)
