"""Integration: the CAUSAL engine is wired into NyxaraCore and the heartbeat, safely.

These exercise the *wiring* NYXARA adds (engine build, the per-turn hot-path hook, the heartbeat
thermodynamic tick, and the config flag) without driving the full heavy reasoning pipeline — that
keeps them fast and deterministic while still proving the integration points.
"""

from __future__ import annotations

import pytest

from nyxara.kernel.config import NyxaraSettings


def _core():
    from nyxara.kernel.orchestrator import NyxaraCore
    return NyxaraCore()


def test_engine_is_built_and_seeded():
    core = _core()
    assert core.causal_engine is not None
    # seeded standing concepts mean a matching query resonates immediately (no cold start)
    hit = core.causal_engine.field.best("help me fix a python code bug in a function")
    assert hit is not None and hit.label == "code"


def test_hot_path_hook_resonates_and_records():
    core = _core()
    thoughts: list = []
    core._causal_engage("reasoning logic and inference to solve a problem", thoughts)
    et = core._last_causal
    assert et is not None
    assert et.resonance                                   # a seeded concept resonated
    assert thoughts                                       # an advisory thought was recorded


def test_hot_path_hook_phase_shifts_on_a_novel_gap():
    core = _core()
    thoughts: list = []
    core._causal_engage("qwibble frobnicate the zorbtar quuxatron gribble", thoughts)
    et = core._last_causal
    assert et.gap is True
    assert et.phase_shift is not None and et.phase_shift["installed"] is True


def test_hot_path_hook_never_raises_even_if_engine_is_broken():
    core = _core()

    class _Boom:
        def turn(self, *a, **k):
            raise RuntimeError("engine exploded")
    core.causal_engine = _Boom()
    # the helper must swallow any engine failure — a turn is never broken by the engine
    core._causal_engage("anything at all", [])


def test_heartbeat_beat_ticks_the_thermodynamic_monitor():
    core = _core()
    if core.heartbeat is None:
        pytest.skip("heartbeat unavailable in this build")
    before = core.causal_engine.thermo.status()["beats"]
    core.heartbeat.beat(dt=1.0)
    after = core.causal_engine.thermo.status()["beats"]
    assert after == before + 1


def test_the_knot_gate_actually_runs_on_a_real_turn():
    """CAUSAL·2 was built, tested and then never called: the hot path passed no claims at all,
    so a Knot Mutation Failure could not happen in production no matter what was said."""
    core = _core()
    core._causal_engage("Heavy rain causes flooding in the valley.", [])
    first = core._last_causal
    assert first.committed == 1                 # the claim was mined and tied in

    thoughts: list = []
    core._causal_engage("Heavy rain prevents flooding in the valley.", thoughts)
    second = core._last_causal
    assert second.consistent is False           # contradicts the previous turn
    assert second.abstain is True
    assert thoughts                             # and she says so in the workspace


def test_turns_without_a_causal_claim_stay_clean():
    core = _core()
    assert core._causal_claims("hello, how are you today") == []
    core._causal_engage("hello, how are you today", [])
    assert core._last_causal.consistent is True
    assert core._last_causal.knot_check is None


def test_the_turn_gate_can_be_switched_off(monkeypatch):
    from nyxara.kernel import config as cfg

    settings = NyxaraSettings()
    settings.causal_engine.knot_gate_on_turns = False
    monkeypatch.setattr(cfg, "get_settings", lambda: settings)

    core = _core()
    assert core._causal_claims("Heavy rain causes flooding.") == []


def test_the_gate_verdict_reaches_the_turn_gates():
    core = _core()
    gates: dict = {}
    core._causal_engage("Heavy rain causes flooding in the valley.", [], gates)
    assert gates["knot"] == "tied"

    gates = {}
    core._causal_engage("Heavy rain prevents flooding in the valley.", [], gates)
    assert gates["knot"] == "mutation-failure"


def test_a_turn_with_no_claims_writes_no_knot_gate():
    core = _core()
    gates: dict = {}
    core._causal_engage("hello there", [], gates)
    assert "knot" not in gates


def test_a_self_correction_is_not_treated_as_a_contradiction():
    """"X causes Y" then "actually X prevents Y" is the Master updating, not a hallucination."""
    core = _core()
    core._causal_engage("Caffeine improves my focus.", [])
    assert core._last_causal.committed == 1

    thoughts: list = []
    core._causal_engage("Actually caffeine reduces my focus.", thoughts, {})
    et = core._last_causal
    assert et.consistent is True and et.abstain is False
    assert et.knot_check is None            # the retraction never reached the gate

    candidate = _respond_candidate(confidence=0.8, belief=0.8)
    assert core._apply_knot_caution(candidate).confidence == 0.8   # and nothing was damped


def test_a_genuine_contradiction_is_still_caught_after_the_correction_rule():
    core = _core()
    core._causal_engage("Caffeine improves my focus.", [])
    core._causal_engage("Caffeine reduces my focus.", [])
    assert core._last_causal.consistent is False
    assert core._last_causal.abstain is True


def test_reading_reports_retractions_separately():
    core = _core()
    report = core.learn_from_text("Heavy rain causes flooding. "
                                  "Actually heavy rain prevents flooding.")
    claims = report["causal_claims"]
    assert claims["claims"] == 1 and claims["contradicted"] == 0
    assert claims["corrections"] == 1


def test_knot_gate_abstains_actually_damps_the_answer():
    """The flag is documented as "abstains more often — FEWER answers" and produced no
    behaviour at all. A measured contradiction now damps confidence so HonestyGuard hedges."""
    core = _core()
    core._causal_engage("Heavy rain causes flooding in the valley.", [])
    core._causal_engage("Heavy rain prevents flooding in the valley.", [])
    assert core._last_causal.abstain is True

    candidate = _respond_candidate(confidence=0.8, belief=0.8)
    damped = core._apply_knot_caution(candidate)
    assert damped.confidence < 0.4 and damped.belief <= 0.4


def test_knot_caution_leaves_a_consistent_turn_alone():
    core = _core()
    core._causal_engage("Heavy rain causes flooding in the valley.", [])
    assert core._last_causal.abstain is False

    candidate = _respond_candidate(confidence=0.8, belief=0.8)
    assert core._apply_knot_caution(candidate).confidence == 0.8


def test_knot_caution_never_touches_an_action_candidate():
    core = _core()
    core._causal_engage("Heavy rain causes flooding in the valley.", [])
    core._causal_engage("Heavy rain prevents flooding in the valley.", [])

    candidate = _respond_candidate(confidence=0.8, belief=0.8)
    candidate.kind = "act"
    assert core._apply_knot_caution(candidate).confidence == 0.8


def _respond_candidate(*, confidence: float, belief: float):
    from nyxara.kernel.orchestrator import Candidate
    return Candidate(text="an answer", kind="respond", confidence=confidence, belief=belief)


def test_reading_a_passage_feeds_the_shared_lattice():
    """growth/text_causal built a throwaway lattice per call, so nothing she read could ever
    contradict anything else she had read — or anything said in the conversation."""
    core = _core()
    first = core.learn_from_text("Heavy rain causes flooding.")
    assert first["causal_claims"]["claims"] == 1
    assert core.causal_engine.lattice.status()["knots"] >= 1

    # a second passage that contradicts the first is refused, not absorbed
    second = core.learn_from_text("Heavy rain prevents flooding.")
    assert second["causal_claims"]["contradicted"] == 1
    assert second["causal_claims"]["claims"] == 0


def test_a_document_and_the_conversation_share_one_lattice():
    core = _core()
    core.learn_from_text("Heavy rain causes flooding.")
    core._causal_engage("Heavy rain prevents flooding.", [])
    assert core._last_causal.consistent is False
    assert core._last_causal.abstain is True


def test_reading_still_works_with_the_engine_off(monkeypatch):
    from nyxara.kernel import config as cfg

    settings = NyxaraSettings()
    settings.causal_engine.enabled = False
    monkeypatch.setattr(cfg, "get_settings", lambda: settings)

    core = _core()
    assert core.causal_engine is None
    report = core.learn_from_text("Heavy rain causes flooding.")
    assert report["causal_claims"]["claims"] == 1      # a local lattice, still better than none


def test_report_surfaces_the_engine_and_its_last_turn():
    """``_last_causal`` was written every turn and read by nothing — the comment beside it
    claimed report() surfaced it, and report() had never mentioned the engine at all."""
    core = _core()
    core._causal_engage("Heavy rain causes flooding in the valley.", [])
    core._causal_engage("Heavy rain prevents flooding in the valley.", [])

    causal = core.report().get("causal")
    assert causal is not None
    assert causal["knot_gate_abstains"] is True
    assert causal["lattice"]["knots"] == 1
    assert causal["last_turn"]["consistent"] is False
    assert causal["last_turn"]["abstain"] is True


def test_report_omits_causal_when_the_engine_is_off(monkeypatch):
    from nyxara.kernel import config as cfg

    settings = NyxaraSettings()
    settings.causal_engine.enabled = False
    monkeypatch.setattr(cfg, "get_settings", lambda: settings)

    core = _core()
    assert "causal" not in core.report()


def test_config_flag_disables_the_engine(monkeypatch):
    from nyxara.kernel import config as cfg

    settings = NyxaraSettings()
    settings.causal_engine.enabled = False
    monkeypatch.setattr(cfg, "get_settings", lambda: settings)

    core = _core()
    assert core.causal_engine is None
    # the hot-path hook is a no-op when the engine is off — nothing raised, nothing recorded
    core._causal_engage("hello", [])
