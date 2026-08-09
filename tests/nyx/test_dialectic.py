"""NYX V.02 · the answer is attacked before you see it.

`workspace` runs a competition for *salience* — the loudest coalition wins. Nothing in V.01 ever
attacked a conclusion after it won and before it shipped. These tests pin the three real
outcomes (survive, weaken, abstain), the fact that a refutation is treated as fatal, and the
honest limit: a finite objection set, not a bulletproof review.
"""

from __future__ import annotations

from nyxara.kernel.config import NyxConfig
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.dialectic import AdversarialSelfDialogue, Objection


class _Refuting:
    """A prover that refutes everything — the sharpest weapon the Skeptic has."""

    @staticmethod
    def prove(claim: str):
        return type("Cert", (), {"verdict": "refuted", "note": f"{claim} is false"})()


class _Silent:
    """A prover that never decides. Its silence must not become an objection."""

    @staticmethod
    def prove(claim: str):
        return type("Cert", (), {"verdict": "inexpressible", "note": "not a formula"})()


class _Brain:
    def __init__(self, prover=None) -> None:
        self.prover = prover


def _dialectic(**kw) -> AdversarialSelfDialogue:
    kw.setdefault("brain", _Brain())
    brain = kw.pop("brain")
    return AdversarialSelfDialogue(brain, **kw)


# --------------------------------------------------------------------------- #
# The three outcomes
# --------------------------------------------------------------------------- #
def test_a_modest_supported_claim_survives_intact():
    got = _dialectic().debate("the flywheel spins because the belt drives it",
                              confidence=0.6, evidence=["measured on the bench"])
    assert got.debated and got.survived and not got.abstained
    assert got.confidence_after == got.confidence_before


def test_an_overreaching_claim_survives_with_less_confidence():
    got = _dialectic().debate("all engines always fail", confidence=0.8)
    assert got.survived and not got.abstained
    assert got.confidence_after < got.confidence_before


def test_a_refuted_claim_makes_her_abstain():
    got = _dialectic(brain=_Brain(_Refuting())).debate("two plus two is five", confidence=0.9)
    assert got.abstained and not got.survived
    assert got.answer == "" and got.confidence_after == 0.0


def test_confidence_going_down_is_recorded_as_the_point_not_as_a_failure():
    note = _dialectic().debate("everything always works", confidence=0.9).note
    assert "which is the point" in note


# --------------------------------------------------------------------------- #
# The Skeptic's weapons
# --------------------------------------------------------------------------- #
def test_a_universal_with_nothing_behind_it_is_objected_to():
    got = _dialectic().debate("all cars are always fast", confidence=0.5)
    assert any(o.kind == "unsupported" for o in got.objections)


def test_evidence_disarms_the_universal_objection():
    got = _dialectic().debate("all cars are always fast", confidence=0.5,
                              evidence=["a hundred measured cars"])
    assert not any(o.kind == "unsupported" for o in got.objections)


def test_an_assertion_without_a_mechanism_is_objected_to():
    got = _dialectic().debate("the flywheel makes the engine run more smoothly overall")
    assert any(o.kind == "no-mechanism" for o in got.objections)


def test_a_mechanism_disarms_that_objection():
    got = _dialectic().debate("the flywheel smooths the engine because its inertia "
                              "resists a change in speed")
    assert not any(o.kind == "no-mechanism" for o in got.objections)


def test_a_hinglish_mechanism_counts_as_a_mechanism():
    got = _dialectic().debate("gaadi ruk gayi kyunki brake dabaya gaya tha aur pahiya jam ho gaya")
    assert not any(o.kind == "no-mechanism" for o in got.objections)


def test_a_verified_answer_is_not_attacked_for_being_unsupported():
    got = _dialectic().debate("all squares always have four sides", confidence=0.9, verified=True)
    assert not any(o.kind in ("unsupported", "no-mechanism") for o in got.objections)


def test_a_prover_that_cannot_decide_produces_no_objection():
    got = _dialectic(brain=_Brain(_Silent())).debate("gravity pulls the apple down")
    assert not any(o.kind == "refuted" for o in got.objections)


def test_the_prover_can_be_switched_off():
    got = _dialectic(brain=_Brain(_Refuting()), use_prover=False).debate("two plus two is five")
    assert not got.abstained


# --------------------------------------------------------------------------- #
# The Proponent
# --------------------------------------------------------------------------- #
def test_a_verified_claim_gets_the_strongest_defence_there_is():
    got = _dialectic().debate("2 + 2 = 4", verified=True)
    assert "derived and checked" in got.defence


def test_a_claim_with_evidence_is_defended_by_its_evidence():
    got = _dialectic().debate("the belt drives it", evidence=["a", "b"])
    assert "2 piece(s)" in got.defence


def test_a_defence_that_has_nothing_says_so_rather_than_inventing_support():
    got = _dialectic().debate("engines fail")
    assert "nothing to offer in its defence" in got.defence


# --------------------------------------------------------------------------- #
# Nothing to attack
# --------------------------------------------------------------------------- #
def test_an_empty_claim_is_not_debated():
    got = _dialectic().debate("")
    assert not got.debated and "nothing to defend" in got.note


def test_a_thought_that_reached_no_answer_is_not_debated():
    got = _dialectic().review(type("T", (), {"winner": None})())
    assert not got.debated and "nothing reached awareness" in got.note


def test_a_turn_that_was_not_debated_records_that_fact():
    """The difference between "attacked and survived" and "never attacked" is never invisible."""
    assert "(not debated" in _dialectic().debate("").render()


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #
def test_the_objections_ride_on_the_result():
    got = _dialectic().debate("all cars are always fast")
    assert got.objections and got.strongest is not None
    assert got.severity >= got.strongest.severity


def test_the_render_shows_what_was_attacked_and_what_was_left():
    rendered = _dialectic().debate("all cars are always fast", confidence=0.8).render()
    assert "claim" in rendered and "confidence" in rendered and "outcome" in rendered


def test_the_result_serialises():
    got = _dialectic().debate("all cars are always fast").to_dict()
    assert got["debated"] and isinstance(got["objections"], list)


def test_an_objection_serialises():
    assert Objection(kind="vagueness", detail="d", severity=0.5).to_dict()["kind"] == "vagueness"


# --------------------------------------------------------------------------- #
# Honesty
# --------------------------------------------------------------------------- #
def test_stats_refuse_the_word_bulletproof():
    note = _dialectic().stats()["note"].lower()
    assert "structured critique" in note and "not a" in note
    assert "walks straight through" in note


def test_stats_count_every_outcome_separately():
    dialectic = _dialectic(brain=_Brain(_Refuting()))
    dialectic.debate("two plus two is five")
    dialectic.debate("")
    assert dialectic.abstentions == 1 and dialectic.skipped == 1


def test_counters_survive_a_round_trip():
    dialectic = _dialectic()
    dialectic.debate("all cars are always fast", confidence=0.9)
    fresh = _dialectic()
    assert fresh.load_dict(dialectic.to_dict())
    assert fresh.debates == dialectic.debates


def test_a_corrupt_sidecar_never_blocks_boot():
    assert not _dialectic().load_dict("not a dict")


# --------------------------------------------------------------------------- #
# Wired into the brain
# --------------------------------------------------------------------------- #
def test_the_brain_exposes_it():
    brain = NyxBrain(NyxConfig())
    assert brain.dialectic is not None and "dialectic" in brain.stats()


def test_the_brain_can_be_told_to_do_without_it():
    brain = NyxBrain(NyxConfig(dialectic_enabled=False))
    assert brain.dialectic is None and brain.debate("anything") is None


def test_a_thought_carries_its_debate():
    brain = NyxBrain(NyxConfig())
    for _ in range(3):
        brain.think("the engine turns the flywheel")
    thought = brain.think("the engine turns the flywheel")
    assert thought.debate is None or thought.debate.debated


def test_the_review_can_be_switched_off_without_losing_the_faculty():
    brain = NyxBrain(NyxConfig(dialectic_review_turns=False))
    for _ in range(3):
        brain.think("the engine turns the flywheel")
    assert brain.think("the engine turns the flywheel").debate is None
    assert brain.debate("all cars are always fast") is not None


# --------------------------------------------------------------------------- #
# The budget — declared in the docstring, and previously never consulted
# --------------------------------------------------------------------------- #
def test_the_prover_is_bounded_by_the_budget_not_by_its_own_timeout():
    """Her prover carries a 3000 ms timeout; a turn's debate is budgeted at 250 ms.

    ``budget_ms`` was stored and never read, so the one expensive weapon could overrun the
    stated budget by an order of magnitude on exactly the claim hardest to decide. It is now
    handed the time that is actually left.
    """
    from nyxara.nyx.theorem_prover import ProofCore

    dialectic = _dialectic(brain=_Brain(ProofCore(timeout_ms=3000)))
    dialectic.budget_ms = 10.0
    got = dialectic.debate("for all real x, x*x >= 0")
    assert got.elapsed_ms <= 60.0            # generous, but nowhere near the prover's own 3000
    assert dialectic.stats()["prover_bounded"] >= 1


def test_a_prover_whose_own_timeout_fits_is_used_unchanged():
    class _Quick:
        timeout_ms = 1
        max_vars = 8

        @staticmethod
        def prove(claim: str):
            return type("Cert", (), {"verdict": "unknown", "note": ""})()

    dialectic = _dialectic(brain=_Brain(_Quick()))
    dialectic.debate("for all real x, x*x >= 0")
    assert dialectic.stats()["prover_bounded"] == 0


def test_the_budget_cuts_the_attack_short_and_says_it_did():
    """Surviving a shortened attack is not the same claim as surviving a full one."""
    import time as _time

    class _Slow:
        @staticmethod
        def attack(claim: str):
            _time.sleep(0.05)
            return type("R", (), {"objections": []})()

    dialectic = _dialectic(brain=_Brain(_Silent()))
    dialectic.budget_ms = 12.0
    dialectic._critic = _Slow()
    got = dialectic.debate("all cars are always fast and every engine always fails")
    assert got.truncated is True
    assert dialectic.stats()["budget_bites"] >= 1
    assert "FEWER weapons" in got.note
    assert "cut short" in got.render()


def test_an_untruncated_debate_says_nothing_about_the_budget():
    got = _dialectic().debate("all cars are always fast and every engine always fails")
    assert got.truncated is False
    assert "cut short" not in got.render()
    assert got.to_dict()["truncated"] is False


def test_the_budget_counters_survive_a_round_trip():
    dialectic = _dialectic(brain=_Brain(_Silent()))
    dialectic.budget_ms = 10.0
    dialectic.debate("for all real x, x*x >= 0")
    fresh = _dialectic()
    assert fresh.load_dict(dialectic.to_dict())
    assert fresh.prover_bounded == dialectic.prover_bounded
