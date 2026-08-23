"""Two clauses of the plan that the stages implementing them left out.

**"Do not re-attack an undecided claim with no new observations — but still ask about it."**
The record is what settles an attack, so running the same four attacks against the same record is
four more coin-flips at the same cost. Skipping is not forgetting: the report still comes back
UNDECIDED so `epistemic.from_attack` can go and get more record.

**The three-way disagreement had no classification of its own.** A speaking seat with support and
a substrate claim with none are not disagreeing about *evidence* — the substrate has none by
construction, it associates rather than derives — so the two-way ladder read every such clash as
EVIDENCE and would have designed an experiment against a difference that is not there.
"""

from __future__ import annotations

from nyxara.njp.adversary import SelfAttacker, Stance
from nyxara.njp.beliefs import BeliefLedger
from nyxara.njp.epistemic import EpistemicCompiler
from nyxara.njp.router import DisagreementKind, Router


class _Link:
    causal = False
    stated = refuted = together = cause_total = effect_total = 0
    lift = strength = 0.0


class _World:
    """A record that settles nothing, and can be made to grow."""

    def __init__(self, observations: int = 5) -> None:
        self._observations = observations

    def link(self, cause, effect):
        return _Link()

    def links(self, causal_only=True):
        return []

    def counterfactual(self, *a, **k):
        return None


def _attacker(world: _World) -> SelfAttacker:
    ledger = BeliefLedger()
    ledger.hold("aag causes garmi", confidence=0.5)
    return SelfAttacker(world=world, beliefs=ledger)


# --------------------------------------------------------------------------- #
# Not re-asking a question the record cannot answer
# --------------------------------------------------------------------------- #

def test_the_same_undecided_claim_is_not_attacked_twice_for_nothing():
    attacker = _attacker(_World())
    attacker.attack("aag", "garmi")
    assert attacker.attacked == 1

    second = attacker.attack("aag", "garmi")
    assert attacker.attacked == 1, "the same four coin-flips ran again against the same record"
    assert attacker.skipped_undecided == 1
    assert second.skipped is True


def test_it_is_attacked_again_once_the_record_grows():
    """The skip must not be permanent — that would be forgetting rather than deferring."""
    world = _World()
    attacker = _attacker(world)
    attacker.attack("aag", "garmi")
    attacker.attack("aag", "garmi")
    assert attacker.attacked == 1

    world._observations = 500
    attacker.attack("aag", "garmi")
    assert attacker.attacked == 2


def test_a_skipped_report_still_says_what_it_is():
    attacker = _attacker(_World())
    attacker.attack("aag", "garmi")
    report = attacker.attack("aag", "garmi")
    assert report.stance == Stance.UNDECIDED
    assert "nothing new" in report.verdict
    assert report.to_dict()["skipped"] is True


def test_a_skipped_claim_can_still_be_turned_into_an_experiment():
    """Skipping the attack is not dropping the question."""
    attacker = _attacker(_World())
    attacker.attack("aag", "garmi")
    skipped = attacker.attack("aag", "garmi")
    compiler = EpistemicCompiler()
    # It has a cause and an effect and was not refuted, which is all `from_attack` needs to ask.
    assert skipped.cause and skipped.effect and not skipped.refuted_by


def test_a_claim_that_was_never_undecided_is_attacked_normally():
    attacker = _attacker(_World())
    attacker.attack("dhuan", "aag")
    assert attacker.attacked == 1
    attacker.attack("baarish", "keechad")
    assert attacker.attacked == 2


def test_the_verdict_recorded_is_derived_from_what_the_attacks_found():
    """`stance` is set only on a skip; reading that empty field recorded every outcome as none.

    The mechanism below it was inert until this was fixed — the undecided record never populated,
    so nothing was ever skipped and nothing was ever capped.
    """
    ledger = BeliefLedger()
    ledger.hold("aag causes garmi", confidence=0.5)
    attacker = SelfAttacker(world=_World(), beliefs=ledger)
    attacker.attack("aag", "garmi")
    belief = ledger.beliefs[next(iter(ledger.beliefs))]
    assert belief.last_stance == Stance.UNDECIDED
    assert belief.observations_at_test == 5


# --------------------------------------------------------------------------- #
# The three-way clash has a name
# --------------------------------------------------------------------------- #

class _Account:
    def __init__(self, text: str, support=()):
        self.text = text
        self.support = list(support)


def test_a_substrate_naming_a_different_filler_is_its_own_kind():
    kind = Router._classify_disagreement(
        _Account("water causes growth", [("a", "b", "c")]),
        _Account("water causes growth", [("a", "b", "c")]),
        "light")
    assert kind == DisagreementKind.SUBSTRATE


def test_a_substrate_that_agrees_does_not_create_a_clash():
    kind = Router._classify_disagreement(
        _Account("water causes growth", [("a", "b", "c")]),
        _Account("water causes growth", [("a", "b", "c")]),
        "water")
    assert kind != DisagreementKind.SUBSTRATE


def test_the_two_way_classification_is_unchanged():
    """The three-way case is added in front of the ladder, not woven through it."""
    both = Router._classify_disagreement(
        _Account("x", [("a", "b", "c")]), _Account("y", [("a", "b", "c")]))
    assert both == DisagreementKind.CAUSAL_MODEL

    disjoint = Router._classify_disagreement(
        _Account("x", [("a", "b", "c")]), _Account("y", [("d", "e", "f")]))
    assert disjoint == DisagreementKind.EVIDENCE


def test_the_substrate_clash_asks_about_the_record_not_about_who_to_believe():
    question = Router._question("what causes growth", "water", "light",
                                DisagreementKind.SUBSTRATE)
    assert "record" in question
    assert "believe" not in question
