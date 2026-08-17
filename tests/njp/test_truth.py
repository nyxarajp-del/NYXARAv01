"""The Truth-Seeking Gauntlet: fail-closed, and soft agreement establishes nothing."""

from __future__ import annotations

from nyxara.njp.ledger import Ledger
from nyxara.njp.truth import (
    ConsistencySource,
    Evidence,
    LedgerSource,
    ObservationSource,
    PredictiveSource,
    Source,
    TruthGauntlet,
    Verdict,
)


def _predictive(ok: bool) -> PredictiveSource:
    return PredictiveSource(predictor=lambda _c, _s: ok, holdout=[1, 2, 3, 4], min_samples=2)


def test_multi_source_with_a_hard_source_establishes():
    g = TruthGauntlet(sources=[_predictive(True),
                               ObservationSource(observations=["an apple falls down"])],
                      min_sources=2)
    j = g.judge("an apple falls down")
    assert j.verdict == Verdict.ESTABLISHED
    assert j.assertable is True
    assert j.hard_supports >= 1


def test_soft_agreement_alone_never_establishes():
    """This is the whole point: consensus without a checkable source is how bias launders itself."""
    g = TruthGauntlet(
        sources=[ConsistencySource(established=["the sky is blue"]),
                 ObservationSource(observations=["an apple falls down"])],
        min_sources=2, require_hard=True)
    j = g.judge("an apple falls down")
    assert j.supports == 2
    assert j.hard_supports == 0
    assert j.verdict == Verdict.SUPPORTED
    assert j.assertable is False


def test_one_refutation_outranks_any_agreement():
    g = TruthGauntlet(sources=[_predictive(False),
                               ObservationSource(observations=["an apple falls down"]),
                               ConsistencySource(established=["an apple falls down"])],
                      min_sources=1)
    j = g.judge("an apple falls down")
    assert j.verdict == Verdict.REFUTED
    assert j.assertable is False


def test_a_contradiction_refutes():
    g = TruthGauntlet(sources=[ConsistencySource(established=["the sky is blue"])],
                      min_sources=1, require_hard=False)
    assert g.judge("the sky is not blue").verdict == Verdict.REFUTED


def test_nothing_to_say_abstains_rather_than_asserting():
    g = TruthGauntlet(sources=[], min_sources=1)
    j = g.judge("zxqv frobnicate wobble")
    assert j.verdict == Verdict.ABSTAINED
    assert j.assertable is False


def test_a_refuted_claim_is_remembered_as_an_error():
    """Learning from your own mistakes only means something if they are still on record."""
    led = Ledger()
    g = TruthGauntlet(sources=[_predictive(False)], min_sources=1, ledger=led)
    g.judge("something wrong")
    assert len(led.errors.records) == 1
    assert led.errors.similar("something wrong")


def test_a_claim_resembling_a_past_error_is_demoted():
    """A past error counts AGAINST a claim; it can never count for one."""
    led = Ledger()
    led.errors.record("the parser handles unicode correctly", verdict="refuted", truth="it did not")
    g = TruthGauntlet(sources=[_predictive(True), LedgerSource(ledger=led)],
                      min_sources=1, ledger=led)
    clean = g.judge("gravity pulls an apple downward")
    tainted = g.judge("the parser handles unicode correctly")
    assert clean.verdict == Verdict.ESTABLISHED
    assert tainted.verdict == Verdict.SUPPORTED
    assert tainted.confidence < clean.confidence


def test_a_broken_source_is_a_silent_source():
    class Boom:
        name, hard = "boom", True

        def check(self, claim, *, context=None):
            raise RuntimeError("this source is broken")

    g = TruthGauntlet(sources=[Boom(), _predictive(True)], min_sources=1)
    assert g.judge("still works").verdict == Verdict.ESTABLISHED


# --------------------------------------------------------------------------- #
# Simulation is never corroboration
# --------------------------------------------------------------------------- #
class _Real(Source):
    name = "real"
    hard = True

    def check(self, claim, *, context=None):
        return Evidence(source=self.name, supports=True, hard=True, weight=0.9)


class _Dream(Source):
    name = "dream"
    hard = False

    def check(self, claim, *, context=None):
        return Evidence(source=self.name, supports=True, simulated=True, weight=1.0)


def test_simulation_alone_never_gets_past_conjecture():
    """The loop this closes: dream a result, store it as evidence, cite it back to yourself.

    Once that closes, more compute buys more certainty and no more truth, and nothing downstream
    can tell the difference. A weight could be accumulated; exclusion cannot, which is why this is
    structural rather than a discount.
    """
    judged = TruthGauntlet(sources=[_Dream(), _Dream()]).judge("x causes y")
    assert judged.verdict == Verdict.CONJECTURE
    assert judged.supports == 0
    assert judged.simulated_supports == 2
    assert not judged.assertable


def test_a_simulated_support_cannot_tip_a_claim_over_the_threshold():
    """The dangerous case is not simulation alone — it is simulation as the deciding vote."""
    judged = TruthGauntlet(sources=[_Real(), _Dream()]).judge("x causes y")
    assert judged.verdict == Verdict.SUPPORTED, "a dream completed a quorum it must not"
    assert judged.supports == 1 and judged.simulated_supports == 1


def test_real_evidence_still_establishes():
    """The rule must not have made establishment unreachable."""
    judged = TruthGauntlet(sources=[_Real(), _Real()]).judge("x causes y")
    assert judged.verdict == Verdict.ESTABLISHED
    assert judged.simulated_supports == 0


def test_a_simulated_reading_is_still_recorded_and_reported():
    """A hypothesis is worth keeping. What it may never do is promote."""
    judged = TruthGauntlet(sources=[_Dream()]).judge("x causes y")
    assert [e for e in judged.evidence if e.simulated]
    assert judged.to_dict()["simulated_supports"] == 1
    assert any(e["simulated"] for e in judged.to_dict()["evidence"])
