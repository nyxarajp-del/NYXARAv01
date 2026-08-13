"""The Truth-Seeking Gauntlet: fail-closed, and soft agreement establishes nothing."""

from __future__ import annotations

from nyxara.njp.ledger import Ledger
from nyxara.njp.truth import (
    ConsistencySource,
    LedgerSource,
    ObservationSource,
    PredictiveSource,
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
