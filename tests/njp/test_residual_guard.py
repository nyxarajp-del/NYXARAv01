"""Residual learning is the sharpest training signal here, which is exactly why it is gated.

Learning ``teacher − njp`` teaches efficiently. Against a teacher that is *wrong* on a case, it
teaches the error efficiently. The gate is the only thing standing between "she learned what she
was missing" and "she learned the teacher's mistakes faster than imitation would have managed".
"""

from __future__ import annotations

import numpy as np

from nyxara.njp.predictive import PredictiveWorldModel
from nyxara.njp.shadow import GapKind, ShadowCognition


def test_no_residual_without_an_independent_check():
    shadow = ShadowCognition()
    assert shadow.residual([1.0, 2.0, 3.0], [0.0, 1.0, 2.0]) is None
    assert shadow.residual([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], established=False) is None
    assert shadow.stats()["residuals_trained"] == 0
    assert shadow.stats()["residuals_refused"] == 2


def test_established_evidence_yields_the_disagreement_not_the_answer():
    """The target is the gap. Copying the teacher's answer teaches nothing on the cases she
    already gets right; the difference points at the specific thing her model lacks."""
    shadow = ShadowCognition()
    got = shadow.residual([1.0, 2.0, 3.0], [0.5, 0.5, 0.5],
                          established=True, evidence="master stated the fact")
    assert got is not None
    assert np.allclose(got, [0.5, 1.5, 2.5])
    assert shadow.stats()["residuals_trained"] == 1


def test_mismatched_shapes_are_refused_rather_than_broadcast():
    shadow = ShadowCognition()
    assert shadow.residual([1.0, 2.0], [1.0], established=True) is None
    assert shadow.residual([], [], established=True) is None
    assert shadow.stats()["residuals_refused"] == 2


def test_a_prediction_gap_is_offered_but_never_trained_by_itself():
    """Finding a gap and being allowed to learn from it are two different events, and only the
    first one happens inside ``compare``."""
    predictor = PredictiveWorldModel()
    for _ in range(8):                    # give her a forward model to actually disagree with
        predictor.observe("cold")
        predictor.observe("warm")
        predictor.observe("hot")
    shadow = ShadowCognition(predictor=predictor)
    reading = shadow.compare("what next?", teacher_text="the bar will bend under heat",
                             njp_text="")
    assert reading.residuals_offered >= 1
    assert reading.residuals_trained == 0
    assert shadow.stats()["residuals_trained"] == 0


def test_having_no_forward_model_is_not_a_maximal_disagreement():
    """The distinction that keeps this counter meaningful: 'she cannot predict here yet' and
    'she predicts something else' are different facts. Scoring the first as a full-magnitude gap
    would make every early turn report maximal disagreement — turning the signal into a measure
    of how young her world model is, exactly while it is being built."""
    shadow = ShadowCognition(predictor=PredictiveWorldModel())
    reading = shadow.compare("what next?", teacher_text="the bar will bend under heat",
                             njp_text="")
    gaps = reading.by_kind(GapKind.PREDICTION)
    assert len(gaps) == 1
    assert gaps[0].magnitude == 0.0
    assert "cannot answer here yet" in gaps[0].detail
    assert reading.residuals_offered == 0
