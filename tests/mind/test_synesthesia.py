"""Tests for nyxara.mind.synesthesia — Cross-Domain Biomimetic Synesthesia.

A pattern's scale-invariant shape is lifted into the shared HDC space; patterns from different domains
that share shape land near each other; a known law is transposed across domains only when it verifiably
fits the target. These tests assert cross-domain matching and the verify-or-abstain transposition.
"""

from __future__ import annotations

import math

import pytest

from nyxara.cognition.hyper_dimensional_vectors import has_numpy
from nyxara.mind.synesthesia import SynestheticMatrix, TranspositionResult

pytestmark = pytest.mark.skipif(not has_numpy(), reason="exercised on the numpy HDC substrate")


def test_best_form_detects_exponential():
    bio = [2.0 * math.exp(0.3 * t) for t in range(16)]
    assert SynestheticMatrix.best_form(bio) == "exponential"


def test_best_form_detects_linear():
    lin = [3.0 * t + 5.0 for t in range(16)]
    assert SynestheticMatrix.best_form(lin) == "linear"


def test_transpose_exponential_across_domains_verified():
    m = SynestheticMatrix(adopt_r2=0.9)
    m.learn("bacterial_growth", "biology", [2.0 * math.exp(0.3 * t) for t in range(16)])
    fin = [100.0 * (1.05 ** t) for t in range(16)]
    res = m.transpose(fin, "finance")
    assert isinstance(res, TranspositionResult)
    assert res.source_domain == "biology" and res.form == "exponential"
    assert res.adopted and res.fit_r2 >= 0.9, res.to_dict()


def test_transpose_abstains_when_law_does_not_fit():
    m = SynestheticMatrix(adopt_r2=0.9)
    m.learn("bacterial_growth", "biology", [2.0 * math.exp(0.3 * t) for t in range(16)])
    osc = [math.sin(0.6 * t) for t in range(20)]
    res = m.transpose(osc, "cardiology")
    assert not res.adopted, res.to_dict()


def test_no_analogue_is_honest():
    m = SynestheticMatrix()
    res = m.transpose([1.0, 2.0, 3.0, 4.0], "anything")
    assert not res.adopted and res.source_label is None


def test_oscillation_matches_oscillation_cross_domain():
    m = SynestheticMatrix()
    m.learn("bacterial_growth", "biology", [2.0 * math.exp(0.3 * t) for t in range(16)])
    m.learn("market_cycle", "economics", [math.sin(0.5 * t + 0.3) for t in range(20)], form="sine")
    iso = m.find_isomorphisms([math.sin(0.6 * t) for t in range(20)], target_domain="cardiology")
    assert iso and iso[0][0].domain == "economics"
