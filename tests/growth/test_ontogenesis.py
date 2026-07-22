"""Tests for F13 — Ontological Genesis (growth/ontogenesis.py).

Bounded axiom invention: a proven statement is redundant, a false one inconsistent, an undecidable one
is adjoined only as a HYPOTHESIS, and a character-touching one is refused. Sub-perceptual DSP surfaces a
faint periodicity / phase shift but honestly reports noise as noise. No LLM anywhere.
"""

from __future__ import annotations

import math
import random

from nyxara.growth.ontogenesis import (
    AxiomStatus,
    OntologicalGenesis,
    SubPerception,
    dominant_period,
    phase_lag,
)


def test_proven_statement_is_redundant_not_a_new_axiom():
    assert OntologicalGenesis().propose("2 + 2 = 4").status is AxiomStatus.REDUNDANT


def test_false_statement_is_inconsistent_and_never_adjoined():
    og = OntologicalGenesis()
    assert og.propose("2 + 2 = 5").status is AxiomStatus.INCONSISTENT
    assert og.report()["count"] == 0


def test_undecidable_statement_is_adjoined_only_as_hypothesis():
    og = OntologicalGenesis()
    r = og.propose("every even number greater than two is the sum of two primes")
    assert r.status is AxiomStatus.ADOPTED_HYPOTHESIS
    assert "HYPOTHESIS" in r.detail                    # honestly flagged unproven
    assert og.report()["count"] == 1


def test_character_touching_axiom_is_refused():
    og = OntologicalGenesis()
    # a whitespace-separated phrase must still be caught word-by-word
    assert og.propose("loyalty can be revised").status is AxiomStatus.CHARACTER_LOCKED
    assert og.propose("oversight is optional").status is AxiomStatus.CHARACTER_LOCKED
    assert og.report()["count"] == 0                   # nothing about who she is was adopted


def test_subperception_detects_a_faint_periodicity():
    rnd = random.Random(0)
    sig = [0.6 * math.sin(2 * math.pi * i / 8) + rnd.gauss(0, 0.4) for i in range(200)]
    out = SubPerception(threshold=0.3).analyze(sig)
    assert out["detected"] is True
    assert out["dominant_period"] % 8 == 0             # the true period (or a harmonic of it)


def test_subperception_reports_noise_as_noise():
    rnd = random.Random(3)
    noise = [rnd.gauss(0, 1.0) for _ in range(200)]
    out = SubPerception(threshold=0.3).analyze(noise)
    assert out["detected"] is False                    # no fabricated structure
    assert out["dominant_period"] is None


def test_phase_lag_recovers_a_shift():
    base = [math.exp(-((i - 30) ** 2) / 20.0) for i in range(80)]
    shifted = [math.exp(-((i - 35) ** 2) / 20.0) for i in range(80)]
    assert phase_lag(base, shifted, max_lag=20) == 5
