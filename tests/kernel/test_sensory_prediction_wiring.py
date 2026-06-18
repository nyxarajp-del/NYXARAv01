"""Tests for sensory prediction wired into the live percept stream.

senses/predictive.py is a real predictor, but it used never to run inside the
cognitive loop. Now each live percept is observed: structural surprise sharpens
attention (salience boost) and novelty colours affect.
"""

from __future__ import annotations

from nyxara.kernel.orchestrator import NyxaraCore
from nyxara.senses.binding import Percept
from nyxara.senses.predictive import PerceptualPredictor


def test_core_builds_sensory_predictor_with_growth():
    nyx = NyxaraCore(enable_growth=True)
    assert isinstance(nyx.sensory_predictor, PerceptualPredictor)


def test_surprising_percept_gets_attention_boost():
    nyx = NyxaraCore(enable_growth=True)
    # a steady stream of plain text percepts establishes the baseline
    for _ in range(8):
        nyx._perceptual_predict(Percept.from_text("routine status update", source="owner"))
    # a structurally very different percept (many threat tags) should be surprising
    odd = Percept.from_text("ACME CORP BREACH", source="net",
                            tags=["threat", "alert", "urgent", "intrusion", "exfil"])
    before = odd.salience
    nyx._perceptual_predict(odd)
    assert odd.salience > before          # surprise sharpened attention


def test_novel_percept_colours_affect():
    nyx = NyxaraCore(enable_growth=True)
    calls = []
    orig = nyx.affect.note_novelty
    nyx.affect.note_novelty = lambda *a, **k: calls.append(a) or orig(*a, **k)
    for _ in range(8):
        nyx._perceptual_predict(Percept.from_text("steady hum", source="owner"))
    nyx._perceptual_predict(Percept.from_text("ALERT", source="net",
                                              tags=["threat", "urgent", "novel", "spike"]))
    assert calls, "a novel percept should feed affect's novelty drive"


def test_perceptual_predict_is_safe_without_predictor():
    nyx = NyxaraCore(enable_growth=False)
    assert nyx.sensory_predictor is None
    # must be a harmless no-op, never raise
    nyx._perceptual_predict(Percept.from_text("hi", source="owner"))
