"""Part 1 → Part 2 compounding: verified deep-reasoned answers feed the own-model corpus.

The always-max deep reasoner (Problem #1) produces NYXARA's best, verifier-scored answers. Those
are exactly the highest-quality supervision for training her *own* model (Pillar A). This test
proves the existing data flywheel accepts such an answer into the foundry-ready corpus and rejects
a degenerate one — so thinking harder today measurably improves her own brain tomorrow. Pure
stdlib: no model, no GPU, no network.
"""

from __future__ import annotations

from pathlib import Path

from nyxara.growth.flywheel import DataFlywheel
from nyxara.mind.router import default_verifier


def _flywheel(tmp_path: Path) -> DataFlywheel:
    return DataFlywheel(store_path=tmp_path / "flywheel.jsonl",
                        min_confidence=0.6, min_chars=8,
                        verifier=default_verifier())


def test_verified_deep_answer_is_collected_for_training(tmp_path: Path):
    fw = _flywheel(tmp_path)
    prompt = "why is the sky blue?"
    deep_answer = ("Rayleigh scattering: the atmosphere scatters short blue wavelengths far more "
                   "strongly than long red ones, so daylight reaches us bluest from every angle.")
    decision = fw.consider(prompt, deep_answer, confidence=0.9)
    assert decision.collected, decision.reason
    # it lands in the corpus the foundry trains the own model on
    docs = fw.training_docs()
    assert docs and any("Rayleigh" in d for d in docs)
    assert fw.count() == 1


def test_degenerate_answer_is_rejected(tmp_path: Path):
    fw = _flywheel(tmp_path)
    # a low-quality answer the verifier scores as degenerate must not pollute the corpus
    decision = fw.consider("why is the sky blue?", "blue blue blue blue blue", confidence=0.9)
    assert not decision.collected
    assert fw.count() == 0


def test_low_confidence_answer_is_rejected(tmp_path: Path):
    fw = _flywheel(tmp_path)
    good = "The sky looks blue because shorter wavelengths scatter more in the atmosphere."
    decision = fw.consider("why is the sky blue?", good, confidence=0.3)
    assert not decision.collected
    assert "confidence" in decision.reason
