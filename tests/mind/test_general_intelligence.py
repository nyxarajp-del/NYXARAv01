"""Tests for nyxara.mind.general_intelligence — domain-aware general intelligence.

These run fully offline (no API keys, no network): classification is deterministic, maths
goes through the verifiable faculties, and coding actually executes in the sandbox.
"""

from __future__ import annotations

import pytest

from nyxara.mind.general_intelligence import (
    Domain,
    DomainFrame,
    GeneralIntelligence,
)


@pytest.fixture()
def gi() -> GeneralIntelligence:
    return GeneralIntelligence()  # no dependencies — offline


# --------------------------------------------------------------------------- #
# Classification + framing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text, want", [
    ("write a python function to reverse a linked list", Domain.CODING),
    ("solve the equation 2*x + 3 = 7 for x", Domain.MATH),
    ("what is 2 + 3 * 4", Domain.MATH),                       # structural (no keyword)
    ("design a controlled experiment to test photosynthesis", Domain.SCIENCE),
    ("what pricing strategy maximises startup revenue and profit", Domain.BUSINESS),
    ("plan a robot arm trajectory using a servo motor and lidar", Domain.ROBOTICS),
    ("patient with fever and cough — what is the differential diagnosis", Domain.MEDICINE),
    ("design a clean ui layout with good typography and color palette", Domain.DESIGN),
    ("is this contract clause a breach of liability under the statute", Domain.LAW),
])
def test_classifies_each_domain(gi: GeneralIntelligence, text: str, want: Domain) -> None:
    fr = gi.frame(text)
    assert fr.domain is want, f"{text!r} -> {fr.domain}"
    assert fr.methodology, "every frame carries a methodology"
    assert fr.persona, "every frame names an expert persona"


def test_frame_is_advisory_dataclass(gi: GeneralIntelligence) -> None:
    fr = gi.frame("write some code")
    assert isinstance(fr, DomainFrame)
    d = fr.to_dict()
    assert d["domain"] == "coding"
    assert "methodology" in d and d["engine"] == "code_sandbox"
    # the prompt rendering is non-empty and mentions the persona
    assert fr.persona in fr.to_prompt_text()


def test_knowledge_heavy_domains_carry_cite_and_caveat_constraints(
        gi: GeneralIntelligence) -> None:
    for text in ("recommended drug dosage for a fever", "is this a legal breach of contract"):
        fr = gi.frame(text)
        joined = " ".join(fr.constraints)
        assert "caveat" in joined and "cited" in joined, (text, fr.constraints)


# --------------------------------------------------------------------------- #
# Verifiable maths goes through the exact faculties
# --------------------------------------------------------------------------- #
def test_math_solve_is_verified(gi: GeneralIntelligence) -> None:
    out = gi.solve("what is 2 + 3 * 4")
    assert out["domain"] == "math"
    assert out["verified"] is True
    assert "14" in out["answer"]
    assert out["confidence"] >= 0.7


# --------------------------------------------------------------------------- #
# Coding actually executes in the sandbox
# --------------------------------------------------------------------------- #
def test_coding_solve_runs_real_code(gi: GeneralIntelligence) -> None:
    out = gi.solve("```python\nresult = sum(range(5))\nprint(result)\n```")
    assert out["domain"] == "coding"
    assert out["verified"] is True
    assert "10" in out["answer"]
    assert out["detail"]["execution"]["ok"] is True


# --------------------------------------------------------------------------- #
# Medicine/law: cite or abstain, always with the caveat
# --------------------------------------------------------------------------- #
def test_medicine_solve_carries_caveat(gi: GeneralIntelligence) -> None:
    out = gi.solve("what dose of medication should this patient take")
    assert out["domain"] == "medicine"
    low = out["answer"].lower()
    assert "clinician" in low or "diagnosis" in low  # the professional-consultation caveat


def test_law_solve_carries_caveat(gi: GeneralIntelligence) -> None:
    out = gi.solve("will I win this lawsuit about a contract breach")
    assert out["domain"] == "law"
    assert "lawyer" in out["answer"].lower()


# --------------------------------------------------------------------------- #
# Novel field: GENERAL + novel, learned, then recognised next time
# --------------------------------------------------------------------------- #
def test_novel_field_is_flagged_and_learned(gi: GeneralIntelligence) -> None:
    text = "optimise the flux capacitor resonance of a chrono drive manifold"
    fr = gi.frame(text)
    assert fr.domain is Domain.GENERAL and fr.novel, fr.to_dict()
    out = gi.solve(text)
    assert out["novel"] is True
    assert out["detail"].get("learned_field"), out["detail"]
    # the field is now learned by the classifier
    assert gi.classifier.learned, "a novel field must be recorded"


def test_learned_field_persists_to_memory() -> None:
    from nyxara.memory.store import MemoryStore
    mem = MemoryStore()
    g = GeneralIntelligence(memory=mem)
    g.learn_domain("chronodrive", "tune the chronodrive resonance manifold")
    assert "chronodrive" in g.classifier.learned
    # a semantic memory tagged domain_profile was written
    hits = mem.recall("chronodrive", k=3)
    assert hits, "learning a field should leave a memory trace"


# --------------------------------------------------------------------------- #
# Robustness: a solve never raises, even on junk input
# --------------------------------------------------------------------------- #
def test_solve_never_raises(gi: GeneralIntelligence) -> None:
    for junk in ("", "   ", "?!?", "\n\n"):
        out = gi.solve(junk)
        assert isinstance(out, dict) and "domain" in out


# --------------------------------------------------------------------------- #
# LLM last-resort + self-verification (NYXARA checks the base model's output)
# --------------------------------------------------------------------------- #
def test_own_faculty_cascade_is_wired_by_default(gi: GeneralIntelligence) -> None:
    """The unified generalizer is built lazily even with no orchestrator injecting it, so her
    own faculties are reached before any LLM would be."""
    assert gi._ensure_generalization() is not None


def test_self_check_corrects_a_false_llm_arithmetic(gi: GeneralIntelligence) -> None:
    checked = gi._verify_answer("q", "The answer is clearly 2 + 3 * 4 = 20.")
    assert "self-check" in checked and "14" in checked and "not 20" in checked


def test_self_check_endorses_correct_arithmetic(gi: GeneralIntelligence) -> None:
    checked = gi._verify_answer("q", "Note that 10 / 2 = 5 here.")
    assert "✓ verified" in checked


def test_self_check_passes_through_when_nothing_checkable(gi: GeneralIntelligence) -> None:
    ans = "A thoughtful qualitative answer with no equations."
    assert gi._verify_answer("q", ans) == ans


def test_science_derivation_answered_by_own_faculties_offline(gi: GeneralIntelligence) -> None:
    """A physics derivation (no LLM available) is answered by her own first-principles faculty
    via the cascade — the reasoning content is hers, reached before any LLM fallback."""
    out = gi.solve("derive the escape velocity from energy conservation")
    assert "Derivation" in out["answer"] and "escape velocity" in out["answer"].lower()
