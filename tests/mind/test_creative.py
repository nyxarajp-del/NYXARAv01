"""Tests for nyxara.mind.creative."""

from __future__ import annotations

import random

from nyxara.mind.creative import CreativeEngine, Idea, Technique
from nyxara.mind.proposal import Proposal


def _eng(seed=7):
    return CreativeEngine(rng=random.Random(seed))


# -------------------- Idea -------------------- #
def test_idea_score_blend():
    i = Idea("x", "scamper", novelty=1.0, usefulness=1.0, feasibility=1.0)
    assert i.score == 1.0
    j = Idea("x", "scamper", novelty=0.0, usefulness=0.0, feasibility=0.0)
    assert j.score == 0.0


def test_idea_to_dict():
    d = Idea("content", "scamper", 0.5, 0.5, 0.5).to_dict()
    assert d["technique"] == "scamper" and "score" in d


# -------------------- SCAMPER -------------------- #
def test_scamper_seven_operators():
    ideas = _eng().scamper("firewall")
    assert len(ideas) == 7
    ops = {i.technique for i in ideas}
    assert {"substitute", "combine", "adapt", "modify", "put_to_other_use",
            "eliminate", "reverse"} <= ops


def test_scamper_mentions_concept():
    ideas = _eng().scamper("password")
    assert all("password" in i.content for i in ideas)


# -------------------- lateral -------------------- #
def test_lateral_techniques():
    ideas = _eng().lateral("encryption")
    techs = {i.technique for i in ideas}
    assert "random_entry" in techs
    assert "provocation" in techs
    assert "lateral_reverse" in techs


def test_random_entry_uses_stimulus():
    i = _eng().random_entry("topic", stimulus="mirror")
    assert "mirror" in i.content


def test_provocation_high_novelty():
    i = _eng().provocation("x")
    assert i.novelty >= 0.85


def test_challenge_assumption():
    i = _eng().challenge_assumption("users always read warnings")
    assert "assumption" in i.content.lower()
    assert i.technique == "challenge"


# -------------------- analogy -------------------- #
def test_analogies_count_and_domains():
    ideas = _eng().analogies("defense", n=3)
    assert len(ideas) == 3
    assert all(i.technique == "analogy" for i in ideas)
    # each frames the topic as a source domain ("«defense» as a {domain}: ...")
    assert all("defense" in i.content and " as a " in i.content for i in ideas)


def test_analogies_respect_n():
    assert len(_eng().analogies("x", n=2)) == 2


# -------------------- blend -------------------- #
def test_blend_combines_concepts():
    i = _eng().blend("chess", "security")
    assert "chess" in i.content and "security" in i.content
    assert i.technique == "blend"


# -------------------- brainstorm + select -------------------- #
def test_brainstorm_multiple_techniques():
    ideas = _eng().brainstorm("topic")
    techs = {i.technique for i in ideas}
    # scamper + lateral + analogy by default
    assert any(t in techs for t in ("substitute", "random_entry", "analogy"))
    assert len(ideas) > 3


def test_brainstorm_with_blend():
    ideas = _eng().brainstorm("a", techniques=[Technique.BLEND], blend_with="b")
    assert any(i.technique == "blend" for i in ideas)


def test_brainstorm_dedups():
    ideas = _eng().brainstorm("topic")
    contents = [i.content for i in ideas]
    assert len(contents) == len(set(contents))


def test_select_returns_top_k():
    eng = _eng()
    ideas = eng.brainstorm("defense")
    top = eng.select(ideas, k=3)
    assert len(top) == 3
    # sorted by novelty-weighted score descending
    keys = [(1 - 0.5) * i.score + 0.5 * i.novelty for i in top]
    assert keys == sorted(keys, reverse=True)


def test_best_returns_one():
    idea = _eng().best("topic")
    assert isinstance(idea, Idea)


# -------------------- reproducibility -------------------- #
def test_reproducible_with_seed():
    a = CreativeEngine(rng=random.Random(42)).brainstorm("x")
    b = CreativeEngine(rng=random.Random(42)).brainstorm("x")
    assert [i.content for i in a] == [i.content for i in b]


# -------------------- generation (offline + llm) -------------------- #
def test_generate_offline_scaffold():
    out = _eng().generate("imagine a new defense")
    assert "scaffold" in out or "defense" in out


def test_write_offline():
    out = _eng().write("a story about loyalty")
    assert "loyalty" in out


def test_code_offline_known_spec_is_real_working_code():
    # offline code-gen now yields REAL working code (via the Capability Foundry recipes),
    # not a NotImplementedError stub, for any spec the foundry recognises.
    code = _eng().code("compute factorial")
    assert "def " in code and "factorial" in code
    assert "NotImplementedError" not in code
    # and it actually computes: exec it and call the generated function
    ns: dict = {}
    exec(code, ns)  # noqa: S102 - generated from a trusted stdlib recipe, test only
    fn = next(v for k, v in ns.items() if k.startswith("compute_factorial"))
    assert fn(5) == 120


def test_code_offline_unknown_spec_is_honest_scaffold():
    # a spec with no matching recipe falls back to a clearly-labelled scaffold
    code = _eng().code("orchestrate a photorealistic dream sequence")
    assert "def " in code
    assert "NotImplementedError" in code


def test_generate_offline_uses_structured_creativity():
    out = _eng().generate("imagine a new defense")
    assert "defense" in out
    assert "structured creativity" in out  # real ideation, not an echo


def test_write_offline_verse_is_themed():
    out = _eng().write("loyalty to the Master", form="haiku")
    assert "loyalty" in out.lower()
    assert out.count("\n") >= 2  # a real multi-line verse, not a placeholder


def test_generate_with_llm():
    from nyxara.kernel.config import NyxaraSettings, Profile
    from nyxara.mind.llm import LLM
    eng = CreativeEngine(llm=LLM(settings=NyxaraSettings.for_profile(Profile.TEST)))
    out = eng.generate("a creative idea")
    assert out  # mock LLM returns something


def test_write_with_llm():
    from nyxara.kernel.config import NyxaraSettings, Profile
    from nyxara.mind.llm import LLM
    eng = CreativeEngine(llm=LLM(settings=NyxaraSettings.for_profile(Profile.TEST)))
    out = eng.write("a haiku")
    assert isinstance(out, str) and out.strip()   # her native own-brain drafts real text, no echo


# -------------------- proposal -------------------- #
def test_propose_returns_proposal():
    prop = _eng().propose("network defense")
    assert isinstance(prop, Proposal)
    assert prop.source_faculty == "creative"
    assert "technique" in prop.metadata
    assert 0.0 <= prop.confidence <= 1.0
