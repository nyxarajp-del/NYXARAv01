"""Tests for nyxara.cognition.sample_efficient — the coordinator faculty + its wiring."""

from __future__ import annotations

from nyxara.cognition.sample_efficient import SampleEfficientMind
from nyxara.memory.store import MemoryStore
from nyxara.mind.lot import Lambda, const, func, pred, var


def _mind():
    return SampleEfficientMind(store=MemoryStore())


# -------------------- single teaching entry point -------------------- #
def test_teach_concept_and_classify():
    mind = _mind()
    mind.teach("weather", "the weather is sunny and warm today", kind="concept")
    mind.teach("cooking", "i love eating pizza and pasta for dinner", kind="concept")
    label, conf = mind.classify("will the weather be rainy today")
    assert label == "weather"


def test_teach_episodic_and_recall():
    mind = _mind()
    mind.teach("what is my name", "your name is JP", kind="episodic")
    assert mind.recall_once("what is my name again") == "your name is JP"


def test_teach_lexeme_and_compose():
    mind = _mind()
    mind.teach("jump", const("JUMP"), kind="lexeme")
    mind.teach("twice", Lambda(var("a"), func("seq", var("a"), var("a"))), kind="lexeme")
    assert str(mind.interpret("jump twice")) == "seq(JUMP, JUMP)"


def test_unknown_teach_kind_raises():
    import pytest
    with pytest.raises(ValueError):
        _mind().teach("x", "y", kind="nonsense")


def test_lexeme_must_be_a_term():
    import pytest
    with pytest.raises(ValueError):
        _mind().teach("jump", "not a term", kind="lexeme")


# -------------------- abstraction passthrough -------------------- #
def test_induce_schema():
    mind = _mind()
    schema = mind.induce([pred("parent", const("tom"), const("bob")),
                          pred("parent", const("ann"), const("sue"))])
    assert schema is not None and len(schema.variables) == 2


# -------------------- grounding block -------------------- #
def test_as_prompt_combines_concept_and_episodic():
    mind = _mind()
    mind.teach("weather", "the weather is sunny and warm today", kind="concept")
    mind.teach("what is my secret", "the secret is forty two", kind="episodic")
    block = mind.as_prompt("what is my secret and the weather today")
    assert "forty two" in block or "weather" in block


def test_as_prompt_empty_when_nothing_learned():
    assert _mind().as_prompt("anything at all") == ""


# -------------------- consolidation closes the loop -------------------- #
def test_consolidate_forms_templates():
    mind = _mind()
    mind.teach("greet", "hello there friend", kind="concept")
    mind.teach("greet", "hello there buddy", kind="concept")
    report = mind.consolidate()
    assert report.concepts_seen >= 1
    # two same-length exemplars share a skeleton → a template with a slot
    assert "greet" in report.templates


def test_stats():
    mind = _mind()
    mind.teach("a", "alpha beta gamma", kind="concept")
    mind.teach("cue", "response", kind="episodic")
    mind.teach("jump", const("JUMP"), kind="lexeme")
    stats = mind.stats()
    assert stats["concepts"] == 1
    assert stats["episodic_pairs"] == 1
    assert stats["lexemes"] == 1


# -------------------- orchestrator wiring -------------------- #
def test_orchestrator_wires_faculty_and_grounds_one_shot():
    from nyxara.kernel.orchestrator import NyxaraCore

    core = NyxaraCore()
    assert core.sample_efficient is not None
    core.sample_efficient.teach("what is my secret code",
                                "the secret code is delta seven", kind="episodic")
    grounds = core._recall_for("please remind me what is my secret code again")
    texts = [g.text() for g in grounds if hasattr(g, "text")]
    assert any("delta seven" in t for t in texts)


def test_orchestrator_registers_tools():
    from nyxara.kernel.orchestrator import NyxaraCore

    core = NyxaraCore()
    names = core.tools.names()
    for tool in ("teach_concept", "classify_concept", "induce_schema", "compose_interpret"):
        assert tool in names
    # the compose tool interprets a novel combination of known primitives
    res = core.tools.get("compose_interpret").handler("jump twice")
    assert res["meaning"] == "seq(JUMP, JUMP)"
