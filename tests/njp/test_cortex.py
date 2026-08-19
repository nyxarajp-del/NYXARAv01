"""The teacher organ: absent honestly, cheap to ask about, and it never eats a reasoning trace."""

from __future__ import annotations

from nyxara.njp.cortex import Cortex, split_think


def test_absent_weights_report_the_reason_and_the_fix():
    cortex = Cortex()
    assert cortex.available() is False
    stats = cortex.stats()
    assert stats["present"] is False
    assert "fetch_qwythos_model.py" in stats["reason"]


def test_availability_never_loads_an_engine():
    """It is called on every turn. Loading 5.6 GB to answer 'are you there' would make the check
    more expensive than the answer."""
    cortex = Cortex()
    for _ in range(50):
        cortex.available()
    assert cortex._ENGINES == {}


def test_a_dead_runtime_latches_instead_of_failing_every_turn():
    cortex = Cortex()
    cortex._dead = "llama_cpp could not load the model: simulated"
    assert cortex.available() is False
    assert "simulated" in cortex.why_unavailable()


def test_asking_an_absent_teacher_is_an_empty_reply_not_a_crash():
    reply = Cortex().ask("what causes rain?")
    assert reply.ok is False
    assert reply.text == ""


def test_reasoning_is_stripped_from_the_answer_and_kept_on_the_reply():
    """njp/genome.py mines these traces for recurring reasoning shapes. Stripping is a rendering
    decision; discarding would throw away the most useful thing the teacher produces."""
    answer, reasoning, truncated = split_think(
        "<think>first this\nthen that</think>The bar bends because the metal expands.")
    assert answer == "The bar bends because the metal expands."
    assert "first this" in reasoning
    assert truncated is False


def test_a_reply_cut_off_mid_reasoning_yields_no_answer_rather_than_the_monologue():
    """The token budget can stop a generation inside <think>. A strip that requires both tags
    leaves it untouched — and hands the Master the raw monologue as though it were the answer."""
    answer, reasoning, truncated = split_think("<think>I am still working through")
    assert answer == ""
    assert truncated is True
    assert "still working" in reasoning


def test_text_without_reasoning_passes_through_unchanged():
    answer, reasoning, truncated = split_think("just an answer")
    assert (answer, reasoning, truncated) == ("just an answer", "", False)


def test_the_organ_states_its_authority_on_every_report():
    cortex = Cortex()
    cortex.calls = 1                       # force the full report shape
    assert cortex.stats()["authority"] == "proposal_only"


def test_the_working_context_and_the_1m_ceiling_are_both_reported():
    """The 1M window is real and it is not what is allocated. Reporting only one of them would
    make an honest capability read as a claim about this host."""
    cortex = Cortex()
    cortex.calls = 1
    stats = cortex.stats()
    assert stats["context_tokens"] < stats["context_ceiling"]
    assert stats["context_ceiling"] == 1_048_576
