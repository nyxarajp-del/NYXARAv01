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
    from nyxara.mind import gguf_assets

    gguf_assets.reset_engines()
    cortex = Cortex()
    for _ in range(50):
        cortex.available()
    assert gguf_assets._ENGINES == {}


def test_the_engine_cache_is_shared_with_the_ladder_rung_not_private():
    """The same weights serve three consumers — the `gguf` rung in mind/llm.py, this organ, and
    njp/graft.py's tensor source. A cache per consumer holds a 5.6-17.9 GB file in memory three
    times over for one copy on disk."""
    from nyxara.mind import gguf_assets
    from nyxara.mind.llm import GGUFProvider

    assert not hasattr(Cortex, "_ENGINES"), "the organ must not keep a private engine cache"
    assert not hasattr(GGUFProvider, "_ENGINES")
    assert hasattr(gguf_assets, "load_engine")


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


def test_a_closing_tag_without_an_opening_one_is_the_shape_the_real_model_produces():
    """Qwen3.5's chat template opens the reasoning block in the PROMPT, so generation starts
    already inside it and the model only ever emits ``</think>``. Measured against the real
    weights: ``contains <think>: False | contains </think>: True``.

    A splitter that requires the opening tag matches nothing here. The Master then receives the
    entire raw monologue as the answer and njp/genome.py gets no trace to mine — both halves of
    the design failing silently while every field on the reply looks populated."""
    answer, reasoning, truncated = split_think(
        "I need to work this out.\nStep 1: 2+2=4.</think>The answer is 4.")
    assert answer == "The answer is 4."
    assert "Step 1" in reasoning
    assert truncated is False


def test_an_unmatched_closing_tag_is_complete_not_truncated():
    """Order matters between the two unmatched cases: an orphan ``</think>`` is a COMPLETE block
    whose opener the template consumed, while an orphan ``<think>`` is a broken one. Checking
    truncation first would mark every real reply truncated."""
    _a, _r, truncated = split_think("reasoning</think>answer")
    assert truncated is False


def test_untagged_output_means_opposite_things_under_the_two_template_conventions():
    """The same bytes, two readings, and only a DECLARED fact can tell them apart.

    Under a pre-opened template — this model's — generation starts inside the reasoning block, so
    a reply carrying no closing tag never left it: the whole thing is reasoning and there is no
    answer yet. Under an ordinary template the same bytes are an answer with no reasoning.

    Measured on the first live shadow run, before this existed: three replies, three reasoning
    monologues returned as finished answers, ``reasoning_chars = 0`` on every one. It is the
    original think-tag bug in a disguise with no syntactic marker to notice."""
    monologue = "The user asks for a two-sentence explanation of why salt changes boiling point."

    answer, reasoning, truncated = split_think(monologue, preopened=True)
    assert answer == ""
    assert reasoning == monologue
    assert truncated is True

    answer, reasoning, truncated = split_think(monologue, preopened=False)
    assert answer == monologue
    assert reasoning == ""
    assert truncated is False


def test_a_completed_reply_splits_identically_under_either_convention():
    """`preopened` must only change the reading of the AMBIGUOUS case. A closing tag is
    unambiguous, and a flag that also moved those would be a flag that hides bugs."""
    for pre in (False, True):
        assert split_think("thinking</think>Four.", preopened=pre) == ("Four.", "thinking", False)
        assert split_think("<think>t</think>Four.", preopened=pre) == ("Four.", "t", False)


def test_text_without_reasoning_passes_through_unchanged():
    # Default convention (no pre-opened block): untagged text is simply an answer.
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
