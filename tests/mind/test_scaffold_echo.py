"""The verifier must not mistake the shape of *not answering* for an answer.

Traced from a live turn. The Master typed "Hii" and got back::

    NYXARA: I'm confident that I understand: Hii

with the rationale *"answered by NYXARA's own model (handoff; verifier 0.74)"*. Nothing had failed:
her forged model produced that text, the intrinsic verifier scored it 0.74, the handoff threshold is
0.6, so she spoke it herself — and never asked the 2.5 GB Gemma that was loaded in the same process.

The text is ``kernel/orchestrator._default_reasoner``'s template. That stand-in runs whenever real
reasoning is unavailable, its output lands in her transcripts, her transcripts are the foundry's
training corpus, and her model learned to emit it fluently about anything.

``answer_quality`` could not see it because it scores **form** — length, coherence, unique-word
ratio, and how much of the prompt is parroted. "I understand: Hii" is well-formed on every one:
three unique real words, only one of them from the prompt. The echo penalty was the right idea
aimed one level too low — it catches a model parroting the *prompt*, not one parroting the
*scaffold*, and a fixed prefix dilutes the overlap ratio (33%) below where the penalty starts (60%).
"""
from __future__ import annotations

import pytest

from nyxara.mind.router import answer_quality, strip_scaffold

THRESHOLD = 0.6          # config default: router.threshold


# --------------------------------------------------------------------------- #
# The exact turn that was reported
# --------------------------------------------------------------------------- #
def test_the_reported_turn_no_longer_clears_the_handoff():
    assert answer_quality("Hii", "I understand: Hii") == 0.0


def test_the_same_shape_in_the_masters_own_words():
    assert answer_quality("Kya kar rhi ho", "I understand: Kya kar rhi ho") == 0.0


def test_the_command_branch_of_the_stand_in_is_caught_too():
    """``_default_reasoner`` has two templates; catching only one would leave half the hole."""
    assert answer_quality("delete the logs", "perform: delete the logs") == 0.0


def test_the_ngram_dialogue_continuation_falls_below_the_threshold():
    """The 532-parameter n-gram's signature: continuing a transcript instead of answering."""
    assert answer_quality("Hii", "The Master says: Hii. NYXARA responds: Hii") < THRESHOLD


# --------------------------------------------------------------------------- #
# ...without breaking real answers
# --------------------------------------------------------------------------- #
def test_a_real_answer_still_passes():
    assert answer_quality("Hii", "Hello. I am here. What is the matter") > THRESHOLD


def test_the_prefix_is_not_itself_the_crime():
    """A reply that opens with the same words but then says something real must still pass —
    otherwise the fix trades one silent failure for another."""
    assert answer_quality(
        "Hii", "I understand: you are greeting me, and I am glad to hear from you") > THRESHOLD


def test_a_substantive_answer_is_unaffected():
    assert answer_quality(
        "what is 2+2", "The answer is 4 because two plus two sums to four") > THRESHOLD


@pytest.mark.parametrize("answer", [
    "The capital is Paris.",
    "I don't know — I have no grounded source for that.",
    "Yes. I finished the migration and the tests are green.",
])
def test_ordinary_replies_keep_their_scores(answer):
    assert answer_quality("tell me", answer) > 0.4


# --------------------------------------------------------------------------- #
# The pre-existing contract must survive
# --------------------------------------------------------------------------- #
def test_empty_is_still_zero():
    assert answer_quality("2+2?", "") == 0.0


def test_degenerate_repetition_is_still_low():
    assert answer_quality("2+2?", "the the the the the") < 0.4


def test_a_bare_prompt_echo_is_still_punished():
    assert answer_quality("what is the capital of france",
                          "what is the capital of france") < THRESHOLD


# --------------------------------------------------------------------------- #
# The stripper
# --------------------------------------------------------------------------- #
def test_strip_reports_what_it_removed():
    body, was = strip_scaffold("I understand: Hii")
    assert was is True and body == "Hii"


def test_strip_is_case_insensitive():
    body, was = strip_scaffold("i UNDERSTAND: something")
    assert was is True and body == "something"


def test_strip_leaves_an_ordinary_reply_alone():
    body, was = strip_scaffold("Hello, Master.")
    assert was is False and body == "Hello, Master."


def test_strip_survives_empty_and_none():
    assert strip_scaffold("") == ("", False)
    assert strip_scaffold(None) == ("", False)      # type: ignore[arg-type]


def test_a_template_with_nothing_after_it_is_worthless():
    assert answer_quality("Hii", "I understand:") == 0.0
