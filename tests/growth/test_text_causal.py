"""Tests for nyxara.growth.text_causal — prose becomes claims, and contradictions get refused."""

from __future__ import annotations

import pytest

from nyxara.causal.causal_knots import CONCORDANT, DISCORDANT, _norm_sign
from nyxara.growth.dataset_causal import learn_causal_graph
from nyxara.growth.text_causal import (_normalize_label, claims_to_graph, extract_claims,
                                       learn_from_text)
from nyxara.mind.causal_world_model import CausalWorldModel


def _pairs(report):
    return {(c.cause, c.effect, c.polarity) for c in report.claims}


# -------------------- the cue grammar -------------------- #
def test_each_cue_family_is_extracted_with_the_right_direction():
    cases = [
        ("Heavy rainfall causes flooding.", "heavy rainfall", "flooding", 1, "causes"),
        # only EDGE function words are trimmed, so the interior "is" survives — labels are
        # normalised, not parsed
        ("The ground is slippery because of the rain.", "rain", "ground is slippery", 1,
         "because_of"),
        ("If the dam fails, then the town floods.", "dam fails", "town floods", 1, "if_then"),
        ("Vaccination prevents severe illness.", "vaccination", "severe illness", -1, "prevents"),
        ("Exercise increases stamina.", "exercise", "stamina", 1, "increases"),
        ("The pressure dropped, therefore the valve closed.", "pressure dropped", "valve closed",
         1, "therefore"),
    ]
    for sentence, cause, effect, polarity, cue in cases:
        claims, _n, _d = extract_claims(sentence)
        assert claims, f"nothing extracted from {sentence!r}"
        claim = claims[0]
        assert (claim.cause, claim.effect, claim.polarity, claim.cue) == \
               (cause, effect, polarity, cue), f"wrong reading of {sentence!r}"


def test_effect_first_cues_do_not_reverse_the_arrow():
    """"Y because of X" means X→Y. Reading it in surface order would invert every such claim."""
    claims, _n, _d = extract_claims("The flooding happened because of heavy rainfall.")
    assert (claims[0].cause, claims[0].effect) == ("heavy rainfall", "flooding happened")


def test_concessive_and_denial_sentences_are_defeated():
    """These contain the causal surface pattern but assert the opposite — the concession wins."""
    text = ("Despite the storm, the harvest caused no losses. "
            "There is no evidence that coffee causes cancer. "
            "Sugar does not cause hyperactivity. "
            "Although rain causes mud, the path stayed dry.")
    claims, n_sentences, defeated = extract_claims(text)
    assert defeated == n_sentences
    assert claims == []


def test_a_sentence_yields_at_most_one_claim():
    claims, _n, _d = extract_claims("Rain causes flooding because of poor drainage.")
    assert len(claims) == 1


def test_self_loops_are_never_claimed():
    assert extract_claims("Rain causes rain.")[0] == []


def test_claims_carry_a_traceable_quote_and_source():
    claims, _n, _d = extract_claims("Smoking leads to lung damage.", source="paper.txt")
    assert claims[0].quote == "Smoking leads to lung damage."
    assert claims[0].source == "paper.txt"


# -------------------- label normalisation -------------------- #
def test_labels_drop_edge_function_words_so_paraphrases_merge():
    assert _normalize_label("the heavy rain") == "heavy rain"
    assert _normalize_label("Heavy Rain.") == "heavy rain"
    assert _normalize_label("of the") == ""


# -------------------- the contradiction gate -------------------- #
def test_a_direct_contradiction_is_refused_not_absorbed():
    report = learn_from_text("Exercise increases stamina. Exercise reduces stamina.")
    assert len(report.claims) == 1
    assert report.contradicted == 1
    assert report.contradictions


def test_a_transitive_contradiction_is_caught_across_a_chain():
    """The lattice reasons over the whole component, not just the pair in front of it."""
    report = learn_from_text("Rain increases moisture. Moisture increases mould. "
                             "Rain reduces mould.")
    assert _pairs(report) == {("rain", "moisture", 1), ("moisture", "mould", 1)}
    assert report.contradicted == 1


def test_consistent_claims_all_survive():
    report = learn_from_text("Rain increases moisture. Moisture increases mould. "
                             "Rain increases mould.")
    assert report.contradicted == 0
    assert len(report.claims) == 3


def test_norm_sign_handles_bool_before_int():
    """bool subclasses int, so False would otherwise take the numeric branch, satisfy 0 >= 0 and
    come back CONCORDANT — silently turning "A prevents B" into "A causes B"."""
    assert _norm_sign(False) == DISCORDANT
    assert _norm_sign(True) == CONCORDANT
    assert _norm_sign(-1) == DISCORDANT
    assert _norm_sign(1) == CONCORDANT


# -------------------- registration -------------------- #
def test_registered_claims_carry_provenance_and_a_capped_confidence():
    model = CausalWorldModel()
    report = learn_from_text("Smoking leads to lung damage.", model=model, source="paper.txt")
    assert report.registered == 1

    link = model._links[("smoking", "lung damage")]
    assert link.evidence["source"] == "text_claim"
    assert link.evidence["cue"] == "causes"
    assert link.evidence["quote"] == "Smoking leads to lung damage."
    assert link.confidence <= 0.35        # an assertion must never outrank a measurement


def test_measured_edges_are_never_overwritten_by_an_assertion():
    """A table-learned edge outranks a sentence claiming the same pair."""
    import random

    rng = random.Random(0)
    rows = [{"smoking": (s := rng.gauss(0, 1)), "lung damage": 2.0 * s + rng.gauss(0, 0.2)}
            for _ in range(60)]
    model = CausalWorldModel()
    learn_causal_graph(["smoking", "lung damage"], rows, model=model,
                       order=["smoking", "lung damage"])
    measured = model._links[("smoking", "lung damage")]

    learn_from_text("Smoking leads to lung damage.", model=model)
    assert model._links[("smoking", "lung damage")] is measured
    assert model._links[("smoking", "lung damage")].evidence["source"] == "dataset_table"


def test_empty_text_is_safe():
    report = learn_from_text("")
    assert report.claims == [] and report.sentences == 0


def test_claims_to_graph_without_a_lattice_still_reports(monkeypatch):
    claims, _n, _d = extract_claims("Rain causes flooding.")
    report = claims_to_graph(claims, model=CausalWorldModel(), lattice=None)
    assert len(report.claims) == 1


# -------------------- provenance must survive into the answer -------------------- #
def test_an_asserted_link_never_reads_like_a_measured_one():
    """Text is the medium in which false causal claims travel. Once prose can register edges,
    anything that reports one has to say which kind it is — otherwise a corpus's folklore becomes
    her confident opinion."""
    model = CausalWorldModel()
    learn_from_text("Vaccines cause autism.", model=model)

    link = model.why("autism")[0]
    assert link.measured is False
    described = link.describe()
    assert "ASSERTED IN TEXT, not measured" in described
    assert "Vaccines cause autism." in described


def test_a_measured_link_carries_no_such_warning():
    import random

    rng = random.Random(0)
    rows = []
    for _ in range(60):
        rain = rng.gauss(0, 1)
        rows.append({"rain": rain, "flood": 2.0 * rain + rng.gauss(0, 0.2)})
    model = CausalWorldModel()
    learn_causal_graph(["rain", "flood"], rows, model=model, order=["rain", "flood"])

    link = model.why("flood")[0]
    assert link.measured is True
    assert "ASSERTED" not in link.describe()


# --------------------------------------------------------------------------- #
# Retractions — a speaker correcting themselves is not a contradiction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sentence", [
    "Actually caffeine reduces my focus.",
    "Actually, caffeine reduces my focus.",
    "I was wrong, caffeine reduces my focus.",
    "Correction: caffeine reduces my focus.",
    "No wait, caffeine reduces my focus.",
    "My mistake, caffeine reduces my focus.",
    "On second thought, caffeine reduces my focus.",
    "I take that back, caffeine reduces my focus.",
    "Maine galat kaha, caffeine reduces my focus.",
    "मैंने ग़लत कहा, caffeine reduces my focus.",
])
def test_a_retraction_yields_no_claim(sentence):
    from nyxara.growth.text_causal import extract_claims, is_correction

    assert is_correction(sentence) is True
    claims, _sentences, defeated = extract_claims(sentence)
    assert claims == [] and defeated == 1


@pytest.mark.parametrize("sentence", [
    "Caffeine reduces my focus.",
    "The drug actually reduces my focus.",        # mid-sentence filler, not a retraction
    "Rain causes flooding.",
])
def test_an_ordinary_assertion_is_not_a_retraction(sentence):
    from nyxara.growth.text_causal import extract_claims, is_correction

    assert is_correction(sentence) is False
    claims, _sentences, _defeated = extract_claims(sentence)
    assert len(claims) == 1


def test_a_retraction_does_not_trip_the_contradiction_gate():
    """"X causes Y" then "actually X prevents Y" is one speaker updating, not a clash."""
    from nyxara.growth.text_causal import learn_from_text

    report = learn_from_text("Caffeine improves my focus. "
                             "Actually caffeine reduces my focus.")
    assert report.contradicted == 0
    assert report.corrections == 1
    assert len(report.claims) == 1          # the first claim stands, unchallenged


def test_a_real_contradiction_is_still_refused():
    from nyxara.growth.text_causal import learn_from_text

    report = learn_from_text("Caffeine improves my focus. Caffeine reduces my focus.")
    assert report.contradicted == 1
    assert report.corrections == 0


def test_the_report_counts_concessions_and_retractions_apart():
    from nyxara.growth.text_causal import learn_from_text

    report = learn_from_text("Rain causes flooding. "
                             "Despite the rain, flooding causes damage. "
                             "Actually rain prevents flooding.")
    assert report.defeated == 1 and report.corrections == 1
    assert report.to_dict()["corrections"] == 1
