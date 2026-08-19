"""The one rule the whole distillation design rests on: the teacher may propose, never grade.

If a teacher's output is ever allowed to score NJP's prediction, the training signal becomes
"agree with the teacher", she converges on its answers including its errors, and
``eval/teacher_removal.py`` passes trivially — because she has become a lossy copy of the thing
that was removed. The experiment would report success in exactly the case where nothing was
learned. These tests exist so that stays impossible rather than merely intended.
"""

from __future__ import annotations

import pytest

from nyxara.njp.adversary import SelfAttacker
from nyxara.njp.concepts import ConceptGenesis
from nyxara.njp.shadow import Gap, GapKind, ShadowCognition, ShadowReading

TEACHER = "Heat causes expansion, and expansion causes bending in the metal lattice."


def _shadow(**kw):
    kw.setdefault("concepts", ConceptGenesis())
    kw.setdefault("adversary", SelfAttacker())
    return ShadowCognition(**kw)


def test_a_reading_has_no_outcomes_and_says_so():
    reading = _shadow().compare("why?", teacher_text=TEACHER, njp_text="it changed")
    assert reading.outcomes == ()
    assert reading.to_dict()["outcomes"] == 0
    assert reading.gaps, "the comparison must actually have found something to be meaningful"


def test_every_gap_is_an_investigation_and_cannot_be_flagged_otherwise():
    reading = _shadow().compare("why?", teacher_text=TEACHER, njp_text="it changed")
    for gap in reading.investigations:
        assert gap.is_outcome is False


def test_is_outcome_is_not_a_constructor_argument():
    """It is a type-level statement, not a default anyone can pass around."""
    with pytest.raises(TypeError):
        Gap(kind=GapKind.SEMANTIC, is_outcome=True)      # type: ignore[call-arg]


def test_the_reading_exposes_no_writable_outcome_channel():
    assert isinstance(ShadowReading.outcomes, property)
    assert ShadowReading.outcomes.fset is None, "a read-only property, with no setter to find"
    with pytest.raises(AttributeError):
        ShadowReading().outcomes = ("anything",)          # type: ignore[misc]


def test_a_teacher_causal_claim_is_challenged_not_believed():
    """Against an empty record the four attacks cannot settle anything, and UNDECIDED is not a
    pass — the claim simply does not enter."""
    shadow = _shadow()
    reading = shadow.compare("why?", teacher_text=TEACHER, njp_text="")
    causal = reading.by_kind(GapKind.CAUSAL)
    assert causal, "the teacher asserted causes; they must have been extracted"
    assert reading.claims_challenged == len(causal)
    assert reading.claims_undecided == len(causal)
    assert reading.claims_refuted == 0


def test_accepted_on_a_causal_gap_means_challenged_not_believed():
    """The counter must never be readable as 'how many teacher claims she took on trust'."""
    reading = _shadow().compare("why?", teacher_text=TEACHER, njp_text="")
    for gap in reading.by_kind(GapKind.CAUSAL):
        assert gap.accepted is True
        assert "refuted=" in gap.detail and "undecided=" in gap.detail


def test_without_an_adversary_a_causal_claim_is_explicitly_not_accepted():
    shadow = ShadowCognition(concepts=ConceptGenesis(), adversary=None)
    reading = shadow.compare("why?", teacher_text=TEACHER, njp_text="")
    for gap in reading.by_kind(GapKind.CAUSAL):
        assert gap.accepted is False
        assert "NOT accepted" in gap.detail


def test_stats_publishes_that_it_produced_no_outcomes():
    shadow = _shadow()
    shadow.compare("why?", teacher_text=TEACHER, njp_text="")
    assert shadow.stats()["outcomes_produced"] == 0


def test_an_absent_teacher_is_reported_not_faked():
    reading = ShadowCognition().compare("why?", teacher_text="")
    assert reading.teacher_present is False
    assert reading.gaps == []


def test_a_semantic_comparison_against_the_same_model_is_skipped_not_reported_as_agreement():
    """``gguf`` leads ``LLM._AUTO_LADDER``, so where the Qwythos weights are present her voice is
    rendered by the very model that is the teacher. Comparing that wording to its own wording
    finds almost nothing missing — which reads as "she already knows all of this" and would make
    the distillation look finished on day one."""
    shadow = _shadow()
    honest = shadow.compare("q", teacher_text=TEACHER, njp_text="it changed")
    circular = shadow.compare("q", teacher_text=TEACHER, njp_text="it changed",
                              njp_rendered_by="gguf")

    assert honest.same_model is False
    assert honest.by_kind(GapKind.SEMANTIC)[0].magnitude > 0

    assert circular.same_model is True
    skipped = circular.by_kind(GapKind.SEMANTIC)[0]
    assert skipped.magnitude == 0.0
    assert skipped.routed_to == "", "a skipped comparison must not be routed anywhere"
    assert "measures the model against itself" in skipped.detail
    assert shadow.stats()["semantic_skipped_as_circular"] == 1


def test_the_circularity_guard_arms_itself_from_the_live_ladder():
    """``gguf`` leads ``_AUTO_LADDER``, so wherever the Qwythos weights exist her voice IS the
    teacher. A guard that only works when a caller remembers to pass the rendering provider is a
    guard that is off in every path nobody updated — so it resolves the provider itself."""
    class _Surface:
        provider = "gguf"

    class _Voice:
        def surface(self):
            return _Surface()

    shadow = ShadowCognition(concepts=ConceptGenesis(), voice=_Voice())
    reading = shadow.compare("q", teacher_text=TEACHER, njp_text="it changed")  # nothing passed

    assert reading.same_model is True
    assert reading.by_kind(GapKind.SEMANTIC)[0].magnitude == 0.0
    assert shadow.stats()["semantic_skipped_as_circular"] == 1


def test_a_different_rendering_provider_does_not_trip_the_circularity_guard():
    shadow = _shadow()
    reading = shadow.compare("q", teacher_text=TEACHER, njp_text="it changed",
                             njp_rendered_by="native")
    assert reading.same_model is False
    assert shadow.stats()["semantic_skipped_as_circular"] == 0
