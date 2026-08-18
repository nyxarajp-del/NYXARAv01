"""The fact store got slower the more it knew, and that was the ceiling on everything above it.

Measured on a real ``NJPBrain`` before this: one ``ground()`` call against a 20,000-fact store
cost 31.7 ms, of which ``_features`` and ``_ungrounded`` were 97% — both full walks of
``self.facts``, one of them run twice per extracted triple. ``recall()`` went 1.3 ms at 5k facts
to 53.9 ms at 200k. Per-sentence and per-question cost grew with everything she had ever been
told, so no organ above the store could be made faster than the store's size allowed.

**Why these tests count work rather than seconds.** A wall-clock threshold on a shared CI runner
measures the runner, not the code, and fails on a noisy neighbour. So the guard is a ``dict``
subclass that counts full traversals of the store: it is deterministic, it names the exact defect,
and it cannot pass by accident on a fast machine. One timing test remains, with a deliberately
loose ratio, purely to catch a regression that keeps the traversal count at zero while making the
per-call cost quadratic some other way.

The second half of the file is about *equivalence*, because a fast answer that differs from the
answer the scan gave is not an optimisation. Every index here is checked against a re-implemented
full scan rather than against a remembered value.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Set, Tuple

import pytest

from nyxara.njp.grounding import GroundedTriple, Grounder


class _CountingFacts(dict):
    """A fact store that records every attempt to walk the whole of it.

    ``items()`` and ``__iter__`` are the two ways the old code reached every key. ``keys()`` and
    ``values()`` are counted too so a rewrite cannot slip past the instrument by spelling the
    same walk differently.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.walks = 0

    def items(self):  # type: ignore[override]
        self.walks += 1
        return super().items()

    def values(self):  # type: ignore[override]
        self.walks += 1
        return super().values()

    def keys(self):  # type: ignore[override]
        self.walks += 1
        return super().keys()

    def __iter__(self):  # type: ignore[override]
        self.walks += 1
        return super().__iter__()


def _fill(grounder: Grounder, n: int, *, predicate: str = "causes") -> None:
    """Assert ``n`` facts over ``n // 2`` subjects, through the real write path."""
    for i in range(n):
        grounder._assert(GroundedTriple(
            subject=f"thing{i % max(1, n // 2)}", predicate=predicate,
            object=f"other{(i * 7) % max(1, n // 2)}", confidence=0.7, source="test"))


def _instrumented(n: int) -> Tuple[Grounder, _CountingFacts]:
    """A filled grounder whose store counts full traversals from here on."""
    grounder = Grounder()
    _fill(grounder, n)
    counting = _CountingFacts(grounder.facts)
    grounder.facts = counting
    counting.walks = 0
    return grounder, counting


# --------------------------------------------------------------------------- #
# The defect: cost that grows with the store
# --------------------------------------------------------------------------- #

def test_grounding_a_sentence_does_not_walk_the_whole_store():
    """`_features` and `_ungrounded` were 97% of `ground()`, and both were full walks."""
    grounder, counting = _instrumented(2000)
    for i in range(20):
        grounder.ground(f"alpha{i} causes beta{i}")
    assert counting.walks == 0, (
        f"grounding 20 sentences walked the whole fact store {counting.walks} times; "
        "per-sentence cost is proportional to everything she has ever been told")


def test_recall_does_not_walk_the_whole_store():
    """`recall` reached `_candidate_subjects`, `_facts_of` and `_general_affinity`, all scans."""
    grounder, counting = _instrumented(2000)
    for i in range(20):
        grounder.recall(f"what about thing{i}", limit=5)
    assert counting.walks == 0, (
        f"20 recalls walked the whole fact store {counting.walks} times")


def test_answering_a_question_does_not_walk_the_whole_store():
    """The `_CAUSE_OF` path went through `_lookup_inverse`, the last of the six scans."""
    grounder, counting = _instrumented(2000)
    for i in range(20):
        grounder.answer(f"what causes other{i}?")
    assert counting.walks == 0, (
        f"20 answers walked the whole fact store {counting.walks} times")


def test_per_sentence_cost_does_not_grow_with_the_store():
    """The loose backstop: catches a quadratic that keeps the traversal count at zero.

    The ratio is deliberately generous because this one *is* wall-clock. Before the indexes the
    measured ratio between these two store sizes was around twentyfold; anything under four is
    comfortably a flat curve and comfortably above CI noise.
    """
    def per_call_ms(n: int) -> float:
        grounder = Grounder()
        _fill(grounder, n)
        started = time.perf_counter()
        for i in range(20):
            grounder.ground(f"alpha{i} causes beta{i}")
        return (time.perf_counter() - started) / 20 * 1000.0

    small, large = per_call_ms(500), per_call_ms(10_000)
    assert large < max(small * 4.0, 2.0), (
        f"grounding cost {small:.2f} ms/sentence at 500 facts and {large:.2f} ms at 10,000 — "
        "the store is still charging her for what she already knows")


# --------------------------------------------------------------------------- #
# Equivalence: the index must answer what the scan answered
# --------------------------------------------------------------------------- #

def _scan_known(grounder: Grounder) -> Set[str]:
    known = {s for s, _p in dict(grounder.facts)}
    for triples in dict(grounder.facts).values():
        for triple in triples:
            known.add(triple.object.lower())
    return known


def _scan_by_subject(grounder: Grounder) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for subject, predicate in dict(grounder.facts):
        out.setdefault(subject, set()).add(predicate)
    return out


def _scan_features(grounder: Grounder, entity: str) -> List[str]:
    """`_features` as it was written before the index, for a golden comparison."""
    out: Set[str] = set()
    low = entity.lower()
    for (subject, predicate), triples in dict(grounder.facts).items():
        if subject == low:
            out.add(predicate)
        for triple in triples:
            if triple.object.lower() == low:
                out.add(f"{predicate}_of")
    return sorted(out)


def _taught() -> Grounder:
    """A store with several relations per subject, shared objects, and multi-word subjects."""
    grounder = Grounder()
    for subject, predicate, obj in (
            ("blue whale", "is_a", "animal"),
            ("blue whale", "has_property", "large"),
            ("sparrow", "is_a", "animal"),
            ("sparrow", "capable_of", "fly"),
            ("crow", "is_a", "animal"),
            ("crow", "capable_of", "fly"),
            ("aag", "causes", "garmi"),
            ("garmi", "causes", "pasina"),
            ("protecting group", "purpose", "protection")):
        grounder._assert(GroundedTriple(subject=subject, predicate=predicate, object=obj,
                                        confidence=0.8, source="test"))
    return grounder


def test_the_indexes_match_a_full_rescan_after_every_write():
    grounder = _taught()
    assert grounder._known == _scan_known(grounder)
    assert grounder._by_subject == _scan_by_subject(grounder)


@pytest.mark.parametrize("entity", ["animal", "sparrow", "fly", "garmi", "nothing_at_all"])
def test_features_returns_exactly_what_the_full_scan_returned(entity):
    """Golden against a re-implementation, not against a remembered list."""
    grounder = _taught()
    assert grounder._features(entity) == _scan_features(grounder, entity)


def test_facts_of_returns_every_live_fact_and_no_superseded_one():
    grounder = _taught()
    stale = GroundedTriple(subject="sparrow", predicate="is_a", object="rock",
                           confidence=0.2, source="test", superseded=True)
    grounder._assert(stale)
    got = grounder._facts_of("sparrow")
    assert {t.object for t in got} == {"animal", "fly"}
    assert stale not in got


def test_lookup_inverse_finds_every_subject_pointing_at_an_object():
    grounder = _taught()
    got = grounder._lookup_inverse("animal", "is_a")
    assert {t.subject for t in got} == {"blue whale", "sparrow", "crow"}
    assert grounder._lookup_inverse("animal", "capable_of") == []


def test_candidate_subjects_finds_a_multi_word_subject_by_one_of_its_words():
    grounder = _taught()
    got = dict(grounder._candidate_subjects({"whale", "blue"}, 0.35))
    assert "blue whale" in got


def test_the_same_question_ranks_candidates_the_same_way_twice():
    """The index is a set, so the order it yields is not the order facts arrived in."""
    grounder = _taught()
    tokens = {"whale", "sparrow", "crow", "animal"}
    first = grounder._candidate_subjects(tokens, 0.35)
    assert first == grounder._candidate_subjects(tokens, 0.35)


# --------------------------------------------------------------------------- #
# The two subtle cases
# --------------------------------------------------------------------------- #

def test_a_superseded_fact_still_counts_as_known():
    """Supersession is not deletion, and `_ungrounded` never filtered on it.

    A word she was told and later corrected about is still a word she has met. Treating a
    retraction as amnesia would put it back on the gap list and have her ask about it again.
    """
    grounder = Grounder()
    grounder._assert(GroundedTriple(subject="jay", predicate="has_name", object="jay",
                                    confidence=0.9, source="test", superseded=True))
    assert grounder._ungrounded(["jay"]) == []


def test_load_dict_rebuilds_the_indexes_a_fresh_store_would_have_built():
    """A reload appends straight into `self.facts`, so nothing maintains the indexes on the way."""
    original = _taught()
    restored = Grounder()
    restored.load_dict(original.to_dict())
    assert restored._known == _scan_known(restored)
    assert restored._by_subject == _scan_by_subject(restored)
    assert restored._features("animal") == _scan_features(restored, "animal")
    assert {t.subject for t in restored._lookup_inverse("animal", "is_a")} == {
        "blue whale", "sparrow", "crow"}


def test_a_word_she_has_met_is_not_reported_as_a_gap():
    grounder = _taught()
    assert grounder._ungrounded(["sparrow", "garmi", "unheard_of"]) == ["unheard_of"]


def test_the_gap_tally_is_bounded():
    """It is a tally of what she could not reach, which on a corpus is an unbounded set."""
    grounder = Grounder()
    for i in range(6000):
        grounder._ungrounded([f"unknown{i}"], tally=True)
    assert len(grounder._ungrounded_seen) <= 4096
