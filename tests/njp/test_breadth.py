"""How much of the world she has anything to say about (NJP V.41).

This file is mostly about not letting the measurement flatter her. Two of the three papers here
produced a meaningless number on their first run and both were the harness, not her.
"""
from __future__ import annotations

import gzip
import json

import pytest

from nyxara.njp.breadth import Breadth, measure, render, sample_subjects


class T:
    __slots__ = ("object", "confidence", "superseded")

    def __init__(self, obj):
        self.object, self.confidence, self.superseded = obj, 1.0, False


class Store:
    def __init__(self, rows=()):
        self.facts = {}
        for a, r, b in rows:
            self.facts.setdefault((a.lower(), r), []).append(T(b))

    def _key(self, text):
        return " ".join(str(text or "").split()).lower()

    def answer(self, question):
        subject = question.lower().replace("what is", "").strip(" ?")
        got = self.facts.get((subject, "is_a"), ())
        return type("A", (), {"text": got[0].object if len(got) == 1 else ""})()


class Brain:
    def __init__(self, rows=()):
        self.grounder = Store(rows)
        self.core = None


ROWS = [("cat", "is_a", "mammal"), ("dog", "is_a", "mammal"), ("mammal", "is_a", "animal")]


# --------------------------------------------------------------------------- #
# Coverage is breadth and nothing else
# --------------------------------------------------------------------------- #
def test_coverage_counts_having_any_fact_at_all():
    got = Breadth(Brain(ROWS)).coverage(["cat", "dog", "keloid", "minniebush"])
    assert got.asked == 4 and got.hit == 2 and got.score == 0.5
    assert "keloid" in got.examples_missed


def test_coverage_is_meaningless_when_the_sample_comes_from_what_was_ingested():
    """The first run scored 1.000 by sampling subjects out of the file it had just loaded.

    Nothing in the module can prevent that — it is a property of how it is *called* — so this test
    exists to state the trap in a place a future caller will read, and to pin the arithmetic that
    makes it obvious: every subject present scores, so a self-sample always scores 1.000.
    """
    got = Breadth(Brain(ROWS)).coverage([a for a, _r, _o in ROWS])
    assert got.score == 1.0, "a self-sample is a tautology, not a measurement"


# --------------------------------------------------------------------------- #
# Reachable asks only about what she has
# --------------------------------------------------------------------------- #
def test_reachable_only_asks_about_subjects_she_holds():
    """A miss must mean the question failed, not that there was no fact."""
    got = Breadth(Brain(ROWS)).reachable(["cat", "keloid", "minniebush"])
    assert got.asked == 1 and got.hit == 1


def test_reachable_records_what_was_said_when_it_misses():
    rows = ROWS + [("cat", "is_a", "pet")]        # two kinds: the answer refuses to choose
    got = Breadth(Brain(rows)).reachable(["cat"])
    assert got.asked == 1 and got.hit == 0
    assert got.examples_missed and "silence" in got.examples_missed[0]


# --------------------------------------------------------------------------- #
# Derived refuses to measure what was not held out
# --------------------------------------------------------------------------- #
def test_derived_drops_any_item_she_was_told():
    """The rule every held-out paper in this package rests on."""
    got = Breadth(Brain(ROWS)).derived([("cat", "is_a", "mammal")])
    assert got.asked == 0, "an item she was told outright is not held out"


def test_derived_drops_any_subject_she_has_never_heard_of():
    """Not a derivation failure — there is nothing to derive from."""
    got = Breadth(Brain(ROWS)).derived([("keloid", "is_a", "scar")])
    assert got.asked == 0


def test_derived_asks_where_the_first_hop_is_present_and_the_answer_is_not():
    got = Breadth(Brain(ROWS)).derived([("cat", "is_a", "animal")])
    assert got.asked == 1


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def test_sampling_is_one_pass_and_deterministic_per_seed(tmp_path):
    path = tmp_path / "src.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for i in range(500):
            handle.write(json.dumps({"subject": f"s{i}", "predicate": "is_a",
                                     "object": f"o{i}"}) + "\n")
    first, triples = sample_subjects(str(path), count=20, seed=7)
    again, _ = sample_subjects(str(path), count=20, seed=7)
    other, _ = sample_subjects(str(path), count=20, seed=8)
    assert first == again and first != other
    assert len(first) == 20 and len(triples) == 20


def test_the_report_names_what_is_not_known():
    report = measure(Brain(ROWS), names=["cat", "keloid"], source="somewhere")
    text = render(report)
    assert "keloid" in text and "coverage" in text and "somewhere" in text


# --------------------------------------------------------------------------- #
# The shipped broad corpus
# --------------------------------------------------------------------------- #
def test_the_broad_corpus_is_shipped_and_is_not_loaded_by_default():
    import os

    from nyxara.njp import general

    here = os.path.join(os.path.dirname(os.path.abspath(general.__file__)), "data")
    assert os.path.exists(os.path.join(here, general._BROAD))
    brain = general.load_brain()
    facts = sum(len(v) for v in brain.grounder.facts.values())
    assert facts < 30_000, "the curated corpus is the default"


def test_the_broad_corpus_multiplies_what_she_has_heard_of():
    from nyxara.njp.general import load_brain

    narrow = load_brain()
    wide = load_brain(broad=True)
    assert len({s for s, _p in wide.grounder.facts}) > \
        10 * len({s for s, _p in narrow.grounder.facts})
